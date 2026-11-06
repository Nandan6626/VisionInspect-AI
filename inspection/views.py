import json
import logging
import os
from datetime import timedelta

from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.db.models import Count, Avg, Q
from django.db.models.functions import TruncDate
from django.utils import timezone
from django.conf import settings

from .models import Inspection
from .forms import ImageUploadForm
from .services.image_validator import (
    get_image_validator,
    INVALID_IMAGE_USER_MESSAGE,
    UNCERTAIN_IMAGE_USER_MESSAGE,
)
from .services.patchcore_service import get_patchcore_service
from .services.anomaly_visualizer import AnomalyVisualizer
from .services.pdf_generator import PDFReportGenerator

logger = logging.getLogger(__name__)


def dashboard_view(request):
    """
    Industrial Quality Dashboard displaying overall PatchCore statistics,
    normal vs anomalous distribution, inspection trends, and recent telemetry.
    """
    valid_inspections = Inspection.objects.filter(validation_status='VALID')
    total_inspections = valid_inspections.count()
    total_normal = valid_inspections.filter(inspection_status='NORMAL').count()
    total_anomalous = valid_inspections.filter(inspection_status='ANOMALOUS').count()
    total_rejected = Inspection.objects.filter(validation_status='INVALID').count()
    total_uncertain = Inspection.objects.filter(validation_status='UNCERTAIN').count()

    anomaly_percentage = round((total_anomalous / total_inspections * 100), 1) if total_inspections > 0 else 0.0
    avg_score = valid_inspections.aggregate(Avg('anomaly_score'))['anomaly_score__avg'] or 0.0

    recent_inspections = Inspection.objects.all().order_by('-created_at')[:8]

    # Ratio Data for Chart.js
    ratio_labels = ['Normal Components', 'Potential Anomalies', 'Rejected (Non-Industrial)']
    ratio_data = [total_normal, total_anomalous, total_rejected]

    # Inspection Trends (Last 7 Days)
    seven_days_ago = timezone.now() - timedelta(days=7)
    daily_stats = (
        Inspection.objects.filter(created_at__gte=seven_days_ago)
        .annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(
            total=Count('id'),
            normal=Count('id', filter=Q(inspection_status='NORMAL')),
            anomalous=Count('id', filter=Q(inspection_status='ANOMALOUS'))
        )
        .order_by('date')
    )

    trend_dates = [s['date'].strftime('%b %d') for s in daily_stats]
    trend_normal = [s['normal'] for s in daily_stats]
    trend_anomalous = [s['anomalous'] for s in daily_stats]

    context = {
        'total_inspections': total_inspections,
        'total_normal': total_normal,
        'total_anomalous': total_anomalous,
        'total_rejected': total_rejected,
        'total_uncertain': total_uncertain,
        'anomaly_percentage': anomaly_percentage,
        'avg_score': round(avg_score, 3),
        'recent_inspections': recent_inspections,
        'ratio_labels_json': json.dumps(ratio_labels),
        'ratio_data_json': json.dumps(ratio_data),
        'trend_dates_json': json.dumps(trend_dates),
        'trend_normal_json': json.dumps(trend_normal),
        'trend_anomalous_json': json.dumps(trend_anomalous),
        'has_inspections': total_inspections > 0,
    }
    return render(request, 'inspection/dashboard.html', context)


def inspect_upload_view(request):
    """
    Handles industrial component upload and PatchCore inspection pipeline:
    1. Validates file format and dimensions.
    2. Runs OpenCLIP Vision-Language industrial validation.
    3. Rejects non-industrial photos (portraits, buildings, animals, food, etc.).
    4. Runs PatchCore anomaly detection on valid industrial images.
    5. Generates pixel-level anomaly heatmaps and side-by-side overlay comparisons.
    6. Stores audit record and redirects to results.
    """
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['image']

            # Create Inspection instance to persist image safely
            inspection = Inspection(
                original_filename=uploaded_file.name,
                uploaded_image=uploaded_file,
                validation_status='VALID',
                inspection_status='NORMAL'
            )
            inspection.save()

            image_path = inspection.uploaded_image.path

            # Step 1: Industrial Image Validation Layer
            validator = get_image_validator()
            val_res = validator.validate(image_path)

            if not val_res.is_valid:
                # Handle REJECTED or UNCERTAIN image
                inspection.validation_status = val_res.status
                inspection.inspection_status = 'REJECTED' if val_res.status == 'INVALID' else 'UNCERTAIN'
                inspection.validation_confidence = val_res.confidence
                inspection.validation_message = val_res.rejection_reason or val_res.user_message
                inspection.save()

                user_msg = val_res.user_message

                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({
                        'success': False,
                        'status': val_res.status,
                        'message': user_msg,
                        'reason': val_res.rejection_reason,
                    }, status=400)

                messages.error(request, user_msg)
                return render(request, 'inspection/upload.html', {
                    'form': form,
                    'error_message': user_msg,
                    'rejected_reason': val_res.rejection_reason,
                    'status': val_res.status,
                })

            # Step 2: Valid Industrial Component -> Run PatchCore Inference
            try:
                patchcore = get_patchcore_service()
                pred = patchcore.predict(image_path)

                inspection.validation_status = 'VALID'
                inspection.validation_confidence = val_res.confidence
                inspection.anomaly_score = pred['anomaly_score']
                inspection.anomaly_threshold = pred['threshold']
                inspection.inspection_status = pred['inspection_status']
                inspection.inference_time_ms = pred['inference_time_ms']

                # Step 3: Anomaly Heatmap & Comparison Generation
                date_path = timezone.now().strftime("%Y/%m/%d")
                heatmaps_dir = os.path.join(settings.MEDIA_ROOT, "heatmaps", date_path)
                os.makedirs(heatmaps_dir, exist_ok=True)

                heatmap_filename = f"heatmap_{inspection.inspection_id}.jpg"
                comparison_filename = f"comparison_{inspection.inspection_id}.jpg"

                heatmap_full_path = os.path.join(heatmaps_dir, heatmap_filename)
                comparison_full_path = os.path.join(heatmaps_dir, comparison_filename)

                AnomalyVisualizer.generate_heatmap(
                    original_image_path=image_path,
                    anomaly_map=pred['anomaly_map'],
                    output_heatmap_path=heatmap_full_path,
                    output_comparison_path=comparison_full_path,
                )

                # Store relative media paths in model
                inspection.anomaly_heatmap.name = f"heatmaps/{date_path}/{heatmap_filename}"
                inspection.comparison_image.name = f"heatmaps/{date_path}/{comparison_filename}"
                inspection.save()

                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({
                        'success': True,
                        'redirect_url': f'/inspections/{inspection.inspection_id}/',
                        'inspection_id': inspection.inspection_id
                    })

                return redirect('inspection_result', inspection_id=inspection.inspection_id)

            except Exception as e:
                logger.exception(f"Error during PatchCore inference: {e}")
                err_msg = f"Inference Error: Could not complete anomaly analysis ({str(e)})"
                inspection.inspection_status = 'UNCERTAIN'
                inspection.validation_message = err_msg
                inspection.save()

                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({'success': False, 'message': err_msg}, status=500)

                messages.error(request, err_msg)
                return render(request, 'inspection/upload.html', {'form': form, 'error_message': err_msg})
        else:
            errors = form.errors.as_text()
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': errors}, status=400)
            messages.error(request, errors)
            return render(request, 'inspection/upload.html', {'form': form})
    else:
        form = ImageUploadForm()

    return render(request, 'inspection/upload.html', {'form': form})


def inspection_result_view(request, inspection_id):
    """
    Displays the detailed diagnostic result for a PatchCore inspection run.
    """
    inspection = get_object_or_404(Inspection, inspection_id=inspection_id)
    return render(request, 'inspection/result.html', {
        'inspection': inspection,
    })


def inspection_history_view(request):
    """
    Searchable and filterable archive of all component inspections.
    """
    query = request.GET.get('q', '').strip()
    status_filter = request.GET.get('status', 'all')

    inspections = Inspection.objects.all().order_by('-created_at')

    if query:
        inspections = inspections.filter(
            Q(inspection_id__icontains=query) |
            Q(original_filename__icontains=query) |
            Q(validation_message__icontains=query)
        )

    if status_filter == 'anomalous':
        inspections = inspections.filter(inspection_status='ANOMALOUS')
    elif status_filter == 'normal':
        inspections = inspections.filter(inspection_status='NORMAL')
    elif status_filter == 'rejected':
        inspections = inspections.filter(validation_status='INVALID')
    elif status_filter == 'uncertain':
        inspections = inspections.filter(validation_status='UNCERTAIN')

    context = {
        'inspections': inspections,
        'query': query,
        'status_filter': status_filter,
        'total_count': inspections.count(),
    }
    return render(request, 'inspection/history.html', context)


def analytics_view(request):
    """
    Industrial PatchCore analytics with anomaly statistics and distribution charts.
    """
    valid_inspections = Inspection.objects.filter(validation_status='VALID')
    total_inspections = valid_inspections.count()
    total_normal = valid_inspections.filter(inspection_status='NORMAL').count()
    total_anomalous = valid_inspections.filter(inspection_status='ANOMALOUS').count()
    total_rejected = Inspection.objects.filter(validation_status='INVALID').count()
    total_uncertain = Inspection.objects.filter(validation_status='UNCERTAIN').count()

    anomaly_percentage = round((total_anomalous / total_inspections * 100), 1) if total_inspections > 0 else 0.0
    avg_score = valid_inspections.aggregate(Avg('anomaly_score'))['anomaly_score__avg'] or 0.0
    avg_latency = valid_inspections.aggregate(Avg('inference_time_ms'))['inference_time_ms__avg'] or 0.0

    pie_labels = ['Normal Components', 'Potential Anomalies', 'Rejected Images', 'Uncertain Quality']
    pie_data = [total_normal, total_anomalous, total_rejected, total_uncertain]

    # Daily trends (last 14 days)
    fourteen_days_ago = timezone.now() - timedelta(days=14)
    daily_stats = (
        Inspection.objects.filter(created_at__gte=fourteen_days_ago)
        .annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(
            total=Count('id'),
            normal=Count('id', filter=Q(inspection_status='NORMAL')),
            anomalous=Count('id', filter=Q(inspection_status='ANOMALOUS'))
        )
        .order_by('date')
    )

    trend_dates = [s['date'].strftime('%b %d') for s in daily_stats]
    trend_normal = [s['normal'] for s in daily_stats]
    trend_anomalous = [s['anomalous'] for s in daily_stats]

    context = {
        'total_inspections': total_inspections,
        'total_normal': total_normal,
        'total_anomalous': total_anomalous,
        'total_rejected': total_rejected,
        'total_uncertain': total_uncertain,
        'anomaly_percentage': anomaly_percentage,
        'avg_score': round(avg_score, 3),
        'avg_latency': round(avg_latency, 1),
        'pie_labels_json': json.dumps(pie_labels),
        'pie_data_json': json.dumps(pie_data),
        'trend_dates_json': json.dumps(trend_dates),
        'trend_normal_json': json.dumps(trend_normal),
        'trend_anomalous_json': json.dumps(trend_anomalous),
        'has_data': total_inspections > 0 or total_rejected > 0,
    }
    return render(request, 'inspection/analytics.html', context)


def download_pdf_view(request, inspection_id):
    """
    Downloads the official PatchCore inspection PDF report.
    Disallows generation for rejected non-industrial images.
    """
    inspection = get_object_or_404(Inspection, inspection_id=inspection_id)

    if inspection.validation_status != 'VALID' or inspection.is_rejected:
        messages.error(request, "A PDF inspection report cannot be generated for an invalid non-industrial image.")
        return redirect('inspection_history')

    pdf_buffer = PDFReportGenerator.generate_inspection_report(inspection)
    filename = f"VisionInspect_PatchCore_{inspection.inspection_id}.pdf"

    response = HttpResponse(pdf_buffer.getvalue(), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response
