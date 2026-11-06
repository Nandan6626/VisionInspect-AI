import io
import os
from django.utils import timezone
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)


class PDFReportGenerator:
    """
    Generates publication-ready PDF inspection reports for PatchCore
    visual defect detection using ReportLab.
    """

    @staticmethod
    def generate_inspection_report(inspection) -> io.BytesIO:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        primary_color = colors.HexColor("#0F172A")    # Deep Navy
        accent_color = colors.HexColor("#0284C7")     # Industrial Cyan
        subtle_text = colors.HexColor("#475569")      # Slate 600
        border_color = colors.HexColor("#CBD5E1")     # Slate 300

        is_anomalous = inspection.is_anomalous
        status_color = colors.HexColor("#DC2626") if is_anomalous else colors.HexColor("#16A34A")
        status_bg = colors.HexColor("#FEE2E2") if is_anomalous else colors.HexColor("#DCFCE7")

        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=primary_color,
        )

        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=accent_color,
        )

        heading_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=15,
            textColor=primary_color,
            spaceAfter=4,
        )

        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=primary_color,
        )

        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            leading=11,
            textColor=subtle_text,
        )

        elements = []

        # 1. Header Banner
        header_table = Table(
            [
                [
                    Paragraph("<b>VISIONINSPECT AI</b>", title_style),
                    Paragraph(
                        f"<b>INSPECTION REPORT</b><br/>"
                        f"<font size=8 color='#64748B'>ID: {inspection.inspection_id}</font>",
                        ParagraphStyle('RightHdr', parent=body_style, alignment=2)
                    )
                ],
                [
                    Paragraph("PatchCore Industrial Surface Defect & Anomaly Diagnostics", subtitle_style),
                    Paragraph(
                        f"<font size=8 color='#64748B'>Generated: {timezone.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</font>",
                        ParagraphStyle('RightSub', parent=body_style, alignment=2)
                    )
                ]
            ],
            colWidths=[4.3 * inch, 3.0 * inch]
        )
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 6))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceBefore=0, spaceAfter=10))

        # 2. Status Banner
        status_text = (
            f"<b><font size=12 color='{status_color.hexval()}'>STATUS: {inspection.status_label.upper()}</font></b><br/>"
            f"<font size=9 color='#1E293B'>Anomaly Score: <b>{inspection.formatted_score} ({inspection.anomaly_percentage})</b> "
            f"| Threshold: <b>{inspection.anomaly_threshold:.2f}</b> "
            f"| Industrial Validation: <b>{inspection.validation_status} ({inspection.validation_confidence:.1f}%)</b></font>"
        )
        status_table = Table([[Paragraph(status_text, body_style)]], colWidths=[7.3 * inch])
        status_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), status_bg),
            ('BOX', (0, 0), (-1, -1), 1, status_color),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(status_table)
        elements.append(Spacer(1, 12))

        # 3. Parameters Table & Original Photo
        orig_img_flowable = None
        if inspection.uploaded_image and os.path.exists(inspection.uploaded_image.path):
            try:
                img_path = inspection.uploaded_image.path
                with PILImage.open(img_path) as pimg:
                    w, h = pimg.size
                    max_w, max_h = 2.8 * inch, 2.2 * inch
                    scale = min(max_w / w, max_h / h, 1.0)
                orig_img_flowable = RLImage(img_path, width=w * scale, height=h * scale)
            except Exception:
                orig_img_flowable = Paragraph("<i>Image preview unavailable</i>", disclaimer_style)
        else:
            orig_img_flowable = Paragraph("<i>Image not found</i>", disclaimer_style)

        meta_data = [
            [Paragraph("<b>Inspection ID:</b>", body_style), Paragraph(str(inspection.inspection_id), body_style)],
            [Paragraph("<b>Inspection Date:</b>", body_style), Paragraph(inspection.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'), body_style)],
            [Paragraph("<b>Original Filename:</b>", body_style), Paragraph(str(inspection.original_filename or 'component.png'), body_style)],
            [Paragraph("<b>Validation Status:</b>", body_style), Paragraph(f"<b>{inspection.validation_status}</b> ({inspection.validation_confidence:.1f}%)", body_style)],
            [Paragraph("<b>Anomaly Score:</b>", body_style), Paragraph(f"<b>{inspection.formatted_score}</b>", body_style)],
            [Paragraph("<b>Decision Threshold:</b>", body_style), Paragraph(f"{inspection.anomaly_threshold:.2f}", body_style)],
            [Paragraph("<b>Inference Latency:</b>", body_style), Paragraph(f"{inspection.inference_time_ms:.1f} ms", body_style)],
            [Paragraph("<b>AI Architecture:</b>", body_style), Paragraph("PatchCore (Wide ResNet-50-2 Backbone)", body_style)],
        ]
        meta_table = Table(meta_data, colWidths=[1.6 * inch, 2.6 * inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))

        side_by_side = Table(
            [
                [Paragraph("<b>Audited Component Photo</b>", heading_style), Paragraph("<b>Diagnostic Telemetry</b>", heading_style)],
                [orig_img_flowable, meta_table]
            ],
            colWidths=[3.0 * inch, 4.3 * inch]
        )
        side_by_side.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(side_by_side)
        elements.append(Spacer(1, 12))

        # 4. Anomaly Heatmap / Comparison Visualization
        elements.append(Paragraph("<b>PatchCore Anomaly Heatmap &amp; Defect Localization Map</b>", heading_style))

        heatmap_flowable = None
        viz_path = inspection.comparison_image.path if inspection.comparison_image and os.path.exists(inspection.comparison_image.path) else (
            inspection.anomaly_heatmap.path if inspection.anomaly_heatmap and os.path.exists(inspection.anomaly_heatmap.path) else None
        )

        if viz_path and os.path.exists(viz_path):
            try:
                with PILImage.open(viz_path) as pimg:
                    w, h = pimg.size
                    max_w, max_h = 7.3 * inch, 2.3 * inch
                    scale = min(max_w / w, max_h / h, 1.0)
                heatmap_flowable = RLImage(viz_path, width=w * scale, height=h * scale)
            except Exception:
                heatmap_flowable = Paragraph("<i>Heatmap visualizer unavailable</i>", disclaimer_style)
        else:
            heatmap_flowable = Paragraph("<i>No anomaly detected or heatmap not generated</i>", disclaimer_style)

        heatmap_box = Table([[heatmap_flowable]], colWidths=[7.3 * inch])
        heatmap_box.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#0F172A")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(heatmap_box)
        elements.append(Spacer(1, 14))

        # 5. Technical Disclaimer
        disclaimer_box = Table(
            [[
                Paragraph(
                    "<b>QUALITY ENGINEERING DISCLAIMER &amp; TECHNICAL NOTICE:</b><br/>"
                    "This automated inspection report was generated by VisionInspect AI using PatchCore anomaly detection. "
                    "PatchCore compares surface patch embeddings against a learned coreset memory bank of normal manufacturing parts. "
                    "Anomaly scores and localized heatmaps identify statistically abnormal surface variations and do not constitute semantic defect classification. "
                    "All components flagged with Potential Surface Anomaly must undergo physical evaluation by a certified quality inspector.",
                    disclaimer_style
                )
            ]],
            colWidths=[7.3 * inch]
        )
        disclaimer_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(KeepTogether(disclaimer_box))

        doc.build(elements)
        buffer.seek(0)
        return buffer
