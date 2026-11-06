from django.contrib import admin
from .models import Inspection


@admin.register(Inspection)
class InspectionAdmin(admin.ModelAdmin):
    list_display = (
        'inspection_id',
        'inspection_status',
        'anomaly_score',
        'validation_status',
        'validation_confidence',
        'inference_time_ms',
        'created_at'
    )
    list_filter = ('inspection_status', 'validation_status', 'created_at')
    search_fields = ('inspection_id', 'original_filename', 'validation_message')
    readonly_fields = ('inspection_id', 'created_at')
