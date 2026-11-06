import uuid
from django.db import models
from django.utils import timezone


class Inspection(models.Model):
    STATUS_CHOICES = [
        ('NORMAL', 'Normal Component'),
        ('ANOMALOUS', 'Potential Surface Anomaly'),
        ('REJECTED', 'Validation Rejected'),
        ('UNCERTAIN', 'Verification Uncertain'),
    ]

    VALIDATION_CHOICES = [
        ('VALID', 'Valid Industrial Component'),
        ('INVALID', 'Non-Industrial Image Rejected'),
        ('UNCERTAIN', 'Uncertain / Low Quality Image'),
    ]

    inspection_id = models.CharField(
        max_length=50,
        unique=True,
        editable=False,
        help_text="Unique audit identifier for this inspection"
    )
    uploaded_image = models.ImageField(
        upload_to='inspections/%Y/%m/%d/',
        help_text="Original uploaded component photograph"
    )
    original_filename = models.CharField(
        max_length=255,
        blank=True,
        help_text="Original client filename"
    )
    validation_status = models.CharField(
        max_length=32,
        choices=VALIDATION_CHOICES,
        default='VALID',
        help_text="Result of the industrial suitability validation layer"
    )
    validation_confidence = models.FloatField(
        default=0.0,
        help_text="Confidence percentage from industrial image validator"
    )
    validation_message = models.TextField(
        blank=True,
        null=True,
        help_text="Diagnostic message or rejection reason"
    )
    anomaly_score = models.FloatField(
        default=0.0,
        help_text="PatchCore anomaly score (0.00 to 1.00)"
    )
    anomaly_threshold = models.FloatField(
        default=0.50,
        help_text="Decision threshold for normal vs anomalous classification"
    )
    inspection_status = models.CharField(
        max_length=32,
        choices=STATUS_CHOICES,
        default='NORMAL',
        help_text="Inspection outcome determined by PatchCore"
    )
    anomaly_heatmap = models.ImageField(
        upload_to='heatmaps/%Y/%m/%d/',
        blank=True,
        null=True,
        help_text="Color-coded pixel anomaly heatmap"
    )
    comparison_image = models.ImageField(
        upload_to='heatmaps/%Y/%m/%d/',
        blank=True,
        null=True,
        help_text="Side-by-side diagnostic comparison (Original | Heatmap | Overlay)"
    )
    inference_time_ms = models.FloatField(
        default=0.0,
        help_text="Inference execution latency in milliseconds"
    )
    created_at = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="Timestamp of component inspection"
    )

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Inspection'
        verbose_name_plural = 'Inspections'

    def save(self, *args, **kwargs):
        if not self.inspection_id:
            date_prefix = timezone.now().strftime("%Y%m%d")
            random_suffix = uuid.uuid4().hex[:6].upper()
            self.inspection_id = f"INSP-{date_prefix}-{random_suffix}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.inspection_id} - {self.inspection_status} (Score: {self.anomaly_score:.3f})"

    @property
    def is_normal(self):
        return self.inspection_status == 'NORMAL'

    @property
    def is_anomalous(self):
        return self.inspection_status == 'ANOMALOUS'

    @property
    def is_rejected(self):
        return self.validation_status == 'INVALID' or self.inspection_status == 'REJECTED'

    @property
    def is_uncertain(self):
        return self.validation_status == 'UNCERTAIN' or self.inspection_status == 'UNCERTAIN'

    @property
    def formatted_score(self):
        return f"{self.anomaly_score:.3f}"

    @property
    def anomaly_percentage(self):
        return f"{self.anomaly_score * 100:.1f}%"

    @property
    def status_label(self):
        if self.is_anomalous:
            return "Potential Surface Anomaly"
        elif self.is_normal:
            return "Normal Component"
        elif self.is_uncertain:
            return "Verification Uncertain"
        return "Validation Rejected"
