import os
import io
from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from inspection.models import Inspection
from inspection.services.patchcore_service import get_patchcore_service
from inspection.services.image_validator import (
    get_image_validator,
    INVALID_IMAGE_USER_MESSAGE,
    UNCERTAIN_IMAGE_USER_MESSAGE,
)
from inspection.services.anomaly_visualizer import AnomalyVisualizer
from inspection.services.pdf_generator import PDFReportGenerator
from PIL import Image
import numpy as np


class VisionInspectPatchCoreTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.patchcore_service = get_patchcore_service()
        self.validator = get_image_validator()

    def test_patchcore_service_initialization(self):
        """Verify PatchCore service loads checkpoint with Wide ResNet-50-2 backbone."""
        self.assertTrue(self.patchcore_service.is_loaded)
        self.assertIsNotNone(self.patchcore_service.model)
        self.assertGreater(self.patchcore_service.threshold, 0.0)

    def test_validator_rejects_portrait(self):
        """Verify OpenCLIP validator rejects human portrait photograph."""
        portrait_path = "test_images/non_industrial_portrait.jpg"
        if os.path.exists(portrait_path):
            result = self.validator.validate(portrait_path)
            self.assertEqual(result.status, "INVALID")
            self.assertFalse(result.is_valid)
            self.assertEqual(result.user_message, INVALID_IMAGE_USER_MESSAGE)
            self.assertIn("Human Portrait", result.rejection_reason)

    def test_validator_accepts_industrial_grid(self):
        """Verify OpenCLIP validator accepts real industrial manufactured component."""
        industrial_path = "test_images/industrial_grid_good.png"
        if os.path.exists(industrial_path):
            result = self.validator.validate(industrial_path)
            self.assertEqual(result.status, "VALID")
            self.assertTrue(result.is_valid)

    def test_patchcore_inference(self):
        """Verify PatchCore inference generates anomaly score and anomaly heatmap."""
        industrial_path = "test_images/industrial_grid_good.png"
        if os.path.exists(industrial_path):
            inf_res = self.patchcore_service.infer(industrial_path)
            self.assertIn("anomaly_score", inf_res)
            self.assertIn("is_anomalous", inf_res)
            self.assertIn("anomaly_map", inf_res)
            self.assertIsInstance(inf_res["anomaly_map"], np.ndarray)
            self.assertGreater(inf_res["inference_time_ms"], 0.0)

    def test_anomaly_visualizer(self):
        """Verify visualizer creates heatmap overlay and side-by-side comparison images."""
        industrial_path = "test_images/industrial_grid_good.png"
        if os.path.exists(industrial_path):
            inf_res = self.patchcore_service.infer(industrial_path)
            tmp_heat = "media/heatmaps/test_heat.png"
            tmp_comp = "media/heatmaps/test_comp.png"
            res = AnomalyVisualizer.generate_heatmap(
                industrial_path,
                inf_res["anomaly_map"],
                tmp_heat,
                tmp_comp,
            )
            self.assertTrue(os.path.exists(tmp_heat))
            self.assertTrue(os.path.exists(tmp_comp))
            if os.path.exists(tmp_heat):
                os.remove(tmp_heat)
            if os.path.exists(tmp_comp):
                os.remove(tmp_comp)

    def test_invalid_image_upload_flow(self):
        """Verify non-industrial portrait image upload is blocked with required message."""
        portrait_path = "test_images/non_industrial_portrait.jpg"
        if not os.path.exists(portrait_path):
            return

        with open(portrait_path, "rb") as f:
            uploaded = SimpleUploadedFile(
                name="portrait.jpg",
                content=f.read(),
                content_type="image/jpeg"
            )

        response = self.client.post(reverse("inspect_upload"), {"image": uploaded}, follow=True)
        self.assertEqual(response.status_code, 200)

        # Confirm the required message is rendered
        self.assertContains(response, INVALID_IMAGE_USER_MESSAGE)

        # Inspection record created as REJECTED
        insp = Inspection.objects.filter(validation_status="INVALID").first()
        self.assertIsNotNone(insp)
        self.assertEqual(insp.inspection_status, "REJECTED")

        # Confirm PDF report cannot be generated for rejected image
        pdf_url = reverse("download_pdf", kwargs={"inspection_id": insp.inspection_id})
        pdf_resp = self.client.get(pdf_url, follow=True)
        self.assertContains(pdf_resp, "A PDF inspection report cannot be generated for an invalid non-industrial image.")

    def test_valid_industrial_upload_flow(self):
        """Verify complete upload and PatchCore pipeline for valid industrial component."""
        industrial_path = "test_images/industrial_grid_good.png"
        if not os.path.exists(industrial_path):
            return

        with open(industrial_path, "rb") as f:
            uploaded = SimpleUploadedFile(
                name="grid_test.png",
                content=f.read(),
                content_type="image/png"
            )

        response = self.client.post(reverse("inspect_upload"), {"image": uploaded}, follow=True)
        self.assertEqual(response.status_code, 200)

        # Inspection record created
        insp = Inspection.objects.filter(validation_status="VALID").first()
        self.assertIsNotNone(insp)
        self.assertEqual(insp.validation_status, "VALID")
        self.assertIn(insp.inspection_status, ["NORMAL", "ANOMALOUS"])
        self.assertIsNotNone(insp.anomaly_score)
        self.assertIsNotNone(insp.anomaly_heatmap)

        # Test PDF generation endpoint
        pdf_url = reverse("download_pdf", kwargs={"inspection_id": insp.inspection_id})
        pdf_resp = self.client.get(pdf_url)
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertEqual(pdf_resp["Content-Type"], "application/pdf")
        self.assertIn(f"VisionInspect_PatchCore_{insp.inspection_id}.pdf", pdf_resp["Content-Disposition"])
        self.assertGreater(len(pdf_resp.content), 5000)

    def test_dashboard_and_analytics_views(self):
        """Test dashboard, history, and analytics page loads."""
        resp = self.client.get(reverse("dashboard"))
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get(reverse("inspection_history"))
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get(reverse("analytics"))
        self.assertEqual(resp.status_code, 200)
