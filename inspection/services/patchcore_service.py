"""
PatchCore AI Anomaly Detection Service for VisionInspect AI.
Loads the trained Anomalib PatchCore model into memory once and performs
high-precision surface defect and anomaly localization.
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T

from anomalib.models import Patchcore
from django.conf import settings

logger = logging.getLogger(__name__)


class PatchCoreInferenceService:
    _instance = None
    _model = None
    _config = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PatchCoreInferenceService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        models_dir = os.path.join(settings.BASE_DIR, "models", "patchcore")
        ckpt_path = os.path.join(models_dir, "patchcore_model.ckpt")
        config_path = os.path.join(models_dir, "config.json")

        if not os.path.exists(ckpt_path):
            logger.warning(f"PatchCore checkpoint not found at {ckpt_path}. Model will need to be trained.")
            self._model = None
            return

        try:
            logger.info(f"Loading PatchCore checkpoint from {ckpt_path}...")
            self._model = Patchcore.load_from_checkpoint(ckpt_path)
            self._model.eval()

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self._config = json.load(f)
            else:
                self._config = {
                    "backbone": "wide_resnet50_2",
                    "anomaly_threshold": 0.50,
                    "category": "industrial_surface",
                }

            self._transform = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize((256, 256)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info("PatchCore model loaded and ready for industrial inspection.")
        except Exception as e:
            logger.exception(f"Failed to load PatchCore model: {e}")
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        return self._model

    @property
    def threshold(self) -> float:
        return float(self._config.get("anomaly_threshold", 0.50))

    def infer(self, image_path: str) -> Dict[str, Any]:
        """Alias for predict."""
        return self.predict(image_path)

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Runs PatchCore anomaly detection on the specified component image.
        Returns anomaly score, normal/anomalous status, and pixel anomaly map.
        """
        if not self._model:
            self._initialize()
            if not self._model:
                raise RuntimeError("PatchCore anomaly detection model is not loaded or checkpoint is missing.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Inspection image file not found at {image_path}")

        start_time = time.time()

        img = Image.open(image_path).convert('RGB')
        tensor = self._transform(img).unsqueeze(0)

        with torch.no_grad():
            out = self._model.model(tensor)

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        # Extract score and anomaly map from InferenceBatch
        raw_score = float(out.pred_score.item())
        am_tensor = out.anomaly_map.squeeze().cpu().numpy()

        # Calibration: Normalize score into a standardized 0.0 - 1.0 range
        # Raw PatchCore distance values are typically in the range [10.0, 50.0]
        # We map distance using a smooth sigmoid/min-max heuristic calibrated around threshold 32.0
        calibrated_score = 1.0 / (1.0 + np.exp(-(raw_score - 32.0) / 4.0))
        calibrated_score = float(np.clip(calibrated_score, 0.01, 0.99))

        threshold = float(self._config.get("anomaly_threshold", 0.50))
        is_anomalous = calibrated_score >= threshold

        return {
            "anomaly_score": round(calibrated_score, 4),
            "raw_score": round(raw_score, 4),
            "threshold": threshold,
            "is_anomalous": is_anomalous,
            "is_normal": not is_anomalous,
            "inspection_status": "ANOMALOUS" if is_anomalous else "NORMAL",
            "status_label": "Potential Surface Anomaly" if is_anomalous else "Normal Component",
            "anomaly_map": am_tensor,
            "inference_time_ms": inference_time_ms,
            "model_name": "PatchCore",
            "backbone": self._config.get("backbone", "wide_resnet50_2"),
        }


def get_patchcore_service() -> PatchCoreInferenceService:
    return PatchCoreInferenceService()
