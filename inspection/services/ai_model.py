import logging
import os
import time
from typing import Dict, Any, List
from django.conf import settings
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class DefectClassificationService:
    """
    Singleton service wrapper around YOLOv8 classification model.
    Loads models/best.pt once into memory and performs classification.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DefectClassificationService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        model_path = getattr(settings, 'YOLO_MODEL_PATH', None)
        if not model_path or not os.path.exists(model_path):
            # Fallback path check
            base_dir = getattr(settings, 'BASE_DIR', None)
            if base_dir:
                model_path = os.path.join(base_dir, 'models', 'best.pt')

        if not os.path.exists(model_path):
            logger.error(f"Defect classification model not found at {model_path}")
            self._model = None
            return

        try:
            logger.info(f"Loading YOLOv8 classification model from {model_path}...")
            self._model = YOLO(str(model_path))
            logger.info(f"YOLOv8 model loaded. Classes: {self._model.names}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_supported_classes(self) -> Dict[int, str]:
        if self._model and hasattr(self._model, 'names'):
            return self._model.names
        return {}

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Runs YOLOv8 classification inference on the specified image file.
        Returns a dictionary with top predicted class, confidence, and top-5 breakdown.
        """
        if not self._model:
            # Attempt re-init
            self._initialize_model()
            if not self._model:
                raise RuntimeError("AI defect classification model is not available or failed to load.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Inspection image not found at {image_path}")

        start_time = time.time()

        # Run Ultralytics classification inference (verbose=False to keep stdout clean)
        results = self._model.predict(
            source=image_path,
            save=False,
            verbose=False
        )

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        if not results:
            raise RuntimeError("Classification model returned no output for the uploaded image.")

        result = results[0]
        probabilities = result.probs

        if probabilities is None:
            raise RuntimeError("Model returned no classification probabilities.")

        # Top 5 classes and scores
        top_indices = probabilities.top5
        top_predictions: List[Dict[str, Any]] = []

        for index in top_indices:
            class_name = self._model.names.get(index, f"class_{index}")
            conf = float(probabilities.data[index]) * 100.0
            top_predictions.append({
                "index": int(index),
                "class": class_name,
                "display_name": class_name.replace('_', ' ').replace('-', ' ').title(),
                "confidence": round(conf, 2),
            })

        # Primary prediction is top-1
        primary = top_predictions[0] if top_predictions else {
            "class": "unknown",
            "display_name": "Unknown",
            "confidence": 0.0
        }

        predicted_class = primary["class"]
        is_normal = (predicted_class.lower() == "normal")

        return {
            "predicted_class": predicted_class,
            "display_class": primary["display_name"],
            "confidence_score": primary["confidence"],
            "is_normal": is_normal,
            "is_defective": not is_normal,
            "top_predictions": top_predictions,
            "inference_time_ms": inference_time_ms,
        }


# Global helper accessor
def get_defect_service() -> DefectClassificationService:
    return DefectClassificationService()

