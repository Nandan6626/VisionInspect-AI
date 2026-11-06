import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

INVALID_IMAGE_USER_MESSAGE = (
    "Invalid Image: The uploaded photograph does not appear to contain a supported "
    "industrial component. Please upload a clear photograph of a manufactured component."
)

UNCERTAIN_IMAGE_USER_MESSAGE = (
    "Unable to verify this image for industrial inspection. "
    "Please upload another clear component photograph."
)


@dataclass
class ValidationResult:
    status: str  # 'VALID', 'INVALID', 'UNCERTAIN'
    is_valid: bool
    is_uncertain: bool
    rejection_reason: str = ""
    user_message: str = ""
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def valid(cls, confidence: float, details: Optional[Dict[str, Any]] = None) -> "ValidationResult":
        return cls(
            status="VALID",
            is_valid=True,
            is_uncertain=False,
            confidence=round(confidence, 2),
            rejection_reason="",
            user_message="",
            details=details or {}
        )

    @classmethod
    def invalid(cls, reason: str, confidence: float = 0.0, details: Optional[Dict[str, Any]] = None) -> "ValidationResult":
        return cls(
            status="INVALID",
            is_valid=False,
            is_uncertain=False,
            confidence=round(confidence, 2),
            rejection_reason=reason,
            user_message=INVALID_IMAGE_USER_MESSAGE,
            details=details or {}
        )

    @classmethod
    def uncertain(cls, reason: str, details: Optional[Dict[str, Any]] = None) -> "ValidationResult":
        return cls(
            status="UNCERTAIN",
            is_valid=False,
            is_uncertain=True,
            confidence=0.0,
            rejection_reason=reason,
            user_message=UNCERTAIN_IMAGE_USER_MESSAGE,
            details=details or {}
        )


class IndustrialCLIPValidator:
    """
    Vision-Language Industrial Image Validator powered by OpenCLIP (ViT-B-32).
    Evaluates whether an uploaded image contains a legitimate industrial manufactured
    component, or should be rejected as an unrelated photograph (selfies, buildings,
    animals, food, landscapes, vehicles, etc.) or flagged as uncertain.
    """
    _instance = None
    _model = None
    _preprocess = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IndustrialCLIPValidator, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        try:
            import open_clip
            logger.info("Initializing OpenCLIP ViT-B-32 validator...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k'
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer('ViT-B-32')

            self.categories = [
                # 0: Industrial Component
                "a close-up photograph of an industrial manufactured metal component, wire mesh, machine part, or workpiece",
                # 1: Human Person / Selfie / Portrait
                "a portrait photograph, selfie, face, or photograph of a person",
                # 2: Buildings / Architecture / Skyscraper
                "a photograph of a building, city architecture, house, or skyscraper like Burj Khalifa",
                # 3: Animal / Wildlife
                "a photograph of an animal, dog, cat, bird, wildlife, or pet",
                # 4: Food / Meal
                "a photograph of food, dish, fruit, snack, or meal",
                # 5: Vehicle / Automobile
                "a photograph of a car, truck, motorcycle, or vehicle",
                # 6: Nature / Landscape
                "a photograph of outdoor nature, trees, forest, mountains, or landscape",
                # 7: Graphics / Cartoon
                "a cartoon illustration, anime, digital drawing, meme, or graphic artwork",
            ]

            text_tokens = self._tokenizer(self.categories)
            with torch.no_grad():
                text_features = self._model.encode_text(text_tokens)
                self._text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logger.info("IndustrialCLIPValidator initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            self._model = None

    def validate(self, image_path: str) -> ValidationResult:
        if not os.path.exists(image_path):
            return ValidationResult.invalid("File does not exist on disk", details={"path": image_path})

        # 1. Basic Format and Resolution Validation
        try:
            cv_img = cv2.imread(image_path)
            if cv_img is None or cv_img.size == 0:
                return ValidationResult.invalid("Corrupt or unreadable image file.")
            h, w = cv_img.shape[:2]
        except Exception as e:
            return ValidationResult.invalid(f"OpenCV cannot decode image: {e}")

        # Size check
        if w < 128 or h < 128:
            return ValidationResult.uncertain(
                f"Image dimensions too small ({w}x{h}). Minimum required for inspection: 128x128 pixels.",
                details={"width": w, "height": h}
            )

        # 2. Sharpness / Blur / Illumination Quality Check
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        if laplacian_var < 15.0:
            return ValidationResult.uncertain(
                f"Image is out of focus, blurry or blank (Laplacian variance: {laplacian_var:.1f}).",
                details={"laplacian_var": laplacian_var, "brightness": brightness}
            )

        if brightness < 15.0 or brightness > 245.0:
            return ValidationResult.uncertain(
                f"Image lighting is too extreme (Brightness: {brightness:.1f}/255).",
                details={"brightness": brightness}
            )

        # 3. Vision-Language CLIP Semantic Validation
        if self._model is None or self._preprocess is None:
            # Fallback if CLIP not loaded
            logger.warning("CLIP model not loaded, passing image with conservative status.")
            return ValidationResult.valid(confidence=50.0, details={"engine": "fallback"})

        try:
            pil_img = Image.open(image_path).convert('RGB')
            tensor = self._preprocess(pil_img).unsqueeze(0)

            with torch.no_grad():
                image_features = self._model.encode_image(tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)[0]

            probs = [round(float(s.item()) * 100, 2) for s in similarity]
            industrial_score = probs[0]
            non_industrial_score = sum(probs[1:])

            logger.info(f"CLIP Validation -> Industrial: {industrial_score}%, Non-Industrial: {non_industrial_score}%")

            # Check for strong non-industrial classification
            highest_idx = int(similarity.argmax().item())
            highest_score = probs[highest_idx]

            details = {
                "industrial_confidence": industrial_score,
                "highest_category_idx": highest_idx,
                "highest_score": highest_score,
                "category_probabilities": {
                    "industrial": industrial_score,
                    "person_selfie": probs[1],
                    "building": probs[2],
                    "animal": probs[3],
                    "food": probs[4],
                    "vehicle": probs[5],
                    "landscape": probs[6],
                    "cartoon": probs[7],
                }
            }

            # If non-industrial is the dominant class or industrial score is very low
            if highest_idx != 0:
                category_names = [
                    "Industrial Component",
                    "Human Portrait / Selfie",
                    "Building / Architecture",
                    "Animal / Wildlife",
                    "Food / Meal",
                    "Vehicle / Automobile",
                    "Landscape / Nature",
                    "Cartoon / Digital Art"
                ]
                detected_subject = category_names[highest_idx]
                return ValidationResult.invalid(
                    reason=f"Identified non-industrial subject: {detected_subject} ({highest_score}% confidence)",
                    confidence=highest_score,
                    details=details
                )

            # If industrial is top class, check confidence margin
            if industrial_score < 40.0:
                return ValidationResult.uncertain(
                    f"Ambiguous visual appearance (Industrial score: {industrial_score}%).",
                    details=details
                )

            return ValidationResult.valid(confidence=industrial_score, details=details)

        except Exception as e:
            logger.exception(f"Error during CLIP validation: {e}")
            return ValidationResult.uncertain(f"Validation analysis failed: {e}")


def get_image_validator() -> IndustrialCLIPValidator:
    return IndustrialCLIPValidator()
