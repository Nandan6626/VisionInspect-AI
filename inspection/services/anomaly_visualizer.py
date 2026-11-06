"""
Anomaly Visualization Service for VisionInspect AI.
Transforms PatchCore anomaly maps into publication-grade heatmaps,
blended overlays, and side-by-side diagnostic comparisons.
"""
import os
import cv2
import numpy as np
from PIL import Image


class AnomalyVisualizer:

    @staticmethod
    def generate_heatmap(
        original_image_path: str,
        anomaly_map: np.ndarray,
        output_heatmap_path: str,
        output_comparison_path: str,
        alpha: float = 0.45,
    ) -> dict:
        """
        Generates:
        1. A normalized jet color heatmap.
        2. A side-by-side diagnostic comparison (Original | Heatmap | Overlay).
        """
        # Read original image
        orig_bgr = cv2.imread(original_image_path)
        if orig_bgr is None:
            raise FileNotFoundError(f"Could not read source image at {original_image_path}")

        h, w = orig_bgr.shape[:2]

        # Anomaly map preprocessing
        if isinstance(anomaly_map, np.ndarray):
            am = anomaly_map.squeeze()
        else:
            am = np.array(anomaly_map).squeeze()

        # Resize anomaly map to original image resolution
        am_resized = cv2.resize(am, (w, h), interpolation=cv2.INTER_LINEAR)

        # Min-max normalization (0 to 255)
        am_min, am_max = am_resized.min(), am_resized.max()
        if am_max > am_min:
            norm_am = ((am_resized - am_min) / (am_max - am_min) * 255.0).astype(np.uint8)
        else:
            norm_am = np.zeros((h, w), dtype=np.uint8)

        # Apply Jet ColorMap (Blue = Normal, Red/Yellow = Anomaly)
        heatmap_bgr = cv2.applyColorMap(norm_am, cv2.COLORMAP_JET)

        # Generate Blended Overlay
        overlay_bgr = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)

        # Create Side-by-Side Comparison: [Original | Heatmap | Overlay]
        # To maintain neat display, scale each panel to height 360
        target_h = 360
        scale = target_h / float(h)
        target_w = int(w * scale)

        panel1 = cv2.resize(orig_bgr, (target_w, target_h))
        panel2 = cv2.resize(heatmap_bgr, (target_w, target_h))
        panel3 = cv2.resize(overlay_bgr, (target_w, target_h))

        # Add title headers to panels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel1, "Original Photograph", (12, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel2, "PatchCore Anomaly Heatmap", (12, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel3, "Defect Region Overlay", (12, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        comparison_img = np.hstack([panel1, panel2, panel3])

        # Ensure destination folders exist
        os.makedirs(os.path.dirname(output_heatmap_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_comparison_path), exist_ok=True)

        # Save files
        cv2.imwrite(output_heatmap_path, heatmap_bgr)
        cv2.imwrite(output_comparison_path, comparison_img)

        return {
            "heatmap_path": output_heatmap_path,
            "comparison_path": output_comparison_path,
            "peak_anomaly_value": float(am_max),
            "mean_anomaly_value": float(am.mean()),
        }


def get_anomaly_visualizer():
    return AnomalyVisualizer()
