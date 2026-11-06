"""
Evaluation script for PatchCore anomaly detection model.
Calculates image-level AUROC, anomaly scores, and validation metrics on test samples.
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torchvision.transforms.v2 as T

from anomalib.models import Patchcore


def evaluate_patchcore(
    checkpoint_path: str = "models/patchcore/patchcore_model.ckpt",
    test_dir: str = "datasets/mvtec_grid/test",
):
    print("=" * 60)
    print(" VISIONINSPECT AI – PATCHCORE EVALUATION")
    print("=" * 60)
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    model = Patchcore.load_from_checkpoint(checkpoint_path)
    model.eval()

    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((256, 256)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    good_dir = os.path.join(test_dir, "good")
    defect_dir = os.path.join(test_dir, "defect")

    scores = []
    labels = []  # 0: normal, 1: anomalous

    # Test Good Samples
    if os.path.exists(good_dir):
        for fname in sorted(os.listdir(good_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(good_dir, fname)
                img = Image.open(fpath).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    out = model.model(tensor)
                    score = float(out.pred_score.item())
                scores.append(score)
                labels.append(0)
                print(f"[Normal Sample]    {fname} -> Score: {score:.4f}")

    # Test Defect Samples
    if os.path.exists(defect_dir):
        for fname in sorted(os.listdir(defect_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(defect_dir, fname)
                img = Image.open(fpath).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    out = model.model(tensor)
                    score = float(out.pred_score.item())
                scores.append(score)
                labels.append(1)
                print(f"[Defective Sample] {fname} -> Score: {score:.4f}")

    if len(set(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
        print("-" * 60)
        print(f"Total Evaluated:   {len(labels)} samples")
        print(f"Normal Samples:    {labels.count(0)}")
        print(f"Defective Samples: {labels.count(1)}")
        print(f"Image-Level AUROC: {auroc * 100:.2f}%")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PatchCore Model")
    parser.add_argument("--checkpoint", type=str, default="models/patchcore/patchcore_model.ckpt")
    parser.add_argument("--test_dir", type=str, default="datasets/mvtec_grid/test")
    args = parser.parse_args()

    evaluate_patchcore(checkpoint_path=args.checkpoint, test_dir=args.test_dir)
