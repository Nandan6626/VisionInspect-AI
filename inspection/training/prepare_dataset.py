"""
Industrial Dataset Preparation Script for VisionInspect AI.
Downloads and organizes MVTec AD industrial benchmark datasets
into training (normal only) and testing (normal and anomalous) structures.
"""
import os
import argparse
import shutil
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download


def prepare_dataset(category: str = 'grid', target_dir: str = 'datasets/mvtec_grid', max_samples: int = 15):
    print(f"==================================================")
    print(f"Preparing MVTec AD Dataset for Category: '{category}'")
    print(f"Target Directory: {target_dir}")
    print(f"==================================================")

    train_good_dir = os.path.join(target_dir, 'train', 'good')
    test_good_dir = os.path.join(target_dir, 'test', 'good')
    test_defect_dir = os.path.join(target_dir, 'test', 'defect')

    os.makedirs(train_good_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)
    os.makedirs(test_defect_dir, exist_ok=True)

    repo_id = f'yiching/MVTec_{category}'
    try:
        files = list_repo_files(repo_id=repo_id, repo_type='dataset')
        png_files = sorted([f for f in files if f.endswith('.png')])
        print(f"Discovered {len(png_files)} files in repository '{repo_id}'.")
    except Exception as e:
        print(f"Could not connect to online repository ({e}). Using existing local files.")
        return

    # Normal training samples (Indices 0 to 9)
    print(f"Downloading normal training images to: {train_good_dir}")
    for i in range(min(10, len(png_files))):
        fname = f"{i:03d}.png"
        hf_hub_download(repo_id=repo_id, filename=fname, repo_type='dataset', local_dir=train_good_dir)

    # Normal test samples (Indices 10 to 14)
    print(f"Downloading normal test images to: {test_good_dir}")
    for i in range(10, min(15, len(png_files))):
        fname = f"{i:03d}.png"
        hf_hub_download(repo_id=repo_id, filename=fname, repo_type='dataset', local_dir=test_good_dir)

    # Anomalous / Defective test samples (Indices 50 to 55)
    print(f"Downloading anomalous test images to: {test_defect_dir}")
    for i in range(50, min(55, len(png_files))):
        fname = f"{i:03d}.png"
        hf_hub_download(repo_id=repo_id, filename=fname, repo_type='dataset', local_dir=test_defect_dir)

    print("\nDataset preparation completed successfully!")
    print(f"- Normal Training Images: {len(os.listdir(train_good_dir))}")
    print(f"- Normal Test Images: {len(os.listdir(test_good_dir))}")
    print(f"- Defect Test Images: {len(os.listdir(test_defect_dir))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare Industrial Dataset for PatchCore")
    parser.add_argument('--category', type=str, default='grid', help='MVTec category name (e.g. grid)')
    parser.add_argument('--target_dir', type=str, default='datasets/mvtec_grid', help='Output directory')
    args = parser.parse_args()

    prepare_dataset(category=args.category, target_dir=args.target_dir)

