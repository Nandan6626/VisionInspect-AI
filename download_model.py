from huggingface_hub import hf_hub_download
import os

os.makedirs("models", exist_ok=True)

print("Downloading industrial defect model...")

model_path = hf_hub_download(
    repo_id="Ashgibbs/Cosmetic_Defect_Detection", filename="best.pt", local_dir="models"
)

print("\nDownload completed!")
print("Model location:", model_path)
