from ultralytics import YOLO
import os

model_path = "models/best.pt"

if not os.path.exists(model_path):
    print("Model file not found!")
    exit()

print("Loading industrial defect model...")

model = YOLO(model_path)

print("\nModel loaded successfully!")
print("\nSupported defect categories:")
for class_id, class_name in model.names.items():
    print(class_id, ":", class_name)
