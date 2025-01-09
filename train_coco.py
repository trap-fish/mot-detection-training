import os
import yaml
from ultralytics import YOLO

# Set dataset paths
train_images_path = "C:/Users/Matthew/Documents/UU/IoT/project/mot-experiments/mot-detection-training/datasets/VisDrone2019-DET-train/images"
val_images_path = "C:/Users/Matthew/Documents/UU/IoT/project/mot-experiments/mot-detection-training/datasets/VisDrone2019-DET-val/images"
coco_annotation_train = "C:/Users/Matthew/Documents/UU/IoT/project/mot-experiments/mot-detection-training/datasets/VisDrone2019-DET-train/annotations_VisDrone_train.json"
coco_annotation_val = "C:/Users/Matthew/Documents/UU/IoT/project/mot-experiments/mot-detection-training/datasets/VisDrone2019-DET-val/annotations_VisDrone_val.json"

# Define YOLO config parameters
config = {
    "train": train_images_path,
    "val": val_images_path,
    "nc": 2,  # Number of classes
    "names": ["pedestrian", "person"]  # Class names
}

# Save the config to a YAML file
yaml_path = "./visdrone_humans.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(config, f)
print(f"YAML config saved to {yaml_path}")

# Train YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano pretrained model (smallest size)
model.train(data=yaml_path, epochs=50, imgsz=640, batch=16)

# Evaluate the model
model.val()

# # Perform inference on a test image or folder of images
# test_image_path = "/path/to/test/image_or_folder"
# model.predict(source=test_image_path, save=True)
