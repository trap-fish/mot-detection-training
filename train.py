import os
import yaml
import shutil

# Step 1: Paths and setup
visdrone_path = "/c/Users/Matthew/Documents/UU/IoT/project/mot-experiments/data/VisDrone2019-DET-train"  # Replace with your VisDrone-DET dataset path
output_path = "/path/to/output"       # Path to save trained models
yolov8_repo_path = "/path/to/yolov8"  # Path to the YOLOv8 repo

# Class IDs for humans in VisDrone: 'pedestrian' and 'person'
human_class_ids = [0, 1]  # 0: pedestrian, 1: person

# Step 2: Filter annotations to keep only human classes
def filter_annotations(visdrone_path):
    """Filter VisDrone annotations to include only humans."""
    input_labels_path = os.path.join(visdrone_path, "annotations")
    output_labels_path = os.path.join(visdrone_path, "labels_filtered")

    os.makedirs(output_labels_path, exist_ok=True)

    for file_name in os.listdir(input_labels_path):
        input_file = os.path.join(input_labels_path, file_name)
        output_file = os.path.join(output_labels_path, file_name)

        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            for line in infile:
                fields = line.strip().split(",")
                class_id = int(fields[5])  # Category is in the 6th column (0-indexed)

                if class_id in human_class_ids:
                    # Keep only <x_min>, <y_min>, <width>, <height>, <category_id>
                    bbox = fields[:5]
                    class_id_new = human_class_ids.index(class_id)  # Remap to 0 (pedestrian) or 1 (person)
                    outfile.write(" ".join(bbox) + f" {class_id_new}\n")

    print(f"Filtered annotations saved to {output_labels_path}")

# Step 3: Create YAML configuration file
def create_yaml_config(yaml_path, visdrone_path):
    """Create a YAML config file for YOLOv8."""
    config = {
        "path": visdrone_path,
        "train": os.path.join(visdrone_path, "images/train"),
        "val": os.path.join(visdrone_path, "images/val"),
        "nc": len(human_class_ids),
        "names": ["pedestrian", "person"]
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)
    print(f"YAML config saved to {yaml_path}")

yaml_path = os.path.join(visdrone_path, "visdrone_humans.yaml")
create_yaml_config(yaml_path, visdrone_path)

# Step 4: Train YOLOv8 model
def train_yolo(yolov8_repo_path, yaml_path, output_path):
    """Train YOLOv8 using the prepared dataset."""
    os.system(
        f"python {os.path.join(yolov8_repo_path, 'train.py')} "
        f"--img 640 --batch 16 --epochs 100 --data {yaml_path} "
        f"--weights yolov8n.pt --project {output_path} --name visdrone_humans"
    )

# Run the script
filter_annotations(visdrone_path)
train_yolo(yolov8_repo_path, yaml_path, output_path)
