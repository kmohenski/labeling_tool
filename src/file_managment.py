import json
import os
import math
from PIL import Image
from pathlib import Path

def save_annotations_obb_dota_json(annotations, image_files, base_path):
    """
    Save annotations in OBB, DOTA, and 4 corners JSON formats

    Args:
        annotations: dict {filename: [{class, x1, y1, x2, y2, x3, y3, x4, y4}, ...]}
        image_files: list of Path objects for images
        base_path: base path for saving (without extension)

    Returns:
        dict: paths to saved files/directories
    """
    try:
        folder_name = Path(base_path).name
        # Create directories
        obb_dir = f"{base_path}/{folder_name}_obb"
        dota_dir = f"{base_path}/{folder_name}_dota"
        os.makedirs(obb_dir, exist_ok=True)
        os.makedirs(dota_dir, exist_ok=True)

        # Create class names file for OBB format
        all_classes = sorted(set(ann['class'] for anns in annotations.values() for ann in anns))

        class_file = os.path.join(obb_dir, 'classes.txt')
        with open(class_file, 'w') as f:
            for class_name in all_classes:
                f.write(f"{class_name}\n")

        # Convert to class IDs for OBB
        class_to_id = {class_name: i for i, class_name in enumerate(all_classes)}

        for filenameW in image_files:
            filename = filenameW.name
            base_filename = filename.rsplit('.', 1)[0]
            print(base_filename)
            if filename in annotations:
                file_annotations = annotations[filename]

                # Get original image size
                img_path = next((f for f in image_files if f.name == filename), None)
                if img_path:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size


                    # Create OBB annotation file (Oriented Bounding Box format)
                    obb_file = os.path.join(obb_dir, base_filename + '.txt')
                    with open(obb_file, 'w') as f:
                        for ann in file_annotations:
                            class_id = class_to_id[ann['class']]

                            center_x = (ann['x1'] + ann['x2'] + ann['x3'] + ann['x4']) / 4 / img_width
                            center_y = (ann['y1'] + ann['y2'] + ann['y3'] + ann['y4']) / 4 / img_height

                            width_pixels = math.sqrt((ann['x2'] - ann['x1'])**2 + (ann['y2'] - ann['y1'])**2)
                            height_pixels = math.sqrt((ann['x4'] - ann['x1'])**2 + (ann['y4'] - ann['y1'])**2)

                            width = width_pixels / img_width
                            height = height_pixels / img_height

                            dx = ann['x2'] - ann['x1']
                            dy = ann['y2'] - ann['y1']
                            angle = math.atan2(dy, dx)

                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {angle:.6f}\n")

                    # Create DOTA annotation file
                    dota_file = os.path.join(dota_dir, base_filename + '.txt')
                    with open(dota_file, 'w') as f:
                        for ann in file_annotations:
                            # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                            f.write(f"{ann['x1']} {ann['y1']} {ann['x2']} {ann['y2']} {ann['x3']} {ann['y3']} {ann['x4']} {ann['y4']} {ann['class']} 0\n")

            else:
                obb_file = os.path.join(base_path, base_filename + '.txt')
                dota_file = os.path.join(dota_dir, base_filename + '.txt')

                open(obb_file, 'w').close()
                open(dota_file, 'w').close()


        # Save 4 corners as JSON for loading
        corners_json_file = f"{base_path}/{folder_name}_4corners.json"
        with open(corners_json_file, 'w') as f:
            json.dump(annotations, f, indent=2)

        return {
            'obb': obb_dir,
            'dota': dota_dir,
            '4corners_json': corners_json_file
        }

    except Exception as e:
        raise Exception(f"Failed to save annotations: {str(e)}")


def load_annotations_json(file_path) -> dict:
    try:
        with open(file_path, 'r') as f:
            annotations = json.load(f)
        return annotations
    except Exception as e:
        raise Exception(f"Failed to load annotations: {str(e)}")

def save_annotations_yolo(annotations, image_files, base_path):
    """
    Save annotations in OBB and 4 corners JSON formats

    Args:
        annotations: dict {filename: [{class, x1, y1, x2, y2, x3, y3, x4, y4}, ...]}
        image_files: list of Path objects for images
        base_path: base path for saving (without extension)

    Returns:
        dict: paths to saved files/directories
    """
    try:

        # Create class names file for OBB format
        all_classes = sorted(set(ann['class'] for anns in annotations.values() for ann in anns))

        class_file = os.path.join(base_path, 'classes.txt')
        with open(class_file, 'w') as f:
            for class_name in all_classes:
                f.write(f"{class_name}\n")

        # Convert to class IDs for OBB
        class_to_id = {class_name: i for i, class_name in enumerate(all_classes)}

        for filenameW in image_files:
            filename = filenameW.name
            base_filename = filename.rsplit('.', 1)[0]
            if filename in annotations:
                file_annotations = annotations[filename]

                # Get original image size
                img_path = next((f for f in image_files if f.name == filename), None)
                if img_path:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size

                    # Create OBB annotation file (Oriented Bounding Box format)
                    obb_file = os.path.join(base_path, base_filename + '.txt')
                    with open(obb_file, 'w') as f:
                        for ann in file_annotations:
                            class_id = class_to_id[ann['class']]

                            center_x = (ann['x1'] + ann['x2'] + ann['x3'] + ann['x4']) / 4 / img_width
                            center_y = (ann['y1'] + ann['y2'] + ann['y3'] + ann['y4']) / 4 / img_height

                            width_pixels = math.sqrt((ann['x2'] - ann['x1'])**2 + (ann['y2'] - ann['y1'])**2)
                            height_pixels = math.sqrt((ann['x4'] - ann['x1'])**2 + (ann['y4'] - ann['y1'])**2)

                            width = width_pixels / img_width
                            height = height_pixels / img_height

                            dx = ann['x2'] - ann['x1']
                            dy = ann['y2'] - ann['y1']
                            angle = math.atan2(dy, dx)

                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {angle:.6f}\n")

            else:
                obb_file = os.path.join(base_path, base_filename + '.txt')
                open(obb_file, 'w').close()

        # Save 4 corners as JSON for loading
        corners_json_file = f"{base_path}_4corners.json"
        with open(corners_json_file, 'w') as f:
            json.dump(annotations, f, indent=2)

        return {
            'yolo': obb_file,
            '4corners_json': corners_json_file
        }

    except Exception as e:
        raise Exception(f"Failed to save annotations: {str(e)}")
