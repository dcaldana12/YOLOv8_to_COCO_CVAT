import logging
import os
import json
import math

import yaml
from ultralytics import YOLO



def calculate_bbox(bbox_id, img_id, r_labels, r_xywhr):
    '''
    Calculate rotated bounding box from YOLO prediction result

    Args:
    - bbox_id: Bounding box id
    - img_id: Image id of the image
    - r_labels: Labels from YOLO
    - r_xywhr: xywhr values from YOLO

    Returns:
    - dict: Bounding box in COCO format
    '''
    category_id = int(r_labels[bbox_id-1].item()) + 1 # object label. COCO format is not zero-indexed
    x, y, width, height, rotation = r_xywhr.tolist()

    # Convertion to degrees
    rotation = math.degrees(rotation)

    area = width * height

    data = {"id": bbox_id, 
            "image_id": img_id, 
            "category_id": category_id, 
            "segmentation":[], 
            "area": area, "bbox": [x-width/2, y-height/2, width, height], 
            "iscrowd": 0, 
            "attributes": {"occluded": False, "rotation": rotation}
    }
    
    return data




def main():
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    with open("constants.yaml", 'r') as stream:
        CONSTANTS = yaml.safe_load(stream)

    # ---------- YAML parse ----------
    model_path = CONSTANTS["model_path"]
    images_path = CONSTANTS["images_path"]
    save_annotations_file_path = CONSTANTS["save_annotations_path"]
    labels = CONSTANTS["labels"]
    save_image = CONSTANTS["yolo_results"]["save_image"]
    save_txt = CONSTANTS["yolo_results"]["save_annotation"]

    # Check if parsing was done correctly
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        raise FileNotFoundError(f"Model path not found: {model_path} was not found")
    
    if not os.path.exists(images_path):
        print(f"Images path not found: {images_path}")
        raise FileNotFoundError(f"Images path not found: {images_path} was not found")

    if save_annotations_file_path is not None and not os.path.exists(save_annotations_file_path):
        print(f"Save annotations path not found: {save_annotations_file_path}")
        raise FileNotFoundError(f"Save annotations path not found: {save_annotations_file_path} was not found")
    else:
        save_annotations_file_path = os.path.curdir
        save_image_path = os.path.join(save_annotations_file_path, "predictions", "images")
        save_txt_file_path = os.path.join(save_annotations_file_path, "predictions", "annotations")
    json_file_path = os.path.join(save_annotations_file_path, "instances.json")

    if not isinstance(labels, list):
        print(f"Incorrect labels format: {labels}")
        raise ValueError(f"Incorrect labels format: {labels} should be a list")

    # Create save directories
    if save_image:
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
    if save_txt:
        if not os.path.exists(save_txt_file_path):
            os.makedirs(save_txt_file_path)

    # ---------- Load model ----------
    logger.debug(f"Loading model from {model_path}")
    model = YOLO(model_path)

    # ---------- Load images ----------
    images_paths = []
    logger.debug(f"Loading images from {images_path}")
    for image in os.listdir(images_path):
        image_path = os.path.join(images_path, image)
        if os.path.isfile(image_path) and image.lower().endswith((".jpg", ".jpeg", ".png")):
            images_paths.append(image_path)
    images_paths = sorted(images_paths)
    logger.debug(f"Found {len(images_paths)} images")

    # ---------- Predict ----------
    logger.debug(f"Predicting on images")
    results = model.predict(images_paths, stream=True)

    # ---------- Save results ----------
    data = {}
    # Add COCO header
    data.update({"licenses": [{"name": "", "id": 0, "url": ""}], "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""}})
    # Add COCO categories
    data.update({"categories": [{"id": i, "name": label, "supercategory": ""} for i, label in enumerate(labels, start=1)]})
    
    data_images = {"images": []}
    data_annotations = {"annotations": []}
    for i, result in enumerate(results, start=1):

        # Save image
        if save_image:
            path_to_save = os.path.join(save_image_path, os.path.basename(result.path))
            result.save(filename=path_to_save)
        # Save txt
        if save_txt:
            path_to_save = os.path.join(save_txt_file_path, os.path.splitext(os.path.basename(result.path))[0] + ".txt")
            result.save_txt(txt_file=path_to_save)

        new_image = {"id": i, "width": result.orig_shape[1], "height": result.orig_shape[0], "file_name": os.path.basename(result.path), "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}
        # Add new image
        data_images["images"].append(new_image)

        # Calculate bounding box for each annotation in result
        id = 1 # bbox id
        for r_xywhr in result.obb.xywhr:
            new_annotation = calculate_bbox(id, i, result.obb.cls, r_xywhr)
            # Add new annotation
            data_annotations["annotations"].append(new_annotation)
            id += 1
    
    logger.debug(f"Saving results")
    # Add images and annotations to data
    data.update(data_images)
    data.update(data_annotations)    
        
    # Generate json file and save data
    with open(json_file_path, 'w') as file:
        json.dump(data, file)

    logger.debug("Done")


if __name__ == "__main__":
    main()