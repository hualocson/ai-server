'''
This script is used to perform custom inferrence on the images, read the data from front end,
perform inferrence on the images and return the output to the front end.
'''
import os
import numpy as np

import cv2

from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


from aistron.config import add_aistron_config
from custom_inferrence_model import VisualizationDemo

from aistron.data.datasets.coco_amodal import register_aistron_cocolike_instances


def get_default_metadata():

    thing_classes = [
        "crystal",
    ]
    thing_colors = [ [0, 0, 255] for _ in thing_classes]
    return {
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }

def setup_cfg(config_path = "/home/locson/workspaces/AI/KLTN_project/server/code/configs/config.yaml",
              default_dataset_name = "crystal_amodal_default",
              default_dataset_json = "/home/locson/workspaces/AI/KLTN_project/server/code/configs/annotations_aistron.json",
              default_image_dir = "",
              model_path = "",
              threshold=0.5):
    # Load the configuration file
    cfg = get_cfg()
    add_aistron_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    # Check in metadataCatalog
    if default_dataset_name not in MetadataCatalog.list():
        register_aistron_cocolike_instances(
        name=default_dataset_name,
        metadata= get_default_metadata(),
        json_file=default_dataset_json,
        image_root= default_image_dir)
    # Set dataset for cfg
    cfg.DATASETS.TEST = (default_dataset_name,)
    #set cuda is cpu
    cfg.MODEL.DEVICE = "cpu"
    return cfg

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Use morphological operations to slightly close gaps and remove noise
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours in the morphologically processed image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the detected contours
    mask = np.zeros_like(gray)

    # Filter out small contours to reduce noise
    min_contour_area = 100  # Adjust this value based on your image
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    # Draw the filtered contours on the mask
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original grayscale image
    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    #convert to RGB
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result

def inference(image, model_path, cfg, threshold=0.5):
    # Load the model
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION, parallel=False)
    # Perform inferrence
    predictions_amodal, visualized_output_amodal = predictor.run_on_image(image,segm_type  = 'amodal')
    predictions_visible, visualized_output_visible = predictor.run_on_image(image,segm_type  = 'visible')
    # convert image to numpy array
    visualized_output_visible = visualized_output_visible.get_image()
    visualized_output_amodal = visualized_output_amodal.get_image()

    return visualized_output_visible, visualized_output_amodal


def handle_inferrence(image_path,model_path,default_dataset_name = "crystal_amodal_default",threshold=0.5):
    # Read the image
    image = cv2.imread(image_path)
    # Preprocess the image
    preprocessed_img = preprocess_image(image)
    # Setup the configuration
    cfg = setup_cfg(model_path = model_path, default_dataset_name = default_dataset_name)
    # Inferrence the image
    visible_image, amodal_image = inference(preprocessed_img,model_path, cfg, threshold=threshold)

    return image, visible_image, amodal_image

def test_API():
    default_dataset_name = "crystal_amodal_default"
    image_path = '/home/locson/workspaces/AI/KLTN_project/server/code/image/1.png'
    model_path = '/home/locson/workspaces/AI/KLTN_project/server/code/train_outputs/model_final.pth'
    image, visible_image, amodal_image = handle_inferrence(image_path,model_path,default_dataset_name = default_dataset_name,threshold=0.5)
    # Write the output images to disk
    cv2.imwrite('image.png', image)
    cv2.imwrite('image_visible.png', visible_image)
    cv2.imwrite('image_amodal.png', amodal_image)

test_API()