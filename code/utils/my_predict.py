import detectron2.config as Cf
import detectron2.model_zoo as ModelZoo
import detectron2.modeling as Modeling
from aistron.config import add_aistron_config
from detectron2.engine import DefaultPredictor
import numpy as np
import PIL.Image
from detectron2.data import MetadataCatalog
from utils.custom_inferrence_model import VisualizationDemo
from detectron2.utils.visualizer import ColorMode
import cv2
from utils.preprocess_image import preprocess_image
import torch

from detectron2.utils.visualizer import (ColorMode, GenericMask, Visualizer)
from utils.calc import detect_and_draw_line_on_image, calculate_average_length


def inference(image, cfg, threshold=0.5):
    # Load the model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION, parallel=False)
    # Perform inferrence
    predictions_amodal, visualized_output_amodal   = predictor.run_on_image(image,segm_type  = 'amodal')

    try:
        instances = predictions_amodal["instances"].to(torch.device("cpu"))
        pred_masks = instances.pred_amodal_masks
        masks = np.asarray(pred_masks)
        masks = [GenericMask(x, 640, 640) for x in masks]
        mask_areas = []
        for mask in masks:
            mask_areas.append(mask.area())
    except Exception as e:
        print("Error calculating mask area:", e)

    predictions_visible, visualized_output_visible = predictor.run_on_image(image,segm_type  = 'visible')
    # convert image to numpy array
    visualized_output_visible = visualized_output_visible.get_image()
    visualized_output_amodal = visualized_output_amodal.get_image()

    num_amodal_instances = len(predictions_amodal["instances"])
    num_visible_instances = len(predictions_visible["instances"])

    return visualized_output_visible, visualized_output_amodal, num_amodal_instances, num_visible_instances, mask_areas

def predict(im_path, cfg, threshold = 0.65) -> PIL.Image.Image:
    # Predict function to process input image and return the model's predictions

    # Read the image
    image = cv2.imread(im_path)
    # Preprocess the image
    # preprocessed_img = preprocess_image(image)
    preprocessed_img = cv2.resize(image, (640, 640))
    visible_image, amodal_image, num_amodal_instances, num_visible_instances, mask_areas  = inference(preprocessed_img, cfg, threshold=threshold)


    return image, visible_image, amodal_image, num_amodal_instances, num_visible_instances, mask_areas