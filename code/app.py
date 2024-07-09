# Serve model as a flask application
from flask import Flask, request, send_file, jsonify
import base64
import detectron2.config as Cf
import detectron2.model_zoo as ModelZoo
import detectron2.modeling as Modeling
from aistron.config import add_aistron_config
from detectron2.engine import DefaultPredictor
import os
from detectron2.data import MetadataCatalog
from utils.custom_inferrence_model import VisualizationDemo
from detectron2.utils.visualizer import ColorMode
import cv2
from utils.my_predict import predict
from utils.calc import measure_average_lengths_of_images
import scipy.stats as ss
import numpy as np
from aistron.projects.aisformer.config import add_aisformer_config

model = None
app = Flask(__name__)


def load_model():
    global model, cfg
    # config_path="/home/locson/workspaces/AI/KLTN_project/server/code/configs/config.yaml"
    config_path="/home/locson/workspaces/AI/KLTN_project/server/code/aistron/projects/AISFormer/configs/COCOA/aisformer_R101_FPN_cocoa_8ep_bs2.yaml"
    model_path = "/home/locson/workspaces/AI/KLTN_project/server/code/train_outputs/model_final.pth"

    cfg = Cf.get_cfg()
    add_aistron_config(cfg)
    add_aisformer_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.DATASETS.TRAIN = ("preprocessed_train",)
    cfg.DATASETS.TEST = ("preprocessed_test",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_path
    cfg = cfg.clone()
    cfg.MODEL.DEVICE = "cpu"  # Change device to run on CPU
    model = Modeling.build_model(cfg)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route("/predict",methods=["POST"])
def main():
    try:
        if request.method=='POST':
            # print("request",request.form)
            image=request.files['image']
            image_name=image.filename

            input_file_name = "input.jpg"
            amodal_file_name = "amodal.jpg"
            visible_file_name = "visible.jpg"

            input_path = os.path.join(os.getcwd(), input_file_name)
            amodal_path = os.path.join(os.getcwd(), amodal_file_name)
            visible_path = os.path.join(os.getcwd(), visible_file_name)

            if '.jpg' in image_name or '.png' in image_name or '.jpeg' in image_name:
                # check if input.jpg exists remove it
                image.save(input_path)
                # load image to model and get prediction
                image, visible_image, amodal_image, num_amodal_instances,  num_visible_instances= predict(input_path, cfg)
                cv2.imwrite(visible_path, visible_image)
                cv2.imwrite(amodal_path, amodal_image)

                with open(amodal_path, 'rb') as image_file:
                    amodal_image_data = base64.b64encode(image_file.read()).decode('utf-8')
                with open(visible_path, 'rb') as image_file:
                    visible_image_data = base64.b64encode(image_file.read()).decode('utf-8')
                # return prediction
                response = {"amodalImage": amodal_image_data, "visibleImage": visible_image_data, "numAmodalInstances": num_amodal_instances, "numVisibleInstances": num_visible_instances}
                return jsonify(response)
            else:
                return {"error":"select you image file"}
    except Exception as e:
        return {"error":str(e)}

@app.route("/get-chart", methods=["GET"])
def get_chart():
    dir_name = "images"
    dir_path = os.path.join(os.getcwd(), dir_name)
    average_lengths, count = measure_average_lengths_of_images(dir_path, cfg, True)
    # handle average_lengths return data for draw size distribution chart with histogram and gamma distribution curve
    histogram_data = {
        "average_lengths": average_lengths,
        "count": count
    }
    average_lengths = np.array(average_lengths) * 5
    average_lengths = average_lengths[~np.isnan(average_lengths)]
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

    counts, bin_edges = np.histogram(average_lengths, bins)
    alpha_est, loc_est, beta_est = ss.gamma.fit(average_lengths, floc=0)  # Fix location to 0

    x = np.linspace(0, max(average_lengths), 1000)
    rv = ss.gamma(alpha_est, loc=loc_est, scale=beta_est)
    pdf = rv.pdf(x)
    max_count = max(counts)
    pdf_scaled = pdf * max_count / max(pdf)

    return jsonify({"histogramData": histogram_data, "pdf": pdf_scaled.tolist(), "x": x.tolist()})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000') # Replace with your Next.js app origin
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    return response


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=8000)
    # enable debug mode