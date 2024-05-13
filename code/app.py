# Serve model as a flask application
from ultralytics import YOLO
import numpy as np
from flask import Flask, request

model = None
app = Flask(__name__)


def load_model():
    global model
    model = YOLO('./model/model_final.pth')


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
            if '.jpg' in image_name:
                image.save(image_name)
                # load image to model and get prediction
                results=model(image_name)
                for result in results:
                    boxes = result.boxes  # Boxes object for bounding box outputs
                    masks = result.masks  # Masks object for segmentation masks outputs
                    keypoints = result.keypoints  # Keypoints object for pose outputs
                    probs = result.probs  # Probs object for classification outputs
                    obb = result.obb  # Oriented boxes object for OBB outputs
                    result.save(filename='result.jpg')  # save to disk
                # return prediction
                return {"response"}
            else:
                return {"error":"select you image file"}
    except Exception as e:
        return {"error":str(e)}



if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)