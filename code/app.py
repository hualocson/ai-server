# Serve model as a flask application
from ultralytics import YOLO
import numpy as np
from flask import Flask, request, send_file

model = None
app = Flask(__name__)


def load_model():
    global model
    # model = YOLO('./model/model_final.pth')
    model = YOLO("yolov8m-seg.pt")


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
            if '.jpg' in image_name or '.png' in image_name or '.jpeg' in image_name:
                image.save(image_name)
                # load image to model and get prediction
                results=model(image_name, save=True)
                # results = model('https://ultralytics.com/images/bus.jpg', save=True)  # predict on an image
                for result in results:
                    boxes = result.boxes  # Boxes object for bounding box outputs
                    masks = result.masks  # Masks object for segmentation masks outputs
                    keypoints = result.keypoints  # Keypoints object for pose outputs
                    probs = result.probs  # Probs object for classification outputs
                    obb = result.obb  # Oriented boxes object for OBB outputs
                    result.save(filename='result.jpg')  # save to disk
                # return prediction
                return send_file("result.jpg",mimetype='image/jpg')
            else:
                return {"error":"select you image file"}
    except Exception as e:
        return {"error":str(e)}

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000') # Replace with your Next.js app origin
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    return response


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)