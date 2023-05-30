from flask import Flask, request, send_file
from flask_cors import *
import numpy as np
import cv2

from os import path

from pipline import Pipeline
from classifier import VGG16

file_path = path.abspath(__file__)
project_path = path.dirname(path.dirname(path.dirname(file_path)))

checkpoint = path.join(project_path, "model/sam_vit_h_4b8939.pth")
classifier = path.join(project_path, "model/classifier_0.pth")
model_type = "vit_h"

app = Flask(__name__)
pipeline = Pipeline(checkpoint, model_type, classifier)

@app.route('/upload', methods=['POST'])
@cross_origin(supports_credentials=True)
def upload_image():
    image = request.files.get("image")
    data = request.form.get("filename")

    print(data)
    print(image)
    if image is None:
        return "image not found"
    image.save(path.join(project_path, r'img/{}.png'.format(data)))
    return r'{}'.format(data)

@app.route("/img/<imageId>.png")
@cross_origin(supports_credentials=True)
def get_frame(imageId):
    resp = send_file(path.join(project_path, r'img/{}.png'.format(imageId)), mimetype="image/png")
    return resp

@app.route('/process/<imageId>.png')
@cross_origin(supports_credentials=True)
def get_sam_image(imageId):
    image = cv2.imread(path.join(project_path, r'img/{}.png'.format(imageId)))
    image, mask1, mask2, mask3 =  pipeline.pipeline(image)
    cv2.imwrite(path.join(project_path, r'img/{}_sam.png'.format(imageId)), image)
    cv2.imwrite(path.join(project_path, r'img/{}_mask1.png'.format(imageId)), mask1)
    cv2.imwrite(path.join(project_path, r'img/{}_mask2.png'.format(imageId)), mask2)
    cv2.imwrite(path.join(project_path, r'img/{}_mask3.png'.format(imageId)), mask3)
    resp = send_file(path.join(project_path, r'img/{}_sam.png'.format(imageId)), mimetype="image/png")
    return resp

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)