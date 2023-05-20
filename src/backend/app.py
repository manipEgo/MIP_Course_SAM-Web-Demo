from flask import Flask, request, Response
from pipline import Pipeline
import numpy as np
import cv2

from os import path

file_path = path.abspath(__file__)
project_path = path.dirname(path.dirname(path.dirname(file_path)))

checkpoint = path.join(project_path, "model/sam_vit_h_4b8939.pth")
model_type = "vit_h"

app = Flask(__name__)
pipeline = Pipeline(checkpoint, model_type)

@app.route('/upload', methods=['POST'])
def upload_image():
    image = request.files.get("image")
    data = request.form.get("filename")

    print(data)
    print(image)
    if image is None:
        return "image not found"
    image.save(path.join(project_path, r'img/{}.png'.format(data)))
    return r'/img/{}.png'.format(data)

@app.route("/img/<imageId>.png")
def get_frame(imageId):
    with open(path.join(project_path, r'img/{}.png'.format(imageId)), 'rb') as f: 	
        image = f.read()
        resp = Response(image, mimetype="image/png")
        return resp

@app.route('/process/<imageId>.png')
def get_sam_image(imageId):
    image = cv2.imread(path.join(project_path, r'img/{}.png'.format(imageId)))
    image =  pipeline.pipeline(image)
    cv2.imwrite(path.join(project_path, r'img/{}_sam.png'.format(imageId)), image)
    with open(path.join(project_path, r'img/{}_sam.png'.format(imageId)), 'rb') as f: 	
        image = f.read()
        resp = Response(image, mimetype="image/png")
        return resp

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)