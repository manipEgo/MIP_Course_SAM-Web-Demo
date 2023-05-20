from flask import Flask, request, Response
from ..pipline import Pipeline
import numpy as np
import cv2

checkpoint = "/home/mpyg/Documents/Codes/MIP_SAM/model/sam_vit_h_4b8939.pth"
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
    image.save("../img/"+data+".png")
    return r'/img/{}.png'.format(data)

@app.route("/img/<imageId>.png")
def get_frame(imageId):
    with open(r'../img/{}.png'.format(imageId), 'rb') as f: 	
        image = f.read()
        resp = Response(image, mimetype="image/png")
        return resp

@app.route('/process/<imageId>.png', methods=['POST'])
def proceed_image(imageId):
    image = cv2.imread(r'../img/{}.png'.format(imageId))
    image =  pipeline.pipeline(image)
    cv2.imwrite(r'../img/{}_sam.png'.format(imageId), image)
    with open(r'../img/{}_sam.png'.format(imageId), 'rb') as f: 	
        image = f.read()
        resp = Response(image, mimetype="image/png")
        return resp

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8080, debug=True)