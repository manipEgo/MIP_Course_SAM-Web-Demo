from flask import Flask, request, Response

app = Flask(__name__)

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

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8080, debug=True)