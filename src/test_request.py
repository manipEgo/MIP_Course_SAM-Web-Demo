import requests
import json
from os import path

file_path = path.abspath(__file__)
project_path = path.dirname(path.dirname(file_path))

# upload
url = 'http://0.0.0.0:8080/upload'
files = {'image':open(path.join(project_path, "img/img_0_clip_0.jpg"), "rb")}
data = {'filename':'test_upload'}
r=requests.post(url,files=files,data=data)
print(r.text)

# proceed
# url = 'http://0.0.0.0:8080/process/test_upload.png'
# r=requests.post(url,files=files,data=data)