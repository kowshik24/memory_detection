import cv2
import io
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify,Response,send_file


app = Flask(__name__)

path = 'F:/DS/Memory_Detection/best.pt'

model = torch.hub._load_local('C:/Users/ASUS/.cache/torch/hub/yolov5','custom',path)

@app.route('/show_image',methods=['GET'])
def show_image(image):
    pass
    

@app.route('/detect_memory', methods=['POST','GET'])
def detect_memory():
# get image file from request
    image_file = request.files['image']
    img_bytes = image_file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Do something with the image (e.g. object detection)
    frame = cv2.resize(img,(1080,720))
    img = model(frame)
    img = np.squeeze(img.render())
    #img = np.array(img).flatten()

    # Return the image as a response
    #img_pil = Image.fromarray(img)
    #img_io = io.BytesIO()
    #img_pil.save(img_io, 'JPEG')
    #img_io.seek(0)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_encoded = np.array(img_encoded)
    response = img_encoded.tostring()
    return Response(response=response,status=200 ,mimetype='image/jpeg')
    
    
if __name__ == '__main__':
    app.run(debug=True,port=5000)