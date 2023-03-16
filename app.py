# import the necessary libraries
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify,Response

# define the Flask app
app = Flask(__name__)

# load the object detection model
# replace this with the path to your trained model
model = torch.load('C:/Users/ASUS/Memory Detection/yolov5/runs/train/exp/weights/best_1.pt', map_location=torch.device('cpu'))


def detect_memory(image):
    # load the image and convert it to grayscale
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply thresholding to convert the image to binary
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # apply morphological operations to remove noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # detect contours in the binary image
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw boxes around the detected contours
    img = model(img)
    img_with_boxes = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_with_boxes

# define the API endpoint
@app.route('/detect_memory', methods=['POST'])
def detect_memory_api():
    # read image file from POST request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    

    # convert image to grayscale and find contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #img = model(img)

    # draw boxes around memory regions
    for i, c in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img = model(img)
    # return the image with boxes drawn
    ret, jpeg = cv2.imencode('.jpg', img)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


# run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=4000)