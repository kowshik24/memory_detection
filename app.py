# import the necessary libraries
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify,Response

# define the Flask app
app = Flask(__name__)

# load the object detection model
# replace this with the path to your trained model
#model = torch.load('C:/Users/ASUS/Memory Detection/yolov5/runs/train/exp/weights/best_1.pt', map_location=torch.device('cpu'))

#model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/ASUS/Memory Detection/yolov5/runs/train/exp/weights/best_1.pt')
force_reload=True
def detect_memory():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/ASUS/Memory Detection/yolov5/runs/train/exp/weights/best_1.pt', force_reload=True)
    return model
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
    #img = model(img)
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
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/ASUS/Memory Detection/yolov5/runs/train/exp/weights/best_1.pt',force_reload=True)
    model = detect_memory()
    #model = torch.load('C:/Users/ASUS/Memory Detection/yolov5/runs/train/exp/weights/best_1.pt',map_location=torch.device('cpu'))

    

    # Load the image and convert to RGB format
    #img = cv2.imread(r'F:\DS\TEST3\no_memory\out2.png')
   # img = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

    # Make a prediction with the model
    #model.eval()
    # if isinstance(model, torch.nn.Module):
        # for module in model.modules():
            # if isinstance(module, torch.nn.Dropout):
                # module.eval()
            # elif isinstance(module, torch.nn.BatchNorm2d):
                # module.eval()
    # pred = model(file)
    
    model.eval()
    pred = model(img)
    bboxes = pred.xyxy

    # Extract the bounding boxes from the prediction
    #bboxes = pred.xyxy
    for bbox in bboxes[0]:
        x1, y1, x2, y2 = map(int, bbox[:4])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return Response(img.tobytes(), mimetype='image/jpeg')


# run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=4000)