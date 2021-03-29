# imports for the web application and the webcam capture 
from flask import Flask, render_template, Response, redirect, url_for
from cv2 import cv2
from PIL import Image

# imports for the artificial intelligence model and other functions 
import torchvision
import torchvision.models as models
import torch 
import torch.nn as nn
import numpy as np

app = Flask(__name__) # instantiates a Flask application
camera = cv2.VideoCapture(0) # used for webcam capture 

# returns a stream of webcam VIDEO capture 
def generate_frames(): 
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes() # converting to bytes otherwise Flask will complain 
            img = (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            yield (img)

# get image for model prediction, ret_type = 'bytes' or 'array'
def get_image(ret_type): 
    success, frame = camera.read()
    if success:
        if (ret_type == 'bytes'):
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes() # converting to bytes otherwise Flask will complain 
            img = (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return img
        elif (ret_type == 'array'):
            return frame

# gets the prediction of the mood from the image 
def prediction():
    # load model 
    vgg_model = models.vgg16(pretrained=False)
    vgg_model.classifier[6] = nn.Linear(4096, 7) # sets output classes to 7 emotions 
    state = torch.load("model", map_location=torch.device('cpu'))
    vgg_model.load_state_dict(state)

    # convert image to right type and dimensions 
    img = get_image('array')
    img = Image.fromarray(img, 'RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = torch.from_numpy(img)
    print(img)
    print(img.shape)

    # put image through model and obtain prediction 
    # pred = vgg_model(img)
    # print(pred)

@app.route('/image')
def image():
    prediction()
    return Response(get_image('bytes'), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/result')
def result():
    return render_template('result.html')
    
@app.route('/button', methods=["GET", "POST"])
def button():
    return redirect(url_for('result'))

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)