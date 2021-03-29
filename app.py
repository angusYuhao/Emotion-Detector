# imports for the web application and the webcam capture 
from flask import Flask, render_template, Response, redirect, url_for
from cv2 import cv2
from PIL import Image

# imports for the artificial intelligence model and other functions 
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch 
import torch.nn as nn
import numpy as np

app = Flask(__name__) # instantiates a Flask application
camera = cv2.VideoCapture(0) # used for webcam capture 

# images used when captured to display and feed into our model
display_img = None
predict_img = None

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
def prediction(img):
    # load model 
    vgg_model = models.vgg16(pretrained=False)
    vgg_model.classifier[6] = nn.Linear(4096, 7) # sets output classes to 7 emotions 
    state = torch.load("model", map_location=torch.device('cpu'))
    vgg_model.load_state_dict(state)

    # convert image to right type and dimensions 
    img = Image.fromarray(img, 'RGB')
    transform_img = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                        transforms.RandomRotation((-10, 10), expand=False, center=None, fill=0, resample=None),
                                        transforms.RandomHorizontalFlip(p=0.5)])
    img = transform_img(img)
    img = np.array(img)
    img = torch.from_numpy(img)
    img.unsqueeze_(0)

    # put image through model and obtain prediction 
    pred = vgg_model(img)
    print(pred)

    max = 0
    label_test = ["angry", "fatigue", "fear", "happy", "neutral", "sad", "surprise"]
    for i, x in enumerate(pred):
        for j, y in enumerate(pred[i]):
            if(y > max):
                max = y
                label = label_test[j]
    print(label)
    print(pred)

@app.route('/image')
def image():
    # instead of grabbing a new image here, pass in display_img and predict_img when we pressed the button
    prediction(predict_img)
    return Response(display_img, mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(prediction(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/result')
def result():
    return render_template('result.html')
    
@app.route('/button', methods=["GET", "POST"])
def button():
    # save the image captured in display_img and the prediction image in predict_img
    global display_img
    global predict_img
    display_img = get_image('bytes')
    predict_img = get_image('array')
    return redirect(url_for('result'))

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)