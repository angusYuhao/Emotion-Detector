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
import torch.nn.functional as F
import numpy as np

app = Flask(__name__) # instantiates a Flask application
camera = cv2.VideoCapture(0) # used for webcam capture 

# images used when captured to display and feed into our model
display_img = None
predict_img = None
display_label = None

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

# get image for model prediction
# returns img_bytes(in bytes), img_arr(in array)
def get_image(): 
    success, frame = camera.read()
    img_arr = frame
    if success:
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes() # converting to bytes otherwise Flask will complain 
        img_bytes = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return img_bytes, img_arr 

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
                                        transforms.RandomHorizontalFlip(p=0.5), transforms.Grayscale(num_output_channels=3)])
    
    img = transform_img(img)
    img = np.array(img)
    img = torch.from_numpy(img)
    img.unsqueeze_(0)

    # put image through model and obtain prediction 
    pred = vgg_model(img)
    prob = F.softmax(pred, dim=1)[0]
    # gets the emotion with the top probability
    first_value, first_index = prob.max(0)
    prob[first_index] = 0
    # gets the emotion with the second best probability
    second_value, second_index = prob.max(0)

    max = 0
    label_test = ["angry", "fatigue", "fear", "happy", "neutral", "sad", "surprise"]

    for i, x in enumerate(pred):
        for j, y in enumerate(pred[i]):
            if(y > max):
                max = y
                label = label_test[j]

    return label, first_value.item() * 100, second_value.item() * 100, label_test[first_index.item()], label_test[second_index.item()]

@app.route('/image')
def image():
    # instead of grabbing a new image here, pass in display_img and predict_img when we pressed the button
    return Response(display_img, mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/result')
def result():
    global display_label
    display_label, first_perc, second_perc, first_value, second_value = prediction(predict_img)
    return render_template('result.html', label=display_label, first_perc=first_perc, first_value=first_value, second_perc=second_perc, second_value=second_value)

# the temporary template that will be loaded when the user presses the capture button, will then render the results route 
@app.route('/loading')
def loading():
    return render_template('loading.html', image=display_img)
    
@app.route('/button_capture', methods=["GET", "POST"])
def button_capture():
    # save the image captured in display_img and the prediction image in predict_img
    global display_img
    global predict_img
    display_img, predict_img = get_image()
    return redirect(url_for('loading'))

@app.route('/button_again', methods=["GET", "POST"])
def button_again():   
    return redirect(url_for('index')) 

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)