import io
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request, redirect
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

MODEL_PATH = './model/dense_model.pt'
device = torch.device('cpu')
class_names = ['covid19', 'normal']

# Defining Model Architecture
def CNN_model(pretrained):
    inf_model = models.densenet121(pretrained=pretrained)
    num_ftrs = inf_model.classifier.in_features
    inf_model.classifier = nn.Linear(num_ftrs, len(class_names))
    inf_model.to(device)
    return inf_model

inf_model = CNN_model(pretrained=False)

# Loading the Model Trained Weights
inf_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
inf_model.eval()
print('Inference model Loaded on CPU')

# Image Transform
def transform_image(image_bytes):
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    if image.shape[-1] == 4:
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[-1] == 3:
        image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif len(image.shape) == 2:
        image_cv = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return test_transforms(image_cv).unsqueeze(0)

# Function to check if the image is grayscale
def is_grayscale(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    if len(image.shape) == 3:
        return np.allclose(image[:,:,0], image[:,:,1]) and np.allclose(image[:,:,1], image[:,:,2])
    return len(image.shape) == 2

# Function to get prediction
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = inf_model(tensor)
    _, prediction = torch.max(outputs, 1)
    return class_names[prediction.item()]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        # Check if the file format is JPEG or PNG
        if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_bytes = file.read()

            # Check if the image is likely an X-ray (grayscale check)
            if is_grayscale(img_bytes):
                prediction_name = get_prediction(img_bytes)
                return render_template('result.html', name=prediction_name)
            else:
                return render_template('error.html', message="Please upload a grayscale image (likely an X-ray).")
        else:
            return render_template('error.html', message="Please upload an image in JPEG or PNG format.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
