import os
import torch
import cv2
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Initialize model and processor
model = AutoModelForImageClassification.from_pretrained("model/")
processor = AutoImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return processor(image, return_tensors="pt").pixel_values.squeeze()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process video
            prediction = process_video(filepath)
            return render_template('index.html', prediction=prediction)
    
    return render_template('index.html')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        pixel_values = preprocess_frame(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            predicted_class_idx = outputs.logits.argmax(-1).item()
            predictions.append(predicted_class_idx)
    
    cap.release()
    average_prediction = sum(predictions) / len(predictions)
    return "Fake Video" if average_prediction > 0.5 else "Real Video"

if _name_ == '_main_':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

