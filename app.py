from flask import Flask, request, jsonify, render_template
import os
import cv2
import subprocess
import torch
import numpy as np
import librosa
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from transformers import AutoModelForImageClassification
from torchvision import transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

AUDIO_MODEL_PATH = "./models_final/audio_classification_model.h5"
IMAGE_MODEL_PATH = "./models_final/distilled_vit_deepfake_model"

# load in the models 
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
image_model = AutoModelForImageClassification.from_pretrained(IMAGE_MODEL_PATH, num_labels=2, ignore_mismatched_sizes=True)
image_model.eval()

LABELS = {0: "Real", 1: "Fake"}

preprocess_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False

def split_audio_video(video_path, audio_output, video_output):
    try:
        subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-acodec", "copy", audio_output], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["ffmpeg", "-i", video_path, "-an", "-vcodec", "copy", video_output], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def preprocess_audio(file_path, sr=None, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs_scaled, axis=0)

def predict_audio(file_path):
    input_data = preprocess_audio(file_path)
    prediction = audio_model.predict(input_data)[0][0]
    return "Fake" if prediction >= 0.5 else "Real"

def extract_frames_from_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = sorted(np.random.choice(range(total_frames), num_frames, replace=False))

    frames = []
    current_idx = 0
    selected_idx = 0

    while cap.isOpened() and selected_idx < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx == frame_indices[selected_idx]:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            selected_idx += 1
        current_idx += 1

    cap.release()
    return frames

def predict_video(frames):
    with torch.no_grad():
        predictions = []
        for frame in frames:
            image = Image.fromarray(frame)
            input_tensor = preprocess_image(image).unsqueeze(0)
            outputs = image_model(input_tensor)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predictions.append(LABELS.get(predicted_class, "Unknown"))
    return max(set(predictions), key=predictions.count)

def cleanup_files(*file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['video']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    audio_path = file_path.replace('.mp4', '.aac')
    video_path = file_path.replace('.mp4', '_video.mp4')

    if not check_ffmpeg() or not split_audio_video(file_path, audio_path, video_path):
        cleanup_files(file_path)
        return jsonify({"error": "FFmpeg error"})
    
    audio_result = predict_audio(audio_path)
    frames = extract_frames_from_video(video_path)
    video_result = predict_video(frames)

    cleanup_files(file_path, audio_path, video_path)

    return jsonify({"audio_result": audio_result, "video_result": video_result})

if __name__ == '__main__':
    app.run(debug=True)
