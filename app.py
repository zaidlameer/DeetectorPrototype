from flask import Flask, request, jsonify, render_template
import os
import cv2
import subprocess
import torch
import numpy as np
# import fer
import librosa
from werkzeug.utils import secure_filename
from transformers import AutoModelForImageClassification, AutoProcessor
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the models
AUDIO_MODEL_PATH = "./modelsV2/audio_classification_model.h5"
VIDEO_MODEL_DIR = "./modelsV2/distilled_deepfake_model"

audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
video_processor = AutoProcessor.from_pretrained(VIDEO_MODEL_DIR)
video_model = AutoModelForImageClassification.from_pretrained(VIDEO_MODEL_DIR)

def check_ffmpeg():
    print("Checking ffmpeg")
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False

def split_audio_video(video_path, audio_output, video_output):
    print("Splitting audio and video")
    audio_cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "copy", audio_output]
    video_cmd = ["ffmpeg", "-i", video_path, "-an", "-vcodec", "copy", video_output]
    
    try:
        subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def preprocess_audio(file_path, sr=None, n_mfcc=40):
    print("Preprocessing audio")
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs_scaled, axis=0)

def predict_audio(file_path):
    print("Predicting audio...")
    input_data = preprocess_audio(file_path)
    prediction = audio_model.predict(input_data)[0][0]
    return "Deepfake" if prediction >= 0.5 else "Real"

def extract_frames_from_video(video_path):
    print("Extracting frames from the video")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

# def extract_emotion_frames_from_video(video_path, frame_skip=5, output_folder="emotion_frames"):
    
#     print(f"Extracting emotion-based frames from {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     detector = FER()
#     frame_count = 0
#     saved_frames = []
#     last_max_emotion = None

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % frame_skip == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             try:
#                 emotions_list = detector.detect_emotions(frame_rgb)
#                 if emotions_list:
#                     max_face_emotion = max(emotions_list, key=lambda x: max(x['emotions'].values()))
#                     max_emotion_label = max(max_face_emotion['emotions'], key=max_face_emotion['emotions'].get)

#                     if last_max_emotion is None or max_emotion_label != last_max_emotion:
#                         x, y, w, h = max_face_emotion['box']
#                         x1, y1, x2, y2 = x - 10, y + 10, x - 10 + w + 20, y + 10 + h
#                         face = frame[y1:y2, x1:x2]
#                         if face is not None and face.size > 0: # Check if the face is valid.
#                             face_resized = cv2.resize(face, (128, 128))
#                             frame_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
#                             cv2.imwrite(frame_filename, face_resized)
#                             saved_frames.append(frame_filename)
#                             print(f"Saved frame {frame_count} with emotion: {max_emotion_label}")
#                             last_max_emotion = max_emotion_label

#             except Exception as e:
#                 print(f"Error processing frame {frame_count}: {e}")

#         frame_count += 1

#     cap.release()
#     print(f"Extracted {len(saved_frames)} emotion-based frames.")
#     return saved_frames


def predict_video(frames):
    print("Predicting video")
    video_model.eval()
    with torch.no_grad():
        predictions = []
        for frame in frames:
            image = Image.fromarray(frame)
            inputs = video_processor(images=image, return_tensors="pt")
            outputs = video_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            labels = video_model.config.id2label
            predictions.append(labels.get(predicted_class, f"Class {predicted_class}"))
    return max(set(predictions), key=predictions.count)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"})
    

    print("file uploaded successfully")
    file = request.files['video']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    audio_path = file_path.replace('.mp4', '.aac')
    video_path = file_path.replace('.mp4', '_video.mp4')
    
    if not check_ffmpeg() or not split_audio_video(file_path, audio_path, video_path):
        return jsonify({"error": "FFmpeg error"})
    
    print("audio video splitting completed")

    audio_result = predict_audio(audio_path)
    frames = extract_frames_from_video(video_path)
    video_result = predict_video(frames)
    print("Prediction completed")
    
    return jsonify({"audio_result": audio_result, "video_result": video_result})

if __name__ == '__main__':
    app.run(debug=True)
