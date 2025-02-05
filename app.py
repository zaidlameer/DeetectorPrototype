from flask import Flask, request, render_template, jsonify
import torch
import librosa
import numpy as np
import cv2
import ffmpeg
import os
from transformers import (
    AutoModelForAudioClassification, AutoFeatureExtractor,
    AutoModelForImageClassification, AutoImageProcessor
)

app = Flask(__name__)

# Define local paths to the saved models
AUDIO_MODEL_PATH = "model/Deepfake_Audio_Detection"
VIDEO_MODEL_PATH = "model/ViT_Deepfake_Detection"

# Load models
audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu").eval()
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_PATH)
video_model = AutoModelForImageClassification.from_pretrained(VIDEO_MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu").eval()
video_processor = AutoImageProcessor.from_pretrained(VIDEO_MODEL_PATH)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def extract_audio_from_video(video_path, audio_output_path):
    """Extracts audio from video using ffmpeg."""
    try:
        ffmpeg.input(video_path).output(audio_output_path, format='wav', acodec='pcm_s16le', ar='16000').run(overwrite_output=True, quiet=True)
        if not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) == 0:
            return False
        return True
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return False

def extract_frames_from_video(video_path):
    """Extracts frames from video using OpenCV."""
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

def predict_audio(audio_file_path):
    """Predict if the audio is deepfake."""
    audio, sample_rate = librosa.load(audio_file_path, sr=16000)
    inputs = audio_feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt").to(audio_model.device)
    
    with torch.no_grad():
        logits = audio_model(**inputs).logits
    
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    class_labels = ["Real", "Fake"]
    predicted_index = np.argmax(probabilities)
    return class_labels[predicted_index], probabilities[predicted_index]

def predict_video(video_file_path):
    """Predict if the video is deepfake."""
    frames = extract_frames_from_video(video_file_path)
    if not frames:
        return None, 0.0
    
    predictions = []
    for frame in frames[::10]:  # Process every 10th frame for efficiency
        inputs = video_processor(images=frame, return_tensors="pt").to(video_model.device)
        with torch.no_grad():
            logits = video_model(**inputs).logits
            predicted_index = logits.argmax(-1).item()
            predictions.append(predicted_index)
    
    avg_prediction = np.mean(predictions)
    return "Fake Video" if avg_prediction > 0.5 else "Real Video", avg_prediction

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    print("break point here")
    
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(video_path)
    
    audio_path = video_path.replace(".mp4", ".wav")
    has_audio = extract_audio_from_video(video_path, audio_path)
    
    try:
        audio_pred, audio_prob = ("No Audio", 0.0) if not has_audio else predict_audio(audio_path)
        video_pred, video_prob = predict_video(video_path)
        
        final_prediction = "Fake" if (audio_pred == "Fake" or video_pred == "Fake Video") else "Real"
        
        result = {
            "audio_prediction": audio_pred,
            "audio_probability": audio_prob,
            "video_prediction": video_pred,
            "video_probability": video_prob,
            "final_prediction": final_prediction
        }
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
