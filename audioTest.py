from flask import Flask, request, render_template, jsonify
import torch
import librosa
import numpy as np
import cv2
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, AutoModelForImageClassification, AutoImageProcessor
import os

app = Flask(__name__)

# Define local paths to the saved models
AUDIO_MODEL_PATH = "model/Deepfake_Audio_Detection"
VIDEO_MODEL_PATH = "model/ViT_Deepfake_Detection"

# Load the audio model and feature extractor
audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_PATH)
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_PATH)

# Load the video model and processor
video_model = AutoModelForImageClassification.from_pretrained(VIDEO_MODEL_PATH)
video_processor = AutoImageProcessor.from_pretrained(VIDEO_MODEL_PATH)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model.to(device)
video_model.to(device)
audio_model.eval()
video_model.eval()

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict_audio(audio_file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_file_path, sr=16000)  # Ensure consistent sample rate

    # Preprocess the audio using the feature extractor
    inputs = audio_feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        logits = audio_model(**inputs).logits

    # Get predicted class probabilities
    predicted_class_probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Get the predicted class label
    predicted_class_index = np.argmax(predicted_class_probs)

    # Define class labels (adjust if needed based on model output)
    class_labels = ["Real", "Fake"]  # Example, adapt to your model
    
    predicted_label = class_labels[predicted_class_index]

    # Return prediction with probability
    return predicted_label, predicted_class_probs[predicted_class_index]

def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = video_processor(images=image, return_tensors="pt").to(device)
    return inputs["pixel_values"]

def predict_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pixel_values = preprocess_frame(frame).unsqueeze(0)
        with torch.no_grad():
            outputs = video_model(pixel_values=pixel_values)
            predicted_class_idx = outputs.logits.argmax(-1).item()
            predictions.append(predicted_class_idx)

    cap.release()
    
    if not predictions:
        return None
    
    # Return the average prediction as a simple fusion strategy
    return np.mean(predictions)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_audio", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        prediction, probability = predict_audio(file_path)
        result = {"prediction": prediction, "probability": probability}
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify(result)

@app.route("/predict_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        prediction = predict_video(file_path)
        result = {"prediction": "Fake Video" if prediction > 0.5 else "Real Video"}
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
