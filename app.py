from flask import Flask, request, render_template, jsonify
import torch
import cv2
import numpy as np
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = "model/ViT_Deepfake_Detection"
MODEL_NAME = "Wvolf/ViT_Deepfake_Detection"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_frame(frame):
    """Convert OpenCV frame to PIL and preprocess."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0).to(device)

def predict_video(video_path):
    """Process video frame-by-frame and predict deepfake likelihood."""
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pixel_values = preprocess_frame(frame).unsqueeze(0)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            predicted_class_idx = outputs.logits.argmax(-1).item()
            predictions.append(predicted_class_idx)

    cap.release()

    if len(predictions) == 0:
        return "Error: No frames processed"

    avg_pred = sum(predictions) / len(predictions)
    return "Fake Video" if avg_pred > 0.5 else "Real Video"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    
    result = predict_video(file_path)

    return jsonify({"prediction": result, "video_path": file_path})

if __name__ == "__main__":
    app.run(debug=True)
