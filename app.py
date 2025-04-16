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
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

AUDIO_MODEL_PATH = "./models_final/audio_classification_model.h5"
IMAGE_MODEL_PATH = "./models_final/distilled_vit_deepfake_model"

# Load models
try:
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
    image_model = AutoModelForImageClassification.from_pretrained(IMAGE_MODEL_PATH, num_labels=2, ignore_mismatched_sizes=True)
    image_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")

LABELS = {0: "Real", 1: "Fake"}

preprocess_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_ffmpeg():
    print("Checking for ffmpeg")
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False

def split_audio_video(video_path, audio_output, video_output):
    print("Splitting Audio and Video")
    try:
        subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-acodec", "copy", audio_output], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["ffmpeg", "-i", video_path, "-an", "-vcodec", "copy", video_output], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def preprocess_audio(file_path, sr=None, n_mfcc=40):
    print("Preprocess audio")
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return np.expand_dims(mfccs_scaled, axis=0)
    except Exception as e:
        raise ValueError(f"Error processing audio file: {e}")

def predict_audio(file_path):
    print("Predicting the Audio")
    try:
        input_data = preprocess_audio(file_path)
        prediction = audio_model.predict(input_data)[0][0]
        label = "Fake" if prediction >= 0.5 else "Real"
        confidence = float(prediction if label == "Fake" else 1 - prediction)
        return label, confidence
    except Exception as e:
        raise RuntimeError(f"Audio prediction failed: {e}")

def has_audio_stream(video_path):
    print("Checking if Video has audio Stream")
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False

# def extract_frames_from_video(video_path, num_frames=100):
#     print("Extracting the frames from the video")
#     try:
#         cap = cv2.VideoCapture(video_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         if total_frames < num_frames:
#             frame_indices = list(range(total_frames))
#         else:
#             frame_indices = sorted(np.random.choice(range(total_frames), num_frames, replace=False))

#         frames = []
#         current_idx = 0
#         selected_idx = 0

#         while cap.isOpened() and selected_idx < len(frame_indices):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if current_idx == frame_indices[selected_idx]:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
#                 selected_idx += 1
#             current_idx += 1

#         cap.release()

#         if not frames:
#             raise ValueError("No frames extracted from video.")
#         return frames

#     except Exception as e:
#         raise RuntimeError(f"Error extracting frames: {e}")

def extract_frames_from_video(video_path, num_frames=50):
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if video_fps > 0:
            target_fps = min(video_fps, 30)
            frame_interval = int(round(video_fps / target_fps))
        else:
            frame_interval = 1 # Default to extracting every frame if fps is not available

        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frame_indices = sorted(list(set(indices))) # Remove potential duplicates

        frames = []
        current_frame_index = 0
        extracted_frame_count = 0

        while cap.isOpened() and extracted_frame_count < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_index in frame_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_frame_count += 1

            current_frame_index += 1

        cap.release()

        if not frames:
            raise ValueError("No frames extracted from video.")
        return frames

    except Exception as e:
        raise RuntimeError(f"Error extracting frames: {e}")

def predict_video(frames):
    print("Predicting the video")
    try:
        with torch.no_grad():
            predictions = []
            confidences = []
            for frame in frames:
                image = Image.fromarray(frame)
                input_tensor = preprocess_image(image).unsqueeze(0)
                outputs = image_model(input_tensor)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = float(torch.max(probabilities).item())
                predictions.append(predicted_class)
                confidences.append(confidence)

            final_label_id = max(set(predictions), key=predictions.count)
            final_label = LABELS.get(final_label_id, "Unknown")
            final_confidence = float(np.mean([confidences[i] for i in range(len(predictions)) if predictions[i] == final_label_id]))

        return final_label, final_confidence

    except Exception as e:
        raise RuntimeError(f"Video prediction failed: {e}")

def cleanup_files(*file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"error": "Empty file name."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: mp4, mov, avi, mkv."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    audio_path = file_path.rsplit('.', 1)[0] + '.aac'
    video_path = file_path.rsplit('.', 1)[0] + '_video.mp4'

    if not check_ffmpeg():
        cleanup_files(file_path)
        return jsonify({"error": "FFmpeg is not installed or not found in PATH."}), 500

    audio_result = "No audio stream found"
    audio_confidence = None

    try:
        if has_audio_stream(file_path):
            if not split_audio_video(file_path, audio_path, video_path):
                raise RuntimeError("Error splitting audio and video. Invalid or corrupted video file.")

            audio_result, audio_confidence = predict_audio(audio_path)

        # Video frame extraction and prediction
        frames = extract_frames_from_video(video_path if os.path.exists(video_path) else file_path)
        video_result, video_confidence = predict_video(frames)

        response = {
            "audio_result": audio_result if audio_confidence is None else audio_result,
            "audio_confidence": None if audio_confidence is None else round(audio_confidence * 100, 2),
            "video_result": video_result,
            "video_confidence": round(video_confidence * 100, 2)
        }

    except Exception as e:
        response = {"error": str(e)}
        return jsonify(response), 500

    finally:
        cleanup_files(file_path, audio_path, video_path)

    return jsonify(response)

if __name__ == '__main__':
    if not check_ffmpeg():
        print("Error: FFmpeg is required but was not found.")
        exit(1)
    app.run(debug=True)
