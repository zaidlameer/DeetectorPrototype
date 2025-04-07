import tensorflow as tf
import numpy as np
import librosa
import argparse
import os

# Load the trained model (updated for .h5 format)
MODEL_PATH = "./audioModel/audio_rnn_model.h5"  # Changed to .h5 extension
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Function to preprocess the audio file (FLAC supported)
def preprocess_audio(file_path, sr=None, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return np.expand_dims(mfccs_scaled, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to predict deepfake or real
def predict_audio(file_path):
    input_data = preprocess_audio(file_path)
    if input_data is None:
        return
    
    prediction = model.predict(input_data)[0][0]
    label = "Deepfake" if prediction >= 0.5 else "Real"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    print(f"Prediction: {label} ({confidence:.2f} confidence)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test deepfake audio detection model on FLAC files.")
    parser.add_argument("audio_path", type=str, help="Path to the FLAC audio file")
    args = parser.parse_args()

    predict_audio(args.audio_path)
