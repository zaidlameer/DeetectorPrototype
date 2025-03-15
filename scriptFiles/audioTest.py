# prompt: use the model "audioModel/Deepfake_Audio_Detection" to test whether the audio is real or fake 

import librosa
import torch
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Define local path to the saved model
local_model_path = "model/Deepfake_Audio_Detection"

# Load the model and feature extractor
model = AutoModelForAudioClassification.from_pretrained(local_model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(local_model_path)

def predict_audio(audio_file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_file_path, sr=16000)  # Ensure consistent sample rate

    # Preprocess the audio using the feature extractor
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted class probabilities
    predicted_class_probs = torch.softmax(logits, dim=-1).numpy()[0]

    # Get the predicted class label
    predicted_class_index = np.argmax(predicted_class_probs)

    # Define class labels (adjust if needed based on model output)
    class_labels = ["Real", "Fake"] # Example, adapt to your model
    
    predicted_label = class_labels[predicted_class_index]

    # Return prediction with probability
    return predicted_label, predicted_class_probs[predicted_class_index]


# Example usage
audio_file = "C:/Users/zaidl/Downloads/DEMONSTRATION/linus-original-DEMO.mp3" # Replace with the actual path to your audio file
prediction, probability = predict_audio(audio_file)

print(f"Prediction: {prediction}")
print(f"Probability: {probability}")
