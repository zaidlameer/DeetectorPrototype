from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Define local path to save the model
local_model_path = "model/Deepfake_Audio_Detection"

# Download and save model
model = AutoModelForAudioClassification.from_pretrained("MelodyMachine/Deepfake-audio-detection-V2")
model.save_pretrained(local_model_path)

# Download and save audio feature extractor instead of processor
feature_extractor = AutoFeatureExtractor.from_pretrained("MelodyMachine/Deepfake-audio-detection-V2")
feature_extractor.save_pretrained(local_model_path)

print("Audio model and feature extractor saved locally!")