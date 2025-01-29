from transformers import AutoModelForImageClassification, AutoImageProcessor

# Define local path to save the model
local_model_path = "model/ViT_Deepfake_Detection"

# Download and save model
model = AutoModelForImageClassification.from_pretrained("Wvolf/ViT_Deepfake_Detection")
model.save_pretrained(local_model_path)

# Download and save image processor
processor = AutoImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
processor.save_pretrained(local_model_path)

print("Model saved locally!")
