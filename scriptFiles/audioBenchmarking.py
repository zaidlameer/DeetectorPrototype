import time
import librosa
import numpy as np
import tensorflow as tf

def load_model(model_path):
    """Load a trained .h5 model."""
    return tf.keras.models.load_model(model_path)

def extract_mfcc(audio_path, sr=22050, n_mfcc=13):
    """Convert raw audio to MFCC features."""
    y, sr = librosa.load(audio_path, sr=sr)  # Load audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    return mfccs

def measure_inference_time(model, audio_path, num_runs=10):
    """Measure inference time on an audio file."""
    mfcc_features = extract_mfcc(audio_path)
    
    # Warm-up run (avoid cold start effects)
    _ = model.predict(mfcc_features)

    # Measure inference time
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(mfcc_features)
        times.append(time.time() - start_time)

    avg_time = np.mean(times)
    print(f"Average Inference Time: {avg_time:.6f} seconds over {num_runs} runs")

if __name__ == "__main__":
    model_path = "./audioModel/audio_classification_model.h5"  # Update with your model path
    audio_path = "C:/Users/zaidl/Downloads/DEMONSTRATION/linus-original-DEMO.mp3"  # Update with your test audio file

    model = load_model(model_path)
    measure_inference_time(model, audio_path)
