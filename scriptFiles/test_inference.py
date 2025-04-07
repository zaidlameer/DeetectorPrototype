import unittest
import numpy as np
import tensorflow as tf
import librosa
import time
from inference_script import load_model, extract_mfcc, measure_inference_time

class TestAudioInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the model once before running tests."""
        cls.model_path = "./audioModel/audio_rnn_model.h5"  # Update with your actual model path
        cls.audio_path = "./output_audio.aac"  # Update with your actual audio file
        cls.model = load_model(cls.model_path)

    def test_model_loading(self):
        """Test if the model loads correctly."""
        self.assertIsInstance(self.model, tf.keras.Model, "Failed to load a Keras model.")

    def test_mfcc_extraction(self):
        """Test MFCC feature extraction from audio."""
        mfccs = extract_mfcc(self.audio_path)
        self.assertIsInstance(mfccs, np.ndarray, "MFCC extraction did not return a NumPy array.")
        self.assertGreater(mfccs.shape[1], 0, "MFCC features are empty.")  # Ensure features exist

    def test_model_inference(self):
        """Test if the model produces an output from MFCC features."""
        mfccs = extract_mfcc(self.audio_path)
        prediction = self.model.predict(mfccs)
        self.assertIsInstance(prediction, np.ndarray, "Model output is not a NumPy array.")
        self.assertGreater(prediction.shape[0], 0, "Model did not return any predictions.")

    def test_inference_speed(self):
        """Test if inference time is within a reasonable range (e.g., < 1 second)."""
        avg_time = measure_inference_time(self.model, self.audio_path, num_runs=5)
        self.assertLess(avg_time, 1.0, f"Inference time is too high: {avg_time:.4f} sec.")

if __name__ == "__main__":
    unittest.main()
