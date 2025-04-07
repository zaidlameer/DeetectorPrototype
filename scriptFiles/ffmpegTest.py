import subprocess
import os

# Define input MP4 file and output paths
input_file = "C:/Users/zaidl/Downloads/DEMONSTRATION/real_billGates.mp4"  # Change this to your actual file
output_audio = "output_audio.aac"
output_video = "output_video.mp4"

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("FFmpeg is installed and working.")
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not added to PATH.")
        return False
    return True

def split_audio_video(input_file, output_audio, output_video):
    """Split audio and video using FFmpeg."""
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return False

    # Extract audio
    audio_cmd = ["ffmpeg", "-i", input_file, "-vn", "-acodec", "copy", output_audio]
    # Extract video (without audio)
    video_cmd = ["ffmpeg", "-i", input_file, "-an", "-vcodec", "copy", output_video]

    try:
        subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("Audio and video extraction completed.")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return False
    return True

def verify_outputs(output_audio, output_video):
    """Check if the extracted files exist."""
    audio_exists = os.path.exists(output_audio)
    video_exists = os.path.exists(output_video)
    
    if audio_exists:
        print(f"✅ Audio file extracted: {output_audio}")
    else:
        print(f"❌ Audio extraction failed.")
    
    if video_exists:
        print(f"✅ Video file extracted: {output_video}")
    else:
        print(f"❌ Video extraction failed.")
    
    return audio_exists and video_exists

if __name__ == "__main__":
    if check_ffmpeg():
        if split_audio_video(input_file, output_audio, output_video):
            verify_outputs(output_audio, output_video)
