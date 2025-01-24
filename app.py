import streamlit as st
import subprocess
import os
from TTS.api import TTS
from PIL import Image

# Streamlit UI
st.set_page_config("DH")
st.title("Digital Human Generator")

# Step 1: Take input from the user
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
user_text = st.text_area("Enter the text for speech synthesis")

audio_output_path = "output_speech.wav"
video_output_path = "final_video.mp4"
wav2lip_path = "Wav2Lip"  # Path to Wav2Lip directory

if st.button("Generate Digital Human"):
    if uploaded_image is None or not user_text:
        st.error("Please upload an image and enter text to proceed.")
    else:
        # Save uploaded image locally
        image_path = "input_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        # Step 2: Generate speech using Coqui TTS
        st.write("Generating speech...")
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        tts.tts_to_file(text=user_text, file_path=audio_output_path)
        st.success("Speech synthesis complete.")

        # Step 3: Generate video with Wav2Lip
        st.write("Generating video...")
        os.chdir(wav2lip_path)
        subprocess.run([
            "python", "inference.py",
            "--checkpoint_path", "checkpoints/wav2lip.pth",
            "--face", f"../{image_path}",
            "--audio", f"../{audio_output_path}",
            "--outfile", f"../{video_output_path}"
        ])
        os.chdir("..")

        st.success("Video generation complete.")

        # Display the final video
        st.video(video_output_path)
