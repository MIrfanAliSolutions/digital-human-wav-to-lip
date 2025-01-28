import streamlit as st
import subprocess
import os
from TTS.api import TTS
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

def resize_image(image_path, target_size=1024):
    """
    Resize image to target size while maintaining aspect ratio and padding with transparency.
    """
    try:
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Check if the image has an alpha channel
        if img.shape[2] == 3:  # No alpha channel, add one
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # Get original dimensions
        height, width = img.shape[:2]
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_size / width, target_size / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create blank canvas with transparency (RGBA)
        final_image = np.zeros((target_size, target_size, 4), dtype=np.uint8)
        
        # Calculate padding
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        
        # Place resized image on canvas
        final_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return final_image
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

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
        try:
            # Save uploaded image locally
            original_image_path = "input_image_original.jpg"
            resized_image_path = "input_image.jpg"
            
            with open(original_image_path, "wb") as f:
                f.write(uploaded_image.read())
            
            # # Resize image to 256x256
            st.write("Resizing image...")
            resized_image = resize_image(original_image_path)
            if resized_image is not None:
                cv2.imwrite(resized_image_path, resized_image)
                st.success("Image resizing complete.")
            else:
                st.error("Failed to resize image.")
                st.stop()

            # Step 2: Generate speech using Coqui TTS
            st.write("Generating speech...")
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            tts.tts_to_file(text=user_text, file_path=audio_output_path)
            st.success("Speech synthesis complete.")

            # Step 3: Generate video with Wav2Lip
            st.write("Generating video...")
            wav2lip_cmd = [
                "python", "inference.py",
                "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
                "--face", f"../{resized_image_path}",
                "--audio", f"../{audio_output_path}",
                "--outfile", f"../{video_output_path}"
            ]
            
            os.chdir(wav2lip_path)
            process = subprocess.run(wav2lip_cmd, capture_output=True, text=True)
            os.chdir("..")
            
            if process.returncode != 0:
                st.error(f"Error in Wav2Lip processing: {process.stderr}")
                st.stop()
            
            st.success("Video generation complete.")

            # Display the final video
            st.video(video_output_path)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Clean up temporary files
            for file_path in [original_image_path, resized_image_path, audio_output_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            # Uncomment the following line if you want to remove the output video as well
            # if os.path.exists(video_output_path):
            #     os.remove(video_output_path)