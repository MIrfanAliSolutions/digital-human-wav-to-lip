from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
from models import Wav2Lip
import platform
import mediapipe as mp

# Initialize Mediapipe utilities
mp_face_mesh = mp.solutions.face_mesh

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos using Wav2Lip models with Mediapipe Face Mesh')

parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to saved Wav2Lip checkpoint')
parser.add_argument('--face', type=str, required=True, help='Filepath of video/image containing faces')
parser.add_argument('--audio', type=str, required=True, help='Filepath of video/audio for lip-syncing')
parser.add_argument('--outfile', type=str, default='results/result_voice.mp4', help='Path to save the result video')
parser.add_argument('--static', type=bool, default=False, help='Use only the first video frame for inference')
parser.add_argument('--fps', type=float, default=25., help='FPS for the output video')
parser.add_argument('--wav2lip_batch_size', type=int, default=128, help='Batch size for Wav2Lip model(s)')
parser.add_argument('--resize_factor', default=1, type=int, help='Resize factor for input video/image')
parser.add_argument('--nosmooth', default=False, action='store_true', help='Disable smoothing of face detections')

args = parser.parse_args()
args.img_size = 96

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for inference.')

def face_detect_with_mediapipe(image):
    """Detect face and lips using Mediapipe Face Mesh."""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise ValueError("Face not detected in the input image/video.")
        
        # Get the bounding box of the detected face
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)

            # Add padding to the bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)

            return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for i, mel in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame = frames[idx].copy()

        # Detect the face using Mediapipe
        face, coords = face_detect_with_mediapipe(frame)

        # Resize the face to the model's input size
        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(mel)
        frame_batch.append(frame)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            yield prepare_batches(img_batch, mel_batch, frame_batch, coords_batch)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        yield prepare_batches(img_batch, mel_batch, frame_batch, coords_batch)

def prepare_batches(img_batch, mel_batch, frame_batch, coords_batch):
    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

    img_masked = img_batch.copy()
    img_masked[:, args.img_size // 2:] = 0

    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

    return img_batch, mel_batch, frame_batch, coords_batch

def load_model(path):
    model = Wav2Lip()
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    return model.to(device).eval()

def main():
    if not os.path.isfile(args.face):
        raise ValueError("--face must be a valid path to a video/image file")

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            full_frames.append(frame)
        video_stream.release()

    print(f"Number of frames available for inference: {len(full_frames)}")

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    for i in range(len(mel[0]) // 16):
        start_idx = int(i * mel_idx_multiplier)
        mel_chunks.append(mel[:, start_idx:start_idx + 16])

    full_frames = full_frames[:len(mel_chunks)]

    model = load_model(args.checkpoint_path)
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    for img_batch, mel_batch, frames, coords in tqdm(datagen(full_frames, mel_chunks)):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            x1, y1, x2, y2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    command = f'ffmpeg -y -i {args.audio} -i temp/result.avi -strict -2 -q:v 1 {args.outfile}'
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    main()
