import os
import re
import json
import numpy as np
import cv2
import tempfile
import whisper
import mediapipe as mp
from moviepy import VideoFileClip, CompositeAudioClip

# Step 1: Extract audio and generate transcript from video using Whisper
def extract_audio_and_transcribe(video_path):
    print("Extracting audio and transcribing...")
    temp_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    # Extract audio using FFmpeg (PyAV can't easily write raw WAV directly)
    os.system(f"ffmpeg -i \"{video_path}\" -vn -acodec pcm_s16le -ar 16000 -ac 1 \"{temp_audio_path}\" -y")

    model = whisper.load_model("base")
    result = model.transcribe(temp_audio_path)
    os.remove(temp_audio_path)

    # Return as list of tuples (start, end, text)
    transcript = [(seg['start'], seg['end'], seg['text']) for seg in result['segments']]
    return transcript

def restore_audio(silent_video_path, video_with_audio_path, new_video_file_name):
    # 1) load the overlaid silent video
    video = VideoFileClip(silent_video_path)

    # 2) grab the audio from the original
    audio = VideoFileClip(video_with_audio_path).audio

    new_audioclip = CompositeAudioClip([audio])
    video.audio = new_audioclip

    video.write_videofile(new_video_file_name)
    print("New video file created!")


# 1) sanitize transcript words
def sanitize_word(w):
    w = w.lower().strip()
    return re.sub(r"[^a-z0-9]", "", w)

# 2) build per-word timings
def build_word_timings(transcript):
    word_times = []
    for start, end, text in transcript:
        raw = text.strip().split()
        dur = end - start
        for i, rw in enumerate(raw):
            w = sanitize_word(rw)
            if not w: 
                continue
            w_start = start + (i/len(raw))*dur
            w_end   = start + ((i+1)/len(raw))*dur
            word_times.append({"word": w, "start": w_start, "end": w_end})
    return word_times

# load GT JSONs
def load_gt_landmarks(folder="landmark_data"):
    gt = {}
    for fn in os.listdir(folder):
        if not fn.endswith(".json"):
            continue
        w = fn[:-5]
        gt[w] = json.load(open(os.path.join(folder, fn)))
    return gt


mp_pose      = mp.solutions.pose
mp_hands     = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

POSE_CONNS = mp_pose.POSE_CONNECTIONS
HAND_CONNS = mp_hands.HAND_CONNECTIONS
FACE_CONNS = mp_face_mesh.FACEMESH_TESSELATION

def create_bold_stickman(flm, width=350, height=350):
    """
    RGBA canvas showing:
     - Pose skeleton in thick white
     - Facial mesh in bold green
     - Hands in thick magenta/cyan
    """
    im = np.zeros((height, width, 4), np.uint8)
    def P(idx_list, idx):
        pt = idx_list[idx]
        return int(pt["x"]*width), int(pt["y"]*height)

    # draw connections
    def draw(conns, pts, color, thick):
        for a,b in conns:
            if a < len(pts) and b < len(pts):
                cv2.line(im, P(pts, a), P(pts, b), color, thick, cv2.LINE_AA)

    # 1) Pose
    if "pose" in flm:
        draw(POSE_CONNS, flm["pose"], (255,255,255,230), thick=6)

    # 2) Face
    if "face" in flm:
        draw(FACE_CONNS, flm["face"], (0,255,0,180), thick=2)
        # highlight eyes/lips
        for i in (33,133,362,263):
            cv2.circle(im, P(flm["face"],i), 4, (0,200,0,255), -1)

    # 3) Left hand
    if flm.get("left_hand"):
        draw(HAND_CONNS, flm["left_hand"], (255,0,255,220), thick=4)
        for p in flm["left_hand"]:
            cv2.circle(im, (int(p["x"]*width), int(p["y"]*height)), 4, (255,0,255,255), -1)

    # 4) Right hand
    if flm.get("right_hand"):
        draw(HAND_CONNS, flm["right_hand"], (0,255,255,220), thick=4)
        for p in flm["right_hand"]:
            cv2.circle(im, (int(p["x"]*width), int(p["y"]*height)), 4, (0,255,255,255), -1)

    return im