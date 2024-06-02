import os
# FFmpeg 경로 설정
os.environ["IMAGEIO_FFMPEG_EXE"] = "C:\\ffmpeg\\bin\\ffmpeg.exe"

import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import ViTImageProcessor, pipeline
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import time
import numpy as np
from google.cloud import storage, speech_v1p1beta1 as speech
from pydub import AudioSegment
import speech_recognition as sr
from konlpy.tag import Okt
from gensim import corpora, models
import moviepy.editor as mp
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import re

from v2021 import SummaryModel


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# GCS 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "interview-400010-9abf1ded3113.json"
BUCKET_NAME = "profile_image_aiinterview"

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

progress_dict = {}

def update_progress(progress):
    global progress_dict
    progress_dict['progress'] = progress

@app.route('/')
def index():
    return "Flask server is running."

# GCS에 영상 저장
def upload_to_gcs(file_path, file_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_name)

    blob.upload_from_filename(file_path)
    # Public URL을 생성
    public_url = f"https://storage.googleapis.com/{bucket.name}/{file_name}"

    return public_url

# 영상 요약
def summarize_video(video_path, summary_time):
    logger.info("Loading video for summarization")
    preprocessor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", size=224)

    SAMPLE_EVERY_SEC = 2

    cap = cv2.VideoCapture(video_path)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_len = n_frames / fps

    logger.info(f"Video length (seconds): {video_len}")

    frames = []
    last_collected = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        second = timestamp // 1000

        if second % SAMPLE_EVERY_SEC == 0 and second != last_collected:
            last_collected = second
            frames.append(frame)

    features = preprocessor(images=frames, return_tensors="pt")["pixel_values"]
    model = SummaryModel.load_from_checkpoint('summary.ckpt')
    model.eval()

    y_pred = []
    for frame in features:
        y_p = model(frame.unsqueeze(0))
        y_p = torch.sigmoid(y_p)
        y_pred.append(y_p.detach().numpy().squeeze())
    y_pred = np.array(y_pred)
    THRESHOLD = 0.2

    clip = VideoFileClip(video_path)
    sorted_indices = np.argsort(-y_pred)
    selected_clips = []
    total_duration = 0

    for idx in sorted_indices:
        sec = idx * SAMPLE_EVERY_SEC
        end_sec = sec + SAMPLE_EVERY_SEC
        if total_duration + SAMPLE_EVERY_SEC <= summary_time:
            selected_clips.append((sec, end_sec))
            total_duration += SAMPLE_EVERY_SEC
        else:
            break

    selected_clips.sort()
    subclips = [clip.subclip(start, end) for start, end in selected_clips]

    if subclips:
        summarized_clip = concatenate_videoclips(subclips, method="compose")
        result_path = "videos/summary_result.mp4"
        summarized_clip.write_videofile(result_path, codec='libx264', audio_codec='aac')

        # GCS에 업로드하고 URL 반환
        gcs_url = upload_to_gcs(result_path, f"summaries/{os.path.basename(result_path)}")

        # 로컬 파일 삭제
        os.remove(result_path)

        return gcs_url
    else:
        return None

# 영상 요약 실행
@app.route('/summarize', methods=['POST'])
def summarize():
    logger.info("Received request at /summarize")
    if 'file' not in request.files or 'summaryTime' not in request.form:
        return jsonify({"error": "No file part or summary time"}), 400

    file = request.files['file']
    summary_time = int(request.form['summaryTime'])
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # # Initialize progress
    # update_progress(0)
    #
    # for i in range(1, 11):
    #     time.sleep(1)  # Simulate time-consuming task
    #     update_progress(i * 10)

    try:
        logger.info("Starting video summarization process")
        result_path = summarize_video(file_path, summary_time)
        logger.info("Video summarization process completed")

        if result_path:
            return jsonify({"result_path": result_path}), 200
        else:
            logger.error("No suitable clips found")
            return jsonify({"error": "No suitable clips found"}), 500
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        return jsonify({"error": str(e)}), 500

# @app.route('/progress', methods=['GET'])
# def get_progress():
#     progress = progress_dict.get('progress', 0)
#     return jsonify({"progress": progress})

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if not os.path.exists("videos"):
        os.makedirs("videos")
    app.run(host='0.0.0.0', port=5000)