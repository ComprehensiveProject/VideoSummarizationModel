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
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip
import time
import numpy as np
from google.cloud import storage, speech_v1p1beta1 as speech
from pydub import AudioSegment
import speech_recognition as sr
from konlpy.tag import Okt
from gensim import corpora, models
import moviepy.editor as mp
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import re
import tempfile
import kss
import dlib

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
    update_progress(10)

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

        # 메모리 해제
        del frame

    cap.release()

    # 메모리 확보
    import gc
    gc.collect()

    update_progress(30)

    if not frames:
        return None

    features = preprocessor(images=frames, return_tensors="pt")["pixel_values"]
    model = SummaryModel.load_from_checkpoint('summary.ckpt')
    model.eval()

    update_progress(40)

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

    update_progress(60)

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

        update_progress(90)

        # GCS에 업로드하고 URL 반환
        gcs_url = upload_to_gcs(result_path, f"summaries/{os.path.basename(result_path)}")

        # 로컬 파일 삭제
        os.remove(result_path)

        update_progress(100)

        return gcs_url
    else:
        return None

@app.route('/progress', methods=['GET'])
def get_progress():
    progress = progress_dict.get('progress', 0)
    return jsonify({"progress": progress})

def extract_audio(video_file):
    video = VideoFileClip(video_file)
    audio_file = video_file.replace('.mp4', '.wav')
    video.audio.write_audiofile(audio_file)
    return audio_file

def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1)
    audio.export("temp.wav", format="wav")

    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)

    client = speech.SpeechClient()
    sample_rate = audio.frame_rate
    chunks = [audio[i:i + 10000] for i in range(0, len(audio), 10000)]
    transcript = ""

    for i, chunk in enumerate(chunks):
        chunk.export(f"chunk{i}.wav", format="wav")
        with open(f"chunk{i}.wav", "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="ko-KR",
        )
        response = client.recognize(config=config, audio=audio)
        transcript += "".join([result.alternatives[0].transcript for result in response.results])
        os.remove(f"chunk{i}.wav")

    os.remove("temp.wav")
    return transcript

def preprocess_text(text):
    processed_text = re.sub(r'[^가-힣\s]', '', text)
    sentences = kss.split_sentences(processed_text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def extract_topics_and_summarize(sentences, max_topics=5):
    if not sentences or len(sentences) < 2:
        return {"error": "Not enough data to extract topics. Please provide more text."}

    # 문장 임베딩 모델 로드
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    sentence_embeddings = model.encode(sentences)

    # 최적의 클러스터 개수 결정
    optimal_num_clusters = find_optimal_clusters(sentence_embeddings, max_topics)

    # 최적의 클러스터 개수를 사용하여 AgglomerativeClustering 수행
    clustering_model = AgglomerativeClustering(n_clusters=optimal_num_clusters)
    clustering_model.fit(sentence_embeddings)
    cluster_assignment = clustering_model.labels_

    # 클러스터별로 문장 그룹화
    clustered_sentences = {i: [] for i in range(optimal_num_clusters)}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(sentences[sentence_id])

    # 각 클러스터에 대해 요약 수행
    summarizer = pipeline("summarization", model="gogamza/kobart-base-v2", tokenizer="gogamza/kobart-base-v2")
    topic_sentences = {}
    for i, sentences in clustered_sentences.items():
        combined_text = ' '.join(sentences)
        if len(combined_text.split()) > 50:  # Summarize only if more than 50 words
            summary = summarizer(combined_text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
        else:
            summary = combined_text
        topic_sentences[f'주제 {i+1}'] = summary

    return topic_sentences

# 최적의 주제 개수를 결정하는 함수
def find_optimal_clusters(embeddings, max_clusters=5):
    scores = []
    for num_clusters in range(2, max_clusters + 1):
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
        cluster_labels = clustering_model.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        scores.append((num_clusters, score))
    optimal_num_clusters = max(scores, key=lambda item: item[1])[0]
    return optimal_num_clusters

def find_relevant_sentences(sentences, topic_summary, max_duration=60):
    okt = Okt()
    topic_nouns = okt.nouns(topic_summary)
    relevant_sentences = [sentence for sentence in sentences if any(noun in sentence for noun in topic_nouns)]

    if not relevant_sentences:
        return [], 0

    total_duration = 0
    relevant_text = []

    for sentence in relevant_sentences:
        duration = len(sentence.split()) / 2.5
        if total_duration + duration > max_duration:
            break
        relevant_text.append(sentence)
        total_duration += duration

    return relevant_text, total_duration

def extract_important_frames(video_file, start_time, end_time, sample_every_sec=5):
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_file)
    frames = []
    timestamps = []
    face_centers = []

    frame_count = 0
    total_frames = int((end_time - start_time) * cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    with tqdm(total=total_frames // sample_every_sec) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp > end_time:
                break
            if int(timestamp - start_time) % sample_every_sec == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                if faces:
                    face = faces[0]
                    center_x = (face.left() + face.right()) // 2
                    face_centers.append(center_x)
                else:
                    face_centers.append(frame.shape[1] // 2)  # 얼굴을 찾지 못한 경우 중앙을 중심으로 설정
                frames.append(frame)
                timestamps.append(timestamp)
                # 메모리 해제
                del frame
                frame_count += 1
                pbar.update(1)
                if frame_count % 50 == 0:  # 메모리 해제 빈도를 조정
                    cv2.waitKey(1)
    cap.release()

    return timestamps, face_centers

def cut_video_with_focus(video_file, start_time, end_time, output_file, focus_frames):
    video = VideoFileClip(video_file).subclip(start_time, end_time)
    target_height = 720
    video = video.resize(height=target_height)

    def crop_center(image, center_x):
        height, width, _ = image.shape
        new_width = int(height * 9 / 16)
        left = max(0, min(center_x - new_width // 2, width - new_width))
        cropped_image = image[:, left:left + new_width]
        return cropped_image

    with tempfile.TemporaryDirectory() as temp_dir:
        frame_files = []

        for t, frame in enumerate(video.iter_frames()):
            idx = int((t / len(focus_frames)) * len(focus_frames))
            idx = min(idx, len(focus_frames) - 1)
            center_x = focus_frames[idx]
            cropped_frame = crop_center(frame, center_x)
            frame_file = os.path.join(temp_dir, f"frame_{t:04d}.png")
            try:
                cv2.imwrite(frame_file, cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))
                if os.path.exists(frame_file):
                    frame_files.append(frame_file)
            except Exception as e:
                print(f"Error saving frame {t}: {e}")

        if not frame_files:
            return

        new_clip = ImageSequenceClip(frame_files, fps=video.fps)
        audio = video.audio
        new_clip = new_clip.set_audio(audio)
        new_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

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

    try:
        logger.info("Starting video summarization process")
        result_path = summarize_video(file_path, summary_time)
        logger.info("Video summarization process completed")

        if result_path:
            update_progress(0)
            return jsonify({"result_path": result_path}), 200
        else:
            logger.error("No suitable clips found")
            return jsonify({"error": "No suitable clips found"}), 500
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    update_progress(10)

    try:
        audio_file = extract_audio(file_path)
        update_progress(20)
        transcript = audio_to_text(audio_file)
        update_progress(50)
        processed_text = preprocess_text(transcript)
        update_progress(80)
        topics = extract_topics_and_summarize(processed_text)

        update_progress(100)
        update_progress(0)

        return jsonify({"topics": topics}), 200
    except Exception as e:
        logger.error(f"Error during topic extraction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/cut', methods=['POST'])
def cut():
    if 'file' not in request.files or 'selectedTopic' not in request.form:
        return jsonify({"error": "No file or selected topic"}), 400

    file = request.files['file']
    selected_topic = request.form['selectedTopic']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    update_progress(10)

    try:
        audio_file = extract_audio(file_path)
        update_progress(20)
        transcript = audio_to_text(audio_file)
        update_progress(30)
        processed_text = preprocess_text(transcript)
        update_progress(50)

        relevant_sentences, duration = find_relevant_sentences(processed_text, selected_topic)
        update_progress(70)
        if not relevant_sentences:
            raise ValueError(f"Selected topic '{selected_topic}' has no relevant sentences.")

        start_time = 0
        end_time = start_time + duration
        timestamps, face_centers = extract_important_frames(file_path, start_time, end_time)
        focus_frame_index = np.argmax(face_centers)
        focus_frame = face_centers[focus_frame_index]

        output_file = 'videos/shorts_output.mp4'
        cut_video_with_focus(file_path, start_time, end_time, output_file, face_centers)
        update_progress(90)

        gcs_url = upload_to_gcs(output_file, f"shorts/{os.path.basename(output_file)}")
        os.remove(output_file)

        update_progress(100)
        update_progress(0)

        return jsonify({"video_url": gcs_url}), 200
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        return jsonify({"error": str(e)}), 500


def find_loudest_sections(video_path, sample_rate=1, top_n=3):
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)

    audio = AudioSegment.from_wav(audio_path)
    os.remove(audio_path)

    update_progress(40)

    loudness = [audio[i * 1000:(i + 1) * 1000].dBFS for i in range(int(audio.duration_seconds))]
    loudness = np.array(loudness)
    sample_every_sec = sample_rate
    loudness = loudness[::sample_every_sec]

    update_progress(50)

    loudest_indices = np.argsort(loudness)[-top_n:][::-1]
    return video, loudest_indices

def create_vertical_clip(video, idx, max_duration=50):
    start_sec = max(0, idx * 1 - max_duration // 2)
    end_sec = min(video.duration, start_sec + max_duration)
    selected_clip = video.subclip(start_sec, end_sec)

    width, height = selected_clip.size
    new_width = int(height * 9 / 16)

    if new_width >= width:
        pad_width = (new_width - width) // 2
        vertical_clip = selected_clip.margin(left=pad_width, right=pad_width, color=(0, 0, 0))
    else:
        new_height = int(width * 16 / 9)
        pad_height = (new_height - height) // 2
        vertical_clip = selected_clip.margin(top=pad_height, bottom=pad_height, color=(0, 0, 0))
        vertical_clip = vertical_clip.resize(height=new_height)

    update_progress(70)

    return vertical_clip

def save_clips(clips, output_dir="videos"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_paths = []
    for i, clip in enumerate(tqdm(clips, desc="Saving clips")):
        final_output = f"{output_dir}/sport_final_result_vertical_top{i + 1}.mp4"
        fps = clip.fps
        clip.write_videofile(final_output, fps=fps, codec='libx264', audio_codec='aac')

        # GCS에 업로드하고 URL 반환
        gcs_url = upload_to_gcs(final_output, f"summaries/{os.path.basename(final_output)}")
        output_paths.append(gcs_url)

        update_progress(90)

        # 로컬 파일 삭제
        os.remove(final_output)

    return output_paths

@app.route('/sports_top_clips', methods=['POST'])
def sports_top_clips_endpoint():
    if 'file' not in request.files or 'topN' not in request.form:
        return jsonify({"error": "No file part or topN"}), 400

    update_progress(10)

    file = request.files['file']
    top_n = int(request.form['topN'])
    summary_time = 50  # 클립의 최대 길이를 50초로 고정

    update_progress(20)

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    update_progress(30)

    try:
        video, loudest_indices = find_loudest_sections(file_path, top_n=top_n)
        clips = [create_vertical_clip(video, idx, max_duration=summary_time) for idx in loudest_indices]
        result_paths = save_clips(clips)

        update_progress(100)
        update_progress(0)
        return jsonify({"result_paths": result_paths}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if not os.path.exists("videos"):
        os.makedirs("videos")
    app.run(host='0.0.0.0', port=5000)
