from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import os
import json
from uuid import uuid4

app = Flask(__name__)
CORS(app)

VIDEO_FOLDER = tempfile.gettempdir()

def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def draw_landmarks_with_annotations(image, results, rep_num, score, knee_angle, back_angle, feedback):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.putText(annotated_image, f"Rep: {rep_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(annotated_image, f"Knee: {int(knee_angle)}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(annotated_image, f"Back: {int(back_angle)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(annotated_image, f"Score: {score}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    for i, note in enumerate(feedback):
        cv2.putText(annotated_image, note, (10, 135 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return annotated_image

def run_analysis_and_return_report(input_path, output_filename):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out_path = os.path.join(VIDEO_FOLDER, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    counter = 0
    stage = None

    summary = {
        "squat_count": 0,
        "bad_reps": 0,
        "total_score": 0.0,
        "rep_scores": [],
        "rep_feedback": []
    }

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                writer.write(frame)
                continue

            try:
                lm = results.pose_landmarks.landmark
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                heel = [lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                foot_index = [lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)
                feedback = []
                penalties = 0
                thigh_drop = knee[1] - hip[1]

                if thigh_drop > 0.12:
                    pass
                elif 0.09 <= thigh_drop <= 0.12:
                    feedback.append("Almost deep enough")
                    penalties += 0.5
                elif 0.06 <= thigh_drop < 0.09:
                    feedback.append("Try deeper")
                    penalties += 1.5
                else:
                    feedback.append("Too shallow")
                    penalties += 3

                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1
                    score = max(4, round(10 - penalties, 1))

                    if back_angle < 150:
                        if thigh_drop > 0.12 and back_angle < 130:
                            feedback.append("Stand straighter")
                            penalties += 1
                        elif thigh_drop <= 0.12 and back_angle < 110:
                            feedback.append("Back too rounded")
                            penalties += 1
                    if heel[1] < foot_index[1] - 0.02:
                        feedback.append("Heels off ground")
                        penalties += 1.5

                    summary["squat_count"] += 1
                    summary["total_score"] += score
                    summary["rep_scores"].append(score)
                    summary["rep_feedback"].append(feedback)
                    if score < 7:
                        summary["bad_reps"] += 1

                score = max(4, round(10 - penalties, 1))
                annotated = draw_landmarks_with_annotations(frame, results, counter, score, knee_angle, back_angle, feedback)
                writer.write(annotated)

            except Exception:
                writer.write(frame)
                continue

    cap.release()
    writer.release()

    if summary["squat_count"] > 0:
        summary["avg_score"] = round(summary["total_score"] / summary["squat_count"], 2)
    else:
        summary["avg_score"] = 0.0

    return summary, output_filename

@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files.get('video')
    if not video:
        return {"error": "No video uploaded"}, 400

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video.save(temp_input.name)

    video_id = str(uuid4())[:8]
    output_filename = f"squat_{video_id}.mp4"

    summary, saved_filename = run_analysis_and_return_report(temp_input.name, output_filename)

    if not summary:
        return {"error": "Video could not be processed"}, 400

    summary["video_url"] = f"http://localhost:10000/download/{saved_filename}"
    return jsonify(summary)

@app.route('/download/<filename>', methods=['GET'])
def download_video(filename):
    filepath = os.path.join(VIDEO_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='video/mp4', as_attachment=True, download_name=filename)
    else:
        return {"error": "File not found"}, 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
