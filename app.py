
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import time

app = Flask(__name__)
CORS(app)

def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def compress_video(input_path, scale=0.4):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return input_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp.name, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        out.write(frame)
    cap.release()
    out.release()
    return temp.name

def run_analysis(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    counter = 0
    good_reps = 0
    bad_reps = 0
    all_scores = []
    reps_feedback = []
    problem_reps = []
    stage = None
    prev_knee_angle = None
    prev_stage = None
    angle_idle = 0
    rep_min_angle = 180
    rep_max_depth = 0
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                continue
            try:
                lm = results.pose_landmarks.landmark
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                heel = [lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                foot_index = [lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)

                hip_to_knee = abs(hip[1] - knee[1])
                knee_to_ankle = abs(knee[1] - ankle[1])
                depth_ratio = hip_to_knee / knee_to_ankle if knee_to_ankle != 0 else 0
                rep_max_depth = round(min(depth_ratio, 1.0) * 10, 1)

                old_stage = stage
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"

                    feedbacks = []

                    # Penalties
                    angle_pen = 0
                    if rep_min_angle > 100:
                        angle_pen = 3
                        feedbacks.append("Very shallow")
                    elif rep_min_angle > 90:
                        angle_pen = 1.5
                        feedbacks.append("Try deeper")
                    elif rep_min_angle > 85:
                        angle_pen = 0.5
                        feedbacks.append("Almost deep enough")

                    depth_pen = 0
                    if rep_max_depth < 7.5:
                        depth_pen = 3
                        if "Very shallow" not in feedbacks:
                            feedbacks.append("Too shallow")
                    elif rep_max_depth < 8.5:
                        depth_pen = 1.5
                        if "Try deeper" not in feedbacks:
                            feedbacks.append("Try deeper")
                    elif rep_max_depth < 9.5:
                        depth_pen = 0.5
                        if "Almost deep enough" not in feedbacks:
                            feedbacks.append("Almost deep enough")

                    back_pen = 0
                    if back_angle < 30:
                        back_pen = 1.5
                        feedbacks.append("Keep your back straighter")

                    heel_pen = 0
                    if heel[0] < foot_index[0] - 0.03:
                        heel_pen = 1.5
                        feedbacks.append("Keep your heels down")

                    top_lock_pen = 0
                    if knee_angle < 165:
                        top_lock_pen = 1
                        feedbacks.append("Incomplete lockout")

                    total_penalty = max(angle_pen, depth_pen) + back_pen + heel_pen + top_lock_pen
                    total_penalty = min(total_penalty, 6)

                    score = round(max(4, 10 - total_penalty) * 2) / 2
                    feedback = "; ".join(feedbacks) if feedbacks else "Perfect form"

                    counter += 1
                    if score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                        problem_reps.append(counter)
                    all_scores.append(score)
                    reps_feedback.append(feedback)
                    rep_min_angle = 180
                    rep_max_depth = 0

                if stage == "down" and old_stage != "down":
                    rep_min_angle = knee_angle
                if stage == "down":
                    rep_min_angle = min(rep_min_angle, knee_angle)

            except Exception:
                continue

    cap.release()
    elapsed_time = time.time() - start_time
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0

    return {
        "squat_count": counter,
        "duration_seconds": round(elapsed_time),
        "technique_score": technique_score,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": reps_feedback,
        "problem_reps": problem_reps
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp.name)
    compressed_path = compress_video(temp.name, scale=0.4)
    try:
        os.remove(temp.name)
    except OSError:
        pass
    result = run_analysis(compressed_path, frame_skip=3, scale=0.4)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

