from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import uuid

app = Flask(__name__)
CORS(app)

MEDIA_DIR = "media"
os.makedirs(MEDIA_DIR, exist_ok=True)

def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def compress_video(input_path, output_path, scale=0.4):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        out.write(frame)

    cap.release()
    out.release()
    return True

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

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue
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
                rep_max_depth = max(rep_max_depth, round(min(depth_ratio, 1.0) * 10, 1))

                old_stage = stage
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"

                    feedbacks = []

                    # Penalties
                    angle_pen = 0
                    if rep_min_angle > 110:
                        angle_pen = 3
                        feedbacks.append("Very shallow")
                    elif rep_min_angle > 100:
                        angle_pen = 1.5
                        feedbacks.append("Try deeper")
                    elif rep_min_angle > 95:
                        angle_pen = 0.5
                        feedbacks.append("Almost deep enough")

                    depth_pen = 0
                    if rep_max_depth < 6.5:
                        depth_pen = 4
                        feedbacks.append("Too shallow")
                    elif rep_max_depth < 7.5:
                        depth_pen = 3
                        feedbacks.append("Too shallow")
                    elif rep_max_depth < 8.5:
                        depth_pen = 1.5
                        feedbacks.append("Try deeper")
                    elif rep_max_depth < 9.5:
                        depth_pen = 0.5
                        feedbacks.append("Almost deep enough")

                    back_pen = 0
                    if back_angle < 35:
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
                    feedback = "; ".join(set(feedbacks)) if feedbacks else "Perfect form"

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
                    rep_max_depth = round(min(depth_ratio, 1.0) * 10, 1)
                if stage == "down":
                    rep_min_angle = min(rep_min_angle, knee_angle)
                    rep_max_depth = max(rep_max_depth, round(min(depth_ratio, 1.0) * 10, 1))

            except Exception:
                continue

    cap.release()
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0

    return {
        "technique_score": technique_score,
        "squat_count": counter,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": reps_feedback,
        "problem_reps": problem_reps,
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp.name)

    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"squat_{unique_id}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)

    success = compress_video(temp.name, output_path, scale=0.4)
    os.remove(temp.name)

    if not success:
        return jsonify({"error": "Video compression failed"}), 500

    result = run_analysis(output_path, frame_skip=3, scale=0.4)
    result["video_url"] = f"/media/{output_filename}"

    return jsonify(result)

@app.route('/media/<filename>')
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080) 
