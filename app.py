from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

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
                heel_y = lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
                foot_y = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y

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

                    feedback_msgs = []

                    # Penalty for knee angle
                    angle_penalty = 0
                    if rep_min_angle > 105:
                        angle_penalty = 3
                        feedback_msgs.append("Knee angle too shallow")
                    elif rep_min_angle > 95:
                        angle_penalty = 1.5
                        feedback_msgs.append("Try to go deeper (angle)")
                    elif rep_min_angle > 90:
                        angle_penalty = 0.5
                        feedback_msgs.append("Almost deep enough (angle)")

                    # Penalty for depth
                    depth_penalty = 0
                    if rep_max_depth < 6.5:
                        depth_penalty = 3
                        feedback_msgs.append("Depth too shallow")
                    elif rep_max_depth < 7.5:
                        depth_penalty = 1.5
                        feedback_msgs.append("Try to go deeper (depth)")
                    elif rep_max_depth < 8.5:
                        depth_penalty = 0.5
                        feedback_msgs.append("Almost deep enough (depth)")

                    # Back angle penalty
                    back_penalty = 0
                    if back_angle < 35:
                        back_penalty = 1.5
                        feedback_msgs.append("Keep your back straighter")

                    # Heel penalty
                    heel_penalty = 0
                    if heel_y < foot_y - 0.03:
                        heel_penalty = 1.5
                        feedback_msgs.append("Keep your heels down")

                    # Total penalty capped at 6
                    total_penalty = min(6, angle_penalty + depth_penalty + back_penalty + heel_penalty)

                    score = round(max(4, 10 - total_penalty) * 2) / 2
                    feedback = "; ".join(feedback_msgs) if feedback_msgs else "Perfect form"

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
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": reps_feedback,
        "problem_reps": problem_reps,
        "squat_count": counter
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


