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

ANNOTATED_DIR = "/tmp/annotated_videos"
os.makedirs(ANNOTATED_DIR, exist_ok=True)

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

def run_analysis_with_output(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    output_path = os.path.join(ANNOTATED_DIR, f"squat_{uuid.uuid4().hex}.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps // frame_skip, (width, height))

    counter = 0
    good_reps = 0
    bad_reps = 0
    all_scores = []
    reps_feedback = []
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
            frame = cv2.resize(frame, (width, height))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            feedbacks = []
            score = 10

            if results.pose_landmarks:
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
                depth_ratio = hip_to_knee / knee_to_ankle if knee_to_ankle else 0
                rep_max_depth = round(min(depth_ratio, 1.0) * 10, 1)

                old_stage = stage
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"
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
                        if "Too shallow" not in feedbacks:
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
                    if heel_y < foot_y - 0.03:
                        heel_pen = 1.5
                        feedbacks.append("Keep your heels down")

                    total_penalty = max(angle_pen, depth_pen) + back_pen + heel_pen
                    score = round(max(4, 10 - total_penalty) * 2) / 2

                    counter += 1
                    if score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                    all_scores.append(score)
                    reps_feedback.append("; ".join(feedbacks))
                    rep_min_angle = 180
                    rep_max_depth = 0

                if stage == "down" and old_stage != "down":
                    rep_min_angle = knee_angle
                if stage == "down":
                    rep_min_angle = min(rep_min_angle, knee_angle)

                cv2.putText(frame, f"Score: {score:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for msg_idx, msg in enumerate(feedbacks):
                    cv2.putText(frame, msg, (10, 60 + 30 * msg_idx), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            out.write(frame)

    cap.release()
    out.release()
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0
    return {
        "technique_score": technique_score,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": reps_feedback,
        "video_url": f"/media/{os.path.basename(output_path)}"
    }

@app.route('/media/<path:filename>')
def media(filename):
    return send_from_directory(ANNOTATED_DIR, filename)

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
    result = run_analysis_with_output(compressed_path)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

