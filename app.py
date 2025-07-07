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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    video_id = str(uuid.uuid4().hex[:8])
    output_filename = f"squat_{video_id}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / frame_skip, (width, height))

    counter = good_reps = bad_reps = 0
    all_scores = []
    reps_feedback = []
    problem_reps = []
    stage = None
    rep_min_angle = 180
    prev_knee_angle = None
    prev_stage = None
    feedback = "Waiting"

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
            annotated = frame.copy()

            if results.pose_landmarks:
                try:
                    lm = results.pose_landmarks.landmark
                    hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    knee_angle = calculate_angle(hip, knee, ankle)
                    back_angle = calculate_angle(shoulder, hip, knee)

                    old_stage = stage
                    if knee_angle < 90:
                        stage = "down"
                    if knee_angle > 160 and stage == "down":
                        stage = "up"
                        feedback = ""
                        penalty = 0

                        if rep_min_angle > 110:
                            penalty += 3
                            feedback = "Very shallow"
                        elif rep_min_angle > 100:
                            penalty += 1.5
                            feedback = "Try deeper"
                        elif rep_min_angle > 95:
                            penalty += 0.5
                            feedback = "Almost deep enough"
                        else:
                            feedback = "Good depth"

                        score = round(max(4, 10 - penalty) * 2) / 2
                        counter += 1
                        if score >= 9.5:
                            good_reps += 1
                        else:
                            bad_reps += 1
                            problem_reps.append(counter)
                        reps_feedback.append(feedback)
                        all_scores.append(score)
                        rep_min_angle = 180

                    if stage == "down" and old_stage != "down":
                        rep_min_angle = knee_angle
                    if stage == "down":
                        rep_min_angle = min(rep_min_angle, knee_angle)

                    # Draw feedback on frame
                    cv2.putText(annotated, f"Stage: {stage or '-'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(annotated, f"Angle: {int(knee_angle)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    cv2.putText(annotated, f"Feedback: {feedback}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                except Exception:
                    pass

            out.write(annotated)

    cap.release()
    out.release()

    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0
    return {
        "technique_score": technique_score,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "squat_count": counter,
        "feedback": reps_feedback,
        "problem_reps": problem_reps,
        "video_url": f"/media/{output_filename}"
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
    return jsonify(result)

@app.route("/media/<path:filename>")
def serve_video(filename):
    return send_from_directory(MEDIA_DIR, filename)

