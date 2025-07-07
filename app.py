
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

def _detect_motion_indices(cap, frame_skip, scale, threshold):
    indices = []
    prev_gray = None
    index = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        index += 1
        if index % frame_skip != 0:
            continue
        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            if np.mean(diff) > threshold:
                indices.append(index)
        prev_gray = gray
    return indices

def run_analysis(video_path, frame_skip=2, scale=0.4, motion_threshold=2.0, angle_epsilon=1.0, max_angle_idle=5, context_frames=6):
    frame_skip = max(1, int(frame_skip))
    scale = min(max(scale, 0.1), 1.0)
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    motion_frames = _detect_motion_indices(cap, frame_skip, scale, motion_threshold)
    cap.release()
    if not motion_frames:
        return {"error": "No clear motion detected"}
    start_frame = max(0, min(motion_frames) - context_frames)
    end_frame = min(frame_count - 1, max(motion_frames) + context_frames)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    counter = good_reps = bad_reps = 0
    all_scores = []
    reps_feedback = []
    problem_reps = []
    stage = None
    prev_knee_angle = prev_stage = None
    angle_idle = 0
    rep_min_angle = 180
    feedback_msgs = []
    start_time = time.time()
    missed_depth_flag = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_index = start_frame - 1
        last_processed_idx = start_frame - frame_skip
        while cap.isOpened() and frame_index < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if frame_index - last_processed_idx < frame_skip:
                continue
            last_processed_idx = frame_index
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
                knee_angle = calculate_angle(hip, knee, ankle)
                old_stage = stage
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    total_penalty = 3 if missed_depth_flag else 0
                    score = max(4, round((10 - total_penalty) * 2) / 2)
                    counter += 1
                    if score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                        problem_reps.append(counter)
                    reps_feedback.append("Too shallow (knee > 100°)" if missed_depth_flag else "Good depth")
                    all_scores.append(score)
                    rep_min_angle = 180
                    missed_depth_flag = False
                    feedback_msgs = []
                if stage == "down" and old_stage != "down":
                    rep_min_angle = knee_angle
                    missed_depth_flag = False
                if stage == "down":
                    rep_min_angle = min(rep_min_angle, knee_angle)
                    if knee_angle > 100:
                        missed_depth_flag = True
                if prev_knee_angle is not None:
                    if abs(knee_angle - prev_knee_angle) < angle_epsilon and stage == prev_stage:
                        angle_idle += 1
                        if angle_idle < max_angle_idle:
                            prev_knee_angle = knee_angle
                            prev_stage = stage
                            continue
                    else:
                        angle_idle = 0
                prev_knee_angle = knee_angle
                prev_stage = stage
                angle_idle = 0
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
    try:
        frame_skip = max(1, int(request.args.get('frame_skip', '2')))
    except ValueError:
        frame_skip = 2
    result = run_analysis(compressed_path, frame_skip=frame_skip, scale=0.4)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
