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
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
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

def detect_motion_frames(video_path, frame_skip=3, scale=0.4, threshold=2.0):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    motion_frames = []
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
                motion_frames.append(index)
        prev_gray = gray
    cap.release()
    return motion_frames

def run_analysis(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    motion_frames = detect_motion_frames(video_path, frame_skip, scale)
    if not motion_frames:
        return {"error": "No squat motion detected"}

    start_frame = max(0, min(motion_frames) - 6)
    end_frame = max(motion_frames) + 6

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    counter = 0
    good_reps = 0
    bad_reps = 0
    all_scores = []
    reps_feedback = []
    problem_reps = []
    stage = None
    rep_min_angle = 180
    rep_max_depth = 0
    prev_stage = None
    prev_knee_angle = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame % frame_skip != 0:
                continue
            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                continue

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

                # Penalties
                feedback = []
                angle_pen = 3 if rep_min_angle > 110 else 1.5 if rep_min_angle > 100 else 0.5 if rep_min_angle > 95 else 0
                if angle_pen == 3: feedback.append("Very shallow")
                elif angle_pen == 1.5: feedback.append("Try deeper")
                elif angle_pen == 0.5: feedback.append("Almost deep enough")

                depth_pen = 4 if rep_max_depth < 6.5 else 3 if rep_max_depth < 7.5 else 1.5 if rep_max_depth < 8.5 else 0.5 if rep_max_depth < 9.5 else 0
                if depth_pen >= 3: feedback.append("Too shallow")
                elif depth_pen == 1.5: feedback.append("Try deeper")
                elif depth_pen == 0.5: feedback.append("Almost deep enough")

                back_pen = 1.5 if back_angle < 35 else 0
                if back_pen > 0: feedback.append("Keep your back straighter")

                heel_pen = 1.5 if heel_y < foot_y - 0.03 else 0
                if heel_pen > 0: feedback.append("Keep your heels down")

                total_pen = max(angle_pen, depth_pen) + back_pen + heel_pen
                total_pen = min(total_pen, 6)
                score = round(max(4, 10 - total_pen) * 2) / 2
                counter += 1
                all_scores.append(score)
                reps_feedback.append("; ".join(set(feedback)) if feedback else "Perfect form")
                if score >= 9.5:
                    good_reps += 1
                else:
                    bad_reps += 1
                    problem_reps.append(counter)
                rep_min_angle = 180
                rep_max_depth = 0

            if stage == "down" and old_stage != "down":
                rep_min_angle = knee_angle
            if stage == "down":
                rep_min_angle = min(rep_min_angle, knee_angle)
                rep_max_depth = max(rep_max_depth, round(min(depth_ratio, 1.0) * 10, 1))

    cap.release()
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0

    return {
        "squat_count": counter,
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
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


    

