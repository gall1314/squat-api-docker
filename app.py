from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

app = Flask(__name__)
CORS(app)

def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

app = Flask(__name__)
CORS(app)

def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def _frame_generator(cap, frame_skip, scale, motion_threshold, idle_frames=5):
    frame_index = 0
    prev_gray = None
    idle = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % frame_skip != 0:
            continue
        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            if np.mean(diff) < motion_threshold:
                idle += 1
                if idle < idle_frames:
                    prev_gray = gray
                    continue
            else:
                idle = 0
        prev_gray = gray
        yield frame

def run_analysis(
    video_path,
    frame_skip=2,
    scale=0.5,
    motion_threshold=2.0,
    angle_epsilon=1.0,
    max_angle_idle=5,
    idle_frames=5,
):
    frame_skip = max(1, int(frame_skip))
    if scale <= 0 or scale > 1:
        scale = 1.0

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    counter = 0
    good_reps = 0
    bad_reps = 0
    all_scores = []
    reps_feedback = []
    stage = None
    start_time = time.time()

    prev_knee_angle = None
    prev_stage = None
    angle_idle = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for frame in _frame_generator(cap, frame_skip, scale, motion_threshold, idle_frames):
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

                old_stage = stage
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"  # ← תוקן ה־if

                if prev_knee_angle is not None:
                    if (
                        abs(knee_angle - prev_knee_angle) < angle_epsilon
                        and stage == prev_stage
                    ):
                        angle_idle += 1
                        if angle_idle < max_angle_idle:
                            prev_knee_angle = knee_angle
                            prev_stage = stage
                            continue
                    else:
                        angle_idle = 0

                if stage == "up" and old_stage == "down":
                    score = max(4, round(10 - penalties, 1))
                    counter += 1
                    if score >= 7:
                        good_reps += 1
                    else:
                        bad_reps += 1
                    all_scores.append(score)
                    reps_feedback.append(feedback)

                prev_knee_angle = knee_angle
                prev_stage = stage
                angle_idle = 0

            except Exception:
                continue

    cap.release()
    elapsed_time = time.time() - start_time

    if counter == 0:
        return {"error": "No clear squat movement detected", "duration_seconds": round(elapsed_time)}

    technique_score = round(np.mean(all_scores), 1)

    return {
        "squat_count": counter,
        "duration_seconds": round(elapsed_time),
        "technique_score": technique_score,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": reps_feedback
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400

    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp.name)

    result = run_analysis(temp.name)

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
