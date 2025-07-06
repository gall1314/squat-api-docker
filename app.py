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
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def run_analysis(video_path):
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

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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

                back_angle = calculate_angle(shoulder, hip, knee)

                feedback = set()
                depth_penalty = 0

                # ✅ NEW: Depth check based on hip vs knee height
                hip_y = hip[1]
                knee_y = knee[1]

                if hip_y > knee_y + 0.04:
                    depth_penalty = 0
                elif knee_y < hip_y <= knee_y + 0.04:
                    depth_penalty = 0.5
                elif hip_y <= knee_y:
                    depth_penalty = 1.5
                    feedback.add("Try to squat deeper")
                else:
                    depth_penalty = 3
                    feedback.add("The squat is too shallow – go deeper")

                # ✅ Heel lifting
                heel_penalty = 0
                if heel[1] < foot_index[1] - 0.02:
                    feedback.add("Keep your heels firmly on the ground")
                    heel_penalty = 2

                # ✅ Back posture depending on stage
                if stage == "up":
                    if back_angle < 150:
                        feedback.add("Try to stand up straighter at the top")
                elif stage == "down":
                    if back_angle < 110:
                        feedback.add("Your back is too rounded – try to stay more upright")

                # ✅ Rep detection (same logic as before)
                if hip_y < knee_y:
                    stage = "down"
                elif hip_y > knee_y + 0.08 and stage == "down":
                    stage = "up"
                    counter += 1
                    total_penalty = depth_penalty + heel_penalty
                    score = max(4, 10 - total_penalty)
                    all_scores.append(score)
                    reps_feedback.append({
                        "rep": counter,
                        "score": score,
                        "issues": list(feedback)
                    })
                    if feedback:
                        bad_reps += 1
                    else:
                        good_reps += 1

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

