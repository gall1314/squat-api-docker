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

def run_analysis(video_path):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    frame_index = 0
    stage = None
    counter = 0
    good_reps = 0
    bad_reps = 0
    all_rep_scores = []
    reps_feedback = []

    start_time = time.time()

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ✅ הקטנת רזולוציה
            frame = cv2.resize(frame, (640, 480))

            if frame_index % 5 != 0:
                frame_index += 1
                continue
            frame_index += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            try:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)

                rep_feedback = set()

                # ✅ הערות טכניקה בזמן הירידה בלבד
                if stage == "down":
                    if back_angle < 130:
                        rep_feedback.add("Try to keep your back a bit straighter as you go down")
                    if knee_angle > 105:
                        rep_feedback.add("It would be better if you squat a little deeper")
                    if heel[1] < foot_index[1] - 0.02:
                        rep_feedback.add("Keep your heels firmly on the ground")

                # ✅ ניתוח חזרות
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1

                    if rep_feedback:
                        bad_reps += 1
                    else:
                        good_reps += 1

                    score = max(0, 10 - len(rep_feedback) * 2)
                    all_rep_scores.append(score)
                    reps_feedback.append({
                        "rep": counter,
                        "issues": list(rep_feedback)
                    })

            except Exception:
                continue

    cap.release()
    elapsed_time = time.time() - start_time

    if counter == 0:
        return {
            "error": "No clear squat movement detected",
            "duration_seconds": round(elapsed_time)
        }

    technique_score = round(np.mean(all_rep_scores), 1)

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

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp_video.name)

    result = run_analysis(temp_video.name)

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

