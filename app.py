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
    counter = 0
    bad_reps = 0
    stage = None
    feedback_log = []

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    start_time = time.time()

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # דילוג על חלק מהפריימים (שיפור ביצועים)
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

                frame_feedback = []

                # תנאי משופר לזיהוי גב בירידה
                if stage == "down" and back_angle < 120:
                    frame_feedback.append("Keep your back straighter during descent")
                elif back_angle < 150:
                    frame_feedback.append("Keep your back straight")

                if knee_angle > 100:
                    frame_feedback.append("Go lower into the squat")
                if heel[1] < foot_index[1] - 0.02:
                    frame_feedback.append("Keep your heels down")

                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1
                    if frame_feedback:
                        bad_reps += 1

                # שמירת פידבקים לפריים
                for issue in frame_feedback:
                    feedback_log.append({"issue": issue})

            except Exception:
                continue

    cap.release()
    elapsed_time = time.time() - start_time

    # ניקוד טכניקה בין 1 ל־10
    good_reps = max(counter - bad_reps, 0)
    score = int((good_reps / counter) * 10) if counter > 0 else 1

    return {
        "squat_count": counter,
        "bad_reps": bad_reps,
        "duration_seconds": round(elapsed_time),
        "technique_score": score,
        "feedback": feedback_log
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp_video.name)

    result = run_analysis(temp_video.name)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

