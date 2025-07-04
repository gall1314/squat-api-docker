from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)
CORS(app)  # מאפשר גישה מ- FlutterFlow

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

@app.route('/analyze', methods=['POST'])
def analyze():
    video_file = request.files.get('video')
    if video_file is None:
        return jsonify({"error": "No video uploaded"}), 400

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp_video.name)

    cap = cv2.VideoCapture(temp_video.name)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    counter = 0
    stage = None
    form_flags = []
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            try:
                landmarks = results.pose_landmarks.landmark

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                back_angle = calculate_angle(right_shoulder, right_hip, right_knee)

                # פידבק על יציבה
                if back_angle < 150:
                    form_flags.append("Keep your back straight")
                if knee_angle > 100:
                    form_flags.append("Go lower into the squat")
                if right_heel[1] < right_foot_index[1] - 0.02:
                    form_flags.append("Keep your heels down")

                # ספירת חזרות
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1

            except Exception as e:
                print("Error:", e)
                continue

    cap.release()
    elapsed_time = time.time() - start_time
    rpm = counter / (elapsed_time / 60) if elapsed_time > 0 else 0

    return jsonify({
        "rep_count": counter,
        "rpm": round(rpm, 2),
        "duration_sec": int(elapsed_time),
        "feedback": list(set(form_flags))
    })

# נדרש עבור Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

