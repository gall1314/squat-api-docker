from flask import Flask, request, jsonify
import tempfile
import requests
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    video_url = data.get('video_url')

    response = requests.get(video_url)
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(response.content)
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)
    mp_pose = mp.solutions.pose
    counter = 0
    stage = None
    form_flags = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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

                feedback = []

                if back_angle < 150:
                    feedback.append("Keep your back straight")
                if knee_angle > 100:
                    feedback.append("Go lower into the squat")
                if heel[1] < foot_index[1] - 0.02:
                    feedback.append("Keep your heels down")

                form_flags += feedback

                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1

            except:
                continue

    cap.release()

    return jsonify({
        "reps": counter,
        "feedback": list(set(form_flags))
    })

# ✅ חשוב! הפעלת Flask כמו ש-Render מצפה
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

