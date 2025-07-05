from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ הוספה חשובה!
import tempfile
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ שורת הקסם שמאפשרת בקשות מהאפליקציה

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

@app.route('/analyze', methods=['POST'])
def analyze():
    video_file = request.files.get('video')
    if video_file is None:
        return jsonify({"error": "No video uploaded"}), 400

    # Save video to temp file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp_video.name)

    # Init pose estimation
    cap = cv2.VideoCapture(temp_video.name)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    results = []
    form_flags = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            def get_coords(index):
                return [landmarks[index].x, landmarks[index].y]

            # נקודות חשובות
            hip = get_coords(mp_pose.PoseLandmark.LEFT_HIP.value)
            knee = get_coords(mp_pose.PoseLandmark.LEFT_KNEE.value)
            ankle = get_coords(mp_pose.PoseLandmark.LEFT_ANKLE.value)
            shoulder = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            ear = get_coords(mp_pose.PoseLandmark.LEFT_EAR.value)

            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            back_angle = calculate_angle(shoulder, hip, ankle)

            result_data = {
                "knee_angle": round(knee_angle, 2),
                "hip_angle": round(hip_angle, 2),
                "back_angle": round(back_angle, 2),
                "feedback": []
            }

            # משוב
            if knee_angle < 70:
                result_data["feedback"].append("Squat deeper")
                form_flags.append("Squat deeper")
            if hip_angle > 160:
                result_data["feedback"].append("Control your descent")
                form_flags.append("Control your descent")
            if back_angle < 30:
                result_data["feedback"].append("Keep your back straighter")
                form_flags.append("Keep your back straighter")

            results.append(result_data)

    cap.release()

    return jsonify({
        "frame_feedback": results,
        "overall_feedback": list(set(form_flags))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

