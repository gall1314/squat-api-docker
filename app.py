+35
-2

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
def run_analysis(video_path, frame_skip=2, scale=0.5):
    """Analyze squat technique in a video.

    Args:
        video_path (str): Path to the video file.
        frame_skip (int, optional): Process every Nth frame. Defaults to 2.
            Values below 1 are treated as 1.
        scale (float, optional): Resize factor for frames (0 < scale <= 1).
            Smaller values speed up processing. Defaults to 0.5.
    Returns:
        dict: Analysis results or an error message.
    """
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

    frame_index = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if frame_index % frame_skip != 0:
                continue
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
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                heel = [lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                foot_index = [lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)

                feedback = []
                penalties = 0

                # עומק סקוואט לפי גובה ירך מול ברך
                thigh_drop = knee[1] - hip[1]
                if thigh_drop > 0.12:
@@ -105,33 +125,46 @@ def run_analysis(video_path):
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
    frame_skip = request.args.get('frame_skip', '2')
    scale = request.args.get('scale', '0.5')
    try:
        frame_skip = max(1, int(frame_skip))
    except ValueError:
        frame_skip = 2
    try:
        scale = float(scale)
        if scale <= 0 or scale > 1:
            scale = 0.5
    except ValueError:
        scale = 0.5

    result = run_analysis(temp.name, frame_skip=frame_skip, scale=scale)

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
