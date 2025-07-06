from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import subprocess

app = Flask(__name__)
CORS(app)

def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def convert_video_to_h264(input_path):
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
            output_path
        ], check=True)
        print("✅ Converted video to H.264:", output_path)
        return output_path
    except subprocess.CalledProcessError:
        print("❌ ffmpeg conversion failed.")
        return None

def run_analysis(video_path):
    print("⚠️ Trying to open video:", video_path)
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("⚠️ Initial open failed, trying conversion...")
        converted_path = convert_video_to_h264(video_path)
        if converted_path:
            cap = cv2.VideoCapture(converted_path)
        if not cap.isOpened():
            print("❌ Failed to open video even after conversion.")
            return {"error": "Video could not be opened even after conversion."}

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

            frame = cv2.resize(frame, (640, 480))
            if frame_index % 5 != 0:
                frame_index += 1
                continue
            frame_index += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                print(f"❌ No pose landmarks detected at frame {frame_index}")
                continue

            try:
                lm = results.pose_landmarks.landmark
                hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                knee_y = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
                ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                heel_y = lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
                foot_index_y = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y

                rep_feedback = set()
                depth_penalty = 0
                back_penalty = 0
                heel_penalty = 0

                # Depth check
                depth_ratio = hip_y / knee_y
                if depth_ratio > 1.05:
                    pass
                elif 1.00 <= depth_ratio <= 1.05:
                    rep_feedback.add("Try to squat slightly deeper")
                    depth_penalty = 1
                else:
                    rep_feedback.add("The squat is too shallow – try going lower")
                    depth_penalty = 3

                # Back angle check
                back_angle = calculate_angle(shoulder, hip, knee)
                if back_angle > 130:
                    pass
                elif 110 <= back_angle <= 130:
                    rep_feedback.add("Try to keep your back a bit straighter")
                    back_penalty = 1.5
                else:
                    rep_feedback.add("Your back is too rounded – try to keep it straighter")
                    back_penalty = 3

                # Heels
                if heel_y < foot_index_y - 0.02:
                    rep_feedback.add("Keep your heels firmly on the ground")
                    heel_penalty = 2

                # Stage detection
                if stage != "down" and depth_ratio > 1.05:
                    stage = "down"

                if depth_ratio < 0.95 and stage == "down":
                    stage = "up"
                    counter += 1
                    total_penalty = depth_penalty + back_penalty + heel_penalty

                    if rep_feedback:
                        bad_reps += 1
                    else:
                        good_reps += 1

                    score = max(4, round(10 - total_penalty, 1))
                    all_rep_scores.append(score)
                    reps_feedback.append({
                        "rep": counter,
                        "issues": list(rep_feedback)
                    })

            except Exception as e:
                print(f"❌ Exception at frame {frame_index}: {e}")
                continue

    cap.release()
    elapsed_time = time.time() - start_time

    if counter == 0:
        print("❌ No squats detected in entire video")
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
        print("❌ No file received")
        return jsonify({"error": "No video uploaded"}), 400

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp_video.name)

    result = run_analysis(temp_video.name)

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


