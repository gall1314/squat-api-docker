import os

def draw_landmarks_with_annotations(image, results, rep_num, score, knee_angle, back_angle, feedback):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    annotated_image = image.copy()

    # Draw pose landmarks
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Add text info
    cv2.putText(annotated_image, f"Rep: {rep_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(annotated_image, f"Knee: {int(knee_angle)} deg", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(annotated_image, f"Back: {int(back_angle)} deg", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(annotated_image, f"Score: {score}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    for i, note in enumerate(feedback):
        cv2.putText(annotated_image, note, (10, 135 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    return annotated_image

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
    if not cap.isOpened():
        return {"error": "Could not open video"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = video_path.replace(".mp4", "_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

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

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                out_writer.write(frame)
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
                    feedback.append("Almost deep enough – try just a bit lower")
                    penalties += 0.5
                elif 0.06 <= thigh_drop < 0.09:
                    feedback.append("Try to squat deeper")
                    penalties += 1.5
                else:
                    feedback.append("The squat is too shallow – go deeper")
                    penalties += 3

                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"

                    if back_angle < 150:
                        if thigh_drop > 0.12 and back_angle < 130:
                            feedback.append("Try to stand up straighter at the top")
                            penalties += 1
                        elif thigh_drop <= 0.12 and back_angle < 110:
                            feedback.append("Your back is too rounded – try to stay more upright")
                            penalties += 1

                    if heel[1] < foot_index[1] - 0.02:
                        feedback.append("Keep your heels firmly on the ground")
                        penalties += 1.5

                    counter += 1
                    score = max(4, round(10 - penalties, 1))
                    all_scores.append(score)
                    reps_feedback.append({"rep": counter, "score": score, "issues": feedback})
                    if feedback:
                        bad_reps += 1
                    else:
                        good_reps += 1

                frame_annotated = draw_landmarks_with_annotations(frame, results, counter, score if counter > 0 else '-', knee_angle, back_angle, feedback)
                out_writer.write(frame_annotated)

            except Exception:
                out_writer.write(frame)
                continue

    cap.release()
    out_writer.release()
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
        "feedback": reps_feedback,
        "annotated_video": out_path
    }

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

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)

                feedback = []
                penalties = 0

                # עומק סקוואט לפי גובה ירך מול ברך
                thigh_drop = knee[1] - hip[1]
                if thigh_drop > 0.12:
                    pass  # מעולה
                elif 0.09 <= thigh_drop <= 0.12:
                    feedback.append("Almost deep enough – try just a bit lower")
                    penalties += 0.5
                elif 0.06 <= thigh_drop < 0.09:
                    feedback.append("Try to squat deeper")
                    penalties += 1.5
                else:
                    feedback.append("The squat is too shallow – go deeper")
                    penalties += 3

                # שמירת שלב התנועה לפי זווית ברך
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"

                    # תנוחת גב לפי שלב
                    if back_angle < 150:
                        if thigh_drop > 0.12 and back_angle < 130:
                            feedback.append("Try to stand up straighter at the top")
                            penalties += 1
                        elif thigh_drop <= 0.12 and back_angle < 110:
                            feedback.append("Your back is too rounded – try to stay more upright")
                            penalties += 1

                    # עקבים
                    if heel[1] < foot_index[1] - 0.02:
                        feedback.append("Keep your heels firmly on the ground")
                        penalties += 1.5

                    counter += 1
                    score = max(4, round(10 - penalties, 1))
                    all_scores.append(score)
                    reps_feedback.append({"rep": counter, "score": score, "issues": feedback})
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
