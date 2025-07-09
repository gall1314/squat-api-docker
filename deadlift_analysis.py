import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle

def run_deadlift_analysis(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []

    stage = None
    min_spine_angle = 180
    top_spine_angle = 0
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
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                # נקודת אמצע עמוד שדרה (מדומה)
                mid_spine = [
                    (shoulder[0] + hip[0]) / 2,
                    (shoulder[1] + hip[1]) / 2
                ]

                spine_angle = calculate_angle(shoulder, mid_spine, hip)

                if stage is None and spine_angle < 160:
                    stage = "down"
                    min_spine_angle = spine_angle

                elif stage == "down" and spine_angle > 165:
                    stage = "up"
                    top_spine_angle = spine_angle

                if stage == "down":
                    min_spine_angle = min(min_spine_angle, spine_angle)

                if stage == "up" and spine_angle > 165:
                    feedbacks = []
                    penalty = 0

                    if min_spine_angle < 155:
                        feedbacks.append("Try to keep your back straighter")
                        penalty += 2

                    if top_spine_angle < 170:
                        feedbacks.append("Finish your rep with a full lockout")
                        penalty += 1.5

                    penalty = min(penalty, 6)
                    score = round(max(4, 10 - penalty) * 2) / 2

                    for f in feedbacks:
                        if f not in overall_feedback:
                            overall_feedback.append(f)

                    counter += 1
                    if score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                        problem_reps.append(counter)
                    all_scores.append(score)

                    stage = None
                    min_spine_angle = 180
                    top_spine_angle = 0

            except Exception:
                continue

    cap.release()
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0

    if not overall_feedback:
        overall_feedback.append("Great form! Keep it up 💪")

    return {
        "technique_score": technique_score,
        "squat_count": counter,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": overall_feedback,
        "problem_reps": problem_reps,
    }

