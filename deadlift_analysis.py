import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle, calculate_body_angle

def run_deadlift_analysis(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []

    rep_in_progress = False
    frame_index = 0
    last_rep_frame = -999
    MIN_FRAMES_BETWEEN_REPS = 10

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

                # נקודות חיוניות
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # בדיקת נראות הברך
                if lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility < 0.5:
                    continue

                body_angle = calculate_body_angle(shoulder, hip)
                knee_angle = calculate_angle(hip, knee, ankle)

                # התחלת חזרה – גב כפוף יחסית + ברך מכופפת
                if not rep_in_progress and body_angle < 140 and knee_angle < 160:
                    rep_in_progress = True

                # סיום חזרה – עמידה זקופה
                elif rep_in_progress and body_angle > 165:
                    if frame_index - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        feedbacks = []
                        penalty = 0

                        # הערכת טכניקה
                        if body_angle < 118:
                            feedbacks.append("Try to keep your back straighter")
                            penalty += 1.5
                        if body_angle < 155:
                            feedbacks.append("Try to finish more upright")
                            penalty += 1

                        score = round(max(4, 10 - penalty) * 2) / 2
                        for f in feedbacks:
                            if f not in overall_feedback:
                                overall_feedback.append(f)

                        counter += 1
                        last_rep_frame = frame_index
                        if score >= 9.5:
                            good_reps += 1
                        else:
                            bad_reps += 1
                            problem_reps.append(counter)
                        all_scores.append(score)

                    rep_in_progress = False

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


