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

    rep_in_progress = False
    frame_index = 0
    last_rep_frame = -999
    MIN_FRAMES_BETWEEN_REPS = 10
    max_delta_x = 0

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

                # נקודות רלוונטיות
                shoulder_x = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
                hip_x = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                knee_angle = calculate_angle(hip, knee, ankle)

                delta_x = abs(hip_x - shoulder_x)

                if rep_in_progress:
                    max_delta_x = max(max_delta_x, delta_x)

                # התחלת חזרה – כשהכתפיים זזות קדימה ביחס לאגן
                if not rep_in_progress:
                    if delta_x > 0.08:
                        rep_in_progress = True
                        max_delta_x = delta_x
                        min_knee_angle = knee_angle

                # סיום חזרה – כתפיים חוזרות לקו עם האגן
                elif rep_in_progress and delta_x < 0.035:
                    if frame_index - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        delta_range = max_delta_x - delta_x

                        if delta_range > 0.05 and min_knee_angle < 160:
                            feedbacks = []
                            penalty = 0

                            if max_delta_x > 0.2:
                                feedbacks.append("Try not to lean too far forward")
                                penalty += 1.5
                            if delta_x > 0.05:
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
                    max_delta_x = 0

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



