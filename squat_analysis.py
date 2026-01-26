# -*- coding: utf-8 -*-
# squat_analysis.py
# Stable counting + HARD depth gate (hip below knee) + improved back logic

import cv2
import numpy as np
import mediapipe as mp

# ===================== FEEDBACK =====================

FB_SEVERITY = {
    "Try to squat deeper": 3,
    "Avoid knee collapse": 3,
    "Try to keep your back a bit straighter": 2,
    "Almost there — go a bit lower": 2,
    "Looking good — just a bit more depth": 1,
}

def pick_strongest_feedback(lst):
    best, score = None, -1
    for f in lst or []:
        s = FB_SEVERITY.get(f, 1)
        if s > score:
            best, score = f, s
    return best

def merge_feedback(a, b):
    cand = pick_strongest_feedback(b)
    if not cand:
        return a
    if not a:
        return cand
    return cand if FB_SEVERITY.get(cand,1) >= FB_SEVERITY.get(a,1) else a

# ===================== SCORE DISPLAY =====================

def score_label(score: float) -> str:
    if score >= 9.5: return "Excellent"
    if score >= 8.5: return "Very good"
    if score >= 7.0: return "Good"
    if score >= 5.5: return "Fair"
    return "Needs work"

def display_half(score: float) -> str:
    q = round(score * 2) / 2
    return str(int(q)) if q.is_integer() else f"{q:.1f}"

# ===================== GEOMETRY =====================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360 - ang if ang > 180 else ang

# ===================== CORE PARAMS =====================

mp_pose = mp.solutions.pose

STAND_KNEE_ANGLE = 160.0
MIN_FRAMES_BETWEEN_REPS = 10

# Back checks
TOP_THR_DEG = 145.0
BOTTOM_THR_DEG = 100.0
TOP_BAD_MIN_SEC = 0.25
BOTTOM_BAD_MIN_SEC = 0.35

# Depth (HARD gate)
DEPTH_MARGIN = 0.015  # tolerance in normalized Y (MediaPipe)

# ===================== MAIN =====================

def run_squat_analysis(
    video_path,
    frame_skip=3,
    scale=0.4,
    output_path="squat_analyzed.mp4",
    feedback_path="squat_feedback.txt",
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0,
            "technique_score": 0.0,
            "technique_score_display": "0",
            "technique_label": "Needs work",
            "good_reps": 0,
            "bad_reps": 0,
            "feedback": ["Could not open video"],
            "reps": [],
            "video_path": "",
            "feedback_path": feedback_path,
        }

    counter = 0
    good_reps = 0
    bad_reps = 0
    rep_reports = []
    all_scores = []

    stage = None
    frame_idx = 0
    last_rep_frame = -999
    session_best_feedback = None

    # per-rep tracking
    rep_start_frame = None
    rep_min_torso_angle = 999.0

    # HARD depth tracking (key part)
    rep_min_hip_y = -1.0          # max Y reached (downwards)
    rep_knee_y_at_bottom = 0.0

    # Back logic counters
    rep_top_bad_frames = 0
    rep_bottom_bad_frames = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    eff_fps = max(1.0, fps / frame_skip)
    dt = 1.0 / eff_fps
    TOP_BAD_MIN_FRAMES = max(2, int(TOP_BAD_MIN_SEC / dt))
    BOTTOM_BAD_MIN_FRAMES = max(2, int(BOTTOM_BAD_MIN_SEC / dt))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, eff_fps, (w, h))

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                out.write(frame)
                continue

            lm = results.pose_landmarks.landmark
            R = mp_pose.PoseLandmark

            hip = np.array([lm[R.RIGHT_HIP.value].x, lm[R.RIGHT_HIP.value].y])
            knee = np.array([lm[R.RIGHT_KNEE.value].x, lm[R.RIGHT_KNEE.value].y])
            ankle = np.array([lm[R.RIGHT_ANKLE.value].x, lm[R.RIGHT_ANKLE.value].y])
            shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])

            knee_angle = calculate_angle(hip, knee, ankle)
            torso_angle = calculate_angle(shoulder, hip, knee)

            # ===== START REP =====
            if knee_angle < 100 and stage != "down":
                stage = "down"
                rep_start_frame = frame_idx
                rep_min_torso_angle = torso_angle
                rep_min_hip_y = hip[1]
                rep_knee_y_at_bottom = knee[1]
                rep_top_bad_frames = 0
                rep_bottom_bad_frames = 0

            # ===== DURING REP =====
            if stage == "down":
                rep_min_torso_angle = min(rep_min_torso_angle, torso_angle)

                # HARD depth collection (this is the fix)
                if hip[1] > rep_min_hip_y:
                    rep_min_hip_y = hip[1]
                    rep_knee_y_at_bottom = knee[1]

                # Back checks
                if hip[1] < rep_knee_y_at_bottom and torso_angle < TOP_THR_DEG:
                    rep_top_bad_frames += 1
                elif hip[1] > rep_knee_y_at_bottom and torso_angle < BOTTOM_THR_DEG:
                    rep_bottom_bad_frames += 1

            # ===== END REP =====
            if knee_angle > STAND_KNEE_ANGLE and stage == "down":
                feedbacks = []
                penalty = 0

                # ---- HARD DEPTH GATE (cannot get 10 without this) ----
                depth_ok = rep_min_hip_y > (rep_knee_y_at_bottom + DEPTH_MARGIN)
                if not depth_ok:
                    feedbacks.append("Try to squat deeper")
                    penalty += 3

                # ---- BACK LOGIC ----
                back_flag = (
                    rep_top_bad_frames >= TOP_BAD_MIN_FRAMES or
                    rep_bottom_bad_frames >= BOTTOM_BAD_MIN_FRAMES
                )
                if back_flag:
                    feedbacks.append("Try to keep your back a bit straighter")
                    penalty += 1

                score = (
                    10.0 if not feedbacks
                    else round(max(4, 10 - min(penalty, 6)) * 2) / 2
                )

                if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                    counter += 1
                    last_rep_frame = frame_idx
                    all_scores.append(score)
                    if score >= 9.5: good_reps += 1
                    else: bad_reps += 1

                rep_reports.append({
                    "rep_index": counter,
                    "score": score,
                    "score_display": display_half(score),
                    "feedback": [pick_strongest_feedback(feedbacks)] if feedbacks else [],
                    "start_frame": rep_start_frame,
                    "end_frame": frame_idx,
                })

                session_best_feedback = merge_feedback(
                    session_best_feedback,
                    [pick_strongest_feedback(feedbacks)] if feedbacks else []
                )

                stage = None

            out.write(frame)

    cap.release()
    if out: out.release()

    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)

    return {
        "squat_count": counter,
        "technique_score": technique_score,
        "technique_score_display": display_half(technique_score),
        "technique_label": score_label(technique_score),
        "form_tip": session_best_feedback,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": [session_best_feedback] if session_best_feedback else ["Great form! Keep it up 💪"],
        "reps": rep_reports,
        "video_path": output_path,
        "feedback_path": feedback_path,
    }

def run_analysis(*args, **kwargs):
    return run_squat_analysis(*args, **kwargs)
