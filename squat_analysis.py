# -*- coding: utf-8 -*-
# squat_analysis.py — FINAL CONSOLIDATED VERSION
# Stable reps + true depth latch + correct back logic + UI-safe output

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

def pick_strongest(feedbacks):
    best, s = None, -1
    for f in feedbacks or []:
        sev = FB_SEVERITY.get(f, 1)
        if sev > s:
            best, s = f, sev
    return best

# ===================== SCORE DISPLAY =====================

def score_label(x):
    if x >= 9.5: return "Excellent"
    if x >= 8.5: return "Very good"
    if x >= 7.0: return "Good"
    if x >= 5.5: return "Fair"
    return "Needs work"

def display_half(x):
    q = round(x * 2) / 2
    return str(int(q)) if q.is_integer() else f"{q:.1f}"

# ===================== GEOMETRY =====================

def angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    r = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(r * 180 / np.pi)
    return 360-ang if ang > 180 else ang

# ===================== PARAMS =====================

mp_pose = mp.solutions.pose

STAND_KNEE = 160
DOWN_ENTER = 100
MIN_FRAMES_BETWEEN_REPS = 10

# depth
MIN_HIP_BELOW_KNEE = 0.015   # normalized Y margin

# back
TOP_THR = 145
BOTTOM_THR = 100
TOP_BAD_SEC = 0.25
BOTTOM_BAD_SEC = 0.35

# ===================== MAIN =====================

def run_squat_analysis(video_path, frame_skip=3, scale=0.4,
                       output_path="squat_out.mp4",
                       feedback_path="feedback.txt"):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"squat_count": 0, "technique_score": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    eff_fps = max(1.0, fps / frame_skip)
    dt = 1 / eff_fps
    TOP_BAD_FR = int(TOP_BAD_SEC / dt)
    BOTTOM_BAD_FR = int(BOTTOM_BAD_SEC / dt)

    counter = good = bad = 0
    last_rep_frame = -999
    stage = None
    frame_idx = 0

    all_scores = []
    reps = []
    session_tip = None

    # rep state
    rep_start = None
    rep_min_torso = 999
    top_bad = bottom_bad = 0

    # depth latch (THIS IS THE FIX)
    bottom_locked = False
    bottom_hip_y = None
    bottom_knee_y = None

    with mp_pose.Pose(1, 0.5, 0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            if scale != 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            R = mp_pose.PoseLandmark

            hip = np.array([lm[R.RIGHT_HIP].x, lm[R.RIGHT_HIP].y])
            knee = np.array([lm[R.RIGHT_KNEE].x, lm[R.RIGHT_KNEE].y])
            ankle = np.array([lm[R.RIGHT_ANKLE].x, lm[R.RIGHT_ANKLE].y])
            shoulder = np.array([lm[R.RIGHT_SHOULDER].x, lm[R.RIGHT_SHOULDER].y])

            knee_ang = angle(hip, knee, ankle)
            torso_ang = angle(shoulder, hip, knee)

            # ===== ENTER REP =====
            if knee_ang < DOWN_ENTER and stage != "down":
                stage = "down"
                rep_start = frame_idx
                rep_min_torso = torso_ang
                top_bad = bottom_bad = 0
                bottom_locked = False
                bottom_hip_y = None
                bottom_knee_y = None

            # ===== DURING DOWN =====
            if stage == "down":
                rep_min_torso = min(rep_min_torso, torso_ang)

                # latch bottom ONCE
                if not bottom_locked:
                    if bottom_hip_y is None or hip[1] > bottom_hip_y:
                        bottom_hip_y = hip[1]
                        bottom_knee_y = knee[1]

                # back logic
                if knee_ang > 120 and torso_ang < TOP_THR:
                    top_bad += 1
                if knee_ang <= 120 and torso_ang < BOTTOM_THR:
                    bottom_bad += 1

            # ===== EXIT REP =====
            if knee_ang > STAND_KNEE and stage == "down":

                feedbacks = []
                penalty = 0

                # -------- DEPTH (CORRECT) --------
                depth_ok = (
                    bottom_hip_y is not None and
                    bottom_hip_y > bottom_knee_y + MIN_HIP_BELOW_KNEE
                )

                if not depth_ok:
                    feedbacks.append("Try to squat deeper")
                    penalty += 3

                # -------- BACK --------
                if top_bad >= TOP_BAD_FR or bottom_bad >= BOTTOM_BAD_FR:
                    feedbacks.append("Try to keep your back a bit straighter")
                    penalty += 1

                score = 10.0 if not feedbacks else max(4, 10 - penalty)
                score = round(score * 2) / 2

                if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                    counter += 1
                    last_rep_frame = frame_idx
                    all_scores.append(score)
                    good += int(score >= 9.5)
                    bad += int(score < 9.5)

                    reps.append({
                        "rep_index": counter,
                        "score": score,
                        "score_display": display_half(score),
                        "feedback": [pick_strongest(feedbacks)] if feedbacks else [],
                        "start_frame": rep_start,
                        "end_frame": frame_idx
                    })

                    tip = pick_strongest(feedbacks)
                    if tip and (not session_tip or FB_SEVERITY[tip] > FB_SEVERITY.get(session_tip, 0)):
                        session_tip = tip

                stage = None

    avg = np.mean(all_scores) if all_scores else 0.0
    avg = round(round(avg * 2) / 2, 2)

    return {
        "squat_count": counter,
        "technique_score": avg,
        "technique_score_display": display_half(avg),
        "technique_label": score_label(avg),
        "form_tip": session_tip,
        "good_reps": good,
        "bad_reps": bad,
        "feedback": [session_tip] if session_tip else ["Great form! Keep it up 💪"],
        "reps": reps,
        "video_path": output_path,
        "feedback_path": feedback_path
    }

def run_analysis(*args, **kwargs):
    return run_squat_analysis(*args, **kwargs)
