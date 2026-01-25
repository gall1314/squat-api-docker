# -*- coding: utf-8 -*-
# squat_analysis.py — בסיס שסופר טוב + Overlay צמוד לפינה
# פאי-גרף "לייב" ללא דיליי (עולה/יורד בזמן אמת), פידבק גב מדויק, ו-RT feedback עם Hold.
import os
import cv2
import math
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / FONTS =====================
BAR_BG_ALPHA = 0.55

DONUT_RADIUS_SCALE = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR = (40, 200, 80)
DEPTH_RING_BG = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE = 18


def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


REPS_FONT = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

mp_pose = mp.solutions.pose

# ===================== FEEDBACK SEVERITY =====================
FB_SEVERITY = {
    "Try to squat deeper": 3,
    "Avoid knee collapse": 3,
    "Try to keep your back a bit straighter": 2,
    "Almost there — go a bit lower": 2,
    "Looking good — just a bit more depth": 1,
}


def pick_strongest_feedback(feedback_list):
    best, score = "", -1
    for f in feedback_list or []:
        s = FB_SEVERITY.get(f, 1)
        if s > score:
            best, score = f, s
    return best


def merge_feedback(global_best, new_list):
    cand = pick_strongest_feedback(new_list)
    if not cand:
        return global_best
    if not global_best:
        return cand
    return cand if FB_SEVERITY.get(cand, 1) >= FB_SEVERITY.get(global_best, 1) else global_best


# ===================== SCORE DISPLAY =====================
def score_label(s):
    s = float(s)
    if s >= 9.5:
        return "Excellent"
    if s >= 8.5:
        return "Very good"
    if s >= 7.0:
        return "Good"
    if s >= 5.5:
        return "Fair"
    return "Needs work"


def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    return str(int(q)) if q.is_integer() else f"{q:.1f}"


# ===================== GEOMETRY =====================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


# ===================== CORE PARAMS =====================
PERFECT_MIN_KNEE_SQ = 60.0
STAND_KNEE_ANGLE = 160.0
MIN_FRAMES_BETWEEN_REPS_SQ = 10

HIP_VEL_THRESH_PCT = 0.014
ANKLE_VEL_THRESH_PCT = 0.017
EMA_ALPHA = 0.65
MOVEMENT_CLEAR_FRAMES = 2


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
    session_best_feedback = ""

    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    movement_free_streak = 0

    start_knee_angle = None
    rep_min_knee_angle = 180.0
    rep_min_torso_angle = 999.0
    rep_start_frame = None
    rep_max_depth = 0.0


    stand_knee_ema = None
    STAND_KNEE_ALPHA = 0.30
    depth_live = 0.0

    # גב
    TOP_THR_DEG = 145.0
    BOTTOM_THR_DEG    = 100.0
    TOP_BAD_MIN_SEC = 0.25
    BOTTOM_BAD_MIN_SEC = 0.35

    rep_top_bad_frames = 0
    rep_bottom_bad_frames = 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / frame_skip)
    dt = 1.0 / effective_fps

    TOP_BAD_MIN_FRAMES = max(2, int(TOP_BAD_MIN_SEC / dt))
    BOTTOM_BAD_MIN_FRAMES = max(2, int(BOTTOM_BAD_MIN_SEC / dt))

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                out.write(frame)
                continue

            lm = results.pose_landmarks.landmark
            R = mp_pose.PoseLandmark

            hip = np.array([lm[R.RIGHT_HIP.value].x, lm[R.RIGHT_HIP.value].y])
            knee = np.array([lm[R.RIGHT_KNEE.value].x, lm[R.RIGHT_KNEE.value].y])
            ankle = np.array([lm[R.RIGHT_ANKLE.value].x, lm[R.RIGHT_ANKLE.value].y])
            shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])
            heel_y = lm[R.RIGHT_HEEL.value].y

            knee_angle = calculate_angle(hip, knee, ankle)
            torso_angle = calculate_angle(shoulder, hip, knee)

            if knee_angle > STAND_KNEE_ANGLE - 5:
                stand_knee_ema = knee_angle if stand_knee_ema is None else (
                    STAND_KNEE_ALPHA * knee_angle + (1 - STAND_KNEE_ALPHA) * stand_knee_ema
                )

            if stand_knee_ema is not None:
                denom = max(10.0, stand_knee_ema - PERFECT_MIN_KNEE_SQ)
                depth_live = np.clip((stand_knee_ema - knee_angle) / denom, 0, 1)

            if knee_angle < 100 and stage != "down":
                stage = "down"
                start_knee_angle = knee_angle
                rep_min_knee_angle = knee_angle
                rep_min_torso_angle = torso_angle
                rep_top_bad_frames = 0
                rep_bottom_bad_frames = 0
                rep_start_frame = frame_idx

            if stage == "down":
                rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                rep_min_torso_angle = min(rep_min_torso_angle, torso_angle)
                rep_max_depth = max(rep_max_depth, depth_live)


                if depth_live <= 0.2 and torso_angle < TOP_THR_DEG:
                    rep_top_bad_frames += 1
                elif depth_live >= 0.6 and torso_angle < BOTTOM_THR_DEG:
                    rep_bottom_bad_frames += 1

            if knee_angle > STAND_KNEE_ANGLE and stage == "down":
                feedbacks = []
                penalty = 0.0

                hip_to_heel_dist = abs(hip[1] - heel_y)
                if hip_to_heel_dist > 0.48:
                    feedbacks.append("Try to squat deeper")
                    penalty += 3
                elif hip_to_heel_dist > 0.45:
                    feedbacks.append("Almost there — go a bit lower")
                    penalty += 2
                elif hip_to_heel_dist > 0.43:
                    feedbacks.append("Looking good — just a bit more depth")
                    penalty += 1

                # 🔥 ריכוך התנאי — זה השינוי היחיד בקובץ
                back_flag = (
                    rep_top_bad_frames >= TOP_BAD_MIN_FRAMES
                    or (
                        rep_bottom_bad_frames >= BOTTOM_BAD_MIN_FRAMES
                        and depth_live < 0.85
                    )
                )

                if back_flag:
                    feedbacks.append("Try to keep your back a bit straighter")
                    penalty += 1

                score = 10.0 if not feedbacks else round(max(4, 10 - min(penalty, 6)) * 2) / 2

                rep_reports.append(
                    {
                        "rep_index": counter + 1,
                        "score": score,
                        "score_display": display_half_str(score),
                        "feedback": [pick_strongest_feedback(feedbacks)] if feedbacks else [],
                        "start_frame": rep_start_frame,
                        "end_frame": frame_idx,
                        "depth_pct": depth_live,
                    }
                )

                session_best_feedback = merge_feedback(
                    session_best_feedback,
                    [pick_strongest_feedback(feedbacks)] if feedbacks else [],
                )

                counter += 1
                all_scores.append(score)
                if score >= 9.5:
                    good_reps += 1
                else:
                    bad_reps += 1

                stage = None

            out.write(frame)

    cap.release()
    if out:
        out.release()

    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)

    return {
        "squat_count": counter,
        "technique_score": technique_score,
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": [session_best_feedback] if session_best_feedback else ["Great form! Keep it up 💪"],
        "reps": rep_reports,
        "video_path": output_path,
        "feedback_path": feedback_path,
    }


def run_analysis(*args, **kwargs):
    return run_squat_analysis(*args, **kwargs)
