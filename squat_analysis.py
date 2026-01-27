# -*- coding: utf-8 -*-
# squat_analysis.py
# FULL FINAL VERSION
# Stable counting + reliable depth (hip/heel) + correct back logic
# Proper reps = score 10 only
# Full overlay + RT feedback + faststart

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

# ===================== FEEDBACK SEVERITY =====================

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

def merge_feedback(global_best, new_list):
    cand = pick_strongest_feedback(new_list)
    if not cand:
        return global_best
    if not global_best:
        return cand
    return cand if FB_SEVERITY.get(cand,1) >= FB_SEVERITY.get(global_best,1) else global_best

# ===================== SCORE DISPLAY =====================

def score_label(s):
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def display_half(x):
    q = round(x * 2) / 2
    return str(int(q)) if q.is_integer() else f"{q:.1f}"

# ===================== GEOMETRY =====================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360 - ang if ang > 180 else ang

# ===================== CORE PARAMS =====================

mp_pose = mp.solutions.pose

PERFECT_MIN_KNEE_SQ = 60.0
STAND_KNEE_ANGLE = 160.0
MIN_FRAMES_BETWEEN_REPS = 10

# Back logic (restored working version)
TOP_THR_DEG = 145.0
BOTTOM_THR_DEG = 100.0
TOP_BAD_MIN_SEC = 0.25
BOTTOM_BAD_MIN_SEC = 0.35

# Depth thresholds (hip vs heel)
HIP_HEEL_BAD = 0.48
HIP_HEEL_OK = 0.45
HIP_HEEL_GOOD = 0.43

# ===================== MAIN =====================

def run_squat_analysis(
    video_path,
    frame_skip=3,
    scale=0.4,
    output_path="squat_analyzed.mp4",
    feedback_path="squat_feedback.txt"
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"squat_count":0,"technique_score":0,"reps":[]}

    counter = 0
    good_reps = 0
    bad_reps = 0
    rep_reports = []
    all_scores = []

    stage = None
    frame_idx = 0
    last_rep_frame = -999

    session_form_tip = None

    stand_knee_ema = None
    STAND_KNEE_ALPHA = 0.3
    depth_live = 0.0

    rep_min_knee = 180
    rep_min_torso = 999
    rep_max_depth = 0.0
    rep_start_frame = None

    rep_top_bad = 0
    rep_bottom_bad = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    eff_fps = max(1.0, fps / frame_skip)
    dt = 1.0 / eff_fps

    TOP_BAD_FRAMES = max(2, int(TOP_BAD_MIN_SEC / dt))
    BOTTOM_BAD_FRAMES = max(2, int(BOTTOM_BAD_MIN_SEC / dt))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    with mp_pose.Pose(model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

            if out is None:
                h,w = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, eff_fps, (w,h))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                out.write(frame)
                continue

            lm = res.pose_landmarks.landmark
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
                    STAND_KNEE_ALPHA * knee_angle + (1-STAND_KNEE_ALPHA)*stand_knee_ema
                )

            if stand_knee_ema:
                denom = max(10.0, stand_knee_ema - PERFECT_MIN_KNEE_SQ)
                depth_live = np.clip((stand_knee_ema - knee_angle)/denom, 0, 1)

            # ---------- START REP ----------
            if knee_angle < 100 and stage != "down":
                stage = "down"
                rep_min_knee = knee_angle
                rep_min_torso = torso_angle
                rep_max_depth = 0
                rep_top_bad = 0
                rep_bottom_bad = 0
                rep_start_frame = frame_idx

            if stage == "down":
                rep_min_knee = min(rep_min_knee, knee_angle)
                rep_min_torso = min(rep_min_torso, torso_angle)
                rep_max_depth = max(rep_max_depth, depth_live)

                if depth_live <= 0.2 and torso_angle < TOP_THR_DEG:
                    rep_top_bad += 1
                elif depth_live >= 0.6 and torso_angle < BOTTOM_THR_DEG:
                    rep_bottom_bad += 1

            # ---------- END REP ----------
            if knee_angle > STAND_KNEE_ANGLE and stage == "down":
                feedbacks = []
                penalty = 0

                hip_heel = abs(hip[1] - heel_y)
                if hip_heel > HIP_HEEL_BAD:
                    feedbacks.append("Try to squat deeper")
                    penalty += 3
                elif hip_heel > HIP_HEEL_OK:
                    feedbacks.append("Almost there — go a bit lower")
                    penalty += 2
                elif hip_heel > HIP_HEEL_GOOD:
                    feedbacks.append("Looking good — just a bit more depth")
                    penalty += 1

                back_flag = (
                    rep_top_bad >= TOP_BAD_FRAMES or
                    (rep_bottom_bad >= BOTTOM_BAD_FRAMES and rep_max_depth < 0.65)
                )

                if back_flag:
                    feedbacks.append("Try to keep your back a bit straighter")
                    penalty += 1

                score = 10.0 if not feedbacks else round(max(4, 10-penalty)*2)/2

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
                    "depth_pct": rep_max_depth
                })

                session_form_tip = merge_feedback(
                    session_form_tip,
                    feedbacks if score < 10 else []
                )

                stage = None

            # ---------- OVERLAY ----------
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            d.text((20,20), f"Reps: {counter}", font=REPS_FONT, fill=(255,255,255))
            d.text((20,55), f"Depth: {int(depth_live*100)}%", font=DEPTH_PCT_FONT, fill=(200,200,200))
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            out.write(frame)

    cap.release()
    out.release()

    avg = np.mean(all_scores) if all_scores else 0.0
    tech = round(round(avg*2)/2,2)

    # faststart
    try:
        tmp = output_path.replace(".mp4","_fs.mp4")
        subprocess.run(["ffmpeg","-y","-i",output_path,"-movflags","+faststart",tmp],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(tmp, output_path)
    except:
        pass

    return {
        "squat_count": counter,
        "technique_score": tech,
        "technique_score_display": display_half(tech),
        "technique_label": score_label(tech),
        "form_tip": session_form_tip,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "reps": rep_reports,
        "video_path": output_path,
        "feedback_path": feedback_path
    }

def run_analysis(*args, **kwargs):
    return run_squat_analysis(*args, **kwargs)
