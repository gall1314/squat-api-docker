# -*- coding: utf-8 -*-
# good_morning_analysis.py — Good Morning analysis aligned to existing standards

import os
import cv2
import math
import numpy as np
import subprocess
from collections import Counter
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / FONTS =====================
BAR_BG_ALPHA         = 0.55
TOP_PAD              = 0
LEFT_PAD             = 0

DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)
DEPTH_RING_BG        = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE   = 18

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

REPS_FONT        = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

mp_pose = mp.solutions.pose

# ===================== SKELETON =====================
_FACE_LMS = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS = tuple()


def _init_skeleton_data():
    global _FACE_LMS, _BODY_CONNECTIONS, _BODY_POINTS
    if not _BODY_CONNECTIONS:
        _FACE_LMS = {
            mp_pose.PoseLandmark.NOSE.value,
            mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
            mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
            mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
            mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
        }
        _BODY_CONNECTIONS = tuple((a, b) for (a, b) in mp_pose.POSE_CONNECTIONS if a not in _FACE_LMS and b not in _FACE_LMS)
        _BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))


def _dyn_thickness(h):
    return max(2, int(round(h * 0.002))), max(3, int(round(h * 0.004)))


def draw_body_only(frame, lms, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    line, dot = _dyn_thickness(h)
    for a, b in _BODY_CONNECTIONS:
        pa, pb = lms[a], lms[b]
        ax, ay = int(pa.x * w), int(pa.y * h)
        bx, by = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, line, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = lms[i]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), dot, color, -1, cv2.LINE_AA)
    return frame

# ===================== FEEDBACK SEVERITY =====================
FB_SEVERITY = {
    "Hinge a bit deeper": 3,
    "Avoid excessive knee bend": 2,
    "Control the lowering": 2,
    "Pause briefly at the bottom": 1,
}

FEEDBACK_CATEGORY = {
    "Hinge a bit deeper": "depth",
    "Avoid excessive knee bend": "knees",
    "Control the lowering": "tempo",
    "Pause briefly at the bottom": "tempo",
}

def pick_strongest_feedback(feedback_list):
    best, score = "", -1
    for f in feedback_list or []:
        s = FB_SEVERITY.get(f, 1)
        if s > score:
            best, score = f, s
    return best

def pick_strongest_per_category(feedback_list):
    best_by_cat = {}
    for f in feedback_list or []:
        cat = FEEDBACK_CATEGORY.get(f, "other")
        best = best_by_cat.get(cat)
        if not best or FB_SEVERITY.get(f, 1) > FB_SEVERITY.get(best, 1):
            best_by_cat[cat] = f
    return list(best_by_cat.values()), best_by_cat

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
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    if abs(q - round(q)) < 1e-9:
        return str(int(round(q)))
    return f"{q:.1f}"

# ===================== OVERLAY =====================
def draw_depth_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius   = int(radius)
    thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def _wrap_two_lines(draw, text, font, max_width):
    words = (text or "").split()
    if not words:
        return [""]

    lines = []
    cur = ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) == 2:
            break

    if cur and len(lines) < 2:
        lines.append(cur)

    if len(lines) >= 2 and draw.textlength(lines[-1], font=font) > max_width:
        last = lines[-1] + "…"
        while draw.textlength(last, font=font) > max_width and len(last) > 1:
            last = last[:-2] + "…"
        lines[-1] = last
    return lines


def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Reps בפינת שמאל-עליון; דונאט ימני-עליון; פידבק תחתון"""
    h, w, _ = frame.shape

    # Match pull-up visual scale across mixed aspect ratios:
    # portrait clips otherwise look oversized compared to typical pull-up landscape clips.
    overlay_scale = min(1.0, max(0.78, w / float(max(1, h))))
    reps_size = max(16, int(round(REPS_FONT_SIZE * overlay_scale)))
    feedback_size = max(14, int(round(FEEDBACK_FONT_SIZE * overlay_scale)))
    depth_label_size = max(11, int(round(DEPTH_LABEL_FONT_SIZE * overlay_scale)))
    depth_pct_size = max(13, int(round(DEPTH_PCT_FONT_SIZE * overlay_scale)))

    ref_axis = min(h, w)
    ref_h = max(int(ref_axis * 0.06), int(reps_size * 1.6))
    r = int(ref_h * DONUT_RADIUS_SCALE)
    th = max(3, int(r * DONUT_THICKNESS_FRAC))
    m = 12
    cx = w - m - r
    cy = max(ref_h + r // 8, r + th // 2 + 2)
    pct = float(np.clip(depth_pct, 0.0, 1.0))

    frame = draw_depth_donut(frame, (cx, cy), r, th, pct)

    reps_font = _load_font(FONT_PATH, reps_size)
    feedback_font = _load_font(FONT_PATH, feedback_size)
    depth_label_font = _load_font(FONT_PATH, depth_label_size)
    depth_pct_font = _load_font(FONT_PATH, depth_pct_size)

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    txt = f"Reps: {int(reps)}"
    pad_x, pad_y = 10, 6
    tw = draw.textlength(txt, font=reps_font)
    thh = reps_font.size
    base = np.array(pil)
    over = base.copy()
    cv2.rectangle(over, (0, 0), (int(tw + 2 * pad_x), int(thh + 2 * pad_y)), (0, 0, 0), -1)
    base = cv2.addWeighted(over, BAR_BG_ALPHA, base, 1 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    draw.text((pad_x, pad_y - 1), txt, font=reps_font, fill=(255, 255, 255))

    gap = max(2, int(r * 0.10))
    label = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    by = cy - (depth_label_font.size + gap + depth_pct_font.size) // 2
    lw = draw.textlength(label, font=depth_label_font)
    pw = draw.textlength(pct_txt, font=depth_pct_font)
    draw.text((cx - int(lw // 2), by), label, font=depth_label_font, fill=(255, 255, 255))
    draw.text((cx - int(pw // 2), by + depth_label_font.size + gap), pct_txt, font=depth_pct_font, fill=(255, 255, 255))

    if feedback:
        max_w = int(w - 2 * 12 - 20)
        lines = _wrap_two_lines(draw, feedback, feedback_font, max_w)
        line_h = feedback_font.size + 6
        block_h = 2 * 8 + len(lines) * line_h + (len(lines) - 1) * 4
        y0 = max(0, h - max(6, int(h * 0.02)) - block_h)
        y1 = h - max(6, int(h * 0.02))
        base2 = np.array(pil)
        over2 = base2.copy()
        cv2.rectangle(over2, (0, y0), (w, y1), (0, 0, 0), -1)
        base2 = cv2.addWeighted(over2, BAR_BG_ALPHA, base2, 1 - BAR_BG_ALPHA, 0)
        pil = Image.fromarray(base2)
        draw = ImageDraw.Draw(pil)
        ty = y0 + 8
        for ln in lines:
            tw2 = draw.textlength(ln, font=feedback_font)
            tx = max(12, (w - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=feedback_font, fill=(255, 255, 255))
            ty += line_h + 4

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ===================== GEOMETRY =====================
def angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    nrm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = float(np.dot(ba, bc) / nrm)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))

def torso_angle_to_vertical(hip, shoulder):
    vec = shoulder - hip
    nrm = np.linalg.norm(vec) + 1e-9
    cosang = float(np.dot(vec, np.array([0.0, -1.0])) / nrm)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))

def _get_side_landmarks(lm):
    PL = mp_pose.PoseLandmark
    sides = {
        "left": [PL.LEFT_SHOULDER, PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE],
        "right": [PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE],
    }
    vis = {}
    for side, idxs in sides.items():
        vis[side] = sum(lm[i.value].visibility for i in idxs)
    side = "left" if vis["left"] >= vis["right"] else "right"
    idxs = sides[side]
    return {
        "shoulder": np.array([lm[idxs[0].value].x, lm[idxs[0].value].y]),
        "hip": np.array([lm[idxs[1].value].x, lm[idxs[1].value].y]),
        "knee": np.array([lm[idxs[2].value].x, lm[idxs[2].value].y]),
        "ankle": np.array([lm[idxs[3].value].x, lm[idxs[3].value].y]),
    }

# ===================== ANALYSIS PARAMS =====================
HINGE_START_ANGLE = 18.0
HINGE_BOTTOM_ANGLE = 55.0
STAND_ANGLE = 12.0
MIN_FRAMES_BETWEEN_REPS = 8
PROG_ALPHA = 0.3

KNEE_MIN_ANGLE = 150.0

MIN_ECC_S = 0.35
MIN_BOTTOM_S = 0.12

MIN_SCORE = 4.0
MAX_SCORE = 10.0

def run_good_morning_analysis(video_path,
                              frame_skip=3,
                              scale=0.4,
                              output_path="good_morning_analyzed.mp4",
                              feedback_path="good_morning_feedback.txt",
                              return_video=True,
                              fast_mode=None):
    """
    Good Morning analysis:
    - Counts reps using torso hinge angle (vertical -> hinged -> vertical)
    - Checks hinge depth, knee bend, and tempo
    - Returns same schema as squat/deadlift analyzers

    Notes:
    1. Back-rounding detection was removed (not reliable enough with MediaPipe landmarks)
    2. Feedback is deduplicated so each message appears only once
    """
    if fast_mode is True:
        return_video = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    counter = good_reps = bad_reps = 0
    all_scores = []
    rep_reports = []
    session_feedbacks = []  # Will be deduplicated at the end
    session_feedback_by_cat = {}
    global_best = ""

    in_rep = False
    bottom_reached = False
    last_rep_frame = -999
    frame_idx = 0

    prog = 0.0
    prev_progress = None

    max_torso_angle = 0.0
    min_knee_angle = 999.0
    down_frames = up_frames = bottom_hold_frames = 0

    rt_fb_msg = None
    rt_fb_hold = 0

    with mp_pose.Pose(model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        _init_skeleton_data()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            work = cv2.resize(frame, (0,0), fx=scale, fy=scale) if scale != 1.0 else frame
            if return_video and out is None:
                h0, w0 = work.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w0, h0))

            rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                if return_video and out is not None:
                    frame_drawn = draw_overlay(work.copy(), reps=counter, feedback=(rt_fb_msg if rt_fb_hold > 0 else None), depth_pct=prog)
                    out.write(frame_drawn)
                continue

            lm = res.pose_landmarks.landmark
            pts = _get_side_landmarks(lm)
            shoulder = pts["shoulder"]
            hip = pts["hip"]
            knee = pts["knee"]
            ankle = pts["ankle"]

            torso_angle = torso_angle_to_vertical(hip, shoulder)
            knee_angle = angle_deg(hip, knee, ankle)

            progress = (torso_angle - STAND_ANGLE) / max(1e-6, (HINGE_BOTTOM_ANGLE - STAND_ANGLE))
            progress = float(np.clip(progress, 0.0, 1.0))
            prog = prog + PROG_ALPHA * (progress - prog)

            if not in_rep and torso_angle >= HINGE_START_ANGLE:
                in_rep = True
                bottom_reached = False
                max_torso_angle = torso_angle
                min_knee_angle = knee_angle
                down_frames = up_frames = bottom_hold_frames = 0
                prev_progress = progress

            if in_rep:
                max_torso_angle = max(max_torso_angle, torso_angle)
                min_knee_angle = min(min_knee_angle, knee_angle)

                if prev_progress is not None:
                    if progress > prev_progress + 1e-4:
                        down_frames += 1
                    elif progress < prev_progress - 1e-4:
                        up_frames += 1
                    else:
                        if progress >= 0.98:
                            bottom_hold_frames += 1
                prev_progress = progress

                if torso_angle >= HINGE_BOTTOM_ANGLE:
                    bottom_reached = True

                if bottom_reached and torso_angle <= STAND_ANGLE and (frame_idx - last_rep_frame) >= MIN_FRAMES_BETWEEN_REPS:
                    last_rep_frame = frame_idx
                    counter += 1

                    down_s = down_frames * dt if down_frames else 0.0
                    bottom_s = bottom_hold_frames * dt if bottom_hold_frames else 0.0

                    feedback = []
                    score = MAX_SCORE

                    if max_torso_angle < (HINGE_BOTTOM_ANGLE - 5.0):
                        feedback.append("Hinge a bit deeper")
                        score -= 2.0

                    if min_knee_angle < KNEE_MIN_ANGLE:
                        feedback.append("Avoid excessive knee bend")
                        score -= 1.5

                    if down_s < MIN_ECC_S:
                        feedback.append("Control the lowering")
                        score -= 1.0

                    if bottom_s < MIN_BOTTOM_S:
                        feedback.append("Pause briefly at the bottom")
                        score -= 0.5

                    score = float(max(MIN_SCORE, min(MAX_SCORE, score)))
                    all_scores.append(score)

                    if score >= 9.0:
                        good_reps += 1
                    else:
                        bad_reps += 1

                    # Add feedback to session list (will be deduplicated later)
                    session_feedbacks.extend(feedback)
                    _, session_feedback_by_cat = pick_strongest_per_category(session_feedbacks)
                    global_best = merge_feedback(global_best, feedback)

                    rep_reports.append({
                        "rep": counter,
                        "score": round(score, 2),
                        "score_display": display_half_str(score),
                        "label": score_label(score),
                        "feedback": feedback,
                        "metrics": {
                            "max_torso_angle": round(max_torso_angle, 2),
                            "min_knee_angle": round(min_knee_angle, 2),
                            "eccentric_s": round(down_s, 2),
                            "bottom_hold_s": round(bottom_s, 2)
                        }
                    })

                    rt_fb_msg = pick_strongest_feedback(feedback)
                    rt_fb_hold = int(0.7 / dt)

                    in_rep = False
                    bottom_reached = False

            if rt_fb_hold > 0:
                rt_fb_hold -= 1

            if return_video and out is not None:
                frame_with_skeleton = draw_body_only(work.copy(), lm)
                frame_drawn = draw_overlay(frame_with_skeleton, reps=counter, feedback=(rt_fb_msg if rt_fb_hold > 0 else None), depth_pct=prog)
                out.write(frame_drawn)

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    if np.isnan(avg) or np.isinf(avg):
        avg = 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    if session_feedbacks and len(session_feedbacks) > 0:
        technique_score = min(technique_score, 9.5)

    # Fixed: Deduplicate feedback - each message appears only once
    # Use dict.fromkeys() to preserve order while removing duplicates
    unique_feedbacks = list(dict.fromkeys(session_feedbacks))
    feedback_list = unique_feedbacks if unique_feedbacks else ["Great form! Keep it up 💪"]

    session_tip = None
    if session_feedback_by_cat:
        if "depth" in session_feedback_by_cat:
            session_tip = "Push the hips farther back to load the hamstrings"
        elif "knees" in session_feedback_by_cat:
            session_tip = "Maintain a soft knee bend to keep it a hinge"
        elif "tempo" in session_feedback_by_cat:
            session_tip = "Use a controlled 2–3s lowering for better tension"
    else:
        session_tip = "Nice consistency! Keep building posterior-chain strength"

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list:
                    f.write(f"- {fb}\n")
    except Exception as e:
        print(f"Warning: Could not write feedback file: {e}")

    final_video_path = ""
    if return_video and output_path:
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                encoded_path
            ], check=False, capture_output=True)

            if os.path.exists(output_path) and os.path.exists(encoded_path):
                os.remove(output_path)
            final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")
        except Exception as e:
            print(f"Warning: FFmpeg error: {e}")
            final_video_path = output_path if os.path.exists(output_path) else ""

    result = {
        "squat_count": int(counter),
        "technique_score": float(technique_score),
        "technique_score_display": str(display_half_str(technique_score)),
        "technique_label": str(score_label(technique_score)),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": [str(f) for f in feedback_list],
        "tip": str(session_tip) if session_tip else None,
        "reps": rep_reports,
        "video_path": str(final_video_path),
        "feedback_path": str(feedback_path)
    }

    return result

def run_analysis(*args, **kwargs):
    return run_good_morning_analysis(*args, **kwargs)
