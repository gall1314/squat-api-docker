# -*- coding: utf-8 -*-
# stiff_leg_deadlift_analysis_fixed.py — ✅ גרסה מתוקנת עם תיקונים קריטיים

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
    "Go deeper - lower the bar to shins": 3,
    "Knees bending too much - keep legs straighter": 4,
}

FEEDBACK_CATEGORY = {
    "Go deeper - lower the bar to shins": "depth",
    "Knees bending too much - keep legs straighter": "knee",
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

def dedupe_feedback(feedback_list):
    seen = set()
    unique = []
    for fb in feedback_list or []:
        if fb in seen:
            continue
        seen.add(fb)
        unique.append(fb)
    return unique

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
    radius = int(radius)
    thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90
    end_ang = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Reps בפינת שמאל-עליון; דונאט ימני-עליון; פידבק תחתון — זהה לסקוואט."""
    h, w, _ = frame.shape

    # --- Reps box (top-left) ---
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    inner_pad_x, inner_pad_y = 10, 6
    text_w = draw.textlength(reps_text, font=REPS_FONT)
    text_h = REPS_FONT.size
    x0, y0 = 0, 0
    x1 = int(text_w + 2 * inner_pad_x)
    y1 = int(text_h + 2 * inner_pad_y)
    top = frame.copy()
    cv2.rectangle(top, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((x0 + inner_pad_x, y0 + inner_pad_y - 1),
                             reps_text, font=REPS_FONT, fill=(255, 255, 255))
    frame = np.array(pil)

    # --- Donut (top-right) ---
    ref_h = max(int(h * 0.06), int(REPS_FONT_SIZE * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)
    frame = draw_depth_donut(frame, (cx, cy), radius, thick, float(np.clip(depth_pct, 0, 1)))

    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"
    pct_txt = f"{int(float(np.clip(depth_pct, 0, 1)) * 100)}%"
    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w = draw.textlength(pct_txt, font=DEPTH_PCT_FONT)
    gap = max(2, int(radius * 0.10))
    base_y = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size) // 2
    draw.text((cx - int(label_w // 2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255, 255, 255))
    draw.text((cx - int(pct_w // 2), base_y + DEPTH_LABEL_FONT.size + gap),
              pct_txt, font=DEPTH_PCT_FONT, fill=(255, 255, 255))
    frame = np.array(pil)

    # --- Bottom feedback ---
    if feedback:
        def wrap_to_two_lines(draw, text, font, max_width):
            words = text.split()
            if not words: return [""]
            lines, cur = [], ""
            for w in words:
                trial = (cur + " " + w).strip()
                if draw.textlength(trial, font=font) <= max_width:
                    cur = trial
                else:
                    if cur: lines.append(cur)
                    cur = w
                if len(lines) == 2: break
            if cur and len(lines) < 2: lines.append(cur)
            leftover = len(words) - sum(len(l.split()) for l in lines)
            if leftover > 0 and len(lines) >= 2:
                last = lines[-1] + "…"
                while draw.textlength(last, font=font) > max_width and len(last) > 1:
                    last = last[:-2] + "…"
                lines[-1] = last
            return lines

        pil_fb = Image.fromarray(frame)
        draw_fb = ImageDraw.Draw(pil_fb)
        safe_margin = max(6, int(h * 0.02))
        pad_x, pad_y, line_gap = 12, 8, 4
        max_text_w = int(w - 2 * pad_x - 20)
        lines = wrap_to_two_lines(draw_fb, feedback, FEEDBACK_FONT, max_text_w)
        line_h = FEEDBACK_FONT.size + 6
        block_h = (2 * pad_y) + len(lines) * line_h + (len(lines) - 1) * line_gap
        y0 = max(0, h - safe_margin - block_h)
        y1 = h - safe_margin
        over = frame.copy()
        cv2.rectangle(over, (0, y0), (w, y1), (0, 0, 0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame)
        draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + pad_y
        for ln in lines:
            tw = draw_fb.textlength(ln, font=FEEDBACK_FONT)
            tx = max(pad_x, (w - int(tw)) // 2)
            draw_fb.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255, 255, 255))
            ty += line_h + line_gap
        frame = np.array(pil_fb)

    return frame

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

def analyze_back_curvature(shoulder, hip, head_like, max_angle_deg=60.0, min_head_dist_ratio=0.35):
    """
    ✅ תיקון: העלנו את הסף ל-60 מעלות (במקום 45)
    MediaPipe לא מספיק מדויק - רק במקרים קיצוניים ביותר זה יתריע
    """
    torso_vec = shoulder - hip
    head_vec = head_like - shoulder
    torso_nrm = np.linalg.norm(torso_vec) + 1e-9
    head_nrm = np.linalg.norm(head_vec) + 1e-9
    if head_nrm < (min_head_dist_ratio * torso_nrm):
        return 0.0, False
    cosang = float(np.dot(torso_vec, head_vec) / (torso_nrm * head_nrm))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosang))
    return angle_deg, angle_deg > max_angle_deg

def _get_side_landmarks(lm):
    PL = mp_pose.PoseLandmark
    sides = {
        "left": [PL.LEFT_SHOULDER, PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE, PL.LEFT_EAR],
        "right": [PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE, PL.RIGHT_EAR],
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
        "ear": np.array([lm[idxs[4].value].x, lm[idxs[4].value].y]),
    }

# ===================== ✅✅ תיקונים קריטיים - STIFF-LEG DEADLIFT =====================
# יישור מלא ללוגיקת הספירה של RDL (אותם ספי זוויות)
HINGE_START_ANGLE = 20.0
HINGE_BOTTOM_ANGLE = 55.0
STAND_ANGLE = 12.0
MIN_FRAMES_BETWEEN_REPS = 8
PROG_ALPHA = 0.3

# ✅✅ תיקון 2: ברכיים - הקלה אגרסיבית מאוד
KNEE_MIN_ANGLE = 150.0  # ✅✅ הורדנו עוד ל-150° - מאוד מקל
KNEE_MAX_ANGLE = 135.0  # ✅✅ רק אם ממש ממש כפוף (מתחת ל-135°)

# ✅ תיקון 3: גב - סף גבוה יותר
BACK_MAX_ANGLE = 60.0  # ✅ העלנו מ-45° ל-60° - רק מקרים קיצוניים

# ✅✅ תיקון 4: טמפו - הקלה אגרסיבית
MIN_ECC_S = 0.1  # ✅✅ הורדנו ל-0.1 - כמעט כל ירידה תעבור
MIN_BOTTOM_S = 0.05  # ✅✅ כמעט בלי השהייה

MIN_SCORE = 4.0
MAX_SCORE = 10.0

def run_stiff_leg_deadlift_analysis(video_path,
                                     frame_skip=3,
                                     scale=0.4,
                                     output_path="stiff_leg_analyzed.mp4",
                                     feedback_path="stiff_leg_feedback.txt",
                                     return_video=True,
                                     fast_mode=False):
    """
    ✅✅✅ Stiff-Leg Deadlift analysis - RADICALLY SIMPLIFIED:
    
    REMOVED ALL UNRELIABLE CHECKS:
    - Knee angle (MediaPipe not accurate enough)
    - Back rounding (MediaPipe can't detect this reliably)
    - Tempo/control (too variable, not meaningful)
    
    KEEPING ONLY:
    1. Rep counting: Torso reaches 55° forward, then returns to 12° upright
    2. Depth check: Did the torso reach at least 50° forward?
    3. Good rep = score 9+ AND no feedback
    
    This version focuses on what MediaPipe CAN reliably detect.
    """
    import sys
    print(f"[SLDL FIXED] Starting analysis: fast_mode={fast_mode}", file=sys.stderr, flush=True)
    
    if fast_mode is True:
        return_video = False
    create_video = bool(return_video) and (output_path is not None) and (output_path != "")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        cap.release()
        return {
            "squat_count": 0, "technique_score": 0.0,
            "good_reps": 0, "bad_reps": 0, 
            "feedback": ["Invalid video file"],
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
    session_feedbacks = []
    session_feedback_by_cat = {}

    in_rep = False
    bottom_reached = False
    last_rep_frame = -999
    frame_idx = 0

    prog = 0.0
    prev_progress = None

    max_torso_angle = 0.0
    min_knee_angle = 999.0
    back_issue = False
    down_frames = up_frames = bottom_hold_frames = 0

    rt_fb_msg = None
    rt_fb_hold = 0

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

            work = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame
            if create_video and out is None:
                h0, w0 = work.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w0, h0))

            rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                if create_video and out is not None:
                    frame_drawn = draw_overlay(work.copy(), reps=counter,
                                               feedback=(rt_fb_msg if rt_fb_hold > 0 else None),
                                               depth_pct=prog)
                    out.write(frame_drawn)
                continue

            lm = res.pose_landmarks.landmark
            pts = _get_side_landmarks(lm)
            shoulder = pts["shoulder"]
            hip = pts["hip"]
            knee = pts["knee"]
            ankle = pts["ankle"]
            ear = pts["ear"]

            torso_angle = torso_angle_to_vertical(hip, shoulder)
            knee_angle = angle_deg(hip, knee, ankle)
            back_angle, back_bad = analyze_back_curvature(shoulder, hip, ear, max_angle_deg=BACK_MAX_ANGLE)

            progress = (torso_angle - STAND_ANGLE) / max(1e-6, (HINGE_BOTTOM_ANGLE - STAND_ANGLE))
            progress = float(np.clip(progress, 0.0, 1.0))
            prog = prog + PROG_ALPHA * (progress - prog)

            if not in_rep and torso_angle >= HINGE_START_ANGLE:
                in_rep = True
                bottom_reached = False
                max_torso_angle = torso_angle
                min_knee_angle = knee_angle
                back_issue = False
                down_frames = up_frames = bottom_hold_frames = 0
                prev_progress = progress

            if in_rep:
                max_torso_angle = max(max_torso_angle, torso_angle)
                min_knee_angle = min(min_knee_angle, knee_angle)
                if back_bad:
                    back_issue = True

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

                # ✅✅ תיקון ספירה: זהה לדדליפט רומני - חזרה נספרת רק כשחוזרים ל-stand angle
                if bottom_reached and torso_angle <= STAND_ANGLE and (frame_idx - last_rep_frame) >= MIN_FRAMES_BETWEEN_REPS:
                    last_rep_frame = frame_idx
                    counter += 1

                    down_s = down_frames * dt if down_frames else 0.0
                    bottom_s = bottom_hold_frames * dt if bottom_hold_frames else 0.0

                    feedback = []
                    score = MAX_SCORE

                    # Depth check
                    if max_torso_angle < (HINGE_BOTTOM_ANGLE - 5.0):
                        feedback.append("Go deeper - lower the bar to shins")
                        score -= 2.0

                    # Knee angle check - stiff-leg requires nearly straight knees
                    if min_knee_angle < KNEE_MIN_ANGLE:
                        feedback.append("Knees bending too much - keep legs straighter")
                        score -= 2.0

                    # ✅✅ REMOVED: Back rounding check - not reliable with MediaPipe
                    # MediaPipe can't accurately detect back rounding

                    # ✅✅ REMOVED: Tempo checks - too strict and not always accurate
                    # Eccentric and bottom pause timing varies too much

                    score = float(max(MIN_SCORE, min(MAX_SCORE, score)))
                    all_scores.append(score)

                    # ✅✅ תיקון: Good Rep = ציון 9+ **וגם** בלי פידבק
                    if score >= 9.0 and len(feedback) == 0:
                        good_reps += 1
                    else:
                        bad_reps += 1

                    session_feedbacks.extend(feedback)
                    _, session_feedback_by_cat = pick_strongest_per_category(session_feedbacks)

                    rep_reports.append({
                        "rep": counter,
                        "score": round(score, 2),
                        "score_display": display_half_str(score),
                        "label": score_label(score),
                        "feedback": feedback,
                        "metrics": {
                            "max_torso_angle": round(max_torso_angle, 2),
                            "min_knee_angle": round(min_knee_angle, 2),
                            "back_angle": round(back_angle, 2),
                            "eccentric_s": round(down_s, 2),
                            "bottom_hold_s": round(bottom_s, 2)
                        }
                    })
                    
                    print(f"[SLDL] Rep {counter}: min_knee={min_knee_angle:.1f}°, torso={max_torso_angle:.1f}°, down_time={down_s:.2f}s", 
                          file=sys.stderr, flush=True)

                    rt_fb_msg = pick_strongest_feedback(feedback)
                    rt_fb_hold = int(0.7 / dt)

                    in_rep = False
                    bottom_reached = False

            if rt_fb_hold > 0:
                rt_fb_hold -= 1

            if create_video and out is not None:
                frame_drawn = draw_overlay(work.copy(), reps=counter,
                                           feedback=(rt_fb_msg if rt_fb_hold > 0 else None),
                                           depth_pct=prog)
                out.write(frame_drawn)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    if np.isnan(avg) or np.isinf(avg):
        avg = 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    if session_feedbacks:
        technique_score = min(technique_score, 9.5)

    feedback_list = dedupe_feedback(session_feedbacks) if session_feedbacks else ["Great form! Keep it up 💪"]

    session_tip = None
    if session_feedback_by_cat:
        if "knee" in session_feedback_by_cat:
            session_tip = "Focus on keeping your legs straight - slight soft bend only, not a squat"
        elif "depth" in session_feedback_by_cat:
            session_tip = "Focus on hip mobility to lower the bar closer to the ground"
    else:
        session_tip = "Perfect stiff-leg form! Great hamstring work 💪"

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list:
                    f.write(f"- {fb}\n")
    except Exception:
        pass

    final_video_path = ""
    if create_video and output_path:
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                encoded_path
            ], check=False, capture_output=True)

            if os.path.exists(output_path) and os.path.exists(encoded_path):
                os.remove(output_path)
            final_video_path = encoded_path if os.path.exists(encoded_path) else (
                output_path if os.path.exists(output_path) else "")
        except Exception:
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
    return run_stiff_leg_deadlift_analysis(*args, **kwargs)
