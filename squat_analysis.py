# -*- coding: utf-8 -*-
# squat_analysis.py  (אפשר גם לשמור כ-combined_analysis.py)
import os
import cv2
import math
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / FONTS =====================
BAR_BG_ALPHA         = 0.55
TOP_PAD              = 0    # צמוד לפינה
LEFT_PAD             = 0    # צמוד לפינה

# Donut (מוקטן ומונמך שלא ייחתך)
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
    if not cand: return global_best
    if not global_best: return cand
    return cand if FB_SEVERITY.get(cand,1) >= FB_SEVERITY.get(global_best,1) else global_best

# ===================== HELPERS =====================
def _text_height(font):
    # חישוב גובה טקסט מדויק
    try:
        bbox = font.getbbox("Ag")
        return bbox[3] - bbox[1]
    except Exception:
        return font.size

def wrap_to_n_lines(draw, text, font, max_width, max_lines=2, ellipsis="…"):
    """
    עוטף טקסט עד max_lines שורות. אם חורג — מוסיף …
    """
    words = (text or "").split()
    lines = []
    curr = []
    for w in words:
        test = (" ".join(curr + [w])).strip()
        if draw.textlength(test, font=font) <= max_width:
            curr.append(w)
        else:
            if curr:
                lines.append(" ".join(curr))
            curr = [w]
            if len(lines) == max_lines - 1:  # השורה הבאה תהיה האחרונה
                break

    # הוסף את השארית
    remainder = " ".join(curr).strip()
    if remainder:
        lines.append(remainder)

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    # אם עדיין יש מילים שלא נכנסו — קצץ אחרונה והוסף …
    if len(words) > 0:
        joined = " ".join(words)
        rebuilt = " ".join(lines)
        if joined != rebuilt:
            # יש חיתוך — ודא שהשורה האחרונה נגישה …
            last = lines[-1]
            while last and draw.textlength(last + ellipsis, font=font) > max_width:
                last = last[:-1].rstrip()
            lines[-1] = (last + ellipsis) if last else ellipsis

    return lines

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

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Reps צמוד לפינת שמאל־עליון, דונאט קטן ומונמך, פידבק בתחתית עד 2 שורות ללא חיתוך."""
    h, w, _ = frame.shape

    # --- Reps box (צמוד לפינה) ---
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    text_w = draw.textlength(reps_text, font=REPS_FONT)
    text_h = _text_height(REPS_FONT)
    pad_x, pad_y = 10, 6
    x0, y0 = LEFT_PAD, TOP_PAD
    x1 = min(int(x0 + text_w + 2*pad_x), w-1)
    y1 = int(y0 + text_h + 2*pad_y)

    top = frame.copy()
    cv2.rectangle(top, (x0, y0), (x1, y1), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)

    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((x0 + pad_x, y0 + pad_y - 1), reps_text, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # --- Donut (ימין-עליון) ---
    ref_h = max(int(h * 0.06), int(REPS_FONT.size * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)
    frame = draw_depth_donut(frame, (cx, cy), radius, thick, float(np.clip(depth_pct,0,1)))

    pil  = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"
    pct_txt   = f"{int(float(np.clip(depth_pct,0,1))*100)}%"
    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt,   font=DEPTH_PCT_FONT)
    gap     = max(2, int(radius * 0.10))
    block_h = _text_height(DEPTH_LABEL_FONT) + gap + _text_height(DEPTH_PCT_FONT)
    base_y  = cy - block_h // 2
    draw.text((cx - int(label_w // 2), base_y),                    label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pct_w   // 2), base_y + _text_height(DEPTH_LABEL_FONT) + gap),
              pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # --- Bottom feedback: עד 2 שורות, ללא חיתוך ---
    if feedback:
        pil2 = Image.fromarray(frame)
        draw2 = ImageDraw.Draw(pil2)

        side_pad = 16
        max_text_w = max(20, w - side_pad * 2)
        lines = wrap_to_n_lines(draw2, feedback, FEEDBACK_FONT, max_text_w, max_lines=2)
        line_h = _text_height(FEEDBACK_FONT) + 4
        text_block_h = len(lines) * line_h
        bottom_pad = 12
        bottom_h = max(int(h * 0.10), text_block_h + bottom_pad * 2)

        over = frame.copy()
        cv2.rectangle(over, (0, h - bottom_h), (w, h), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)

        pil2 = Image.fromarray(frame)
        draw2 = ImageDraw.Draw(pil2)

        y = h - bottom_h + bottom_pad
        for ln in lines:
            tw = draw2.textlength(ln, font=FEEDBACK_FONT)
            tx = max(side_pad, (w - int(tw)) // 2)  # מרכז + שוליים כדי לא לגעת בקצה
            draw2.text((tx, y), ln, font=FEEDBACK_FONT, fill=(255,255,255))
            y += line_h

        frame = np.array(pil2)

    return frame

# ===================== BODY-ONLY SKELETON =====================
_FACE_LMS = {
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EE.value if hasattr(mp_pose.PoseLandmark, "RIGHT_EE") else mp_pose.PoseLandmark.RIGHT_EYE.value,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
    mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
    mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
}
_BODY_CONNECTIONS = tuple(
    (a, b) for (a, b) in mp_pose.POSE_CONNECTIONS
    if a not in _FACE_LMS and b not in _FACE_LMS
)
_BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

def draw_body_only(frame, landmarks, color=(255,255,255)):
    h, w = frame.shape[:2]
    for a, b in _BODY_CONNECTIONS:
        pa = landmarks[a]; pb = landmarks[b]
        ax, ay = int(pa.x * w), int(pa.y * h)
        bx, by = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = landmarks[i]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
    return frame

# ===================== GEOMETRY =====================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ===================== SQUAT (Bodyweight) =====================
PERFECT_MIN_KNEE_SQ = 60.0
STAND_KNEE_ANGLE    = 160.0
MIN_FRAMES_BETWEEN_REPS_SQ = 10
DEPTH_ALPHA_SQ      = 0.35

def run_squat_analysis(video_path,
                       frame_skip=3,
                       scale=0.4,
                       output_path="squat_analyzed.mp4",
                       feedback_path="squat_feedback.txt"):
    mp_pose_mod = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0, "good_reps": 0, "bad_reps": 0,
            "feedback": ["Could not open video"], "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    counter = 0
    good_reps = 0
    bad_reps = 0
    rep_reports = []
    all_scores = []

    stage = None
    frame_idx = 0
    last_rep_frame = -999
    session_best_feedback = ""   # ← נשמור פה את החמור ביותר של הסשן

    start_knee_angle = None
    rep_min_knee_angle = 180.0
    rep_max_knee_angle = -999.0
    rep_min_torso_angle = 999.0
    rep_start_frame = None

    depth_smooth = 0.0
    peak_hold = 0
    def update_depth(dt, target):
        nonlocal depth_smooth, peak_hold
        depth_smooth = DEPTH_ALPHA_SQ * target + (1 - DEPTH_ALPHA_SQ) * depth_smooth
        if peak_hold > 0: peak_hold -= 1
        else: depth_smooth *= 0.985
        depth_smooth = float(np.clip(depth_smooth, 0.0, 1.0))
        return depth_smooth

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    with mp_pose_mod.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue
            if scale != 1.0: frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=update_depth(dt, 0.0))
                out.write(frame); continue

            try:
                lm = results.pose_landmarks.landmark
                R = mp_pose_mod.PoseLandmark
                hip      = np.array([lm[R.RIGHT_HIP.value].x,      lm[R.RIGHT_HIP.value].y])
                knee     = np.array([lm[R.RIGHT_KNEE.value].x,     lm[R.RIGHT_KNEE.value].y])
                ankle    = np.array([lm[R.RIGHT_ANKLE.value].x,    lm[R.RIGHT_ANKLE.value].y])
                shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])
                heel_y   = lm[R.RIGHT_HEEL.value].y

                knee_angle   = calculate_angle(hip, knee, ankle)
                torso_angle  = calculate_angle(shoulder, hip, knee)

                # התחלת ירידה
                if knee_angle < 100:
                    if stage != "down":
                        start_knee_angle = float(knee_angle)
                        rep_min_knee_angle = 180.0
                        rep_max_knee_angle = -999.0
                        rep_min_torso_angle = 999.0
                        rep_start_frame = frame_idx
                    stage = "down"

                # תוך כדי ירידה
                if stage == "down":
                    rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                    rep_max_knee_angle = max(rep_max_knee_angle, knee_angle)
                    rep_min_torso_angle = min(rep_min_torso_angle, torso_angle)
                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE_SQ))
                        depth_target = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))
                        update_depth(dt, depth_target)

                # סיום חזרה
                if knee_angle > STAND_KNEE_ANGLE and stage == "down":
                    feedbacks = []
                    penalty = 0.0

                    # עומק
                    hip_to_heel_dist = abs(hip[1] - heel_y)
                    if   hip_to_heel_dist > 0.48: feedbacks.append("Try to squat deeper");            penalty += 3
                    elif hip_to_heel_dist > 0.45: feedbacks.append("Almost there — go a bit lower");  penalty += 2
                    elif hip_to_heel_dist > 0.43: feedbacks.append("Looking good — just a bit more depth"); penalty += 1

                    # גב
                    if torso_angle < 140:
                        feedbacks.append("Try to keep your back a bit straighter"); penalty += 1.0

                    # נעילה (הערה משנית, לא מציגים כטקסט בזמן אמת)
                    if knee_angle < 160:
                        penalty += 0.5

                    # ניקח רק את החמור ביותר לחזרה הזו
                    chosen_fb = pick_strongest_feedback(feedbacks)
                    per_rep_feedback = [chosen_fb] if chosen_fb else []

                    # ציון
                    score = 10.0 if not feedbacks else round(max(4, 10 - min(penalty,6)) * 2) / 2

                    depth_pct = 0.0
                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE_SQ))
                        depth_pct = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))

                    rep_reports.append({
                        "rep_index": counter + 1,
                        "score": round(float(score), 1),
                        "feedback": per_rep_feedback,
                        "start_frame": rep_start_frame or 0,
                        "end_frame": frame_idx,
                        "start_knee_angle": round(float(start_knee_angle or knee_angle), 2),
                        "min_knee_angle": round(float(rep_min_knee_angle), 2),
                        "max_knee_angle": round(float(rep_max_knee_angle), 2),
                        "torso_min_angle": round(float(rep_min_torso_angle), 2),
                        "depth_pct": depth_pct
                    })

                    # עדכן פידבק הסשן (החמור ביותר עד כה)
                    session_best_feedback = merge_feedback(session_best_feedback, per_rep_feedback)

                    start_knee_angle = None
                    stage = "up"

                    # debounce
                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS_SQ:
                        counter += 1
                        last_rep_frame = frame_idx
                        if score >= 9.5: good_reps += 1
                        else: bad_reps += 1
                        all_scores.append(score)

                # שלד – גוף בלבד
                frame = draw_body_only(frame, lm)

                # פידבק בזמן אמת על המסך – מציגים רק החמור ביותר מבין המועמדים
                rt_candidates = []
                if stage == "down" and rep_min_torso_angle < 140:
                    rt_candidates.append("Try to keep your back a bit straighter")
                rt_feedback = pick_strongest_feedback(rt_candidates) or None

                frame = draw_overlay(
                    frame, reps=counter,
                    feedback=rt_feedback,
                    depth_pct=depth_smooth
                )
                out.write(frame)

            except Exception:
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=update_depth(dt, 0.0))
                if out is not None: out.write(frame)
                continue

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    feedback_list = [session_best_feedback] if session_best_feedback else ["Great form! Keep it up 💪"]

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list: f.write(f"- {fb}\n")
    except Exception:
        pass

    # faststart encode
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
            encoded_path
        ], check=False)
        if os.path.exists(output_path) and os.path.exists(encoded_path):
            os.remove(output_path)
    except Exception:
        pass
    final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")

    return {
        "squat_count": counter,
        "technique_score": technique_score,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": feedback_list,  # ← פריט יחיד, החמור ביותר
        "reps": rep_reports,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }

# ===== Alias לשמירה על תאימות ל-app.py =====
def run_analysis(*args, **kwargs):
    return run_squat_analysis(*args, **kwargs)
