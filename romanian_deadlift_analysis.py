# -*- coding: utf-8 -*-
# romanian_deadlift_analysis_fixed.py — תיקון סף ירידה (eccentric) - גרסה מתוקנת

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
    "Bend your knees a bit more": 3,
    "Too much knee bend": 3,
    "Try to keep your back neutral": 2,
    "Control the lowering": 2,
    "Pause at the bottom": 1,
}

FEEDBACK_CATEGORY = {
    "Bend your knees a bit more": "knees",
    "Too much knee bend": "knees",
    "Try to keep your back neutral": "back",
    "Control the lowering": "tempo",
    "Pause at the bottom": "tempo",
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
    h, w, _ = frame.shape
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    
    # Reps counter
    reps_text = f"Reps: {reps}"
    inner_pad_x, inner_pad_y = 10, 6
    text_w = draw.textlength(reps_text, font=REPS_FONT)
    text_h = REPS_FONT.size
    x0, y0 = 0, 0
    x1 = int(text_w + 2 * inner_pad_x)
    y1 = int(text_h + 2 * inner_pad_y)
    overlay = Image.new('RGBA', (x1, y1), (0, 0, 0, int(255 * BAR_BG_ALPHA)))
    pil.paste(overlay, (x0, y0), overlay)
    draw.text((x0 + inner_pad_x, y0 + inner_pad_y), reps_text, font=REPS_FONT, fill=(255, 255, 255))

    # Depth donut
    donut_r = int(h * DONUT_RADIUS_SCALE * 0.09)
    thickness = max(6, int(donut_r * DONUT_THICKNESS_FRAC))
    cx = w - donut_r - 15
    cy = donut_r + 15
    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    frame = draw_depth_donut(frame, (cx, cy), donut_r, thickness, depth_pct)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)

    draw.text((cx, cy - 10 - DEPTH_LABEL_FONT_SIZE), "DEPTH", font=DEPTH_LABEL_FONT, fill=(255, 255, 255), anchor="mm")
    draw.text((cx, cy + 10), f"{int(depth_pct * 100)}%", font=DEPTH_PCT_FONT, fill=(255, 255, 255), anchor="mm")

    # Feedback bar
    if feedback:
        fb_text = feedback
        text_h = FEEDBACK_FONT.size
        fb_pad_x, fb_pad_y = 10, 6
        fb_h = int(text_h + 2 * fb_pad_y)
        overlay = Image.new('RGBA', (w, fb_h), (0, 0, 0, int(255 * BAR_BG_ALPHA)))
        pil.paste(overlay, (0, h - fb_h), overlay)
        draw.text((fb_pad_x, h - fb_h + fb_pad_y), fb_text, font=FEEDBACK_FONT, fill=(255, 255, 255))

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

def analyze_back_curvature(shoulder, hip, head_like, max_angle_deg=45.0, min_head_dist_ratio=0.35):
    """
    ✅ תיקון: העלנו את הסף ל-45 מעלות
    MediaPipe לא מספיק מדויק לזיהוי back rounding עדין
    רק במקרים קיצוניים זה יתריע
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

# ===================== תיקון פרמטרים - גרסה סופית =====================
HINGE_START_ANGLE = 20.0
HINGE_BOTTOM_ANGLE = 55.0
STAND_ANGLE = 12.0
MIN_FRAMES_BETWEEN_REPS = 8
PROG_ALPHA = 0.3

# ✅ ברכיים בדדליפט רומני - צריך עיקום קל של 15-20 מעלות
KNEE_MIN_ANGLE = 155.0  # מעל לזה = ברכיים ישרות מדי
KNEE_OPTIMAL_MIN = 160.0
KNEE_OPTIMAL_MAX = 170.0
KNEE_MAX_ANGLE = 140.0  # מתחת לזה = יותר מדי כיפוף

# ✅ גב - MediaPipe לא מדויק מספיק, סף גבוה
BACK_MAX_ANGLE = 45.0

# ✅✅✅ תיקון קריטי: הקלה עוד יותר משמעותית בדרישות הזמן
# רק נתריע במקרים קיצוניים של נפילה חופשית ממש
MIN_ECC_S = 0.08  # ✅ ירידה מבוקרת (רק נגד נפילה חופשית קיצונית)
MIN_BOTTOM_S = 0.03  # ✅ השהייה מינימלית בתחתית

MIN_SCORE = 4.0
MAX_SCORE = 10.0

def run_romanian_deadlift_analysis(video_path,
                                   frame_skip=3,
                                   scale=0.4,
                                   output_path="romanian_deadlift_analyzed.mp4",
                                   feedback_path="romanian_deadlift_feedback.txt",
                                   return_video=True,
                                   fast_mode=False):
    """
    Romanian Deadlift analysis - FIXED:
    ✅ ברכיים: מותר עיקום 160-170 מעלות (עיקום קל)
    ✅ גב: מותר נטייה עד 45 מעלות
    ✅✅✅ זמן ירידה: סף מינימלי מאוד + 3 פריימים (רק נגד נפילה חופשית קיצונית)
    ✅ fast_mode: במצב מהיר - אותה לוגיקה בדיוק, רק ללא ציור וידאו
    """
    import sys
    print(f"[RDL] Starting analysis: fast_mode={fast_mode}, return_video={return_video}", file=sys.stderr, flush=True)
    print(f"[RDL] Video path: {video_path}", file=sys.stderr, flush=True)
    
    mp_pose_mod = mp.solutions.pose

    if fast_mode is True:
        return_video = False
    create_video = bool(return_video) and (output_path is not None) and (output_path != "")
    
    print(f"[RDL] create_video={create_video}", file=sys.stderr, flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[RDL ERROR] Cannot open video: {video_path}", file=sys.stderr, flush=True)
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"[RDL] Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration", file=sys.stderr, flush=True)
    
    if total_frames < 10:
        print(f"[RDL ERROR] File has only {total_frames} frames - this is not a valid video!", file=sys.stderr, flush=True)
        cap.release()
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, 
            "feedback": [f"Invalid video file - only {total_frames} frame(s). Please upload an actual video."],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    if fast_mode:
        effective_frame_skip = frame_skip
        effective_scale = scale * 0.85
        model_complexity = 0
        print(f"[RDL FAST] frame_skip={effective_frame_skip}, scale={effective_scale:.2f}, model=lite", file=sys.stderr, flush=True)
    else:
        effective_frame_skip = frame_skip
        effective_scale = scale
        model_complexity = 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, effective_frame_skip))
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
    max_knee_angle = 0.0
    back_issue = False
    down_frames = up_frames = bottom_hold_frames = 0

    rt_fb_msg = None
    rt_fb_hold = 0

    print(f"[RDL] Starting pose detection loop", file=sys.stderr, flush=True)
    
    import time
    loop_start_time = time.time()
    MAX_PROCESSING_TIME = 180
    frames_processed = 0
    
    with mp_pose_mod.Pose(model_complexity=model_complexity,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed = time.time() - loop_start_time
            if elapsed > MAX_PROCESSING_TIME:
                print(f"[RDL ERROR] Processing timeout after {elapsed:.1f}s (processed {frames_processed} frames)", file=sys.stderr, flush=True)
                break
            
            frame_idx += 1
            if frame_idx % effective_frame_skip != 0:
                continue
            
            frames_processed += 1
            
            if frames_processed % 30 == 0:
                print(f"[RDL] Progress: {frames_processed} frames, {counter} reps, {elapsed:.1f}s", file=sys.stderr, flush=True)

            work = cv2.resize(frame, (0, 0), fx=effective_scale, fy=effective_scale) if effective_scale != 1.0 else frame
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
                max_knee_angle = knee_angle
                back_issue = False
                down_frames = up_frames = bottom_hold_frames = 0
                prev_progress = progress

            if in_rep:
                max_torso_angle = max(max_torso_angle, torso_angle)
                min_knee_angle = min(min_knee_angle, knee_angle)
                max_knee_angle = max(max_knee_angle, knee_angle)
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

                if bottom_reached and torso_angle <= STAND_ANGLE and (frame_idx - last_rep_frame) >= MIN_FRAMES_BETWEEN_REPS:
                    last_rep_frame = frame_idx
                    counter += 1

                    down_s = down_frames * dt if down_frames else 0.0
                    bottom_s = bottom_hold_frames * dt if bottom_hold_frames else 0.0

                    feedback = []
                    score = MAX_SCORE

                    # ✅ בדיקת ברכיים
                    if max_knee_angle > KNEE_MIN_ANGLE:
                        feedback.append("Bend your knees a bit more")
                        score -= 1.5
                    elif min_knee_angle < KNEE_MAX_ANGLE:
                        feedback.append("Too much knee bend")
                        score -= 2.0

                    # ✅ בדיקת גב - רק במקרים קיצוניים
                    if back_issue and back_angle > BACK_MAX_ANGLE:
                        feedback.append("Try to keep your back neutral")
                        score -= 1.0

                    # ✅✅✅ בדיקת טמפו - תנאי עוד יותר רגוע (רק נגד נפילה חופשית קיצונית)
                    # דורש לפחות 3 פריימים של ירידה, או סף זמן מינימלי מאוד
                    min_ecc_s = max(MIN_ECC_S, 3 * dt)
                    if down_s < min_ecc_s:
                        feedback.append("Control the lowering")
                        score -= 0.05  # ✅ עונש מינימלי מאוד

                    if bottom_s < MIN_BOTTOM_S:
                        feedback.append("Pause at the bottom")
                        score -= 0.15  # ✅ עונש מינימלי

                    score = float(max(MIN_SCORE, min(MAX_SCORE, score)))
                    all_scores.append(score)

                    if score >= 9.0:
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
                            "max_knee_angle": round(max_knee_angle, 2),
                            "back_angle": round(back_angle, 2),
                            "eccentric_s": round(down_s, 2),
                            "bottom_hold_s": round(bottom_s, 2)
                        }
                    })
                    
                    print(f"[RDL] Rep {counter}: min_knee={min_knee_angle:.1f}°, max_knee={max_knee_angle:.1f}°, torso={max_torso_angle:.1f}°, down_time={down_s:.2f}s", 
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
    
    print(f"[RDL] Video processing complete. Reps={counter}", file=sys.stderr, flush=True)

    avg = np.mean(all_scores) if all_scores else 0.0
    if np.isnan(avg) or np.isinf(avg):
        avg = 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    if session_feedbacks and len(session_feedbacks) > 0:
        technique_score = min(technique_score, 9.5)

    feedback_list = dedupe_feedback(session_feedbacks) if session_feedbacks else ["Perfect form! 🔥"]

    session_tip = None
    if session_feedback_by_cat:
        if "knees" in session_feedback_by_cat:
            session_tip = "Romanian deadlifts need a soft knee bend throughout the movement"
        elif "back" in session_feedback_by_cat:
            session_tip = "Keep your core tight and chest up"
        elif "tempo" in session_feedback_by_cat:
            session_tip = "Lower the bar slowly to maximize hamstring activation"
    else:
        session_tip = "Great technique! Keep building that posterior chain 💪"

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
    if create_video and output_path:
        print(f"[RDL] Starting FFmpeg encoding", file=sys.stderr, flush=True)
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
            print(f"[RDL] FFmpeg encoding complete", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Warning: FFmpeg error: {e}")
            final_video_path = output_path if os.path.exists(output_path) else ""
    else:
        print(f"[RDL] Skipping video encoding (create_video={create_video})", file=sys.stderr, flush=True)

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

    print(f"[RDL] Returning result (video_path length={len(final_video_path)})", file=sys.stderr, flush=True)

    return result

def run_analysis(*args, **kwargs):
    return run_romanian_deadlift_analysis(*args, **kwargs)
