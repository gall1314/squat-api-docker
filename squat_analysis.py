# -*- coding: utf-8 -*-
# squat_analysis.py — חסימת ספירה בתנועה (מוקלת), חלון סבלנות בסיום, בלי לשנות את לוגיקת הספירה
import os
import cv2
import math
import numpy as np
import subprocess
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / FONTS =====================
BAR_BG_ALPHA         = 0.55
TOP_PAD              = 8
LEFT_PAD             = 8

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
    h, w, _ = frame.shape

    # Reps box (שמאל-עליון)
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    text_w = draw.textlength(reps_text, font=REPS_FONT)
    text_h = REPS_FONT.size
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

    # Donut (ימין-עליון)
    ref_h = max(int(h * 0.06), int(REPS_FONT.size * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)
    frame = draw_depth_donut(frame, (cx, cy), radius, thick, float(np.clip(depth_pct,0,1)))

    pil  = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"; pct_txt = f"{int(float(np.clip(depth_pct,0,1))*100)}%"
    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt,   font=DEPTH_PCT_FONT)
    gap     = max(2, int(radius * 0.10))
    base_y  = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size) // 2
    draw.text((cx - int(label_w // 2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pct_w // 2), base_y + DEPTH_LABEL_FONT.size + gap),
              pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Bottom feedback (שורה אחת, אופציונלי)
    if feedback:
        line_h = FEEDBACK_FONT.size + 6
        bottom_h = max(int(h * 0.08), line_h + 14)
        over = frame.copy()
        cv2.rectangle(over, (0, h - bottom_h), (w, h), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
        pil2 = Image.fromarray(frame)
        draw2 = ImageDraw.Draw(pil2)
        tw = draw2.textlength(feedback, font=FEEDBACK_FONT)
        tx = max(10, (w - int(tw)) // 2)
        ty = h - bottom_h + (bottom_h - line_h) // 2
        draw2.text((tx, ty), feedback, font=FEEDBACK_FONT, fill=(255,255,255))
        frame = np.array(pil2)

    return frame

# ===================== BODY-ONLY SKELETON =====================
_FACE_LMS = {
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
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

# ===================== SQUAT CORE PARAMS =====================
PERFECT_MIN_KNEE_SQ = 60.0
STAND_KNEE_ANGLE    = 160.0
MIN_FRAMES_BETWEEN_REPS_SQ = 10
DEPTH_ALPHA_SQ      = 0.35

# --------- תנועה גלובלית: ספים מקלים + חלון סיום סובלני ---------
HIP_VEL_THRESH_PCT    = 0.015  # ↑ מקל
ANKLE_VEL_THRESH_PCT  = 0.018  # ↑ מקל
EMA_ALPHA             = 0.30   # ↓ פחות רגיש לפיקים
END_STABILITY_WINDOW  = 5      # כמה פריימים אחורה בודקים בסיום
END_ALLOWED_MOVING_FRAMES = 1  # מתיר עד פריים תנועה אחד בחלון

def _euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ===================== OPTIONAL FORM TIP (לוגיקה נשמרת, לא משפיע על ציון) =====================
FORM_TIPS = [
    "Pause for 1–2s at the bottom to boost hypertrophy",
    "Control the eccentric; go down a bit slower",
    "Brace your core before the descent",
    "Keep your stance slightly wider for better stability",
]
def choose_session_tip(cands):
    if not cands: return FORM_TIPS[0]
    from collections import Counter
    return Counter(cands).most_common(1)[0][0]

# ===================== MAIN =====================
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
            "feedback": ["Could not open video"], "tips": [], "reps": [], "video_path": "", "feedback_path": feedback_path
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

    # גלובל-מושן
    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    move_hist = deque(maxlen=END_STABILITY_WINDOW)

    # עומק/ירידה
    start_knee_angle = None
    rep_min_knee_angle = 180.0
    rep_max_knee_angle = -999.0
    rep_min_torso_angle = 999.0
    rep_start_frame = None
    rep_down_start_idx = None

    # דונאט חלק
    depth_smooth = 0.0
    peak_hold = 0
    def update_depth(dt, target):
        nonlocal depth_smooth, peak_hold
        depth_smooth = DEPTH_ALPHA_SQ * target + (1 - DEPTH_ALPHA_SQ) * depth_smooth
        if peak_hold > 0: peak_hold -= 1
        else: depth_smooth *= 0.985
        depth_smooth = float(np.clip(depth_smooth, 0.0, 1.0))
        return depth_smooth

    tip_candidates_session = []

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
                l_ankle  = np.array([lm[R.LEFT_ANKLE.value].x,     lm[R.LEFT_ANKLE.value].y])

                # מהירויות גלובליות
                hip_px = (hip[0]*w, hip[1]*h)
                la_px  = (l_ankle[0]*w, l_ankle[1]*h)
                ra_px  = (ankle[0]*w,  ankle[1]*h)
                if prev_hip is None:
                    prev_hip, prev_la, prev_ra = hip_px, la_px, ra_px
                hip_vel = math.hypot(hip_px[0]-prev_hip[0], hip_px[1]-prev_hip[1]) / max(w, h)
                an_vel  = max(math.hypot(la_px[0]-prev_la[0], la_px[1]-prev_la[1]),
                              math.hypot(ra_px[0]-prev_ra[0], ra_px[1]-prev_ra[1])) / max(w, h)
                hip_vel_ema   = EMA_ALPHA*hip_vel + (1-EMA_ALPHA)*hip_vel_ema
                ankle_vel_ema = EMA_ALPHA*an_vel  + (1-EMA_ALPHA)*ankle_vel_ema
                prev_hip, prev_la, prev_ra = hip_px, la_px, ra_px

                movement_block = (hip_vel_ema > HIP_VEL_THRESH_PCT) or (ankle_vel_ema > ANKLE_VEL_THRESH_PCT)
                move_hist.append(1 if movement_block else 0)
                stable_enough_for_end = (sum(move_hist) <= END_ALLOWED_MOVING_FRAMES)

                # לוגיקה קיימת — עם תנאים רכים לתנועה
                knee_angle   = calculate_angle(hip, knee, ankle)
                torso_angle  = calculate_angle(shoulder, hip, knee)

                # התחלת ירידה — רק שלא זזים כרגע (בלי רצף המתנה)
                if (knee_angle < 100) and (stage != "down") and (not movement_block):
                    start_knee_angle = float(knee_angle)
                    rep_min_knee_angle = 180.0
                    rep_max_knee_angle = -999.0
                    rep_min_torso_angle = 999.0
                    rep_start_frame = frame_idx
                    rep_down_start_idx = frame_idx
                    stage = "down"

                # תוך כדי ירידה (לא חוסמים בגלל תנועה חולפת)
                if stage == "down":
                    rep_min_knee_angle  = min(rep_min_knee_angle, knee_angle)
                    rep_max_knee_angle  = max(rep_max_knee_angle, knee_angle)
                    rep_min_torso_angle = min(rep_min_torso_angle, torso_angle)
                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE_SQ))
                        depth_target = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))
                        update_depth(1.0/max(1.0, (cap.get(cv2.CAP_PROP_FPS) or 25)/max(1,frame_skip)), depth_target)

                # סיום חזרה — צריך יציבות "מספיק" בחלון האחרון (לאו דווקא עכשיו)
                if (knee_angle > STAND_KNEE_ANGLE) and (stage == "down") and stable_enough_for_end:
                    feedbacks = []
                    penalty = 0.0

                    hip_to_heel_dist = abs(hip[1] - heel_y)
                    if   hip_to_heel_dist > 0.48: feedbacks.append("Try to squat deeper");            penalty += 3
                    elif hip_to_heel_dist > 0.45: feedbacks.append("Almost there — go a bit lower");  penalty += 2
                    elif hip_to_heel_dist > 0.43: feedbacks.append("Looking good — just a bit more depth"); penalty += 1

                    if rep_min_torso_angle < 140:
                        feedbacks.append("Try to keep your back a bit straighter"); penalty += 1.0

                    score = 10.0 if not feedbacks else round(max(4, 10 - min(penalty,6)) * 2) / 2

                    depth_pct = 0.0
                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE_SQ))
                        depth_pct = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))

                    # טיפ פר-רפ (לא משפיע על ציון)
                    down_frames = (frame_idx - (rep_down_start_idx or frame_idx))
                    fast_eccentric = down_frames < max(10, int(0.25 / (cap.get(cv2.CAP_PROP_FPS) or 25)))
                    per_rep_tip = None
                    if fast_eccentric:
                        per_rep_tip = "Control the eccentric; go down a bit slower"
                    elif rep_min_torso_angle < 145:
                        per_rep_tip = "Brace your core before the descent"
                    if per_rep_tip:
                        tip_candidates_session.append(per_rep_tip)

                    rep_reports.append({
                        "rep_index": counter + 1,
                        "score": round(float(score), 1),
                        "feedback": ([pick_strongest_feedback(feedbacks)] if feedbacks else []),
                        "tip": per_rep_tip,
                        "start_frame": rep_start_frame or 0,
                        "end_frame": frame_idx,
                        "start_knee_angle": round(float(start_knee_angle or knee_angle), 2),
                        "min_knee_angle": round(float(rep_min_knee_angle), 2),
                        "max_knee_angle": round(float(rep_max_knee_angle), 2),
                        "torso_min_angle": round(float(rep_min_torso_angle), 2),
                        "depth_pct": depth_pct
                    })

                    session_best_feedback = merge_feedback(session_best_feedback, [pick_strongest_feedback(feedbacks)] if feedbacks else [])

                    start_knee_angle = None
                    rep_down_start_idx = None
                    stage = "up"

                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS_SQ:
                        counter += 1
                        last_rep_frame = frame_idx
                        if score >= 9.5: good_reps += 1
                        else: bad_reps += 1
                        all_scores.append(score)

                # שלד + אוברליי
                frame = draw_body_only(frame, lm)
                rt_feedback = None
                if stage == "down" and rep_min_torso_angle < 140:
                    rt_feedback = "Try to keep your back a bit straighter"
                frame = draw_overlay(frame, reps=counter, feedback=rt_feedback, depth_pct=depth_smooth)
                out.write(frame)

            except Exception:
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=0.0)
                if out is not None: out.write(frame)
                continue

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    feedback_list = [session_best_feedback] if session_best_feedback else ["Great form! Keep it up 💪"]

    session_tip = choose_session_tip(tip_candidates_session)
    tips = [session_tip] if session_tip else []

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list: f.write(f"- {fb}\n")
            if tips:
                f.write("Form Tip:\n")
                for t in tips: f.write(f"- {t}\n")
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
        "feedback": feedback_list,
        "tips": tips,
        "reps": rep_reports,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }

# Alias
def run_analysis(*args, **kwargs):
    return run_squat_analysis(*args, **kwargs)


