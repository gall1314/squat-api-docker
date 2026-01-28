# -*- coding: utf-8 -*-
# squat_analysis.py — בסיס שסופר טוב + Overlay צמוד לפינה
# פאי-גרף "לייב" ללא דיליי (עולה/יורד בזמן אמת), פידבק גב מדויק, ו-RT feedback עם Hold.
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
TOP_PAD              = 0     # צמוד לקצה
LEFT_PAD             = 0     # צמוד לקצה

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
FEEDBACK_CATEGORY = {
    "Try to squat deeper": "depth",
    "Almost there — go a bit lower": "depth",
    "Looking good — just a bit more depth": "depth",
    "Try to keep your back a bit straighter": "back",
    "Avoid knee collapse": "knees",
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
    return list(best_by_cat.values())

def merge_feedback(global_best, new_list):
    cand = pick_strongest_feedback(new_list)
    if not cand: return global_best
    if not global_best: return cand
    return cand if FB_SEVERITY.get(cand,1) >= FB_SEVERITY.get(global_best,1) else global_best

# ===================== תצוגה מילולית/מספרית לציון =====================
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
        return str(int(round(q)))  # "10"
    return f"{q:.1f}"            # "9.5"

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
    """Reps בפינת שמאל-עליון (0,0) בלי פאדינג חיצוני; דונאט ימני-עליון; פידבק תחתון שלא נחתך (עד 2 שורות)."""
    h, w, _ = frame.shape

    # --- Reps box: צמוד לפינה ---
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    inner_pad_x, inner_pad_y = 10, 6
    text_w = draw.textlength(reps_text, font=REPS_FONT)
    text_h = REPS_FONT.size
    x0, y0 = 0, 0
    x1 = int(text_w + 2*inner_pad_x)
    y1 = int(text_h + 2*inner_pad_y)
    top = frame.copy()
    cv2.rectangle(top, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((x0 + inner_pad_x, y0 + inner_pad_y - 1),
                             reps_text, font=REPS_FONT, fill=(255, 255, 255))
    frame = np.array(pil)

    # --- Donut (ימין-עליון) ---
    ref_h = max(int(h * 0.06), int(REPS_FONT_SIZE * 1.6))
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

    # --- Bottom feedback (לא נחתך; עד 2 שורות; עם safe area) ---
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

        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        safe_margin = max(6, int(h * 0.02))
        pad_x, pad_y, line_gap = 12, 8, 4
        max_text_w = int(w - 2*pad_x - 20)
        lines = wrap_to_two_lines(draw_fb, feedback, FEEDBACK_FONT, max_text_w)
        line_h = FEEDBACK_FONT.size + 6
        block_h = (2*pad_y) + len(lines)*line_h + (len(lines)-1)*line_gap
        y0 = max(0, h - safe_margin - block_h); y1 = h - safe_margin
        over = frame.copy()
        cv2.rectangle(over, (0, y0), (w, y1), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + pad_y
        for ln in lines:
            tw = draw_fb.textlength(ln, font=FEEDBACK_FONT)
            tx = max(pad_x, (w - int(tw)) // 2)
            draw_fb.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255,255,255))
            ty += line_h + line_gap
        frame = np.array(pil_fb)

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

def angle_between_vectors(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

# ===================== SQUAT CORE PARAMS =====================
STAND_KNEE_ANGLE    = 160.0
MIN_FRAMES_BETWEEN_REPS_SQ = 10

# --------- תנועה גלובלית: חסימת ספירה בזמן הליכה + soft-start לחזרה הראשונה ---------
HIP_VEL_THRESH_PCT    = 0.014
ANKLE_VEL_THRESH_PCT  = 0.017
EMA_ALPHA             = 0.65
MOVEMENT_CLEAR_FRAMES = 2   # רצף קצר של שקט כדי לסיים חזרה

def _euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

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
            "feedback": ["Could not open video"], "tips": [], "reps": [], "video_path": "", "feedback_path": feedback_path,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0)
        }

    counter = 0
    good_reps = 0
    bad_reps = 0
    rep_reports = []
    all_scores = []

    stage = None
    frame_idx = 0
    last_rep_frame = -999
    session_feedbacks = []

    # גלובל-מושן
    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    movement_free_streak = 0

    # משתני ירידה/עומק לחישובי ציון
    start_knee_angle = None
    rep_min_knee_angle = 180.0
    rep_max_knee_angle = -999.0
    rep_max_back_angle = -999.0
    rep_start_frame = None
    rep_down_start_idx = None
    rep_max_hip_knee_delta = -999.0

    # עומק "לייב" דו-כיווני (גם בירידה וגם בעלייה)
    depth_live = 0.0

    # פידבק גב — סינון לפי משך ואזור
    TOP_BACK_MAX_DEG    = 40.0   # Top: זווית נטייה מקסימלית מול אנכי
    BOTTOM_BACK_MAX_DEG = 65.0   # Bottom: מתריע רק אם ממש קיצוני
    TOP_BAD_MIN_SEC   = 0.25    # צריך לפחות משך זה ב-Top כדי להתריע
    BOTTOM_BAD_MIN_SEC= 0.35    # ובתחתית אפילו יותר
    rep_top_bad_frames = 0
    rep_bottom_bad_frames = 0

    # פידבק בזמן אמת עם hold
    RT_FB_HOLD_SEC = 0.8
    rt_fb_msg = None
    rt_fb_hold = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)
    TOP_BAD_MIN_FRAMES    = max(2, int(TOP_BAD_MIN_SEC / dt))
    BOTTOM_BAD_MIN_FRAMES = max(2, int(BOTTOM_BAD_MIN_SEC / dt))
    RT_FB_HOLD_FRAMES     = max(2, int(RT_FB_HOLD_SEC / dt))

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
                depth_live = 0.0
                if rt_fb_hold > 0: rt_fb_hold -= 1
                frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
                out.write(frame); continue

            try:
                lm = results.pose_landmarks.landmark
                R = mp_pose_mod.PoseLandmark
                hip      = np.array([lm[R.RIGHT_HIP.value].x,      lm[R.RIGHT_HIP.value].y])
                knee     = np.array([lm[R.RIGHT_KNEE.value].x,     lm[R.RIGHT_KNEE.value].y])
                ankle    = np.array([lm[R.RIGHT_ANKLE.value].x,    lm[R.RIGHT_ANKLE.value].y])
                shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])
                l_hip    = np.array([lm[R.LEFT_HIP.value].x,       lm[R.LEFT_HIP.value].y])
                l_knee   = np.array([lm[R.LEFT_KNEE.value].x,      lm[R.LEFT_KNEE.value].y])
                l_ankle  = np.array([lm[R.LEFT_ANKLE.value].x,     lm[R.LEFT_ANKLE.value].y])
                l_shldr  = np.array([lm[R.LEFT_SHOULDER.value].x,  lm[R.LEFT_SHOULDER.value].y])

                mid_hip      = (hip + l_hip) / 2.0
                mid_knee     = (knee + l_knee) / 2.0
                mid_ankle    = (ankle + l_ankle) / 2.0
                mid_shoulder = (shoulder + l_shldr) / 2.0

                # --- מהירויות גלובליות ---
                hip_px = (hip[0]*w, hip[1]*h)
                la_px  = (l_ankle[0]*w, l_ankle[1]*h)
                ra_px  = (ankle[0]*w,  ankle[1]*h)
                if prev_hip is None:
                    prev_hip, prev_la, prev_ra = hip_px, la_px, ra_px
                def _d(a,b): return math.hypot(a[0]-b[0], a[1]-b[1]) / max(w, h)
                hip_vel = _d(hip_px, prev_hip); an_vel = max(_d(la_px, prev_la), _d(ra_px, prev_ra))
                hip_vel_ema   = EMA_ALPHA*hip_vel + (1-EMA_ALPHA)*hip_vel_ema
                ankle_vel_ema = EMA_ALPHA*an_vel  + (1-EMA_ALPHA)*ankle_vel_ema
                prev_hip, prev_la, prev_ra = hip_px, la_px, ra_px

                movement_block = (hip_vel_ema > HIP_VEL_THRESH_PCT) or (ankle_vel_ema > ANKLE_VEL_THRESH_PCT)
                if movement_block: movement_free_streak = 0
                else:              movement_free_streak = min(MOVEMENT_CLEAR_FRAMES, movement_free_streak + 1)

                # --- זוויות ---
                knee_angle   = calculate_angle(hip, knee, ankle)
                back_angle   = angle_between_vectors(mid_shoulder - mid_hip, np.array([0.0, -1.0]))

                # --- התחלת ירידה (soft start) ---
                soft_start_ok = (hip_vel_ema < HIP_VEL_THRESH_PCT * 1.25) and (ankle_vel_ema < ANKLE_VEL_THRESH_PCT * 1.25)
                if (knee_angle < 100) and (stage != "down") and soft_start_ok:
                    start_knee_angle = float(knee_angle)
                    rep_min_knee_angle = 180.0
                    rep_max_knee_angle = -999.0
                    rep_max_back_angle = -999.0
                    rep_max_hip_knee_delta = -999.0
                    rep_top_bad_frames = 0
                    rep_bottom_bad_frames = 0
                    rep_start_frame = frame_idx
                    rep_down_start_idx = frame_idx
                    stage = "down"

                # --- עומק "לייב" גם בירידה וגם בעלייה ---
                knee_to_ankle = max(1e-6, abs(mid_ankle[1] - mid_knee[1]))
                hip_knee_delta = mid_hip[1] - mid_knee[1]
                depth_ratio = max(0.0, hip_knee_delta) / knee_to_ankle
                depth_live = float(np.clip(depth_ratio / 0.35, 0, 1))

                # --- תוך כדי ירידה: מדדי רפ + סיווג גב לפי עומק ---
                if stage == "down":
                    rep_min_knee_angle   = min(rep_min_knee_angle, knee_angle)
                    rep_max_knee_angle   = max(rep_max_knee_angle, knee_angle)
                    rep_max_back_angle   = max(rep_max_back_angle, back_angle)
                    rep_max_hip_knee_delta = max(rep_max_hip_knee_delta, hip_knee_delta)

                    # Top: עומק קטן → דורש זקיפות יחסית; Bottom: עומק גדול → סלחני יותר
                    if depth_live <= 0.25 and back_angle > TOP_BACK_MAX_DEG:
                        rep_top_bad_frames += 1
                        # RT feedback עם hold
                        if rt_fb_msg != "Try to keep your back a bit straighter":
                            rt_fb_msg = "Try to keep your back a bit straighter"
                            rt_fb_hold = RT_FB_HOLD_FRAMES
                        else:
                            rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
                    elif depth_live >= 0.80 and back_angle > BOTTOM_BACK_MAX_DEG:
                        rep_bottom_bad_frames += 1
                        # בזמן אמת לא נצעק בתחתית כדי לא להציק; נשמור לסוף רפ
                    else:
                        if rt_fb_hold > 0:
                            rt_fb_hold -= 1
                # --- סיום חזרה (כמו שעבד) ---
                if (knee_angle > STAND_KNEE_ANGLE) and (stage == "down") and (movement_free_streak >= MOVEMENT_CLEAR_FRAMES):
                    feedbacks = []
                    penalty = 0.0

                    # עומק
                    depth_ratio = 0.0
                    if knee_to_ankle > 1e-6:
                        depth_ratio = max(0.0, rep_max_hip_knee_delta) / knee_to_ankle
                    if   depth_ratio < 0.10: feedbacks.append("Try to squat deeper");            penalty += 3
                    elif depth_ratio < 0.16: feedbacks.append("Almost there — go a bit lower");  penalty += 2
                    elif depth_ratio < 0.22: feedbacks.append("Looking good — just a bit more depth"); penalty += 1

                    # גב — מתריעים רק אם נצברה חריגה למשך מינימום
                    back_flag = (rep_top_bad_frames >= TOP_BAD_MIN_FRAMES) or (rep_bottom_bad_frames >= BOTTOM_BAD_MIN_FRAMES)
                    if back_flag:
                        feedbacks.append("Try to keep your back a bit straighter")
                        penalty += 1.0

                    # ציון
                    score = 10.0 if not feedbacks else round(max(4, 10 - min(penalty,6)) * 2) / 2

                    # עומק סופי (שיא העומק בחזרה)
                    depth_pct = float(np.clip(depth_ratio, 0, 1))

                    # דוח רפ
                    rep_feedbacks = pick_strongest_per_category(feedbacks)
                    rep_reports.append({
                        "rep_index": counter + 1,
                        "score": round(float(score), 1),
                        "score_display": display_half_str(score),  # <-- נוסף להצגה
                        "feedback": rep_feedbacks,
                        "tip": None,
                        "start_frame": rep_start_frame or 0,
                        "end_frame": frame_idx,
                        "start_knee_angle": round(float(start_knee_angle or knee_angle), 2),
                        "min_knee_angle": round(float(rep_min_knee_angle), 2),
                        "max_knee_angle": round(float(rep_max_knee_angle), 2),
                        "torso_min_angle": round(float(rep_max_back_angle), 2),
                        "depth_pct": depth_pct,
                        "top_bad_frames": int(rep_top_bad_frames),
                        "bottom_bad_frames": int(rep_bottom_bad_frames)
                    })

                    # פידבק-סשן
                    for fb in rep_feedbacks:
                        if fb not in session_feedbacks:
                            session_feedbacks.append(fb)

                    start_knee_angle = None
                    rep_down_start_idx = None
                    stage = "up"

                    # debounce
                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS_SQ:
                        counter += 1
                        last_rep_frame = frame_idx
                        if score >= 9.5: good_reps += 1
                        else: bad_reps += 1
                        all_scores.append(score)

                # --- ציור שלד + אוברליי ---
                frame = draw_body_only(frame, lm)
                frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
                out.write(frame)

            except Exception:
                if rt_fb_hold > 0: rt_fb_hold -= 1
                depth_live = 0.0
                frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
                if out is not None: out.write(frame)
                continue

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    feedback_list = session_feedbacks if session_feedbacks else ["Great form! Keep it up 💪"]

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

    # -------- החזרה המעודכנת (לוגיקה לא שונתה) --------
    return {
        "squat_count": counter,
        "technique_score": technique_score,                             # double לחישובים/גרפים
        "technique_score_display": display_half_str(technique_score),   # מחרוזת להצגה (ללא .0 אם שלם)
        "technique_label": score_label(technique_score),                # ציון מילולי
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": feedback_list,
        "reps": rep_reports,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }

# תאימות
def run_analysis(*args, **kwargs):
    return run_squat_analysis(*args, **kwargs)
