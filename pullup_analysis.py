# -*- coding: utf-8 -*-
# pullup_analysis.py — Ascents-only counting (peaks). Fast like Bulgarian (no random seek).
# Overlay תואם סקוואט: Reps בפינה, דונאט HEIGHT ימין-עליון (לייב בעלייה/ירידה), פידבק תחתון עם hold 0.8s.
# בחירת צד דינמית לפי visibility. Gate תנועה גלובלית כדי לא לספור בזמן הליכה למתח/ממנו.
import os, math, subprocess, collections
import cv2, numpy as np
from PIL import ImageFont, ImageDraw, Image

# ===================== STYLE / FONTS (כמו בסקוואט) =====================
BAR_BG_ALPHA         = 0.55
REPS_FONT_SIZE       = 28
FEEDBACK_FONT_SIZE   = 22
DEPTH_LABEL_FONT_SIZE= 14
DEPTH_PCT_FONT_SIZE  = 18
FONT_PATH            = "Roboto-VariableFont_wdth,wght.ttf"

DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
ASCENT_COLOR         = (40, 200, 80)  # BGR
DONUT_RING_BG        = (70, 70, 70)

RT_FB_HOLD_SEC       = 0.8

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

REPS_FONT        = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

# ===================== MEDIAPIPE =====================
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    MP_OK = True
except Exception:
    MP_OK = False

# ===================== ציונים להצגה =====================
def score_label(s):
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

# ===================== BODY-ONLY (ללא פנים) =====================
_FACE_LMS = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS = tuple()
if MP_OK:
    _FACE_LMS = {
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
        mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
    }
    _BODY_CONNECTIONS = tuple(
        (a,b) for (a,b) in mp_pose.POSE_CONNECTIONS
        if a not in _FACE_LMS and b not in _FACE_LMS
    )
    _BODY_POINTS = tuple(sorted({i for c in _BODY_CONNECTIONS for i in c}))

def draw_body_only(frame, landmarks, color=(255,255,255)):
    h, w = frame.shape[:2]
    for a, b in _BODY_CONNECTIONS:
        pa, pb = landmarks[a], landmarks[b]
        ax, ay = int(pa.x*w), int(pa.y*h); bx, by = int(pb.x*w), int(pb.y*h)
        cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = landmarks[i]; x, y = int(p.x*w), int(p.y*h)
        cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
    return frame

# ===================== Overlay (Reps, HEIGHT donut, Feedback) =====================
def draw_height_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius = int(radius); thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DONUT_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90; end_ang = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang, ASCENT_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def _wrap_two_lines(draw, text, font, max_width):
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

def draw_overlay(frame, reps=0, feedback=None, height_pct=0.0):
    h, w, _ = frame.shape
    # Reps box (0,0)
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    pad_x, pad_y = 10, 6
    tw = draw.textlength(reps_text, font=REPS_FONT); th = REPS_FONT.size
    x0, y0 = 0, 0; x1, y1 = int(tw + 2*pad_x), int(th + 2*pad_y)
    top = frame.copy(); cv2.rectangle(top, (x0,y0), (x1,y1), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame); ImageDraw.Draw(pil).text((x0+pad_x, y0+pad_y-1), reps_text, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Donut HEIGHT ימין-עליון
    ref_h = max(int(h*0.06), int(REPS_FONT_SIZE*1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12; cx = w - margin - radius; cy = max(ref_h + radius//8, radius + thick//2 + 2)
    frame = draw_height_donut(frame, (cx, cy), radius, thick, float(np.clip(height_pct,0,1)))

    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    label_txt = "HEIGHT"; pct_txt = f"{int(float(np.clip(height_pct,0,1))*100)}%"
    gap = max(2, int(radius * 0.10))
    base_y = cy - (DEPTH_LABEL_FONT_SIZE + gap + DEPTH_PCT_FONT_SIZE) // 2
    lw = draw.textlength(label_txt, font=DEPTH_LABEL_FONT); pw = draw.textlength(pct_txt, font=DEPTH_PCT_FONT)
    draw.text((cx - int(lw//2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pw//2), base_y + DEPTH_LABEL_FONT_SIZE + gap), pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Feedback תחתון (wrap עד 2 שורות)
    if feedback:
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        safe = max(6, int(h*0.02)); padx, pady, gap = 12, 8, 4
        max_w = int(w - 2*padx - 20)
        lines = _wrap_two_lines(draw_fb, feedback, FEEDBACK_FONT, max_w)
        line_h = FEEDBACK_FONT.size + 6
        block_h = 2*pady + len(lines)*line_h + (len(lines)-1)*gap
        y0 = max(0, h - safe - block_h); y1 = h - safe
        over = frame.copy(); cv2.rectangle(over, (0,y0), (w,y1), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1 - BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + pady
        for ln in lines:
            tw = draw_fb.textlength(ln, font=FEEDBACK_FONT); tx = max(padx, (w-int(tw))//2)
            draw_fb.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255,255,255))
            ty += line_h + gap
        frame = np.array(pil_fb)
    return frame

# ===================== עזר/זווית =====================
def _ang(a,b,c):
    ba = np.array([a[0]-b[0], a[1]-b[1]]); bc = np.array([c[0]-b[0], c[1]-b[1]])
    den = (np.linalg.norm(ba)*np.linalg.norm(bc))+1e-9
    cosang = float(np.clip(np.dot(ba, bc)/den, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

# ===================== פרמטרי ספירת "עליות בלבד" =====================
ELBOW_TOP_THRESHOLD   = 75.0    # טופ כשהמרפק סגור יחסית
HEAD_MIN_ASCENT       = 0.03    # עלייה מינימלית מה-baseline של העלייה
RESET_DESCENT         = 0.015   # כמה לרדת (y גדל) כדי לאפשר פסגה נוספת
RESET_ELBOW           = 150.0   # או לפתוח מרפק מעל ערך זה לריסט
TOP_HOLD_FRAMES       = 1       # כמה פריימים על הטופ כדי לאשר פסגה
REFRACTORY_FRAMES     = 5       # דיבאונס בין פסגות

# — אופציונלי: טריגר "מתחיל לעלות" (לעדכן baseline) —
HEAD_VEL_UP_TINY      = 0.0005

# Gate תנועה גלובלית (כמו בסקוואט/בולגרי): אל תספר כשזזים למתח/ממנו
HIP_VEL_THRESH_PCT    = 0.014
ANKLE_VEL_THRESH_PCT  = 0.017
MOTION_EMA_ALPHA      = 0.65
MOVEMENT_CLEAR_FRAMES = 2

# ===================== MAIN =====================
def run_pullup_analysis(video_path,
                        frame_skip=1,
                        scale=1.0,
                        output_path="pullup_analyzed.mp4",
                        feedback_path="pullup_feedback.txt"):
    if not MP_OK:
        return {
            "squat_count": 0, "technique_score": 0.0, "good_reps": 0, "bad_reps": 0,
            "feedback": ["Mediapipe not available"], "tips": [], "reps": [], "video_path": "", "feedback_path": feedback_path,
            "technique_score_display": display_half_str(0.0), "technique_label": score_label(0.0)
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0, "good_reps": 0, "bad_reps": 0,
            "feedback": ["Could not open video"], "tips": [], "reps": [], "video_path": "", "feedback_path": feedback_path,
            "technique_score_display": display_half_str(0.0), "technique_label": score_label(0.0)
        }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_no = 0

    # מונים/דוחות
    rep_count = 0
    rep_reports = []
    good_reps = 0
    bad_reps  = 0
    all_scores = []

    # בחירת צד דינמית
    def _pick_side_dyn(lms):
        def vis(i):
            try: return float(lms[i].visibility or 0.0)
            except Exception: return 0.0
        vL = vis(mp_pose.PoseLandmark.LEFT_SHOULDER.value) + vis(mp_pose.PoseLandmark.LEFT_ELBOW.value) + vis(mp_pose.PoseLandmark.LEFT_WRIST.value)
        vR = vis(mp_pose.PoseLandmark.RIGHT_SHOULDER.value) + vis(mp_pose.PoseLandmark.RIGHT_ELBOW.value) + vis(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        return "LEFT" if vL >= vR else "RIGHT"

    # סטייט לפסגות
    allow_new_peak = True
    peak_hold = 0
    last_peak_frame = -999999
    asc_base_head = None   # baseline לכל עלייה
    baseline_head_y_global = None
    ascent_live = 0.0

    # פידבק בזמן אמת
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)
    RT_FB_HOLD_FRAMES = max(2, int(RT_FB_HOLD_SEC / dt))
    rt_fb_msg = None
    rt_fb_hold = 0

    # תנועה גלובלית
    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    movement_free_streak = 0

    last_head_y = None
    last_elbow_angle = None

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            frame_no += 1
            if frame_skip > 1 and (frame_no % frame_skip) != 0: continue
            if scale != 1.0: frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(image_rgb)

            elbow_angle = None
            head_y = None
            lms = None

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                # צד דינמי
                side = _pick_side_dyn(lms)
                S = getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value
                E = getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value
                W = getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value
                NOSE = mp_pose.PoseLandmark.NOSE.value

                vis_ok = min(lms[NOSE].visibility, lms[S].visibility, lms[E].visibility, lms[W].visibility) >= 0.35
                if vis_ok:
                    head_y = float(lms[NOSE].y)  # normalized (0=top)
                    elbow_angle = _ang((lms[S].x, lms[S].y), (lms[E].x, lms[E].y), (lms[W].x, lms[W].y))
                    if baseline_head_y_global is None:
                        baseline_head_y_global = head_y

                # Gate תנועה גלובלית (ירך/קרסוליים)
                hip_px    = (lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x * w,
                             lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y * h)
                l_ankle_px= (lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                             lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h)
                r_ankle_px= (lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                             lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
                if prev_hip is None:
                    prev_hip, prev_la, prev_ra = hip_px, l_ankle_px, r_ankle_px
                def _d(a,b): return math.hypot(a[0]-b[0], a[1]-b[1]) / max(w, h)
                hip_vel = _d(hip_px, prev_hip); an_vel = max(_d(l_ankle_px, prev_la), _d(r_ankle_px, prev_ra))
                hip_vel_ema   = MOTION_EMA_ALPHA*hip_vel + (1-MOTION_EMA_ALPHA)*hip_vel_ema
                ankle_vel_ema = MOTION_EMA_ALPHA*an_vel  + (1-MOTION_EMA_ALPHA)*ankle_vel_ema
                prev_hip, prev_la, prev_ra = hip_px, l_ankle_px, r_ankle_px
                movement_block = (hip_vel_ema > HIP_VEL_THRESH_PCT) or (ankle_vel_ema > ANKLE_VEL_THRESH_PCT)
                if movement_block: movement_free_streak = 0
                else:              movement_free_streak = min(MOVEMENT_CLEAR_FRAMES, movement_free_streak + 1)
            else:
                movement_block = False
                movement_free_streak = 0

            # מהירות ראש (שלילי=עולה)
            head_vel = 0.0
            if last_head_y is not None and head_y is not None:
                head_vel = head_y - last_head_y

            # ---- ספירה לפי עליות בלבד (פסגות) ----
            if (not movement_block) and (head_y is not None) and (elbow_angle is not None):
                # קביעת baseline לעלייה הנוכחית: כשהראש מתחיל לעלות בעדינות
                if asc_base_head is None:
                    if head_vel < -HEAD_VEL_UP_TINY:
                        asc_base_head = head_y
                else:
                    # אם ירדנו הרבה — נעדכן כדי לא לפספס עלייה חדשה
                    if (head_y - asc_base_head) > RESET_DESCENT * 2:
                        asc_base_head = head_y

                # תנאי טופ (פסגה): מרפק סגור + עלייה יחסית מספקת
                top_ok = False
                if asc_base_head is not None:
                    asc_amount = (asc_base_head - head_y)  # חיובי כשעולים
                    if (elbow_angle <= ELBOW_TOP_THRESHOLD) and (asc_amount >= HEAD_MIN_ASCENT):
                        top_ok = True

                can_count_again = (frame_no - last_peak_frame) >= REFRACTORY_FRAMES

                if top_ok and allow_new_peak and can_count_again:
                    peak_hold += 1
                    if peak_hold >= TOP_HOLD_FRAMES:
                        # --- ספר פסגה ---
                        rep_count += 1
                        rep_reports.append({
                            "rep_index": rep_count,
                            "peak_head_y": float(head_y),
                            "asc_from": float(asc_base_head),
                            "top_elbow": float(elbow_angle)
                        })
                        good_reps += 1
                        all_scores.append(10.0)  # ספירה בלבד — אין ענישת איכות
                        last_peak_frame = frame_no
                        allow_new_peak = False
                        peak_hold = 0
                else:
                    peak_hold = 0

                # תנאי "ריסט" לשחרור פסגה הבאה (בלי לדרוש ירידה מלאה)
                reset_by_descent = (asc_base_head is not None) and ((head_y - asc_base_head) >= RESET_DESCENT)
                reset_by_elbow   = (elbow_angle >= RESET_ELBOW)
                if reset_by_descent or reset_by_elbow:
                    allow_new_peak = True
                    asc_base_head = head_y  # עדכן בסיס לעלייה הבאה

                # RT feedback עדין (לא משפיע על ספירה)
                cur_rt = None
                if asc_base_head is not None:
                    asc_amount_live = asc_base_head - head_y
                    if asc_amount_live < HEAD_MIN_ASCENT * 0.7 and head_vel < -HEAD_VEL_UP_TINY:
                        cur_rt = "Go a bit higher (chin over bar)"
                if cur_rt:
                    if cur_rt != rt_fb_msg:
                        rt_fb_msg = cur_rt; rt_fb_hold = RT_FB_HOLD_FRAMES
                    else:
                        rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
                else:
                    if rt_fb_hold > 0: rt_fb_hold -= 1
            else:
                if rt_fb_hold > 0: rt_fb_hold -= 1

            # --- דונאט HEIGHT לייב (גם בעלייה וגם בירידה) ---
            if baseline_head_y_global is not None and head_y is not None:
                raw = baseline_head_y_global - head_y
                ascent_live = float(np.clip(raw / max(0.12, HEAD_MIN_ASCENT*1.2), 0.0, 1.0))
            else:
                ascent_live = 0.0

            # ציור שלד + אוברליי
            if res.pose_landmarks:
                frame = draw_body_only(frame, res.pose_landmarks.landmark)
            frame = draw_overlay(frame, reps=rep_count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=ascent_live)
            out.write(frame)

            if head_y is not None: last_head_y = head_y
            if elbow_angle is not None: last_elbow_angle = elbow_angle

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    # ציון כולל להצגה (ספירה בלבד → 10)
    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)

    # טיפ אחד (לא משפיע על הציון)
    session_tip = "Slow down the lowering phase to maximize hypertrophy"

    # קובץ תקציר
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {rep_count}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            f.write(f"Tip: {session_tip}\n")
    except Exception:
        pass

    # faststart encode (ffmpeg)
    encoded_path = output_path.replace('.mp4', '_encoded.mp4')
    try:
        subprocess.run([
            'ffmpeg','-y','-i', output_path,
            '-c:v','libx264','-preset','fast','-movflags','+faststart','-pix_fmt','yuv420p',
            encoded_path
        ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final_path = encoded_path if os.path.isfile(encoded_path) else output_path
    except Exception:
        final_path = output_path

    if not os.path.isfile(final_path) and os.path.isfile(output_path):
        final_path = output_path

    return {
        "squat_count": int(rep_count),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": ["Great form! Keep it up 💪"],
        "tips": [session_tip],
        "reps": rep_reports,
        "video_path": final_path,
        "feedback_path": feedback_path
    }

# תאימות
def run_analysis(*args, **kwargs):
    return run_pullup_analysis(*args, **kwargs)

