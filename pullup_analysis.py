# -*- coding: utf-8 -*-
# pullup_analysis.py — מיושר 1:1 לטמפלייט הסקוואט (Overlay/Return/Flow)
# דונאט "ASCENT" בלייב (גם בעלייה וגם בירידה), RT feedback עם Hold 0.8s,
# חסימת ספירה בזמן הליכה/תנועה כללית, שלד גוף בלבד (ללא פנים), ו-faststart encode.

import os
import cv2
import math
import time
import json
import subprocess
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# ===================== STYLE / FONTS (כמו בסקוואט) =====================
BAR_BG_ALPHA         = 0.55
FONT_PATH            = "Roboto-VariableFont_wdth,wght.ttf"

REPS_FONT_SIZE       = 28
FEEDBACK_FONT_SIZE   = 22
DEPTH_LABEL_FONT_SIZE= 14
DEPTH_PCT_FONT_SIZE  = 18

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

REPS_FONT        = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

# דונאט
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)   # ירוק כמו בסקוואט
DEPTH_RING_BG        = (70, 70, 70)

# ===================== FEEDBACK MAPPING =====================
FB_SEVERITY = {
    "Aim for chin over the bar": 3,
    "Try to fully extend at the bottom": 2,
    "Avoid excessive swinging": 2,
    "Nice work — keep reps controlled": 1,
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

# ===================== Donut + Overlay =====================
def draw_ascent_donut(frame, center, radius, thickness, pct):
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

def draw_overlay(frame, reps=0, feedback=None, ascent_pct=0.0):
    """Reps שמאל-עליון, דונאט ימין-עליון, פידבק בתחתית — זהה לסקוואט."""
    h, w, _ = frame.shape

    # Reps box
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
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

    # Donut (ASCENT)
    ref_h = max(int(h * 0.06), int(REPS_FONT_SIZE * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)
    frame = draw_ascent_donut(frame, (cx, cy), radius, thick, float(np.clip(ascent_pct,0,1)))

    pil  = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    label_txt = "ASCENT"; pct_txt = f"{int(float(np.clip(ascent_pct,0,1))*100)}%"
    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt,   font=DEPTH_PCT_FONT)
    gap     = max(2, int(radius * 0.10))
    base_y  = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size) // 2
    draw.text((cx - int(label_w // 2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pct_w // 2), base_y + DEPTH_LABEL_FONT.size + gap),
              pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Bottom feedback — עד 2 שורות, עם אליפסות
    if feedback:
        def wrap_to_two_lines(draw, text, font, max_width):
            words = text.split()
            if not words: return [""]
            lines, cur = [], ""
            for w_ in words:
                trial = (cur + " " + w_).strip()
                if draw.textlength(trial, font=font) <= max_width:
                    cur = trial
                else:
                    if cur: lines.append(cur)
                    cur = w_
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

# ===================== MediaPipe + שלד גוף בלבד =====================
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

mp_pose = mp.solutions.pose

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

# ===================== עזר גיאומטרי =====================
def _angle(a, b, c):
    try:
        ba = (a[0]-b[0], a[1]-b[1]); bc = (c[0]-b[0], c[1]-b[1])
        den = ( (ba[0]**2 + ba[1]**2)**0.5 * (bc[0]**2 + bc[1]**2)**0.5 ) + 1e-9
        cosang = (ba[0]*bc[0] + ba[1]*bc[1]) / den
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))
    except Exception:
        return 180.0

# ===================== ספים/פרמטרים למתח =====================
ELBOW_START_THRESHOLD   = 150.0
ELBOW_TOP_THRESHOLD     = 65.0
ELBOW_BOTTOM_THRESHOLD  = 160.0
HEAD_MIN_ASCENT         = 0.06
HEAD_VEL_UP_THRESH      = 0.0025
HEAD_TOP_STICK_FRAMES   = 2
BOTTOM_STICK_FRAMES     = 2

# חסימת ספירה בזמן תנועה גלובלית (הליכה/קפיצה)
HIP_VEL_THRESH_PCT      = 0.014
ANKLE_VEL_THRESH_PCT    = 0.017
EMA_ALPHA               = 0.65
MOVEMENT_CLEAR_FRAMES   = 2

# RT feedback hold (זהה לסקוואט)
RT_FB_HOLD_SEC_DEFAULT  = 0.8

# ===================== אנלייזר =====================
class PullUpAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rep_count = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.in_rep = False
        self.seen_top_frames = 0
        self.seen_bottom_frames = 0
        self.min_head_y_in_rep = None
        self.rep_started_elbow = None
        self.last_head_y = None
        self.last_elbow = None

        self.session_best_feedback = ""
        self.reps_meta = []
        self.all_scores = []

        # גלובל מושן (חסימת ספירה בזמן הליכה)
        self.prev_hip = self.prev_la = self.prev_ra = None
        self.hip_vel_ema = 0.0
        self.ankle_vel_ema = 0.0
        self.movement_free_streak = 0

        # דונאט (עלייה בלייב)
        self.baseline_head_y_global = None
        self.ascent_live = 0.0

        # RT feedback עם hold
        self.rt_fb_msg = None
        self.rt_fb_hold = 0

        # fps/זמנים
        self.RT_FB_HOLD_FRAMES = 20  # יתעדכן לפי fps האפקטיבי

    def process(self, input_path, output_path=None, frame_skip=3, scale=0.4):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not available")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return self._empty_result("Could not open video", output_path)

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        eff_fps = max(1.0, fps_in / max(1, frame_skip))
        self.RT_FB_HOLD_FRAMES = max(2, int(RT_FB_HOLD_SEC_DEFAULT / (1.0/eff_fps)))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        frame_idx = 0

        # הכנת כתיבה
        ret, test_frame = cap.read()
        if not ret:
            cap.release()
            return self._empty_result("Empty video", output_path)
        if scale != 1.0:
            test_frame = cv2.resize(test_frame, (0,0), fx=scale, fy=scale)
        h, w = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, eff_fps, (w, h))

        with mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = cap.read()
                if not ok: break
                frame_idx += 1
                if frame_idx % frame_skip != 0: continue
                if scale != 1.0:
                    frame = cv2.resize(frame, (w, h))

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # אין לנדמרקס → רק אוברליי עם שמירת hold
                if not results.pose_landmarks:
                    self.ascent_live = 0.0
                    if self.rt_fb_hold > 0: self.rt_fb_hold -= 1
                    frame = draw_overlay(frame, reps=self.rep_count,
                                         feedback=(self.rt_fb_msg if self.rt_fb_hold>0 else None),
                                         ascent_pct=self.ascent_live)
                    if out is not None: out.write(frame)
                    continue

                lm = results.pose_landmarks.landmark
                R = mp.solutions.pose.PoseLandmark

                # נקודות רלוונטיות
                hip      = np.array([lm[R.RIGHT_HIP.value].x,      lm[R.RIGHT_HIP.value].y])
                l_ankle  = np.array([lm[R.LEFT_ANKLE.value].x,     lm[R.LEFT_ANKLE.value].y])
                r_ankle  = np.array([lm[R.RIGHT_ANKLE.value].x,    lm[R.RIGHT_ANKLE.value].y])
                shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])
                elbow    = np.array([lm[R.RIGHT_ELBOW.value].x,    lm[R.RIGHT_ELBOW.value].y])
                wrist    = np.array([lm[R.RIGHT_WRIST.value].x,    lm[R.RIGHT_WRIST.value].y])
                nose_y   = lm[R.NOSE.value].y

                # --- מהירויות גלובליות (כמו בסקוואט)---
                hip_px = (hip[0]*w, hip[1]*h)
                la_px  = (l_ankle[0]*w, l_ankle[1]*h)
                ra_px  = (r_ankle[0]*w, r_ankle[1]*h)
                if self.prev_hip is None:
                    self.prev_hip, self.prev_la, self.prev_ra = hip_px, la_px, ra_px

                def _d(a,b): return math.hypot(a[0]-b[0], a[1]-b[1]) / max(w, h)

                hip_vel = _d(hip_px, self.prev_hip)
                an_vel  = max(_d(la_px, self.prev_la), _d(ra_px, self.prev_ra))
                self.hip_vel_ema   = EMA_ALPHA*hip_vel + (1-EMA_ALPHA)*self.hip_vel_ema
                self.ankle_vel_ema = EMA_ALPHA*an_vel  + (1-EMA_ALPHA)*self.ankle_vel_ema
                self.prev_hip, self.prev_la, self.prev_ra = hip_px, la_px, ra_px

                movement_block = (self.hip_vel_ema > HIP_VEL_THRESH_PCT) or (self.ankle_vel_ema > ANKLE_VEL_THRESH_PCT)
                if movement_block: self.movement_free_streak = 0
                else:              self.movement_free_streak = min(MOVEMENT_CLEAR_FRAMES, self.movement_free_streak + 1)

                # --- חישובי מרפק/ראש ---
                elbow_angle = _angle(shoulder, elbow, wrist)  # ABC
                head_y_norm = float(nose_y)                   # 0 למעלה, 1 למטה

                # בייסליין ראש עולמי (לתקופת הסט): ה־head_y בתחילת הסט
                if self.baseline_head_y_global is None:
                    self.baseline_head_y_global = head_y_norm

                # --- עלייה "לייב" (ASCENT) — גם בעלייה וגם בירידה ---
                if self.baseline_head_y_global is not None:
                    self.ascent_live = float(np.clip(self.baseline_head_y_global - head_y_norm, 0.0, 1.0))
                else:
                    self.ascent_live = 0.0

                # --- מהירות ראש ---
                head_vel = 0.0
                if self.last_head_y is not None and head_y_norm is not None:
                    head_vel = head_y_norm - self.last_head_y  # שלילי = עולה

                # --- RT feedback בסיסי + HOLD ---
                if self.ascent_live < 0.03:
                    if self.rt_fb_msg != "Aim for chin over the bar":
                        self.rt_fb_msg = "Aim for chin over the bar"
                        self.rt_fb_hold = self.RT_FB_HOLD_FRAMES
                    else:
                        self.rt_fb_hold = max(self.rt_fb_hold, self.RT_FB_HOLD_FRAMES)
                else:
                    if self.rt_fb_hold > 0:
                        self.rt_fb_hold -= 1

                # --- לוגיקת ספירה (עם חסימת תנועה כללית) ---
                if not self.in_rep:
                    if (not movement_block) and (elbow_angle < ELBOW_START_THRESHOLD) and (head_vel < -HEAD_VEL_UP_THRESH):
                        self.in_rep = True
                        self.min_head_y_in_rep = head_y_norm
                        self.seen_top_frames = 0
                        self.rep_started_elbow = elbow_angle
                else:
                    self.min_head_y_in_rep = min(self.min_head_y_in_rep, head_y_norm)

                    top_ok = ( (self.baseline_head_y_global - head_y_norm) >= HEAD_MIN_ASCENT ) and (elbow_angle <= ELBOW_TOP_THRESHOLD)
                    if top_ok:
                        self.seen_top_frames += 1
                    else:
                        self.seen_top_frames = 0

                    if self.seen_top_frames >= HEAD_TOP_STICK_FRAMES and (not movement_block):
                        # קובעים פידבקים/ענישה לחזרה
                        feedbacks = []
                        penalty = 0.0

                        # ROM
                        ascent_peak = max(0.0, self.baseline_head_y_global - self.min_head_y_in_rep)
                        if ascent_peak < 0.08:
                            feedbacks.append("Aim for chin over the bar"); penalty += 2.5
                        elif ascent_peak < HEAD_MIN_ASCENT:
                            feedbacks.append("Aim for chin over the bar"); penalty += 2.0

                        # הארכה בתחתית (לפני התחלה)
                        if self.rep_started_elbow is not None and self.rep_started_elbow < ELBOW_BOTTOM_THRESHOLD - 5:
                            feedbacks.append("Try to fully extend at the bottom"); penalty += 1.0

                        # קיפינג/נדנוד
                        if abs(head_vel) > 0.02 and self.ascent_live > 0.02:
                            feedbacks.append("Avoid excessive swinging"); penalty += 0.5

                        # ציון
                        score = 10.0 if not feedbacks else round(max(4, 10 - min(penalty,6)) * 2) / 2

                        # טיפ אחד לסשן (לא משפיע על ציון)
                        tip = "Slow down the lowering phase to maximize hypertrophy"

                        # Report רפ
                        self.reps_meta.append({
                            "rep_index": self.rep_count + 1,
                            "score": round(float(score), 1),
                            "score_display": display_half_str(score),
                            "feedback": ([pick_strongest_feedback(feedbacks)] if feedbacks else []),
                            "tip": tip,
                            "ascent_peak": float(ascent_peak),
                        })

                        # מיזוג פידבק סשן
                        self.session_best_feedback = merge_feedback(self.session_best_feedback, [pick_strongest_feedback(feedbacks)] if feedbacks else [])

                        # עדכוני מונים/צה
                        self.rep_count += 1
                        if score >= 9.5: self.good_reps += 1
                        else: self.bad_reps += 1
                        self.all_scores.append(score)

                        # איפוס לחזרה הבאה
                        self.in_rep = False
                        self.seen_top_frames = 0
                        self.min_head_y_in_rep = None
                        self.rep_started_elbow = None
                        # הבייסליין נשאר לסט

                # --- ציור שלד + אוברליי ---
                frame = draw_body_only(frame, lm)
                frame = draw_overlay(frame,
                                     reps=self.rep_count,
                                     feedback=(self.rt_fb_msg if self.rt_fb_hold>0 else None),
                                     ascent_pct=self.ascent_live)
                if out is not None:
                    out.write(frame)

                # עדכוני “אחרון”
                self.last_elbow = elbow_angle
                self.last_head_y = head_y_norm

        cap.release()
        if out is not None:
            out.release()

        # faststart encode
        final_video_path = ""
        if output_path:
            encoded_path = output_path.replace(".mp4", "_encoded.mp4")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_path,
                    "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                    encoded_path
                ], check=False)
                if os.path.exists(output_path) and os.path.exists(encoded_path):
                    os.remove(output_path)
                final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")
            except Exception:
                final_video_path = output_path if os.path.exists(output_path) else ""

        # סיכום ציון סשן + פידבק/טיפים (אחד)
        avg = np.mean(self.all_scores) if self.all_scores else 0.0
        technique_score = round(round((avg if avg>0 else 0.0) * 2) / 2, 2)
        feedback_list = [self.session_best_feedback] if self.session_best_feedback else ["Great form! Keep it up 💪"]
        tips = ["Slow down the lowering phase to maximize hypertrophy"]

        return {
            "squat_count": int(self.rep_count),                      # ← שם גלובלי אחיד
            "technique_score": float(technique_score),
            "technique_score_display": display_half_str(technique_score),
            "technique_label": score_label(technique_score),
            "good_reps": int(self.good_reps),
            "bad_reps": int(self.bad_reps),
            "feedback": feedback_list,
            "tips": tips,
            "reps": self.reps_meta,
            "video_path": final_video_path,
            "feedback_path": "pullup_feedback.txt"
        }

    def _empty_result(self, msg, output_path):
        return {
            "squat_count": 0,
            "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0,
            "bad_reps": 0,
            "feedback": [msg],
            "tips": [],
            "reps": [],
            "video_path": output_path or "",
            "feedback_path": "pullup_feedback.txt"
        }

# ===================== API-Facing =====================
def run_pullup_analysis(input_path, frame_skip=3, scale=0.4, output_path=None):
    """
    שימוש:
      run_pullup_analysis(path, frame_skip=3, scale=0.4, output_path="out.mp4")
    """
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    t0 = time.time()
    analyzer = PullUpAnalyzer()
    res = analyzer.process(
        input_path=input_path,
        output_path=output_path,
        frame_skip=frame_skip,
        scale=scale,
    )
    res["elapsed_sec"] = round(time.time() - t0, 3)
    if output_path and "video_path" not in res:
        res["video_path"] = output_path
    return res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", default="")
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--skip", type=int, default=3)
    args = ap.parse_args()

    out_path = args.output if args.output else None
    result = run_pullup_analysis(
        input_path=args.input,
        frame_skip=args.skip,
        scale=args.scale,
        output_path=out_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

