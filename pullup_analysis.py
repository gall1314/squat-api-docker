# -*- coding: utf-8 -*-
# pullup_analysis.py — שמירת ספירה מקורית 1:1 + אוברליי וטמפלייט תואמים לסקוואט
# דונאט ASCENT חי (במתח זה מייצג עלייה — כלומר ככל שהראש גבוה יותר כך הפאי מתמלא),
# RT-feedback עם HOLD ~0.8s רק *בתוך חזרה*, החזרה (return) בפורמט זהה לסקוואט כולל
# "squat_count" כשם שדה גלובלי לכל התרגילים.

import os
import cv2
import math
import time
import json
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image

# ==============================
# ספים/הגדרות (כמו המקור שלך)
# ==============================
ELBOW_START_THRESHOLD = 150.0   # התחלת עליה: מרפק מתחת לסף
ELBOW_TOP_THRESHOLD   = 65.0    # טופ: מרפק "סגור"
ELBOW_BOTTOM_THRESHOLD= 160.0   # חזרה לתחתית (היסטורזיס)
HEAD_MIN_ASCENT       = 0.06    # עליה נטו (0..1 מגובה הפריים)
HEAD_VEL_UP_THRESH    = 0.0025  # מהירות עליה מינימלית (שלילי)
HEAD_TOP_STICK_FRAMES = 2
BOTTOM_STICK_FRAMES   = 2

BASE_FRAME_SKIP_IDLE  = 3       # מנוחה
BASE_FRAME_SKIP_MOVE  = 1       # תנועה
EMA_ALPHA_DEPTH       = 0.2
FPS_FALLBACK          = 25.0

# ===================== STYLE / FONTS (תואם סקוואט) =====================
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

# ===================== פונקציות עזר =====================
def _angle(a, b, c):
    try:
        ba = (a[0]-b[0], a[1]-b[1]); bc = (c[0]-b[0], c[1]-b[1])
        den = ( (ba[0]**2 + ba[1]**2)**0.5 * (bc[0]**2 + bc[1]**2)**0.5 ) + 1e-9
        cosang = (ba[0]*bc[0] + ba[1]*bc[1]) / den
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))
    except Exception:
        return 180.0

def _ema(prev, new, alpha):
    return new if prev is None else (alpha*new + (1-alpha)*prev)

def _round_score_half(x):
    return round(x*2)/2.0

def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    if abs(q - round(q)) < 1e-9:
        return str(int(round(q)))
    return f"{q:.1f}"

def score_label(s):
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

# ===================== MediaPipe Pose =====================
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

if MP_AVAILABLE:
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

POSE_IDXS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
}

def _pick_side(lms):
    """בחירת צד יציבה לכל הסשן (לנעול בצד חזק יותר בפריימים הראשונים).
    בחרנו לוגיקה עדינה שמעדיפה ימין אם אין הכרעה — כמו במקור שהיה "right" קבוע.
    """
    rs = lms[POSE_IDXS["right_shoulder"]].visibility
    ls = lms[POSE_IDXS["left_shoulder"]].visibility
    re = lms[POSE_IDXS["right_elbow"]].visibility
    le = lms[POSE_IDXS["left_elbow"]].visibility
    count_r = sum(v > 0.4 for v in (rs, re))
    count_l = sum(v > 0.4 for v in (ls, le))
    return "right" if count_r >= count_l else "left"

def _get_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h, lm.visibility)

def _safe_vis(*vals, min_v=0.4):
    return all(v >= min_v for v in vals)

def draw_body_only(frame, landmarks, color=(255,255,255)):
    """שלד גוף בלבד (ללא פנים)."""
    if not MP_AVAILABLE:
        return frame
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

# ===================== Overlay (תואם סקוואט) =====================
def draw_ascent_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius = int(radius); thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, ascent_pct=0.0):
    """Reps שמאל־עליון; דונאט ASCENT ימין־עליון; פידבק תחתון (עד 2 שורות, עם אליפסות)."""
    h, w = frame.shape[:2]

    # Reps (שמאל־עליון צמוד לקצה)
    rep_text = f"Reps: {int(reps)}"
    (tw, th), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    pad = 8
    bg = frame.copy()
    cv2.rectangle(bg, (0, 0), (tw + 2*pad, th + int(1.8*pad)), (0,0,0), -1)
    frame = cv2.addWeighted(bg, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
    cv2.putText(frame, rep_text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    # Donut (ימין־עליון)
    cx = w - int(70)
    cy = int(60)
    radius = int(min(80, w*0.09) * DONUT_RADIUS_SCALE)
    thick  = max(8, int(radius * DONUT_THICKNESS_FRAC))
    frame = draw_ascent_donut(frame, (cx, cy), radius, thick, ascent_pct)

    # מלל עומק קטן במרכז הדונאט
    depth_pct = int(round(ascent_pct * 100))
    lbl = Image.fromarray(frame)
    d = ImageDraw.Draw(lbl)
    t1 = "Ascent"
    t2 = f"{depth_pct}%"
    t1w = d.textlength(t1, font=DEPTH_LABEL_FONT)
    t2w = d.textlength(t2, font=DEPTH_PCT_FONT)
    d.text((cx - t1w/2, cy - 20), t1, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    d.text((cx - t2w/2, cy + 2),  t2, font=DEPTH_PCT_FONT,   fill=(255,255,255))
    frame = np.array(lbl)

    # Bottom feedback (2 שורות מקס' + אליפסות)
    if feedback:
        def wrap_two_lines(draw, text, font, max_w):
            words = text.split(); lines, cur = [], ""
            for w_ in words:
                trial = (cur + " " + w_).strip()
                if draw.textlength(trial, font=font) <= max_w: cur = trial
                else:
                    if cur: lines.append(cur)
                    cur = w_
                if len(lines) == 2: break
            if cur and len(lines) < 2: lines.append(cur)
            leftover = len(words) - sum(len(x.split()) for x in lines)
            if leftover > 0 and len(lines) >= 2:
                last = lines[-1] + "…"
                while draw.textlength(last, font=font) > max_w and len(last) > 1:
                    last = last[:-2] + "…"
                lines[-1] = last
            return lines

        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        safe_m, bx, by, gap = max(6, int(h*0.02)), 12, 8, 4
        max_w = int(w - 2*bx - 20)
        lines = wrap_two_lines(draw_fb, feedback, FEEDBACK_FONT, max_w)
        line_h = FEEDBACK_FONT.size + 6
        block_h = (2*by) + len(lines)*line_h + (len(lines)-1)*gap
        y0 = max(0, h - safe_m - block_h); y1 = h - safe_m
        over = frame.copy()
        cv2.rectangle(over, (0, y0), (w, y1), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + by
        for ln in lines:
            tw = draw_fb.textlength(ln, font=FEEDBACK_FONT)
            tx = max(bx, (w - int(tw)) // 2)
            draw_fb.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255,255,255))
            ty += line_h + gap
        frame = np.array(pil_fb)

    return frame

# =========================
# PullUp Analyzer (ספירה מקורית + תיקון צד יציב + לוגים)
# =========================
class PullUpAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rep_count = 0
        self.in_rep = False
        self.seen_top_frames = 0
        self.seen_bottom_frames = 0
        self.min_head_y_in_rep = None
        self.rep_started_elbow = None
        self.last_head_y = None
        self.last_elbow = None

        # ציון/דוחות
        self.reps_meta = []
        self.all_scores = []
        self.session_best_feedback = ""

        # עומק לייב (דונאט)
        self.depth_ema = None
        self.ascent_live = 0.0

        # צד נעול לסשן (מונע קפיצות מצד לצד שמחרבות ספירה)
        self.side_locked = None

        # RT-feedback עם HOLD רק בתוך חזרה
        self.rt_fb_msg = None
        self.rt_fb_hold = 0
        self.RT_FB_HOLD_FRAMES = 20  # ~0.8s ב-25fps (מותאם דינמית בהמשך)

    # === עזר לזיהוי תנועה (כמו במקור)
    def _movement_detected(self, elbow_angle, head_y):
        mov = False
        if self.last_elbow is not None and elbow_angle is not None:
            if abs(elbow_angle - self.last_elbow) > 1.5:
                mov = True
        if self.last_head_y is not None and head_y is not None:
            if abs(head_y - self.last_head_y) > 0.002:
                mov = True
        return mov

    # ===== אימותים =====
    def _confirm_top(self, elbow_angle, head_y, rep_baseline_head_y):
        elbow_ok = elbow_angle is not None and (elbow_angle <= ELBOW_TOP_THRESHOLD)
        head_ok = (rep_baseline_head_y is not None and head_y is not None and
                   (rep_baseline_head_y - head_y) >= HEAD_MIN_ASCENT)
        return elbow_ok and head_ok

    def _confirm_start(self, elbow_angle, head_vel):
        elbow_ok = elbow_angle is not None and elbow_angle < ELBOW_START_THRESHOLD
        head_speed_ok = head_vel < -HEAD_VEL_UP_THRESH
        return elbow_ok and head_speed_ok

    def _confirm_bottom(self, elbow_angle):
        return elbow_angle is not None and elbow_angle >= ELBOW_BOTTOM_THRESHOLD

    # =========================
    # לולאת העיבוד
    # =========================
    def process(self, input_path, output_path=None, frame_skip=3, scale=0.4, overlay_enabled=True):
        print("[pullup] process start", input_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return self._empty_result("Could not open video", output_path)

        # FPS/מידות
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        if fps <= 1: fps = FPS_FALLBACK
        self.RT_FB_HOLD_FRAMES = max(12, int(0.8 * fps))  # ~0.8s

        ok, first = cap.read()
        if not ok:
            cap.release()
            return self._empty_result("Empty video", output_path)

        h0, w0 = first.shape[:2]
        out_w = int(w0 * (scale if scale > 0 else 1.0))
        out_h = int(h0 * (scale if scale > 0 else 1.0))

        # פותחים writer רק אחרי שמוגדרים מימדים
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        # נחזיר את הפריים הראשון אחורה כי כבר קראנו אותו
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        baseline_head_y_global = None
        rep_baseline_head_y = None

        # Pose
        if not MP_AVAILABLE:
            cap.release()
            if out is not None: out.release()
            return self._empty_result("MediaPipe is not available", output_path)

        with mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True) as pose:
            base_skip_idle = max(1, frame_skip)
            base_skip_move = BASE_FRAME_SKIP_MOVE

            frame_idx = 0
            while True:
                ok, frame_orig = cap.read()
                if not ok:
                    break
                frame_idx += 1

                # ריסייז יציב
                frame = cv2.resize(frame_orig, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(frame_rgb)

                elbow_angle = None
                head_y = None

                if res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    # נעל צד פעם אחת בסשן — מונע קפיצות צדדים שמפרקות את התנאים
                    if self.side_locked is None:
                        self.side_locked = _pick_side(lms)
                    side = self.side_locked

                    sh_idx = POSE_IDXS["right_shoulder"] if side == "right" else POSE_IDXS["left_shoulder"]
                    el_idx = POSE_IDXS["right_elbow"]    if side == "right" else POSE_IDXS["left_elbow"]
                    wr_idx = POSE_IDXS["right_wrist"]    if side == "right" else POSE_IDXS["left_wrist"]

                    nose = _get_xy(lms, POSE_IDXS["nose"], out_w, out_h)
                    shld = _get_xy(lms, sh_idx, out_w, out_h)
                    elbw = _get_xy(lms, el_idx, out_w, out_h)
                    wrst = _get_xy(lms, wr_idx, out_w, out_h)

                    if _safe_vis(nose[2], shld[2], elbw[2], wrst[2], min_v=0.4):
                        head_y = nose[1] / out_h  # 0 למעלה, 1 למטה
                        elbow_angle = _angle(shld[:2], elbw[:2], wrst[:2])
                        if baseline_head_y_global is None:
                            baseline_head_y_global = head_y

                # ASCENT לייב לדונאט (במתח: ככל שהראש גבוה יותר — ערך גדול יותר)
                ascent_live = 0.0
                if baseline_head_y_global is not None and head_y is not None:
                    ascent_live = float(np.clip(baseline_head_y_global - head_y, 0.0, 1.0))
                self.ascent_live = ascent_live

                # Adaptive skip (כמו במקור)
                if elbow_angle is not None and head_y is not None:
                    moving = self._movement_detected(elbow_angle, head_y)
                    step_now = base_skip_move if moving else base_skip_idle
                else:
                    step_now = base_skip_idle

                # מהירות ראש
                head_vel = 0.0
                if self.last_head_y is not None and head_y is not None:
                    head_vel = head_y - self.last_head_y  # שלילי = עולה

                # ===== לוגיקת ספירה =====
                if elbow_angle is not None and head_y is not None:
                    if not self.in_rep and rep_baseline_head_y is None:
                        rep_baseline_head_y = head_y

                    if not self.in_rep:
                        if self._confirm_start(elbow_angle, head_vel):
                            self.in_rep = True
                            self.min_head_y_in_rep = head_y
                            self.seen_top_frames = 0
                            self.rep_started_elbow = elbow_angle
                            # לוג
                            print(f"[pullup] start rep at f{frame_idx} elbow={elbow_angle:.1f} head_y={head_y:.3f}")
                    else:
                        # עדכון מינימום ראש בתוך הסט (הכי גבוה — ערך y קטן)
                        if self.min_head_y_in_rep is None:
                            self.min_head_y_in_rep = head_y
                        else:
                            self.min_head_y_in_rep = min(self.min_head_y_in_rep, head_y)

                        # בדיקת טופ
                        if self._confirm_top(elbow_angle, head_y, rep_baseline_head_y):
                            self.seen_top_frames += 1
                        else:
                            self.seen_top_frames = 0

                        if self.seen_top_frames >= HEAD_TOP_STICK_FRAMES:
                            # נספרה חזרה — ניקוד/פידבק כמו בסקוואט (לא משפיע על ספירה)
                            ascent_peak = max(0.0, (baseline_head_y_global or head_y) - (self.min_head_y_in_rep or head_y))
                            penalty = 0.0
                            fb = None
                            if ascent_peak < 0.08:
                                fb = "Aim for chin over the bar"; penalty += 2.5
                            elif ascent_peak < HEAD_MIN_ASCENT:
                                fb = "Aim for chin over the bar"; penalty += 2.0
                            if self.rep_started_elbow is not None and self.rep_started_elbow < ELBOW_BOTTOM_THRESHOLD - 5:
                                fb = fb or "Try to fully extend at the bottom"; penalty += 1.0

                            score = 10.0 if penalty == 0.0 else max(4.0, 10.0 - min(penalty, 6.0))
                            score = _round_score_half(score)

                            self.rep_count += 1
                            print(f"[pullup] REP {self.rep_count} peak_ascent={ascent_peak:.3f} top_elbow={elbow_angle:.1f}")
                            self.reps_meta.append({
                                "rep_index": self.rep_count,
                                "score": float(score),
                                "score_display": display_half_str(score),
                                "feedback": ([fb] if fb else []),
                                "ascent_peak": float(ascent_peak),
                                "start_elbow": float(self.rep_started_elbow) if self.rep_started_elbow is not None else None,
                                "top_elbow": float(elbow_angle),
                            })
                            self.all_scores.append(score)
                            if fb and not self.session_best_feedback:
                                self.session_best_feedback = fb

                            # איפוס לחזרה הבאה
                            self.in_rep = False
                            rep_baseline_head_y = None
                            self.min_head_y_in_rep = None
                            self.rep_started_elbow = None
                            self.seen_top_frames = 0
                            self.seen_bottom_frames = 0

                # RT-feedback — רק בתוך חזרה, לא לפני שהתחילה
                if self.in_rep:
                    if self.ascent_live < 0.03:
                        if self.rt_fb_msg != "Aim for chin over the bar":
                            self.rt_fb_msg = "Aim for chin over the bar"
                            self.rt_fb_hold = self.RT_FB_HOLD_FRAMES
                        else:
                            self.rt_fb_hold = max(self.rt_fb_hold, self.RT_FB_HOLD_FRAMES)
                    else:
                        if self.rt_fb_hold > 0:
                            self.rt_fb_hold -= 1
                else:
                    # מחוץ לחזרה אל תציג RT
                    self.rt_fb_msg = None
                    self.rt_fb_hold = 0

                # החלקת עומק (לא חובה לציור)
                if baseline_head_y_global is not None and head_y is not None:
                    depth_val = max(0.0, min(1.0, (baseline_head_y_global - head_y)))
                    self.depth_ema = _ema(self.depth_ema, depth_val, EMA_ALPHA_DEPTH)

                # ציור שלד + אוברליי
                if res.pose_landmarks is not None:
                    frame = draw_body_only(frame, res.pose_landmarks.landmark)
                if overlay_enabled:
                    frame = draw_overlay(
                        frame,
                        reps=self.rep_count,
                        feedback=(self.rt_fb_msg if self.rt_fb_hold > 0 else None),
                        ascent_pct=self.ascent_live
                    )

                if out is not None:
                    out.write(frame)

                # עדכוני "אחרון"
                if elbow_angle is not None: self.last_elbow = elbow_angle
                if head_y is not None:     self.last_head_y = head_y

                # קפיצת פריימים אדפטיבית (לא חוסמת ספירה!)
                if step_now > 1:
                    cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cur + (step_now - 1))

                # לוג קל כל ~50 פריימים לזמינות
                if frame_idx % 50 == 0:
                    eh = f"{elbow_angle:.1f}" if elbow_angle is not None else "NA"
                    hy = f"{head_y:.3f}" if head_y is not None else "NA"
                    print(f"[pullup] f{frame_idx} reps={self.rep_count} in_rep={self.in_rep} elbow={eh} head_y={hy} step={step_now}")

        cap.release()
        if out is not None:
            out.release()

        # ניקוד טכניקה ממוצע (כמו בסקוואט — ממוצע רפ־סקור, עיגול לחצאים)
        avg = float(np.mean(self.all_scores)) if self.all_scores else 0.0
        technique_score = round(round(avg * 2) / 2, 2)

        # good/bad לפי ציון הרפ (>=9.5 טוב)
        good_reps = sum(1 for s in self.all_scores if s >= 9.5)
        bad_reps  = max(0, self.rep_count - good_reps)

        feedback_list = [self.session_best_feedback] if self.session_best_feedback else ["Great form! Keep it up 💪"]
        tips = ["Slow down the lowering phase to maximize hypertrophy"]  # טיפ אחד לסשן — לא משפיע על הציון

        # קידוד faststart כמו בסקוואט
        final_video_path = ""
        if output_path:
            encoded_path = output_path.replace(".mp4", "_encoded.mp4")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_path,
                    "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                    encoded_path
                ], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(output_path) and os.path.exists(encoded_path):
                    os.remove(output_path)
                final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")
            except Exception:
                final_video_path = output_path if os.path.exists(output_path) else ""

        return {
            "squat_count": int(self.rep_count),                                 # שם גלובלי אחיד לכל התרגילים
            "technique_score": float(technique_score),                          # double לחישובים/גרפים
            "technique_score_display": display_half_str(technique_score),       # תצוגה (ללא .0 אם שלם)
            "technique_label": score_label(technique_score),                    # ציון מילולי באותן מדרגות
            "good_reps": int(good_reps),
            "bad_reps": int(bad_reps),
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

# ===========================
# API
# ===========================
def run_pullup_analysis(input_path, frame_skip=3, scale=0.4, output_path=None, overlay_enabled=True):
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
        overlay_enabled=overlay_enabled
    )
    res["elapsed_sec"] = round(time.time() - t0, 3)
    if output_path and "video_path" not in res:
        res["video_path"] = output_path
    print(f"[pullup] done in {res['elapsed_sec']}s reps={res.get('squat_count')}")
    return res

# ===========================
# הרצה ידנית
# ===========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", default="")
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    out_path = args.output if args.output else None
    result = run_pullup_analysis(
        input_path=args.input,
        frame_skip=args.skip,
        scale=args.scale,
        output_path=out_path,
        overlay_enabled=(not args.no_overlay)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
