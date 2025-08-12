# pullup_analysis.py
# -*- coding: utf-8 -*-
import os
import cv2
import math
import time
import json
import numpy as np

# ==============================
# הגדרות וספים (ניתנים לטיונינג)
# ==============================
ELBOW_START_THRESHOLD = 150.0   # מעל זה ~ יד "פתוחה". תחילת תנועה כאשר יורד מתחת.
ELBOW_TOP_THRESHOLD   = 65.0    # מתחת לזה ~ יד "סגורה" (בחלק העליון).
ELBOW_BOTTOM_THRESHOLD= 160.0   # חזרה לתחתית (פתוח מאוד) לסיום היסטורזיס.
HEAD_MIN_ASCENT       = 0.06    # ירידה בציר Y (תמונה) = עלייה פיזית. נדרש בתוך הסט (יחסי לגובה הפריים)
HEAD_VEL_UP_THRESH    = 0.0025  # מהירות עלייה מינימלית (delta Y/frame שלילי) לזיהוי התחלה
HEAD_TOP_STICK_FRAMES = 2       # כמה פריימים רצופים צריך להיות "למעלה" כדי לאשר טופ (חזרה מהירה → נמוך)
BOTTOM_STICK_FRAMES   = 2       # כמה פריימים רצופים בתחתית לאיפוס מצב

BASE_FRAME_SKIP_IDLE  = 3       # מנוחה
BASE_FRAME_SKIP_MOVE  = 1       # בזמן תנועה

EMA_ALPHA_DEPTH       = 0.2     # החלקה לוגית למדד עומק/ROM, אם תרצה לצייר
FPS_FALLBACK          = 25.0

RETURN_SERIES         = False   # ברירת מחדל: לא להחזיר סדרות ארוכות (compact)

# =================================
# עוזרים גיאומטריים/מספריים פשוטים
# =================================
def _angle(a, b, c):
    # זווית ABC במעלות; a,b,c זה (x,y)
    try:
        ba = (a[0]-b[0], a[1]-b[1])
        bc = (c[0]-b[0], c[1]-b[1])
        cosang = (ba[0]*bc[0] + ba[1]*bc[1]) / (
            math.hypot(*ba) * math.hypot(*bc) + 1e-9
        )
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))
    except Exception:
        return 180.0

def _ema(prev, new, alpha):
    return new if prev is None else (alpha*new + (1-alpha)*prev)

def _round_score_half(x):
    # עיגול לציון .0/.5
    return round(x*2)/2.0

# ===========================
# MediaPipe Pose (light load)
# ===========================
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# מיפוי שמות לנקודות של MediaPipe
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_IDXS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    # ברירת מחדל נשתמש בימין; אם confidence נמוך נעשה fallback לשמאל
}

def _pick_side(landmarks):
    # לפשט: בחר יד עם נראות טובה יותר; פה נאסוף ימין כברירת מחדל
    return "right"

def _get_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h, lm.visibility)

def _safe_vis(*vals, min_v=0.5):
    return all(v >= min_v for v in vals)

# =========================
# PullUp Analyzer (משודרג)
# =========================
class PullUpAnalyzer:
    def __init__(self, include_series=RETURN_SERIES):
        self.include_series = include_series
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
        self.rom_series = [] if self.include_series else None
        self.reps_meta = []   # נאסוף מידע תמציתי על כל חזרה

        # ניקוד/פידבק כללי
        self.errors_set = set()
        self.tips_set = set(["Control the lowering phase for better muscle growth"])

        # עומק/ROM מחולק (לא חובה לצייר)
        self.depth_ema = None

    def _movement_detected(self, elbow_angle, head_y):
        # מזהה תנועה כדי לקבוע frame_skip אדפטיבי
        mov = False
        if self.last_elbow is not None and abs(elbow_angle - self.last_elbow) > 1.5:
            mov = True
        if self.last_head_y is not None and abs(head_y - self.last_head_y) > 0.002:
            mov = True
        return mov

    def _update_series(self, rom_val):
        if self.include_series:
            self.rom_series.append(rom_val)

    def _confirm_top(self, elbow_angle, head_y, rep_baseline_head_y):
        """
        תנאי טופ (חלק עליון): מרפק סגור + הראש עלה משמעותית לעומת תחילת הסט.
        """
        elbow_ok = elbow_angle <= ELBOW_TOP_THRESHOLD
        head_ok = (rep_baseline_head_y - head_y) >= HEAD_MIN_ASCENT
        return elbow_ok and head_ok

    def _confirm_start(self, elbow_angle, head_y, head_vel):
        """
        תנאי התחלה: ירידת מרפק מתחת ל-START + מהירות עלייה מינימלית של הראש (להימנע מניעור ידיים).
        """
        elbow_ok = elbow_angle < ELBOW_START_THRESHOLD
        head_speed_ok = head_vel < -HEAD_VEL_UP_THRESH  # שלילי = עולה
        return elbow_ok and head_speed_ok

    def _confirm_bottom(self, elbow_angle):
        return elbow_angle >= ELBOW_BOTTOM_THRESHOLD

    def process(self, input_path, output_path=None, frame_skip=3, scale=0.4, draw_overlay=True):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not available in environment")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                          smooth_landmarks=True) as pose:
            idx = 0
            base_skip_idle = max(1, frame_skip)
            base_skip_move = 1  # בזמן תנועה תמיד 1

            # baseline ראש בתחילת וידאו (למקרה הצורך)
            baseline_head_y_global = None
            rep_baseline_head_y = None

            # counters להחזקה (hysteresis)
            self.seen_top_frames = 0
            self.seen_bottom_frames = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if scale != 1.0:
                    frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    # עדכון w/h לשרשרת החישוב/ציור
                    sh, sw = frame.shape[:2]
                else:
                    sh, sw = h, w

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(frame_rgb)

                elbow_angle = None
                head_y = None

                if res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    side = _pick_side(lms)

                    sh_idx  = POSE_IDXS["right_shoulder"] if side == "right" else POSE_IDXS["left_shoulder"]
                    el_idx  = POSE_IDXS["right_elbow"]    if side == "right" else POSE_IDXS["left_elbow"]
                    wr_idx  = POSE_IDXS["right_wrist"]    if side == "right" else POSE_IDXS["left_wrist"]

                    nose = _get_xy(lms, POSE_IDXS["nose"], sw, sh)
                    shld = _get_xy(lms, sh_idx, sw, sh)
                    elbw = _get_xy(lms, el_idx, sw, sh)
                    wrst = _get_xy(lms, wr_idx, sw, sh)

                    if _safe_vis(nose[2], shld[2], elbw[2], wrst[2], min_v=0.4):
                        head_y = nose[1] / sh  # נורמליזציה (0-1), למעלה קטן
                        elbow_angle = _angle(shld[:2], elbw[:2], wrst[:2])

                        if baseline_head_y_global is None:
                            baseline_head_y_global = head_y

                # Adaptive skip: אם אין נתונים → מנוחה; אם יש תנועה → step 1
                if elbow_angle is not None and head_y is not None:
                    moving = self._movement_detected(elbow_angle, head_y)
                    step_now = base_skip_move if moving else base_skip_idle
                else:
                    step_now = base_skip_idle

                # חישוב מהירות ראש (לסינון ניעורים)
                head_vel = 0.0
                if self.last_head_y is not None and head_y is not None:
                    head_vel = head_y - self.last_head_y  # שלילי = עלייה

                # לוגיקת ספירה
                if elbow_angle is not None and head_y is not None:
                    # baseline לסט ספציפי
                    if not self.in_rep and rep_baseline_head_y is None:
                        rep_baseline_head_y = head_y

                    if not self.in_rep:
                        # בדיקת התחלה: מרפק התחיל להיסגר + הראש באמת עולה (מהירות שלילית)
                        if self._confirm_start(elbow_angle, head_y, head_vel):
                            self.in_rep = True
                            self.min_head_y_in_rep = head_y  # נעקוב אחרי המינימום (הכי גבוה)
                            self.seen_top_frames = 0
                            self.rep_started_elbow = elbow_angle
                    else:
                        # בתוך סט: עדכון מינימום Y של הראש
                        if self.min_head_y_in_rep is None:
                            self.min_head_y_in_rep = head_y
                        else:
                            self.min_head_y_in_rep = min(self.min_head_y_in_rep, head_y)

                        # האם הגענו לטופ (הכי גבוה + מרפק סגור)
                        if self._confirm_top(elbow_angle, head_y, rep_baseline_head_y):
                            self.seen_top_frames += 1
                        else:
                            # אם לא בטופ בפריים הזה – אפס מונה
                            self.seen_top_frames = 0

                        # כללים: נספור חזרה בזמן הטופ, כדי לא לפספס בקצב גבוה
                        if self.seen_top_frames >= HEAD_TOP_STICK_FRAMES:
                            # ספר חזרה
                            self.rep_count += 1

                            # מטא‑דאטה קצרה
                            self.reps_meta.append({
                                "rep_index": self.rep_count,
                                "peak_head_y": float(self.min_head_y_in_rep),
                                "start_elbow": float(self.rep_started_elbow) if self.rep_started_elbow is not None else None,
                                "top_elbow": float(elbow_angle),
                            })

                            # איפוס מצב להמשך
                            self.in_rep = False
                            rep_baseline_head_y = None
                            self.min_head_y_in_rep = None
                            self.rep_started_elbow = None
                            self.seen_top_frames = 0
                            self.seen_bottom_frames = 0

                    # החלקת "עומק" לוגית (לא חובה, שומר על עקביות אם תרצה דונאט)
                    # עומק נורמלי 0..1: ככל שהראש גבוה יותר (קטן יותר ב-Y), העומק גדול יותר
                    if baseline_head_y_global is not None:
                        depth_val = max(0.0, min(1.0, (baseline_head_y_global - head_y)))
                        self.depth_ema = _ema(self.depth_ema, depth_val, EMA_ALPHA_DEPTH)
                        self._update_series(self.depth_ema if self.depth_ema is not None else depth_val)

                # ציור (קליל; אם תרצה לכבות – draw_overlay=False)
                if draw_overlay:
                    txt = f"Reps: {self.rep_count}"
                    cv2.putText(frame, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                    if elbow_angle is not None:
                        cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                if out is not None:
                    # שים לב: אם שינית scale, אנחנו כותבים את ה-frame בגודל scaled.
                    # כדי לשמור בפורמט המקורי, אפשר לרנדר ל-size המקורי, כאן משאירים scaled למען ביצועים.
                    out.write(frame)

                # עדכוני "אחרון"
                self.last_elbow = elbow_angle if elbow_angle is not None else self.last_elbow
                self.last_head_y = head_y if head_y is not None else self.last_head_y

                # קפיצת פריימים אדפטיבית
                if step_now > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + (step_now - 1))
                idx += step_now

        cap.release()
        if out is not None:
            out.release()

        # ניקוד טכניקה (פשוט; תוכל להחליף לוגיקה שלך)
        # כאן לא מורידים מתחת ל‑4, ומעגלים ל‑.0/.5
        base_score = 9.0
        penalties = 0.0

        # דוגמה: אם חזרות מעטות/מהירות מאוד → אפשר להוסיף penalty קטן
        if self.rep_count == 0:
            penalties += 3.0

        tech_score = max(4.0, base_score - penalties)
        tech_score = _round_score_half(tech_score)

        # אריזה לתוצאה
        result = {
            "rep_count": int(self.rep_count),
            "technique_score": float(tech_score),
            "errors": sorted(list(self.errors_set)),
            "tips": sorted(list(self.tips_set)),
        }

        if self.include_series:
            result["rom_series"] = self.rom_series

        if output_path:
            result["video_path"] = output_path

        return result


# ===========================
# API פונקציה ידידותית ל-Flask
# ===========================
def run_pullup_analysis(input_path, frame_skip=3, scale=0.4, output_path=None,
                        overlay=True, include_series=RETURN_SERIES):
    """
    תואם ל-API שלך:
    - input_path: וידאו מקור
    - output_path: וידאו מעובד (אם None, לא מייצר קובץ)
    - מחזיר dict עם video_path (אם הוגדר), rep_count, technique_score, וכו'.
    """
    # ייצוב OpenCV בת’רד אחד (מונע קריסות בשירותים קלים)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    t0 = time.time()
    analyzer = PullUpAnalyzer(include_series=include_series)
    res = analyzer.process(
        input_path=input_path,
        output_path=output_path,
        frame_skip=frame_skip,
        scale=scale,
        draw_overlay=overlay
    )
    res["elapsed_sec"] = round(time.time() - t0, 3)
    # וודא שתמיד יש video_path אם ביקשו פלט
    if output_path and "video_path" not in res:
        res["video_path"] = output_path
    return res


# ===========================
# הרצה ידנית (לבדיקה מקומית)
# ===========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", default="")
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--series", action="store_true", help="Return ROM series (heavy)")
    args = ap.parse_args()

    out = args.output if args.output else None
    result = run_pullup_analysis(
        input_path=args.input,
        frame_skip=args.skip,
        scale=args.scale,
        output_path=out,
        overlay=not args.no_overlay,
        include_series=args.series
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

