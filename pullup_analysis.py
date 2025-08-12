# pullup_analysis.py
# -*- coding: utf-8 -*-
import os
import cv2
import math
import time
import json

# ==============================
# הגדרות וספים (טיונינג קל)
# ==============================
ELBOW_START_THRESHOLD = 150.0   # מעל זה ~ יד "פתוחה". תחילת תנועה כשהופך לפחות.
ELBOW_TOP_THRESHOLD   = 65.0    # מתחת לזה ~ יד "סגורה" (חלק עליון).
ELBOW_BOTTOM_THRESHOLD= 160.0   # חזרה ל"תחתית" (פתוח מאוד) לאיפוס היסטורזיס.
HEAD_MIN_ASCENT       = 0.06    # עלייה נטו של הראש (ביחס לגובה הפריים 0..1) בתוך הסט.
HEAD_VEL_UP_THRESH    = 0.0025  # מהירות עלייה מינימלית (delta Y/frame שלילי) כדי לאשר התחלה.
HEAD_TOP_STICK_FRAMES = 2       # כמה פריימים רצופים בטופ כדי לאשר (במהיר → אפשר להוריד ל-1).
BOTTOM_STICK_FRAMES   = 2

BASE_FRAME_SKIP_IDLE  = 3       # מנוחה
BASE_FRAME_SKIP_MOVE  = 1       # תנועה

EMA_ALPHA_DEPTH       = 0.2
FPS_FALLBACK          = 25.0

RETURN_SERIES         = False   # ברירת מחדל: פלט קומפקטי (לא מחזיר סדרות כבדות)

# ===========================
# עוזרים גיאומטריים/מספריים
# ===========================
def _angle(a, b, c):
    # זווית ABC במעלות; a,b,c הם (x,y)
    try:
        ba = (a[0]-b[0], a[1]-b[1])
        bc = (c[0]-b[0], c[1]-b[1])
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

# ===========================
# MediaPipe Pose
# ===========================
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

POSE_IDXS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
}

def _pick_side(landmarks):
    # לשם פשטות – יד ימין; אפשר לשפר לפי visibility
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
        self.reps_meta = []

        self.errors_set = set()
        self.tips_set = set(["Control the lowering phase for better muscle growth"])

        self.depth_ema = None

    def _movement_detected(self, elbow_angle, head_y):
        mov = False
        if self.last_elbow is not None and elbow_angle is not None:
            if abs(elbow_angle - self.last_elbow) > 1.5:
                mov = True
        if self.last_head_y is not None and head_y is not None:
            if abs(head_y - self.last_head_y) > 0.002:
                mov = True
        return mov

    def _update_series(self, rom_val):
        if self.include_series:
            self.rom_series.append(rom_val)

    def _confirm_top(self, elbow_angle, head_y, rep_baseline_head_y):
        # טופ = מרפק סגור + הראש עלה משמעותית יחסית לבסיס הסט
        elbow_ok = elbow_angle is not None and (elbow_angle <= ELBOW_TOP_THRESHOLD)
        head_ok = (rep_baseline_head_y is not None and head_y is not None and
                   (rep_baseline_head_y - head_y) >= HEAD_MIN_ASCENT)
        return elbow_ok and head_ok

    def _confirm_start(self, elbow_angle, head_vel):
        # התחלה = ירידת מרפק מתחת ל-START + מהירות עלייה מינ' של הראש (שלילי)
        elbow_ok = elbow_angle is not None and elbow_angle < ELBOW_START_THRESHOLD
        head_speed_ok = head_vel < -HEAD_VEL_UP_THRESH
        return elbow_ok and head_speed_ok

    def _confirm_bottom(self, elbow_angle):
        return elbow_angle is not None and elbow_angle >= ELBOW_BOTTOM_THRESHOLD

    def process(self, input_path, output_path=None, frame_skip=3, scale=0.4, draw_overlay=True):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not available")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ===== קובעים מימדי פלט קבועים (פותר FFmpeg 'Failed to write frame') =====
        out_w = int(src_w * scale) if scale != 1.0 else src_w
        out_h = int(src_h * scale) if scale != 1.0 else src_h

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
            if not out.isOpened():
                # נסיונות קודק חלופיים
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
                if not out.isOpened():
                    alt_path = os.path.splitext(output_path)[0] + ".avi"
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    out = cv2.VideoWriter(alt_path, fourcc, fps, (out_w, out_h))
                    output_path = alt_path  # מעדכן את הנתיב אם עברנו ל-AVI

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True) as pose:
            base_skip_idle = max(1, frame_skip)
            base_skip_move = BASE_FRAME_SKIP_MOVE

            baseline_head_y_global = None
            rep_baseline_head_y = None

            self.seen_top_frames = 0
            self.seen_bottom_frames = 0

            while True:
                ok, frame_orig = cap.read()
                if not ok:
                    break

                # resize חד-ערכי לגודל הפלט
                frame = cv2.resize(frame_orig, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

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

                    nose = _get_xy(lms, POSE_IDXS["nose"], out_w, out_h)
                    shld = _get_xy(lms, sh_idx, out_w, out_h)
                    elbw = _get_xy(lms, el_idx, out_w, out_h)
                    wrst = _get_xy(lms, wr_idx, out_w, out_h)

                    if _safe_vis(nose[2], shld[2], elbw[2], wrst[2], min_v=0.4):
                        head_y = nose[1] / out_h  # 0 למעלה, 1 למטה
                        elbow_angle = _angle(shld[:2], elbw[:2], wrst[:2])
                        if baseline_head_y_global is None:
                            baseline_head_y_global = head_y

                # Adaptive skip
                if elbow_angle is not None and head_y is not None:
                    moving = self._movement_detected(elbow_angle, head_y)
                    step_now = base_skip_move if moving else base_skip_idle
                else:
                    step_now = base_skip_idle

                # מהירות ראש
                head_vel = 0.0
                if self.last_head_y is not None and head_y is not None:
                    head_vel = head_y - self.last_head_y  # שלילי = עולה

                # לוגיקת ספירה
                if elbow_angle is not None and head_y is not None:
                    if not self.in_rep and rep_baseline_head_y is None:
                        rep_baseline_head_y = head_y

                    if not self.in_rep:
                        if self._confirm_start(elbow_angle, head_vel):
                            self.in_rep = True
                            self.min_head_y_in_rep = head_y
                            self.seen_top_frames = 0
                            self.rep_started_elbow = elbow_angle
                    else:
                        # עדכון מינימום ראש בתוך הסט (הכי גבוה)
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
                            # ספר חזרה
                            self.rep_count += 1
                            self.reps_meta.append({
                                "rep_index": self.rep_count,
                                "peak_head_y": float(self.min_head_y_in_rep),
                                "start_elbow": float(self.rep_started_elbow) if self.rep_started_elbow is not None else None,
                                "top_elbow": float(elbow_angle),
                            })
                            # איפוס
                            self.in_rep = False
                            rep_baseline_head_y = None
                            self.min_head_y_in_rep = None
                            self.rep_started_elbow = None
                            self.seen_top_frames = 0
                            self.seen_bottom_frames = 0

                    # החלקת עומק (לא חובה לציור)
                    if baseline_head_y_global is not None:
                        depth_val = max(0.0, min(1.0, (baseline_head_y_global - head_y)))
                        self.depth_ema = _ema(self.depth_ema, depth_val, EMA_ALPHA_DEPTH)
                        self._update_series(self.depth_ema if self.depth_ema is not None else depth_val)

                # Overlay קליל
                if draw_overlay:
                    cv2.putText(frame, f"Reps: {self.rep_count}", (16, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                    if elbow_angle is not None:
                        cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (16, 68),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                # כתיבה – תמיד בגודל (out_w,out_h)
                if out is not None:
                    # frame כבר בגודל הנכון, כי ריסייזנו קודם
                    out.write(frame)

                # עדכוני "אחרון"
                if elbow_angle is not None:
                    self.last_elbow = elbow_angle
                if head_y is not None:
                    self.last_head_y = head_y

                # קפיצת פריימים אדפטיבית
                if step_now > 1:
                    cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cur + (step_now - 1))

        cap.release()
        if out is not None:
            out.release()

        # ניקוד טכניקה בסיסי (אפשר להחליף)
        base_score = 9.0
        penalties = 0.0
        if self.rep_count == 0:
            penalties += 3.0
        tech_score = max(4.0, base_score - penalties)
        tech_score = _round_score_half(tech_score)

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
# API ידידותי ל-Flask
# ===========================
def run_pullup_analysis(input_path, frame_skip=3, scale=0.4, output_path=None,
                        overlay=True, include_series=RETURN_SERIES):
    """
    שימוש:
      run_pullup_analysis(path, frame_skip=3, scale=0.4, output_path="out.mp4")
    מחזיר dict: {"video_path": ..., "rep_count": ..., "technique_score": ...}
    """
    # לייצב ת׳רדינג של OpenCV בקונטיינרים
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
    if output_path and "video_path" not in res:
        res["video_path"] = output_path
    return res


# ===========================
# הרצה ידנית מקומית
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

    out_path = args.output if args.output else None
    result = run_pullup_analysis(
        input_path=args.input,
        frame_skip=args.skip,
        scale=args.scale,
        output_path=out_path,
        overlay=not args.no_overlay,
        include_series=args.series
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
