# -*- coding: utf-8 -*-
# deadlift_analysis.py — SIMPLE & ROBUST Deadlift rep counter (FSM + hinge dx)
# נקי מרעשים, בלי YOLO/קלמן. API תואם לגרסאות הקודמות.

import os, cv2, math, numpy as np, subprocess
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ---------------- UI / FONTS ----------------
BAR_BG_ALPHA         = 0.55
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)
DEPTH_RING_BG        = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE   = 18

def _font(path,size):
    try: return ImageFont.truetype(path,size)
    except: return ImageFont.load_default()

REPS_FONT        = _font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

# ---------------- Helpers ----------------
mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

def display_half_str(x: float) -> str:
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

def score_label(s: float) -> str:
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def _wrap2(draw, text, font, maxw):
    words = text.split()
    if not words: return [""]
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
        if len(lines) == 2: break
    if cur and len(lines) < 2: lines.append(cur)
    leftover = len(words) - sum(len(l.split()) for l in lines)
    if leftover > 0 and len(lines) >= 2:
        last = lines[-1] + "…"
        while draw.textlength(last, font=font) > maxw and len(last) > 1:
            last = last[:-2] + "…"
        lines[-1] = last
    return lines

def _donut(frame, c, r, t, p):
    p = float(np.clip(p, 0, 1))
    cx, cy = int(c[0]), int(c[1]); r = int(r); t = int(t)
    cv2.circle(frame, (cx,cy), r, DEPTH_RING_BG, t, cv2.LINE_AA)
    cv2.ellipse(frame, (cx,cy), (r,r), 0, -90, -90 + int(360*p), DEPTH_COLOR, t, cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, progress_pct=0.0):
    h, w, _ = frame.shape
    # Top-left reps box
    pil = Image.fromarray(frame); d = ImageDraw.Draw(pil)
    txt = f"Reps: {reps}"; padx, pady = 10, 6
    tw = d.textlength(txt, font=REPS_FONT); th = REPS_FONT.size
    x0, y0 = 0, 0; x1 = int(tw + 2*padx); y1 = int(th + 2*pady)
    top = frame.copy(); cv2.rectangle(top, (x0,y0), (x1,y1), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1-BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((x0+padx, y0+pady-1), txt, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Top-right donut (100% = נעול למעלה)
    ref_h = max(int(h*0.06), int(REPS_FONT_SIZE*1.6))
    r = int(ref_h * DONUT_RADIUS_SCALE)
    thick = max(3, int(r * DONUT_THICKNESS_FRAC))
    m = 12; cx = w - m - r; cy = max(ref_h + r//8, r + thick//2 + 2)
    frame = _donut(frame, (cx,cy), r, thick, float(np.clip(progress_pct,0,1)))

    pil = Image.fromarray(frame); d = ImageDraw.Draw(pil)
    label = "DEPTH"; pct = f"{int(float(np.clip(progress_pct,0,1))*100)}%"
    lw = d.textlength(label, font=DEPTH_LABEL_FONT); pw = d.textlength(pct, font=DEPTH_PCT_FONT)
    gap = max(2, int(r*0.10)); base = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size)//2
    d.text((cx-int(lw//2), base), label, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    d.text((cx-int(pw//2), base+DEPTH_LABEL_FONT.size+gap), pct, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Bottom feedback
    if feedback:
        pil_fb = Image.fromarray(frame); dfb = ImageDraw.Draw(pil_fb)
        safe = max(6, int(h*0.02)); pad_x, pad_y, lg = 12, 8, 4
        maxw = int(w - 2*pad_x - 20)
        lines = _wrap2(dfb, feedback, FEEDBACK_FONT, maxw)
        lh = FEEDBACK_FONT.size + 6
        block = (2*pad_y) + len(lines)*lh + (len(lines)-1)*lg
        y0 = max(0, h - safe - block); y1 = h - safe
        over = frame.copy(); cv2.rectangle(over, (0,y0), (w,y1), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1-BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame); dfb = ImageDraw.Draw(pil_fb); ty = y0 + pad_y
        for ln in lines:
            t_w = dfb.textlength(ln, font=FEEDBACK_FONT); tx = max(pad_x, (w - int(t_w)) // 2)
            dfb.text((tx,ty), ln, font=FEEDBACK_FONT, fill=(255,255,255)); ty += lh + lg
        frame = np.array(pil_fb)
    return frame

# ---------------- Rep FSM (simple & robust) ----------------
class HingeRepFSM:
    """
    States:
      'UPRIGHT' -> waiting at lockout (dx near top_ref)
      'DESCENT' -> dx rising
      'ASCENT'  -> dx falling towards lockout
      'LOCK_HOLD' -> short hold after lockout to avoid double-count
    """
    def __init__(self, dt, start_thr=0.09, end_thr=0.045, min_rom=0.055,
                 min_frames_between=8, lock_hold_frames=4,
                 ema_alpha=0.35, median_win=5, max_freeze=6):
        self.state = 'UPRIGHT'
        self.dt = float(dt)
        self.start_thr = float(start_thr)
        self.end_thr = float(end_thr)
        self.min_rom = float(min_rom)
        self.min_frames_between = int(min_frames_between)
        self.lock_hold_frames = int(lock_hold_frames)

        self.top_ref = 0.03  # יכויל בתחילת הסרטון
        self.bottom_dx = None
        self.rep_ready = False
        self.last_rep_end_frame = -999999

        self.median_win = int(median_win)
        self.mq = deque(maxlen=self.median_win)
        self.ema_alpha = float(ema_alpha)
        self.ema = None
        self.prev = None
        self.progress = 1.0  # 1=upright

        self.freeze_left = 0
        self.max_freeze = int(max_freeze)
        self.frame_idx = 0

        # per-rep timers
        self.down_frames = 0
        self.top_frames  = 0

        # outputs
        self.count = 0
        self.reps = []
        self.all_scores = []
        self.good = 0
        self.bad  = 0

    def _smooth(self, x):
        # median then EMA
        self.mq.append(x)
        med = float(np.median(self.mq))
        self.ema = med if self.ema is None else (self.ema_alpha*med + (1-self.ema_alpha)*self.ema)
        return self.ema

    def _progress(self, dx):
        # 1 at upright (top_ref), 0 at bottom_dx
        if self.bottom_dx is None or self.bottom_dx <= self.top_ref + 1e-6:
            return 1.0
        return float(np.clip(1.0 - (dx - self.top_ref) / (self.bottom_dx - self.top_ref + 1e-9), 0.0, 1.0))

    def calibrate_top(self, dx):
        # קח min רץ של dx בתחילת הסרטון כדי להגדיר top_ref (נעילה)
        self.top_ref = 0.9*self.top_ref + 0.1*dx

    def update(self, dx_raw, vis_ok, allow_count=True):
        self.frame_idx += 1

        if not vis_ok:
            # אין מדידות — תקפיא קצר
            if self.freeze_left < self.max_freeze:
                self.freeze_left += 1
            dx = self.prev if self.prev is not None else dx_raw
        else:
            self.freeze_left = 0
            dx = dx_raw

        # כיול ראשוני אם אנחנו בתחילת הסרטון/בנעילה
        if self.frame_idx < 15 or self.state == 'UPRIGHT':
            self.calibrate_top(dx)

        dx_s = self._smooth(dx)
        slope = 0.0 if self.prev is None else (dx_s - self.prev)
        self.prev = dx_s

        # עדכון progress
        if self.state in ('DESCENT', 'ASCENT'):
            if self.bottom_dx is None:
                self.bottom_dx = dx_s
            else:
                self.bottom_dx = max(self.bottom_dx, dx_s)
        self.progress = self._progress(dx_s)

        # FSM
        if self.state == 'UPRIGHT':
            self.bottom_dx = None
            self.top_frames += 1
            # יציאה לירידה רק אם חצינו start_thr מעל top_ref
            if (dx_s - self.top_ref) > self.start_thr:
                self.state = 'DESCENT'
                self.down_frames = 0
                self.top_frames = 0

        elif self.state == 'DESCENT':
            # סופרים זמן ירידה (ל-tip)
            if slope >  1e-4: self.down_frames += 1
            # אם התחלנו לעלות מספיק פריימים → ASCENT
            if slope < -1e-4 and (self.bottom_dx is not None and (self.bottom_dx - dx_s) > 1e-3):
                self.state = 'ASCENT'

        elif self.state == 'ASCENT':
            # תנאי נעילה: יורדים מתחת end_thr מעל top_ref + ROM מספיק
            rom_ok = (self.bottom_dx is not None) and ((self.bottom_dx - dx_s) >= self.min_rom)
            lock_ok = (dx_s - self.top_ref) <= self.end_thr
            if rom_ok and lock_ok:
                # מניעת דאבל־קאונט (מרחק פריימים)
                if allow_count and (self.frame_idx - self.last_rep_end_frame) > self.min_frames_between:
                    self.count += 1
                    # ניקוד פשוט: 10 בסיס, ענישה על ROM נמוך/ללא נעילה/זמן קצר בירידה או החזקה למעלה
                    penalty = 0.0
                    fb = []
                    if not rom_ok:
                        penalty += 1.0; fb.append("Hinge a bit deeper")
                    if (dx_s - self.top_ref) > (self.end_thr + 0.01):
                        penalty += 1.0; fb.append("Try to finish more upright")
                    # טיפים (לא מורידים ציון)
                    down_s = self.down_frames * self.dt
                    tip = None
                    if down_s < 0.35: tip = "Slow the lowering to ~2–3s for more hypertrophy"
                    # ציון
                    score = round(max(4.0, 10.0 - penalty)*2)/2
                    if score >= 9.5: self.good += 1
                    else: self.bad += 1
                    # אין פידבק? → מושלם
                    rep_fb = fb if score < 10.0 else []
                    self.reps.append({
                        "rep_index": self.count,
                        "score": float(score),
                        "score_display": display_half_str(score),
                        "feedback": rep_fb,
                        "tip": tip or "Keep the bar close and move smoothly"
                    })
                    self.all_scores.append(score)

                # נועל top_ref רק בלוקאאוט (שומר יציבות)
                self.top_ref = 0.9*self.top_ref + 0.1*dx_s
                self.last_rep_end_frame = self.frame_idx
                self.state = 'LOCK_HOLD'
                self.top_frames = 0

        elif self.state == 'LOCK_HOLD':
            self.top_frames += 1
            if self.top_frames >= self.lock_hold_frames:
                self.state = 'UPRIGHT'

        return dx_s, self.progress

# ---------------- Run analysis ----------------
def run_deadlift_analysis(video_path,
                          frame_skip=2,
                          scale=0.5,
                          output_path="deadlift_analyzed.mp4",
                          feedback_path="deadlift_feedback.txt",
                          detector_onnx_path=None,      # נשמר לשמירת API; לא בשימוש
                          detector_input_size=640,
                          detector_conf_thres=0.25,
                          detector_iou_thres=0.45,
                          detector_class_ids=None,
                          return_video=True,
                          fast_mode=None,
                          side_preference='auto'  # 'auto'|'right'|'left'
                          ):
    """
    מינימליסטי, מדויק ועמיד.
    - fast_mode=True == return_video=False
    - מחזיר שדות זהים לקודם (כולל squat_count)
    """
    if fast_mode is True:
        return_video = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    fsm = HingeRepFSM(
        dt=dt,
        start_thr=0.09,  # נדרש לעזוב נעילה
        end_thr=0.045,   # להיכנס לנעילה
        min_rom=0.055,   # עומק מינימלי לספירה
        min_frames_between= max(6, int(0.18 / dt)),
        lock_hold_frames= max(3, int(0.12 / dt)),
        ema_alpha=0.35,
        median_win=5,
        max_freeze=6
    )

    def _vis(lm, idx): 
        try: return float(lm[idx].visibility)
        except: return 0.0
    def _pt(lm, idx): 
        return np.array([lm[idx].x, lm[idx].y], dtype=float)

    def pick_side(lm, pref):
        if pref in ('left','right'):
            return pref
        vL = _vis(lm, PL.LEFT_SHOULDER.value) + _vis(lm, PL.LEFT_HIP.value)
        vR = _vis(lm, PL.RIGHT_SHOULDER.value) + _vis(lm, PL.RIGHT_HIP.value)
        return 'left' if vL >= vR else 'right'

    with mp_pose.Pose(model_complexity=2,  # דיוק גבוה
                      smooth_landmarks=True,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue

            work = cv2.resize(frame, (0,0), fx=scale, fy=scale) if scale != 1.0 else frame
            if return_video and out is None:
                h0, w0 = work.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w0, h0))

            rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            feedback_rt = None
            progress = fsm.progress

            if not res.pose_landmarks:
                # אין שלד — שומר פרוגרס קיים
                if return_video:
                    out.write(draw_overlay(work.copy(), reps=fsm.count, feedback=None, progress_pct=progress))
                continue

            lm = res.pose_landmarks.landmark
            side = pick_side(lm, side_preference)
            if side == 'right':
                s_idx, h_idx = PL.RIGHT_SHOULDER.value, PL.RIGHT_HIP.value
            else:
                s_idx, h_idx = PL.LEFT_SHOULDER.value, PL.LEFT_HIP.value

            s_vis, h_vis = _vis(lm, s_idx), _vis(lm, h_idx)
            vis_ok = (s_vis > 0.45 and h_vis > 0.45)

            s = _pt(lm, s_idx)
            h = _pt(lm, h_idx)
            dx_raw = abs(h[0] - s[0])

            _, progress = fsm.update(dx_raw, vis_ok, allow_count=True)

            # ציור
            if return_video:
                # נצייר רק “שלד מזערי” עבור הכתף/ירך והדונאט
                draw = work.copy()
                hh, ww = draw.shape[:2]
                sx, sy = int(s[0]*ww), int(s[1]*hh)
                hx, hy = int(h[0]*ww), int(h[1]*hh)
                cv2.circle(draw, (sx,sy), 4, (255,255,255), -1, cv2.LINE_AA)
                cv2.circle(draw, (hx,hy), 4, (255,255,255), -1, cv2.LINE_AA)
                cv2.line(draw, (sx,sy), (hx,hy), (255,255,255), 2, cv2.LINE_AA)
                draw = draw_overlay(draw, reps=fsm.count, feedback=feedback_rt, progress_pct=progress)
                out.write(draw)

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    # ציונים וסיכום
    avg = float(np.mean(fsm.all_scores)) if fsm.all_scores else 10.0  # אם אין פידבקים/חזרות – נניח טוב
    technique_score = round(round(avg * 2) / 2, 2)
    # אוספים רק בעיות ברמת אימון (אם היו)
    seen = set(); feedback_list = []
    for r in fsm.reps:
        if float(r.get("score") or 0.0) >= 10.0: continue
        for msg in (r.get("feedback") or []):
            if msg and msg not in seen:
                seen.add(msg); feedback_list.append(msg)
    # אם לא נאסף שום פידבק → מושלם
    if not feedback_list:
        feedback_list = ["Great form, keep it up"]

    final_video_path = ""
    if return_video:
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run([
                "ffmpeg","-y","-i", output_path,
                "-c:v","libx264","-preset","fast","-movflags","+faststart","-pix_fmt","yuv420p",
                encoded_path
            ], check=False)
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                os.remove(output_path)
        except Exception:
            pass
        final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")

    return {
        "squat_count": fsm.count,                    # שמרתי את המפתח שקיים אצלך
        "technique_score": technique_score,
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": fsm.good,
        "bad_reps": fsm.bad,
        "feedback": feedback_list,                  # אם אין בעיות → ["Great form, keep it up"]
        "reps": fsm.reps,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }
