# -*- coding: utf-8 -*-
# pullup_analysis.py — Simple & robust pull-up rep counter (FSM + nose/chin y)
# - Supports return_video / fast_mode / output_path=None safely
# - Median+EMA smoothing, short freeze when landmarks missing
# - API-compatible dict: squat_count, technique_score, feedback, reps, video_path, feedback_path

import os, cv2, numpy as np, subprocess
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
    # Top-left reps
    pil = Image.fromarray(frame); d = ImageDraw.Draw(pil)
    txt = f"Reps: {reps}"; padx, pady = 10, 6
    tw = d.textlength(txt, font=REPS_FONT); th = REPS_FONT.size
    x0, y0 = 0, 0; x1 = int(tw + 2*padx); y1 = int(th + 2*pady)
    top = frame.copy(); cv2.rectangle(top, (x0,y0), (x1,y1), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1-BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((x0+padx, y0+pady-1), txt, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Top-right donut: 0=בתחתית, 100%=סנטר גבוה (למעלה)
    ref_h = max(int(h*0.06), int(REPS_FONT_SIZE*1.6))
    r = int(ref_h * DONUT_RADIUS_SCALE)
    thick = max(3, int(r * DONUT_THICKNESS_FRAC))
    m = 12; cx = w - m - r; cy = max(ref_h + r//8, r + thick//2 + 2)
    frame = _donut(frame, (cx,cy), r, thick, float(np.clip(progress_pct,0,1)))

    pil = Image.fromarray(frame); d = ImageDraw.Draw(pil)
    label = "HEIGHT"; pct = f"{int(float(np.clip(progress_pct,0,1))*100)}%"
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

# ---------------- Pull-up FSM ----------------
class PullupFSM:
    """
    States:
      'BOTTOM'  -> תלייה (y קרוב לתחתית; y גדול כי ציר Y למטה)
      'ASCENT'  -> עלייה (y קטן)
      'TOP_HOLD'-> נעילה קצרה למעלה
      'DESCENT' -> ירידה (y גדל)
    מדד תנועה: y של NOSE (או CHIN אם קיים), עם סינון median+EMA.
    """

    def __init__(self, dt,
                 start_thr=0.06,     # כמה צריך לעלות מעל תחתית כדי להיחשב התחלת עלייה
                 top_thr=0.035,      # קרוב מספיק ל"מעלה"
                 min_rom=0.09,       # תנועה מינימלית בין תחתית לשיא
                 min_frames_between=8,
                 top_hold_frames=3,
                 median_win=5, ema_alpha=0.35,
                 max_freeze=6):
        self.dt = float(dt)
        self.state = 'BOTTOM'

        self.start_thr = float(start_thr)
        self.top_thr   = float(top_thr)
        self.min_rom   = float(min_rom)
        self.min_frames_between = int(min_frames_between)
        self.top_hold_frames = int(top_hold_frames)

        self.median_win = int(median_win)
        self.ema_alpha  = float(ema_alpha)
        self.max_freeze = int(max_freeze)

        self.mq = deque(maxlen=self.median_win)
        self.ema = None
        self.prev = None
        self.freeze_left = 0

        self.frame_idx = 0
        self.bottom_ref = None   # y בתחתית (גדול)
        self.top_ref    = None   # y בשיא (קטן)
        self.rep_bottom = None   # y תחתית חזרה נוכחית
        self.rep_top    = None   # y שיא חזרה נוכחית

        self.top_frames = 0
        self.last_rep_end_frame = -999999

        self.count = 0
        self.reps = []
        self.all_scores = []
        self.good = 0
        self.bad  = 0
        self.progress = 0.0  # 0 בתחתית; 1 בשיא

    def _smooth(self, y_raw):
        self.mq.append(float(y_raw))
        med = float(np.median(self.mq))
        self.ema = med if self.ema is None else (self.ema_alpha*med + (1-self.ema_alpha)*self.ema)
        return self.ema

    def _progress_from(self, y):
        # יחסית ל-bottom_ref/top_ref (y נמוך = גבוה פיזית)
        if self.rep_bottom is None or self.rep_top is None:
            return 0.0
        denom = max(1e-5, self.rep_bottom - self.rep_top)
        p = (self.rep_bottom - y) / denom
        return float(np.clip(p, 0.0, 1.0))

    def update(self, y_raw, vis_ok, allow_count=True):
        self.frame_idx += 1

        if not vis_ok:
            if self.freeze_left < self.max_freeze:
                self.freeze_left += 1
            y = self.prev if self.prev is not None else y_raw
        else:
            self.freeze_left = 0
            y = y_raw

        y_s = self._smooth(y)
        slope = 0.0 if self.prev is None else (y_s - self.prev)  # חיובי = יורד למטה, שלילי = עולה למעלה
        self.prev = y_s

        # כיול ראשוני של bottom_ref/top_ref בפריימים הראשונים
        if self.frame_idx < 15:
            self.bottom_ref = y_s if (self.bottom_ref is None) else max(self.bottom_ref, y_s)
            self.top_ref    = y_s if (self.top_ref    is None) else min(self.top_ref,    y_s)

        if self.state == 'BOTTOM':
            # עדכון תחתית כללית וגם תחתית חזרה נוכחית
            self.bottom_ref = y_s if (self.bottom_ref is None) else max(self.bottom_ref, y_s)
            self.rep_bottom = y_s if (self.rep_bottom is None) else max(self.rep_bottom, y_s)
            self.rep_top = None
            # התחלת עלייה כשיצאנו מספיק מהתחתית
            if (self.rep_bottom - y_s) > self.start_thr:
                self.state = 'ASCENT'

        elif self.state == 'ASCENT':
            # מעדכון שיא (y קטן)
            self.rep_top = y_s if (self.rep_top is None) else min(self.rep_top, y_s)
            # אם עברנו מספיק קרוב ל"מעלה" (top_thr מהשיא או מה-top_ref) → TOP_HOLD
            top_target = (self.rep_top if self.rep_top is not None else y_s) + self.top_thr
            if y_s <= top_target:
                self.state = 'TOP_HOLD'
                self.top_frames = 0

        elif self.state == 'TOP_HOLD':
            self.top_frames += 1
            self.rep_top = y_s if (self.rep_top is None) else min(self.rep_top, y_s)
            # ננעלנו מספיק זמן → ירידה
            if self.top_frames >= self.top_hold_frames:
                self.state = 'DESCENT'

        elif self.state == 'DESCENT':
            # מחכים שנחזור קרוב לתחתית (תנודת y כלפי מטה)
            # תנאי ספירה: ROM מספיק + מניעת דאבל קאונט + קרוב לתחתית
            rom_ok = (self.rep_bottom is not None and self.rep_top is not None and
                      (self.rep_bottom - self.rep_top) >= self.min_rom)
            near_bottom = (y_s >= (self.rep_bottom - self.start_thr/2.0))

            if rom_ok and near_bottom:
                if allow_count and (self.frame_idx - self.last_rep_end_frame) > self.min_frames_between:
                    self.count += 1
                    # ניקוד פשוט: 10 אם ROM טוב; 9.5 אם גבולי; פחות אם לא חזר ממש לתחתית
                    fb = []
                    rom = (self.rep_bottom - self.rep_top)
                    penalty = 0.0
                    if rom < (self.min_rom + 0.01):
                        penalty += 0.5; fb.append("Increase range of motion")
                    if not near_bottom:
                        penalty += 0.5; fb.append("Return fully to dead-hang")

                    score = round(max(4.0, 10.0 - penalty)*2)/2
                    if score >= 9.5: self.good += 1
                    else: self.bad += 1

                    self.reps.append({
                        "rep_index": self.count,
                        "score": float(score),
                        "score_display": display_half_str(score),
                        "feedback": fb if score < 10.0 else [],
                        "tip": "Drive elbows down, chest to bar"
                    })
                    self.all_scores.append(score)

                self.last_rep_end_frame = self.frame_idx
                # אתחול לחזרה הבאה
                self.state = 'BOTTOM'
                self.rep_bottom = None
                self.rep_top = None

        # progress מתוך תחתית/שיא של החזרה הנוכחית
        # אם לא הוגדרו—נחשב מול bottom_ref/top_ref
        if self.rep_bottom is None or self.rep_top is None:
            rb = self.bottom_ref if self.bottom_ref is not None else y_s + 0.001
            rt = self.top_ref if self.top_ref is not None else y_s - 0.001
            denom = max(1e-5, rb - rt)
            self.progress = float(np.clip((rb - y_s)/denom, 0.0, 1.0))
        else:
            self.progress = self._progress_from(y_s)

        return y_s, self.progress

# ---------------- Run analysis ----------------
def run_pullup_analysis(video_path,
                        frame_skip=2,
                        scale=0.5,
                        output_path="pullup_analyzed.mp4",
                        feedback_path="pullup_feedback.txt",
                        return_video=True,
                        fast_mode=None):
    """
    - fast_mode=True => return_video=False (לא נוצרת כתיבה או קידוד)
    - אם output_path הוא None או return_video=False, לא ניצור VideoWriter בכלל.
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
    write_video = (return_video is True) and bool(output_path)
    out = None

    fsm = PullupFSM(
        dt=dt,
        start_thr=0.06,
        top_thr=0.035,
        min_rom=0.09,
        min_frames_between=max(6, int(0.18 / dt)),
        top_hold_frames=max(3, int(0.10 / dt)),
        median_win=5, ema_alpha=0.35, max_freeze=6
    )

    mp_pose = mp.solutions.pose
    PL = mp_pose.PoseLandmark

    def _vis(lm, idx): 
        try: return float(lm[idx].visibility)
        except: return 0.0
    def _pt(lm, idx): 
        return np.array([lm[idx].x, lm[idx].y], dtype=float)

    with mp_pose.Pose(model_complexity=2,
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

            if write_video and out is None:
                h0, w0 = work.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w0, h0))

            rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            progress = fsm.progress
            if not res.pose_landmarks:
                if write_video:
                    out.write(draw_overlay(work.copy(), reps=fsm.count, feedback=None, progress_pct=progress))
                continue

            lm = res.pose_landmarks.landmark

            # בוחרים "פנים" ל־y: Nose כמחדל; אם אין—סנטר (לא קיים ב-MP) → ניקח Mouth/Chin proxy
            nose_v = _vis(lm, PL.NOSE.value)
            if nose_v > 0.4:
                y_raw = _pt(lm, PL.NOSE.value)[1]
            else:
                # fallback: ממוצע פה-אוזניים אם יש
                ys = []
                for idx in (PL.MOUTH_LEFT.value, PL.MOUTH_RIGHT.value, PL.LEFT_EAR.value, PL.RIGHT_EAR.value):
                    if _vis(lm, idx) > 0.5: ys.append(_pt(lm, idx)[1])
                y_raw = float(np.median(ys)) if ys else _pt(lm, PL.LEFT_SHOULDER.value)[1]

            vis_ok = (nose_v > 0.4) or (len([1 for idx in (PL.MOUTH_LEFT.value, PL.MOUTH_RIGHT.value, PL.LEFT_EAR.value, PL.RIGHT_EAR.value) if _vis(lm, idx) > 0.5]) >= 2)

            _, progress = fsm.update(y_raw, vis_ok, allow_count=True)

            if write_video:
                # ציור מינימלי: נקודת פנים וקו ויזואלי לגובה
                draw = work.copy()
                hh, ww = draw.shape[:2]
                ypix = int(y_raw * hh)
                cv2.line(draw, (0, ypix), (ww, ypix), (255,255,255), 1, cv2.LINE_AA)
                cv2.circle(draw, (ww//2, ypix), 4, (255,255,255), -1, cv2.LINE_AA)
                draw = draw_overlay(draw, reps=fsm.count, feedback=None, progress_pct=progress)
                out.write(draw)

    cap.release()
    if write_video and out is not None:
        out.release()
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()

    # ציונים וסיכום
    avg = float(np.mean(fsm.all_scores)) if fsm.all_scores else 10.0
    technique_score = round(round(avg * 2) / 2, 2)

    # איסוף פידבק מצומצם (רק בעיות)
    seen = set(); feedback_list = []
    for r in fsm.reps:
        if float(r.get("score") or 0.0) >= 10.0: continue
        for msg in (r.get("feedback") or []):
            if msg and msg not in seen:
                seen.add(msg); feedback_list.append(msg)
    if not feedback_list:
        feedback_list = ["Great form, keep it up"]

    final_video_path = ""
    if write_video and output_path:
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
        "squat_count": fsm.count,
        "technique_score": technique_score,
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": fsm.good,
        "bad_reps": fsm.bad,
        "feedback": feedback_list,
        "reps": fsm.reps,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }

