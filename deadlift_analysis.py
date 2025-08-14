# -*- coding: utf-8 -*-
# deadlift_analysis.py — דדליפט מיושר סטנדרט + דונאט דו־כיווני + החלקת לנדמרקים (EMA + hold-last)
import os
import cv2
import math
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / FONTS (כמו בסקוואט) =====================
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

def _load_font(path, size):
    try:    return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

REPS_FONT        = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

mp_pose = mp.solutions.pose

# ===================== ציונים — אותו סטנדרט =====================
def score_label(s):
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    if abs(q - round(q)) < 1e-9: return str(int(round(q)))
    return f"{q:.1f}"

# ===================== שלד גוף בלבד (ללא פנים) =====================
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

def draw_body_only_from_dict(frame, pts_norm, color=(255,255,255)):
    """pts_norm: dict[idx] = (x_norm, y_norm) — מצייר רק חיבורים/נקודות שקיימים במפה."""
    h, w = frame.shape[:2]
    for a, b in _BODY_CONNECTIONS:
        if a in pts_norm and b in pts_norm:
            ax, ay = int(pts_norm[a][0] * w), int(pts_norm[a][1] * h)
            bx, by = int(pts_norm[b][0] * w), int(pts_norm[b][1] * h)
            cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    for i in _BODY_POINTS:
        if i in pts_norm:
            x, y = int(pts_norm[i][0] * w), int(pts_norm[i][1] * h)
            cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
    return frame

# ===================== גיאומטריה =====================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ===================== פרמטרים לדדליפט =====================
HINGE_START_THRESH = 0.08      # התחלת חזרה (delta_x)
STAND_DELTA_TARGET = 0.025     # דלתא אופקית בעמידה (קטנה)
END_THRESH         = 0.035     # סיום חזרה (חזרה לעמידה)
MIN_FRAMES_BETWEEN_REPS = 10

# גב
BACK_ROUND_CURV_THR = 0.04

# דונאט דו־כיווני (קרבה לעמידה)
PROG_ALPHA = 0.35

# ===================== החלקת לנדמרקים =====================
SMOOTH_ALPHA   = 0.55      # EMA — משקל לערך הנוכחי
VIS_THRESH     = 0.50      # אם visibility מתחת — מתעלמים מהתצפית
MAX_JUMP_NORM  = 0.06      # "קפיצה" מקס' (נורמליזציה על אלכסון הפריים)

class LandmarkSmoother:
    def __init__(self, alpha=SMOOTH_ALPHA, vis_thr=VIS_THRESH, max_jump=MAX_JUMP_NORM):
        self.alpha = float(alpha)
        self.vis_thr = float(vis_thr)
        self.max_jump = float(max_jump)
        self.prev = {}  # idx -> (x,y)

    @staticmethod
    def _dist_norm(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])  # כבר בנורמליזציה [0..1]

    def get(self, lm_list, idx):
        """מחזיר נקודה מוחלקת/מחוזקת: אם אין/חלש/קופץ חזק → מחזיק ערך אחרון."""
        p = lm_list[idx]
        has_prev = idx in self.prev
        if (p.visibility or 0.0) < self.vis_thr:
            # לא יציב → חזור לערך קודם אם יש
            return self.prev.get(idx, None)

        cur = (float(p.x), float(p.y))
        if has_prev:
            if self._dist_norm(cur, self.prev[idx]) > self.max_jump:
                # קפיצה לא סבירה (למשל ברבל מסתיר) → החזק הקודם
                return self.prev[idx]
            # EMA
            sm = (self.alpha * cur[0] + (1-self.alpha) * self.prev[idx][0],
                  self.alpha * cur[1] + (1-self.alpha) * self.prev[idx][1])
            self.prev[idx] = sm
            return sm
        else:
            self.prev[idx] = cur
            return cur

# ===================== Overlay (כמו בסקוואט; “DEPTH” נשאר לאחידות) =====================
def _wrap_to_two_lines(draw, text, font, max_width):
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

def _draw_depth_donut(frame, center, radius, thickness, pct):
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

def draw_overlay(frame, reps=0, feedback=None, progress_pct=0.0):
    h, w, _ = frame.shape
    # Reps box
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    inner_pad_x, inner_pad_y = 10, 6
    text_w = draw.textlength(reps_text, font=REPS_FONT)
    text_h = REPS_FONT.size
    x0, y0 = 0, 0
    x1 = int(text_w + 2*inner_pad_x); y1 = int(text_h + 2*inner_pad_y)
    top = frame.copy()
    cv2.rectangle(top, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((x0 + inner_pad_x, y0 + inner_pad_y - 1),
                             reps_text, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Donut (ימין־עליון) — "קרבה לעמידה" (100% = עמידה)
    ref_h = max(int(h * 0.06), int(REPS_FONT_SIZE * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)
    frame = _draw_depth_donut(frame, (cx, cy), radius, thick, float(np.clip(progress_pct,0,1)))

    pil  = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"; pct_txt = f"{int(float(np.clip(progress_pct,0,1))*100)}%"
    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt,   font=DEPTH_PCT_FONT)
    gap     = max(2, int(radius * 0.10))
    base_y  = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size) // 2
    draw.text((cx - int(label_w // 2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pct_w // 2), base_y + DEPTH_LABEL_FONT.size + gap),
              pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Bottom feedback (עד 2 שורות)
    if feedback:
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        safe_margin = max(6, int(h * 0.02))
        pad_x, pad_y, line_gap = 12, 8, 4
        max_text_w = int(w - 2*pad_x - 20)
        lines = _wrap_to_two_lines(draw_fb, feedback, FEEDBACK_FONT, max_text_w)
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

# ===================== עזר לעקמומיות גב =====================
def analyze_back_curvature(shoulder, hip, head_like, threshold=BACK_ROUND_CURV_THR):
    line_vec = hip - shoulder
    nrm = np.linalg.norm(line_vec) + 1e-9
    line_unit = line_vec / nrm
    proj_len = np.dot(head_like - shoulder, line_unit)
    proj_point = shoulder + proj_len * line_unit
    offset_vec = head_like - proj_point
    direction_sign = np.sign(offset_vec[1]) * -1  # inward = negative
    signed_curv = float(direction_sign * np.linalg.norm(offset_vec))
    is_rounded = signed_curv < -threshold
    return signed_curv, is_rounded

# ===================== MAIN =====================
def run_deadlift_analysis(video_path,
                          frame_skip=3,
                          scale=0.4,
                          output_path="deadlift_analyzed.mp4",
                          feedback_path="deadlift_feedback.txt"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    mp_pose_mod = mp.solutions.pose
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    counter = good_reps = bad_reps = 0
    all_scores, reps_report, overall_feedback = [], [], []

    # מצב רפ
    rep_in_progress = False
    last_rep_frame = -999
    frame_idx = 0

    # דונאט — קרבה לעמידה דו־כיווני
    top_ref = STAND_DELTA_TARGET
    bottom_est = None
    progress_ema = 1.0

    # גב: דרישה למינימום פריימים לפני התרעה
    BACK_WARN_FRAMES_MIN = max(2, int(0.25 / dt))
    back_warn_frames = 0

    smoother = LandmarkSmoother()

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

            rt_feedback = None
            if not results.pose_landmarks:
                # אין שלד — שמור החלקה קיימת, אל תקפוץ
                frame = draw_overlay(frame, reps=counter, feedback=None, progress_pct=progress_ema)
                out.write(frame); continue

            try:
                lm = results.pose_landmarks.landmark
                R = mp_pose_mod.PoseLandmark

                # --- נקודות מוחלקות (EMA + hold-last) ---
                idxs = {
                    "RS": R.RIGHT_SHOULDER.value,
                    "RH": R.RIGHT_HIP.value,
                    "RK": R.RIGHT_KNEE.value,
                    "RA": R.RIGHT_ANKLE.value,
                    "LE": R.LEFT_EAR.value,
                    "RE": R.RIGHT_EAR.value,
                    "NO": R.NOSE.value,
                }
                pts = {k: smoother.get(lm, idxs[k]) for k in idxs}
                # בחר "ראש" מהנקודות עם ערך קיים
                head_point = None
                for k in ("RE","LE","NO"):
                    if pts[k] is not None:
                        head_point = np.array(pts[k], dtype=float)
                        break
                if (pts["RS"] is None) or (pts["RH"] is None) or (head_point is None):
                    frame = draw_overlay(frame, reps=counter, feedback=None, progress_pct=progress_ema)
                    out.write(frame); continue

                shoulder = np.array(pts["RS"])
                hip      = np.array(pts["RH"])
                # לברך/קרסול ניקח מוחלק — ואם חסר, לא נכשיל את הלוגיקה (דדליפט: רגליים כמעט קבועות)
                knee     = np.array(pts["RK"]) if pts["RK"] is not None else None
                ankle    = np.array(pts["RA"]) if pts["RA"] is not None else None

                # פרוקסי עומק הינג’ — דלתא אופקית כתף-ירך
                delta_x = abs(hip[0] - shoulder[0])

                # עדכון top_ref כשברור שמדובר בעמידה
                if delta_x < (STAND_DELTA_TARGET * 1.4):
                    top_ref = 0.9*top_ref + 0.1*delta_x

                # עקמומיות גב (עם נק’ מוחלקות)
                mid_spine = (shoulder + hip) * 0.5 * 0.4 + head_point * 0.6
                _, is_rounded = analyze_back_curvature(shoulder, hip, mid_spine)
                if is_rounded: back_warn_frames += 1
                else:          back_warn_frames = max(0, back_warn_frames - 1)

                # התחלת חזרה
                if (not rep_in_progress) and (delta_x > HINGE_START_THRESH) and (frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS):
                    rep_in_progress = True
                    bottom_est = delta_x
                    back_warn_frames = 0

                # תוך כדי חזרה — עדכון תחתית דינמית + קרבה לעמידה דו־כיוונית
                if rep_in_progress:
                    bottom_est = max(bottom_est or delta_x, delta_x)
                    denom = max(1e-4, (bottom_est - top_ref))
                    progress_raw = 1.0 - ((delta_x - top_ref) / denom)  # 1=עמידה, 0=תחתית
                    progress_raw = float(np.clip(progress_raw, 0.0, 1.0))
                    progress_ema = PROG_ALPHA*progress_raw + (1-PROG_ALPHA)*progress_ema

                    if is_rounded:
                        rt_feedback = "Keep your back straighter"
                else:
                    # בין חזרות — חזרה עדינה ל-100%
                    progress_ema = PROG_ALPHA*1.0 + (1-PROG_ALPHA)*progress_ema

                # סיום חזרה
                if rep_in_progress and (delta_x < END_THRESH):
                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        penalty = 0.0
                        fb = []
                        if delta_x > (top_ref + 0.02):
                            fb.append("Try to finish more upright"); penalty += 1.0
                        if back_warn_frames >= max(2, int(0.25/dt)):
                            fb.append("Try to keep your back a bit straighter"); penalty += 1.5

                        score = round(max(4, 10 - penalty) * 2) / 2
                        if fb:
                            for f in fb:
                                if f not in overall_feedback: overall_feedback.append(f)

                        moved_enough = (bottom_est - delta_x) > 0.05
                        if moved_enough:
                            counter += 1
                            if score >= 9.5: good_reps += 1
                            else: bad_reps += 1
                            all_scores.append(score)
                            reps_report.append({
                                "rep_index": counter,
                                "score": float(score),
                                "score_display": display_half_str(score),
                                "feedback": fb,
                                "tip": None
                            })

                        last_rep_frame = frame_idx

                    # reset
                    rep_in_progress = False
                    bottom_est = None
                    back_warn_frames = 0

                # ציור שלד: רק מה שיש לנו במפה המוחלקת
                # נרכיב dict של אינדקסים מוחלקים כדי לצייר גוף
                sm_dict = {}
                for idx in _BODY_POINTS:
                    p = smoother.prev.get(idx, None)
                    if p is not None:
                        sm_dict[idx] = p
                frame = draw_body_only_from_dict(frame, sm_dict)
                frame = draw_overlay(frame, reps=counter, feedback=rt_feedback, progress_pct=progress_ema)
                out.write(frame)

            except Exception:
                frame = draw_overlay(frame, reps=counter, feedback=None, progress_pct=progress_ema)
                if out is not None: out.write(frame)
                continue

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    # סיכום
    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    feedback_list = overall_feedback[:] if overall_feedback else ["Great form! Keep your spine neutral and hinge smoothly. 💪"]

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list: f.write(f"- {fb}\n")
    except Exception:
        pass

    # קידוד H.264 faststart
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
        "squat_count": counter,                     # תאימות ל-UI שלך
        "technique_score": technique_score,
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": feedback_list,
        "reps": reps_report,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }


