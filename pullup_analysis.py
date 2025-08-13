# -*- coding: utf-8 -*-
# pullup_analysis.py — Fixed: no random seeking, fast like Bulgarian.
# Overlay תואם סקוואט, דונאט HEIGHT, RT feedback 0.8s, ספירה פשוטה (אלבו↓ + ראש↑), החזרה עם squat_count.

import os, math, subprocess
import cv2, numpy as np
from PIL import ImageFont, ImageDraw, Image

# ===================== STYLE / FONTS (כמו בסקוואט) =====================
BAR_BG_ALPHA         = 0.55
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
ASCENT_COLOR         = (40, 200, 80)
DONUT_RING_BG        = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE       = 28
FEEDBACK_FONT_SIZE   = 22
DEPTH_LABEL_FONT_SIZE= 14
DEPTH_PCT_FONT_SIZE  = 18

def _load_font(path, size):
    try: return ImageFont.truetype(path, size)
    except Exception: return ImageFont.load_default()

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

# ===================== BODY-ONLY SKELETON =====================
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
    _BODY_CONNECTIONS = tuple((a,b) for (a,b) in mp_pose.POSE_CONNECTIONS if a not in _FACE_LMS and b not in _FACE_LMS)
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

# ===================== OVERLAY (כמו סקוואט; כאן HEIGHT) =====================
def draw_ascent_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius, thickness = int(radius), int(thickness)
    cv2.circle(frame, (cx, cy), radius, DONUT_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90; end_ang = start_ang + int(360*pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang, ASCENT_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, ascent_pct=0.0):
    h, w, _ = frame.shape
    # Reps box (0,0)
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    pad_x, pad_y = 10, 6
    tw = draw.textlength(reps_text, font=REPS_FONT); th = REPS_FONT.size
    x0, y0 = 0, 0; x1, y1 = int(tw+2*pad_x), int(th+2*pad_y)
    top = frame.copy(); cv2.rectangle(top, (x0,y0), (x1,y1), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1-BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame); ImageDraw.Draw(pil).text((x0+pad_x, y0+pad_y-1), reps_text, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Donut ימין-עליון
    ref_h = max(int(h*0.06), int(REPS_FONT_SIZE*1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12; cx = w - margin - radius; cy = max(ref_h + radius//8, radius + thick//2 + 2)
    frame = draw_ascent_donut(frame, (cx, cy), radius, thick, float(np.clip(ascent_pct,0,1)))

    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    label_txt = "HEIGHT"; pct_txt = f"{int(float(np.clip(ascent_pct,0,1))*100)}%"
    gap = max(2, int(radius*0.10))
    base_y = cy - (DEPTH_LABEL_FONT_SIZE + gap + DEPTH_PCT_FONT_SIZE)//2
    lw = draw.textlength(label_txt, font=DEPTH_LABEL_FONT); pw = draw.textlength(pct_txt, font=DEPTH_PCT_FONT)
    draw.text((cx - int(lw//2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pw//2), base_y + DEPTH_LABEL_FONT_SIZE + gap), pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Feedback תחתון (עד 2 שורות)
    if feedback:
        def wrap2(draw, text, font, max_w):
            words = text.split(); lines=[]; cur=""
            for w_ in words:
                t = (cur+" "+w_).strip()
                if draw.textlength(t, font=font) <= max_w: cur=t
                else:
                    if cur: lines.append(cur); cur=w_
                if len(lines)==2: break
            if cur and len(lines)<2: lines.append(cur)
            leftover = len(words) - sum(len(l.split()) for l in lines)
            if leftover>0 and len(lines)>=2:
                last = lines[-1]+"…"
                while draw.textlength(last, font=font)>max_w and len(last)>1: last = last[:-2]+"…"
                lines[-1]=last
            return lines
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        safe = max(6, int(h*0.02)); padx, pady, gap = 12, 8, 4
        max_w = int(w - 2*padx - 20)
        lines = wrap2(draw_fb, feedback, FEEDBACK_FONT, max_w)
        line_h = FEEDBACK_FONT.size + 6
        block_h = 2*pady + len(lines)*line_h + (len(lines)-1)*gap
        y0 = max(0, h - safe - block_h); y1 = h - safe
        over = frame.copy(); cv2.rectangle(over, (0,y0), (w,y1), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1-BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + pady
        for ln in lines:
            tw = draw_fb.textlength(ln, font=FEEDBACK_FONT); tx = max(padx, (w-int(tw))//2)
            draw_fb.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255,255,255)); ty += line_h + gap
        frame = np.array(pil_fb)
    return frame

# ===================== ציונים =====================
def score_label(s):
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def display_half_str(x):
    q = round(float(x)*2)/2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

# ===================== ספירה פשוטה (אלבו↓ + ראש↑) =====================
ELBOW_START_THRESHOLD = 155.0
ELBOW_TOP_THRESHOLD   = 75.0
HEAD_MIN_ASCENT       = 0.03
HEAD_VEL_UP_THRESH    = 0.0015
HEAD_TOP_STICK_FRAMES = 1

RT_FB_HOLD_SEC        = 0.8

def _ang(a,b,c):
    ba = np.array([a[0]-b[0], a[1]-b[1]]); bc = np.array([c[0]-b[0], c[1]-b[1]])
    den = (np.linalg.norm(ba)*np.linalg.norm(bc))+1e-9
    cosang = float(np.clip(np.dot(ba, bc)/den, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

# ===================== MAIN =====================
def run_pullup_analysis(input_path,
                        frame_skip=3,
                        scale=0.4,
                        output_path="pullup_analyzed.mp4",
                        feedback_path="pullup_feedback.txt"):
    if not MP_OK:
        return {"squat_count":0,"technique_score":0.0,"technique_score_display":display_half_str(0.0),
                "technique_label":score_label(0.0),"good_reps":0,"bad_reps":0,"feedback":["Mediapipe not available"],
                "tips":[],"reps":[],"video_path":"","feedback_path":feedback_path}

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"squat_count":0,"technique_score":0.0,"technique_score_display":display_half_str(0.0),
                "technique_label":score_label(0.0),"good_reps":0,"bad_reps":0,"feedback":["Could not open video"],
                "tips":[],"reps":[],"video_path":"","feedback_path":feedback_path}

    # --- FPS אפקטיבי כמו בבולגרי ---
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    frame_no = 0
    counter = 0
    good_reps = 0
    bad_reps  = 0
    rep_reports = []
    all_scores = []

    in_rep = False
    rep_baseline_head_y = None
    min_head_y_in_rep = None
    rep_started_elbow = None
    seen_top_frames = 0

    baseline_head_y_global = None
    ascent_live = 0.0

    rt_fb_msg = None
    rt_fb_hold = 0
    RT_FB_HOLD_FRAMES = max(2, int(RT_FB_HOLD_SEC / dt))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        last_head_y = None
        last_elbow = None

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            frame_no += 1
            # --------- דילוג פריימים מהיר (בלי seek!) כמו בבולגרי ---------
            if frame_skip > 1 and (frame_no % frame_skip) != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
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
                R = mp_pose.PoseLandmark
                sh = lms[R.RIGHT_SHOULDER.value]; el = lms[R.RIGHT_ELBOW.value]; wr = lms[R.RIGHT_WRIST.value]
                nose = lms[R.NOSE.value]
                if min(nose.visibility, sh.visibility, el.visibility, wr.visibility) >= 0.35:
                    head_y = nose.y
                    elbow_angle = _ang((sh.x,sh.y),(el.x,el.y),(wr.x,wr.y))
                    if baseline_head_y_global is None:
                        baseline_head_y_global = head_y

            # מהירות ראש (שלילי=עולה)
            head_vel = 0.0
            if last_head_y is not None and head_y is not None:
                head_vel = head_y - last_head_y

            # -------- ספירת חזרות פשוטה --------
            if head_y is not None and elbow_angle is not None:
                if not in_rep and rep_baseline_head_y is None:
                    rep_baseline_head_y = head_y

                if not in_rep:
                    elbow_drop_ok = (elbow_angle < ELBOW_START_THRESHOLD) or \
                                    (last_elbow is not None and (last_elbow - elbow_angle) > 1.0)
                    head_speed_ok = (head_vel < -HEAD_VEL_UP_THRESH)
                    if elbow_drop_ok and head_speed_ok:
                        in_rep = True
                        min_head_y_in_rep = head_y
                        seen_top_frames = 0
                        rep_started_elbow = elbow_angle
                else:
                    min_head_y_in_rep = min(min_head_y_in_rep, head_y) if min_head_y_in_rep is not None else head_y
                    ascent_ok = (rep_baseline_head_y is not None and (rep_baseline_head_y - head_y) >= HEAD_MIN_ASCENT)
                    elbow_top_ok = (elbow_angle <= ELBOW_TOP_THRESHOLD)
                    if ascent_ok and elbow_top_ok:
                        seen_top_frames += 1
                    else:
                        seen_top_frames = 0

                    if seen_top_frames >= HEAD_TOP_STICK_FRAMES:
                        counter += 1
                        rep_reports.append({
                            "rep_index": counter,
                            "score": 10.0,
                            "score_display": display_half_str(10.0),
                            "feedback": [],
                            "tip": None,
                        })
                        all_scores.append(10.0); good_reps += 1
                        in_rep = False
                        rep_baseline_head_y = None
                        min_head_y_in_rep = None
                        rep_started_elbow = None
                        seen_top_frames = 0

                # RT feedback עדין
                cur_rt = None
                if in_rep and rep_baseline_head_y is not None and (rep_baseline_head_y - head_y) < HEAD_MIN_ASCENT*0.7:
                    cur_rt = "Go a bit higher (chin over bar)"
                elif not in_rep and elbow_angle is not None and elbow_angle >= 160.0:
                    cur_rt = "Nice reps — keep the rhythm"
                elif not in_rep and elbow_angle is not None and elbow_angle < 150.0:
                    cur_rt = "Fully extend your arms at the bottom"

                if cur_rt:
                    if rt_fb_msg != cur_rt:
                        rt_fb_msg = cur_rt; rt_fb_hold = RT_FB_HOLD_FRAMES
                    else:
                        rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
                else:
                    if rt_fb_hold > 0: rt_fb_hold -= 1
            else:
                if rt_fb_hold > 0: rt_fb_hold -= 1

            # --- דונאט HEIGHT לייב ---
            if baseline_head_y_global is not None and head_y is not None:
                raw = baseline_head_y_global - head_y
                ascent_live = float(np.clip(raw / max(0.12, HEAD_MIN_ASCENT*1.2), 0.0, 1.0))
            else:
                ascent_live = 0.0

            # ציור
            if lms is not None:
                frame = draw_body_only(frame, lms)
            frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), ascent_pct=ascent_live)
            out.write(frame)

            if head_y is not None: last_head_y = head_y
            if elbow_angle is not None: last_elbow = elbow_angle

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    # ציון כולל להצגה
    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    tip_for_session = "Slow down the lowering phase to maximize hypertrophy"
    feedback_list = ["Great form! Keep it up 💪"]

    # קובץ תקציר אופציונלי
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            f.write("Tip: " + tip_for_session + "\n")
    except Exception:
        pass

    # faststart encode
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg","-y","-i", output_path,
            "-c:v","libx264","-preset","fast","-movflags","+faststart","-pix_fmt","yuv420p",
            encoded_path
        ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final_video = encoded_path if os.path.exists(encoded_path) else output_path
    except Exception:
        final_video = output_path

    return {
        "squat_count": counter,
        "technique_score": technique_score,
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": feedback_list,
        "tips": [tip_for_session],
        "reps": rep_reports,
        "video_path": final_video,
        "feedback_path": feedback_path
    }

# תאימות
def run_analysis(*args, **kwargs):
    return run_pullup_analysis(*args, **kwargs)


