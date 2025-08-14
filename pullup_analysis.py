# -*- coding: utf-8 -*-
# Pull-ups — squat-style visuals, fast overlay, on-bar/off-bar gating, AND-counting (head ascent + elbow ≤ TOP_ANGLE).
# Exact look & feel like squat (fonts, donut, skeleton). Feedback on-screen is mirrored in returned feedback + txt file.

import os, cv2, math, numpy as np, subprocess
from PIL import ImageFont, ImageDraw, Image

# ===================== STYLE / FONTS (like squat) =====================
BAR_BG_ALPHA         = 0.55
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)   # BGR
DEPTH_RING_BG        = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE   = 18

def _load_font(path, size):
    try: return ImageFont.truetype(path, size)
    except Exception: return ImageFont.load_default()

REPS_FONT        = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

# ===================== MediaPipe =====================
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None

def score_label(s):
    s=float(s)
    if s>=9.5: return "Excellent"
    if s>=8.5: return "Very good"
    if s>=7.0: return "Good"
    if s>=5.5: return "Fair"
    return "Needs work"

def display_half_str(x):
    q=round(float(x)*2)/2.0
    return str(int(round(q))) if abs(q-round(q))<1e-9 else f"{q:.1f}"

# ===================== BODY-ONLY skeleton (like squat) =====================
_FACE_LMS = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS = tuple()
if mp_pose:
    _FACE_LMS = {
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
        mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
    }
    _BODY_CONNECTIONS = tuple((a,b) for (a,b) in mp_pose.POSE_CONNECTIONS if a not in _FACE_LMS and b not in _FACE_LMS)
    _BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

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

# ===================== Overlay (single PIL pass) =====================
def draw_overlay(frame, reps=0, feedback=None, height_pct=0.0):
    """Draws rectangles & donut with OpenCV, then texts in ONE PIL pass (perf)."""
    h, w, _ = frame.shape
    # --- OpenCV shapes ---
    # Refs
    ref_h = max(int(h*0.06), int(REPS_FONT_SIZE*1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12; cx = w - margin - radius; cy = max(ref_h + radius//8, radius + thick//2 + 2)
    # Reps bar bg
    (tw, th), _ = cv2.getTextSize("X"*8, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)  # rough pad ref (unused for text)
    pad_x, pad_y = 10, 6
    # Fixed-size bar by measured text width in PIL below; here just paint big enough area:
    cv2.rectangle(frame, (0,0), (int(w*0.25), int(REPS_FONT_SIZE + 2*pad_y + 8)), (0,0,0), -1)
    frame = cv2.addWeighted(frame, 1.0, frame, 0.0, 0)  # noop to keep path similar
    # Donut ring
    pct = float(np.clip(height_pct,0,1))
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thick, lineType=cv2.LINE_AA)
    start_ang = -90; end_ang = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang, DEPTH_COLOR, thick, lineType=cv2.LINE_AA)

    # --- ONE PIL pass for texts ---
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    # Reps
    reps_text = f"Reps: {reps}"
    tw = draw.textlength(reps_text, font=REPS_FONT); th = REPS_FONT.size
    top = np.array(pil)  # to draw exact bg to fit text
    cv2.rectangle(top, (0,0), (int(tw + 2*pad_x), int(th + 2*pad_y)), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, np.array(pil), 1 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    draw.text((pad_x, pad_y-1), reps_text, font=REPS_FONT, fill=(255,255,255))
    # Donut labels
    label_txt = "HEIGHT"; pct_txt = f"{int(pct*100)}%"
    gap = max(2, int(radius*0.10))
    base_y = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size)//2
    lw = draw.textlength(label_txt, font=DEPTH_LABEL_FONT); pw = draw.textlength(pct_txt, font=DEPTH_PCT_FONT)
    draw.text((cx - int(lw//2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pw//2), base_y + DEPTH_LABEL_FONT.size + gap), pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    # Feedback bottom (single line or short)
    if feedback:
        lines = _wrap_two_lines(draw, feedback, FEEDBACK_FONT, int(w - 2*12 - 20))
        line_h = FEEDBACK_FONT.size + 6
        block_h = 2*8 + len(lines)*line_h + (len(lines)-1)*4
        y0 = max(0, h - max(6, int(h*0.02)) - block_h); y1 = h - max(6, int(h*0.02))
        base = np.array(pil)
        over = base.copy(); cv2.rectangle(over, (0,y0), (w,y1), (0,0,0), -1)
        base = cv2.addWeighted(over, BAR_BG_ALPHA, base, 1 - BAR_BG_ALPHA, 0)
        pil = Image.fromarray(base); draw = ImageDraw.Draw(pil)
        ty = y0 + 8
        for ln in lines:
            tw = draw.textlength(ln, font=FEEDBACK_FONT); tx = max(12, (w-int(tw))//2)
            draw.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255,255,255)); ty += line_h + 4
    return np.array(pil)

def _wrap_two_lines(draw, text, font, max_width):
    words=text.split()
    if not words: return [""]
    lines, cur = [], ""
    for w in words:
        trial=(cur+" "+w).strip()
        if draw.textlength(trial, font=font) <= max_width: cur=trial
        else:
            if cur: lines.append(cur)
            cur=w
        if len(lines)==2: break
    if cur and len(lines)<2: lines.append(cur)
    if len(lines)>=2 and draw.textlength(lines[-1], font=font) > max_width:
        last = lines[-1]+"…"
        while draw.textlength(last, font=font) > max_width and len(last)>1:
            last = last[:-2]+"…"
        lines[-1] = last
    return lines

# ===================== Helpers =====================
def _ang(a,b,c):
    ba=np.array([a[0]-b[0], a[1]-b[1]]); bc=np.array([c[0]-b[0], c[1]-b[1]])
    den=(np.linalg.norm(ba)*np.linalg.norm(bc))+1e-9
    cos=float(np.clip(np.dot(ba,bc)/den, -1, 1))
    return float(np.degrees(np.arccos(cos)))

def _ema(prev,new,alpha):
    return float(new) if prev is None else (alpha*float(new) + (1-alpha)*float(prev))

# ===================== Logic params =====================
ELBOW_TOP_ANGLE      = 100.0   # ≤100° = מספיק כפוף (שנה ל-90° אם תרצה קשיח)
HEAD_MIN_ASCENT      = 0.0075  # ~0.75% מגובה הפריים
RESET_DESCENT        = 0.0045
RESET_ELBOW          = 135.0
REFRACTORY_FRAMES    = 2
HEAD_VEL_UP_TINY     = 0.0002
ELBOW_EMA_ALPHA      = 0.35
HEAD_EMA_ALPHA       = 0.30

# On/Off-bar gating (כניסה/יציאה)
VIS_THR_STRICT       = 0.30
WRIST_VIS_THR        = 0.20
WRIST_ABOVE_HEAD_MARGIN = 0.02
TORSO_X_THR          = 0.010
ONBAR_MIN_FRAMES     = 2
OFFBAR_MIN_FRAMES    = 6
AUTO_STOP_AFTER_EXIT_SEC = 1.2
TAIL_NOPOSE_STOP_SEC = 1.0

# ===================== MAIN =====================
def run_pullup_analysis(video_path,
                        frame_skip=3,   # כמו סקווט
                        scale=0.4,      # כמו סקווט
                        output_path="pullup_analyzed.mp4",
                        feedback_path="pullup_feedback.txt"):
    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _ret_err("Could not open video", feedback_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_idx = 0

    # Counters
    rep_count=0; good_reps=0; bad_reps=0
    rep_reports=[]; all_scores=[]

    # Dynamic side indices (pre-calc)
    LSH=mp_pose.PoseLandmark.LEFT_SHOULDER.value;  RSH=mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LE =mp_pose.PoseLandmark.LEFT_ELBOW.value;     RE =mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW =mp_pose.PoseLandmark.LEFT_WRIST.value;     RW =mp_pose.PoseLandmark.RIGHT_WRIST.value
    LH =mp_pose.PoseLandmark.LEFT_HIP.value;       RH =mp_pose.PoseLandmark.RIGHT_HIP.value
    NOSE=mp_pose.PoseLandmark.NOSE.value

    def _pick_side_dyn(lms):
        vL=lms[LSH].visibility + lms[LE].visibility + lms[LW].visibility
        vR=lms[RSH].visibility + lms[RE].visibility + lms[RW].visibility
        return ("LEFT", LSH,LE,LW) if vL>=vR else ("RIGHT", RSH,RE,RW)

    # State
    elbow_ema=None; head_ema=None; head_prev=None
    asc_base_head=None; baseline_head_y_global=None
    allow_new_peak=True; last_peak_frame=-99999

    # On/Off-bar
    onbar=False; onbar_streak=0; offbar_streak=0
    prev_torso_cx=None
    offbar_frames_since_any_rep = 0
    nopose_frames_since_any_rep = 0

    # Feedback session collection (לשקף למסך ולקובץ)
    session_feedback=set()
    rt_fb_msg=None; rt_fb_hold=0

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    sec_to_frames = lambda s: max(1, int(s * effective_fps))
    OFFBAR_STOP_FRAMES = sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES = sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES  = sec_to_frames(0.8)

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue
            if scale != 1.0:
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w,h))

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            height_live = 0.0

            if not res.pose_landmarks:
                # No pose tail-cut
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep + 1) if rep_count>0 else 0
                if rep_count>0 and nopose_frames_since_any_rep >= NOPOSE_STOP_FRAMES:
                    break
                frame = draw_overlay(frame, reps=rep_count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=0.0)
                out.write(frame)
                if rt_fb_hold>0: rt_fb_hold-=1
                continue

            nopose_frames_since_any_rep = 0
            lms = res.pose_landmarks.landmark
            side, S,E,W = _pick_side_dyn(lms)

            # Visibility (מחמיר)
            min_vis = min(lms[NOSE].visibility, lms[S].visibility, lms[E].visibility, lms[W].visibility)
            vis_strict_ok = (min_vis >= VIS_THR_STRICT)

            # Kinematics
            head_raw = float(lms[NOSE].y)
            raw_elbow = _ang((lms[S].x, lms[S].y), (lms[E].x, lms[E].y), (lms[W].x, lms[W].y))
            elbow_ema = _ema(elbow_ema, raw_elbow, ELBOW_EMA_ALPHA)
            head_ema  = _ema(head_ema,  head_raw,  HEAD_EMA_ALPHA)
            head_y = head_ema; elbow_angle = elbow_ema
            if baseline_head_y_global is None: baseline_head_y_global = head_y
            height_live = float(np.clip((baseline_head_y_global - head_y)/max(0.12, HEAD_MIN_ASCENT*1.2), 0.0, 1.0))

            # Torso X (walking)
            torso_cx = np.mean([lms[LSH].x, lms[RSH].x, lms[LH].x, lms[RH].x]) * w
            torso_dx_norm = 0.0 if prev_torso_cx is None else abs(torso_cx - prev_torso_cx)/max(1.0, w)
            prev_torso_cx = torso_cx

            # On/Off-bar
            lw_vis = lms[LW].visibility; rw_vis = lms[RW].visibility
            lw_above = (lw_vis >= WRIST_VIS_THR) and (lms[LW].y < lms[NOSE].y - WRIST_ABOVE_HEAD_MARGIN)
            rw_above = (rw_vis >= WRIST_VIS_THR) and (lms[RW].y < lms[NOSE].y - WRIST_ABOVE_HEAD_MARGIN)
            grip = (lw_above or rw_above)

            if vis_strict_ok and grip and (torso_dx_norm <= TORSO_X_THR):
                onbar_streak += 1; offbar_streak = 0
            else:
                offbar_streak += 1; onbar_streak = 0

            if not onbar and onbar_streak >= ONBAR_MIN_FRAMES:
                onbar = True; asc_base_head = None; allow_new_peak = True

            if onbar and offbar_streak >= OFFBAR_MIN_FRAMES:
                onbar = False; offbar_frames_since_any_rep = 0

            if not onbar and rep_count>0:
                offbar_frames_since_any_rep += 1
                if offbar_frames_since_any_rep >= OFFBAR_STOP_FRAMES:
                    break

            # Counting (only ON-BAR)
            head_vel = 0.0 if head_prev is None else (head_y - head_prev)
            if onbar and vis_strict_ok:
                if asc_base_head is None:
                    if head_vel < -HEAD_VEL_UP_TINY:
                        asc_base_head = head_y
                else:
                    if (head_y - asc_base_head) > (RESET_DESCENT * 2):
                        asc_base_head = head_y

                ascent_amt = 0.0 if asc_base_head is None else (asc_base_head - head_y)
                at_top   = (elbow_angle <= ELBOW_TOP_ANGLE) and (ascent_amt >= HEAD_MIN_ASCENT)
                can_cnt  = (frame_idx - last_peak_frame) >= REFRACTORY_FRAMES

                if at_top and allow_new_peak and can_cnt:
                    rep_count += 1; good_reps += 1; all_scores.append(10.0)
                    rep_reports.append({
                        "rep_index": rep_count,
                        "top_elbow": float(elbow_angle),
                        "ascent_from": float(asc_base_head if asc_base_head is not None else head_y),
                        "peak_head_y": float(head_y)
                    })
                    last_peak_frame = frame_idx
                    allow_new_peak = False

                reset_by_desc = (asc_base_head is not None) and ((head_y - asc_base_head) >= RESET_DESCENT)
                reset_by_elb  = (elbow_angle >= RESET_ELBOW)
                if reset_by_desc or reset_by_elb:
                    allow_new_peak = True
                    asc_base_head = head_y

                # RT feedback (collect + show)
                cur_rt = None
                if asc_base_head is not None and ascent_amt < HEAD_MIN_ASCENT*0.7 and head_vel < -HEAD_VEL_UP_TINY:
                    cur_rt = "Go a bit higher (chin over bar)"
                if cur_rt:
                    session_feedback.add(cur_rt)
                    if cur_rt != rt_fb_msg:
                        rt_fb_msg = cur_rt; rt_fb_hold = RT_FB_HOLD_FRAMES
                    else:
                        rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
                else:
                    if rt_fb_hold > 0: rt_fb_hold -= 1
            else:
                asc_base_head = None; allow_new_peak = True
                if rt_fb_hold > 0: rt_fb_hold -= 1

            # Draw skeleton + overlay
            frame = draw_body_only(frame, lms)
            frame = draw_overlay(frame, reps=rep_count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=height_live)
            out.write(frame)

            if head_y is not None: head_prev = head_y

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    # Technique score (counting-only)
    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg*2)/2, 2)

    # Build feedback list to return + file (mirror on-screen cues)
    if session_feedback:
        feedback_list = sorted(session_feedback)
    else:
        feedback_list = ["Great form! Keep it up 💪"]

    # Write feedback file
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            if feedback_list:
                f.write("Feedback:\n")
                for ln in feedback_list:
                    f.write(f"- {ln}\n")
    except Exception:
        pass

    # faststart encode (like squat)
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg","-y","-i", output_path,
            "-c:v","libx264","-preset","fast","-movflags","+faststart","-pix_fmt","yuv420p",
            encoded_path
        ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final_path = encoded_path if os.path.exists(encoded_path) else output_path
        if os.path.exists(output_path) and os.path.exists(encoded_path):
            try: os.remove(output_path)
            except: pass
    except Exception:
        final_path = output_path if os.path.exists(output_path) else ""

    return {
        "squat_count": int(rep_count),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": feedback_list,
        "tips": [],  # טיפים כלליים נפרדים מהערות-על מסך; נוסיף בהמשך אם תרצה
        "reps": rep_reports,
        "video_path": final_path,
        "feedback_path": feedback_path
    }

def _ret_err(msg, feedback_path):
    try:
        with open(feedback_path, "w", encoding="utf-8") as f: f.write(msg+"\n")
    except Exception:
        pass
    return {
        "squat_count": 0, "technique_score": 0.0,
        "technique_score_display": display_half_str(0.0),
        "technique_label": score_label(0.0),
        "good_reps": 0, "bad_reps": 0,
        "feedback": [msg], "tips": [],
        "reps": [], "video_path": "", "feedback_path": feedback_path
    }

# תאימות
def run_analysis(*args, **kwargs):
    return run_pullup_analysis(*args, **kwargs)

