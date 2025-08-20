# -*- coding: utf-8 -*-
# pullup_analysis.py
# Pull-ups — squat-style visuals, on-bar/off-bar gating, AND-counting (head ascent + elbow ≤ TOP_ANGLE).
# Feedback shown on video is mirrored in API + txt, and lowers the technique score (10.0 only if no cues).
#
# Usage example (original quality):
#   run_pullup_analysis(
#       "/path/to/video.mp4",
#       preserve_quality=True,   # עיבוד ורינדור באיכות מקורית
#       encode_crf=18            # קידוד איכותי (ברירת מחדל ב-preserve_quality)
#   )

import os, cv2, math, numpy as np, subprocess
from PIL import ImageFont, ImageDraw, Image

# ===================== STYLE / FONTS (like squat) =====================
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SMALL_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

TITLE_FONT_SIZE = 28
LABEL_FONT_SIZE = 18
DEPTH_FONT_SIZE = 18
DEPTH_PCT_FONT_SIZE   = 18

def _load_font(path, size, fallback=ImageFont.load_default):
    try:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        pass
    return fallback()

TITLE_FONT = _load_font(FONT_PATH, TITLE_FONT_SIZE)
LABEL_FONT = _load_font(FONT_SMALL_PATH, LABEL_FONT_SIZE)
DEPTH_FONT = _load_font(FONT_SMALL_PATH, DEPTH_FONT_SIZE)
DEPTH_PCT_FONT = _load_font(FONT_SMALL_PATH, DEPTH_PCT_FONT_SIZE)

def _draw_text_pil(img_bgr, text, org, font, color=(255,255,255), anchor=None):
    # img_bgr: np.ndarray BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        draw.text(org, text, font=font, fill=color, anchor=anchor)
    except Exception:
        draw.text(org, text, fill=color)
    new_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return new_bgr

# ===================== GEOMETRY / UTILS (like squat) =====================

def _ang(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    v1x, v1y = ax-bx, ay-by
    v2x, v2y = cx-bx, cy-by
    d1 = math.hypot(v1x, v1y)
    d2 = math.hypot(v2x, v2y)
    if d1<1e-6 or d2<1e-6: return 180.0
    v1x /= d1; v1y /= d1
    v2x /= d2; v2y /= d2
    dot = v1x*v2x + v1y*v2y
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def _ema(prev, x, alpha):
    return x if prev is None else (alpha * x + (1.0 - alpha) * prev)

def _half_floor(x):
    # round down to nearest 0.5
    q = math.floor(float(x) * 2.0) / 2.0
    return max(0.0, min(10.0, q))

# ===================== THRESHOLDS / PARAMS (as previously used) =====================

VIS_THR_STRICT = 0.55
HEAD_EMA_ALPHA = 0.35
ELBOW_EMA_ALPHA = 0.35

HEAD_MIN_ASCENT = 0.06       # normalized by image height
HEAD_VEL_UP_TINY = -0.0008

ELBOW_TOP_ANGLE = 72.0
RESET_ELBOW = 158.0
RESET_DESCENT = 0.04

ONBAR_MIN_FRAMES = 2
OFFBAR_MIN_FRAMES = 3

WRIST_VIS_THR = 0.45
WRIST_ABOVE_HEAD_MARGIN = 0.04

TORSO_X_THR = 0.012
SWING_THR = 0.012
SWING_MIN_STREAK = 6

REFRACTORY_FRAMES = 8

AUTO_STOP_AFTER_EXIT_SEC = 1.5
TAIL_NOPOSE_STOP_SEC = 1.2

# bottom-phase extension check
BOTTOM_EXT_MIN_ANGLE = 168.0
BOTTOM_HYST_DEG      = 2.0
BOTTOM_FAIL_MIN_REPS = 2

# ===================== FEEDBACK / SCORING =====================

FB_CUE_HIGHER = "Try to pull a little higher"
FB_CUE_BOTTOM = "Fully extend arms at the bottom"
FB_CUE_SWING  = "Try to reduce momentum"

FB_WEIGHTS = {
    FB_CUE_HIGHER: 0.5,
    FB_CUE_BOTTOM: 0.7,
    FB_CUE_SWING:  0.7,
}
FB_DEFAULT_WEIGHT = 0.5
PENALTY_MIN_IF_ANY = 0.5

def score_label(s):
    s = float(s)
    if s>=9.0: return "Excellent"
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
    _BODY_CONNECTIONS = (
        # arms
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
        (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
        (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
        # shoulders
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        # torso
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    )
    _BODY_POINTS = {
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value,
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
    }

def draw_body_only(img_bgr, lms):
    h, w = img_bgr.shape[:2]
    pts = []
    for i, lm in enumerate(lms):
        if i not in _BODY_POINTS: 
            continue
        if lm.visibility < 0.2: 
            continue
        x = int(lm.x * w); y = int(lm.y * h)
        pts.append((i, x, y))
    # lines
    for a, b in _BODY_CONNECTIONS:
        pa = next((p for p in pts if p[0]==a), None)
        pb = next((p for p in pts if p[0]==b), None)
        if pa and pb:
            cv2.line(img_bgr, (pa[1], pa[2]), (pb[1], pb[2]), (255,255,255), 2)
    # joints
    for _, x, y in pts:
        cv2.circle(img_bgr, (x,y), 3, (255,255,255), -1)
    return img_bgr

def draw_overlay(img_bgr, reps=0, feedback=None, height_pct=0.0):
    h, w = img_bgr.shape[:2]
    # Title
    img_bgr = _draw_text_pil(img_bgr, f"Pull-ups", (16,16), TITLE_FONT, color=(255,255,255))
    # Reps box
    box_text = f"Reps: {reps}"
    img_bgr = _draw_text_pil(img_bgr, box_text, (16, 56), LABEL_FONT, color=(220,220,220))
    # Height bar (progress-like)
    bar_w = 10
    filled = int(max(0.0, min(1.0, height_pct)) * (h - 120))
    y0 = 100
    cv2.rectangle(img_bgr, (w-32, y0), (w-32+bar_w, y0 + h - 120), (255,255,255), 2)
    cv2.rectangle(img_bgr, (w-32, y0 + (h-120 - filled)), (w-32+bar_w, y0 + h - 120), (255,255,255), -1)
    # feedback bubble
    if feedback:
        img_bgr = _draw_text_pil(img_bgr, feedback, (16, h-40), LABEL_FONT, color=(255,255,255))
    return img_bgr

# ===================== MAIN =====================

def run_pullup_analysis(video_path,
                        frame_skip=3,   # כמו סקווט
                        scale=0.4,      # כמו סקווט
                        output_path="pullup_analyzed.mp4",
                        feedback_path="pullup_feedback.txt",
                        preserve_quality=False,
                        encode_crf=None,
                        return_video=True,
                        fast_mode=None):
    """
    preserve_quality=True  =>  scale=1.0, frame_skip=1, קידוד CRF=18 (אם encode_crf לא סופק).
    """
    if fast_mode is True:
        return_video = False

    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    if preserve_quality:
        scale = 1.0
        frame_skip = 1
        if encode_crf is None:
            encode_crf = 18
    else:
        if encode_crf is None:
            encode_crf = 23

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _ret_err("Could not open video", feedback_path)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    # effective fps after frame skipping:
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    sec_to_frames = lambda s: max(1, int(s * effective_fps))

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

    # Feedback session collection
    session_feedback=set()
    rt_fb_msg=None; rt_fb_hold=0

    # Swing state
    swing_streak=0
    swing_already_reported=False
    bottom_already_reported=False

    # bottom-phase elbow tracking (both arms)
    bottom_phase_max_elbow = None
    bottom_fail_count = 0

    OFFBAR_STOP_FRAMES = sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES = sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES  = sec_to_frames(0.8)

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            if frame_idx % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            h, w = frame.shape[:2]
            if return_video and out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w,h))

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not res.pose_landmarks:
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep + 1) if rep_count>0 else 0
                if rep_count>0 and nopose_frames_since_any_rep >= NOPOSE_STOP_FRAMES:
                    break
                if return_video and out is not None:
                    frame = draw_overlay(frame, reps=rep_count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=0.0)
                    out.write(frame)
                if rt_fb_hold>0: rt_fb_hold-=1
                continue

            nopose_frames_since_any_rep = 0
            lms = res.pose_landmarks.landmark
            side, S,E,W = _pick_side_dyn(lms)

            # strict visibility
            min_vis = min(lms[NOSE].visibility, lms[S].visibility, lms[E].visibility, lms[W].visibility)
            vis_strict_ok = (min_vis >= VIS_THR_STRICT)

            # positions/angles
            head_raw = float(lms[NOSE].y)
            raw_elbow_L = _ang((lms[LSH].x, lms[LSH].y), (lms[LE].x, lms[LE].y), (lms[LW].x, lms[LW].y))
            raw_elbow_R = _ang((lms[RSH].x, lms[RSH].y), (lms[RE].x, lms[RE].y), (lms[RW].x, lms[RW].y))
            raw_elbow = raw_elbow_L if side == "LEFT" else raw_elbow_R

            elbow_ema = _ema(elbow_ema, raw_elbow, ELBOW_EMA_ALPHA)
            head_ema  = _ema(head_ema,  head_raw,  HEAD_EMA_ALPHA)
            head_y = head_ema; elbow_angle = elbow_ema
            if baseline_head_y_global is None: baseline_head_y_global = head_y
            height_live = float(np.clip((baseline_head_y_global - head_y)/max(0.12, HEAD_MIN_ASCENT*1.2), 0.0, 1.0))

            # torso cx (to detect walking/swing)
            torso_cx = np.mean([lms[LSH].x, lms[RSH].x, lms[LH].x, lms[RH].x]) * w
            torso_dx_norm = 0.0 if prev_torso_cx is None else abs(torso_cx - prev_torso_cx)/max(1.0, w)
            prev_torso_cx = torso_cx

            # On/Off-bar gating
            lw_vis = lms[LW].visibility; rw_vis = lms[RW].visibility
            lw_above = (lw_vis >= WRIST_VIS_THR) and (lms[LW].y < lms[NOSE].y - WRIST_ABOVE_HEAD_MARGIN)
            rw_above = (rw_vis >= WRIST_VIS_THR) and (lms[RW].y < lms[NOSE].y - WRIST_ABOVE_HEAD_MARGIN)
            grip = (lw_above or rw_above)

            if vis_strict_ok and grip and (torso_dx_norm <= TORSO_X_THR):
                onbar_streak += 1; offbar_streak = 0
            else:
                offbar_streak += 1; onbar_streak = 0

            onbar_prev = onbar
            if not onbar and onbar_streak >= ONBAR_MIN_FRAMES:
                onbar = True; asc_base_head = None; allow_new_peak = True
                swing_streak = 0

            if onbar and offbar_streak >= OFFBAR_MIN_FRAMES:
                onbar = False; offbar_frames_since_any_rep = 0

            if (not onbar) and rep_count>0:
                offbar_frames_since_any_rep += 1
                if offbar_frames_since_any_rep >= OFFBAR_STOP_FRAMES:
                    break

            # Counting (ON-BAR)
            head_vel = 0.0 if head_prev is None else (head_y - head_prev)
            cur_rt = None

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
                    bottom_already_reported = False
                    bottom_phase_max_elbow = max(raw_elbow_L, raw_elbow_R)

                if (allow_new_peak is False):
                    cand = max(raw_elbow_L, raw_elbow_R)
                    bottom_phase_max_elbow = cand if bottom_phase_max_elbow is None else max(bottom_phase_max_elbow, cand)

                reset_by_desc = (asc_base_head is not None) and ((head_y - asc_base_head) >= RESET_DESCENT)
                reset_by_elb  = (elbow_angle >= RESET_ELBOW)
                if reset_by_desc or reset_by_elb:
                    if not bottom_already_reported:
                        effective_max = bottom_phase_max_elbow if bottom_phase_max_elbow is not None else max(raw_elbow_L, raw_elbow_R)
                        if effective_max < (BOTTOM_EXT_MIN_ANGLE - BOTTOM_HYST_DEG):
                            bottom_fail_count += 1
                            if bottom_fail_count >= BOTTOM_FAIL_MIN_REPS:
                                session_feedback.add(FB_CUE_BOTTOM)
                                cur_rt = cur_rt or FB_CUE_BOTTOM
                        else:
                            bottom_fail_count = max(0, bottom_fail_count - 1)
                        bottom_already_reported = True

                    allow_new_peak = True
                    asc_base_head = head_y
                    bottom_phase_max_elbow = None

                if (cur_rt is None) and (asc_base_head is not None) and (ascent_amt < HEAD_MIN_ASCENT*0.7) and (head_vel < -HEAD_VEL_UP_TINY):
                    session_feedback.add(FB_CUE_HIGHER); cur_rt = FB_CUE_HIGHER

                if torso_dx_norm > SWING_THR:
                    swing_streak += 1
                else:
                    swing_streak = max(0, swing_streak-1)
                if (cur_rt is None) and (swing_streak >= SWING_MIN_STREAK) and (not swing_already_reported):
                    session_feedback.add(FB_CUE_SWING); cur_rt = FB_CUE_SWING; swing_already_reported = True
            else:
                asc_base_head = None; allow_new_peak = True
                swing_streak = 0
                bottom_phase_max_elbow = None

            if cur_rt:
                if cur_rt != rt_fb_msg:
                    rt_fb_msg = cur_rt; rt_fb_hold = RT_FB_HOLD_FRAMES
                else:
                    rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold > 0: rt_fb_hold -= 1

            # draw/write only when returning video
            if return_video and out is not None:
                frame = draw_body_only(frame, lms)
                frame = draw_overlay(frame, reps=rep_count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=height_live)
                out.write(frame)

            if head_y is not None: head_prev = head_y

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    # ===== TECHNIQUE SCORE =====
    if rep_count == 0:
        technique_score = 0.0
    else:
        penalty = 0.0
        if session_feedback:
            penalty = sum(FB_WEIGHTS.get(msg, FB_DEFAULT_WEIGHT) for msg in session_feedback)
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
        raw_score = max(0.0, 10.0 - penalty)
        technique_score = _half_floor(raw_score)

    feedback_list = sorted(session_feedback) if session_feedback else ["Great form! Keep it up 💪"]

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

    final_path = ""
    if return_video:
        # ========== Encode faststart with chosen quality ==========
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run([
                "ffmpeg","-y","-i", output_path,
                "-c:v","libx264","-preset","medium",
                "-crf", str(int(encode_crf)),
                "-movflags","+faststart","-pix_fmt","yuv420p",
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
        "tips": [],
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

