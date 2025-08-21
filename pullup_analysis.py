# -*- coding: utf-8 -*-
# pullup_analysis.py
# שמות ה-JSON נשמרים כמו במקורי (squat_count וכו').

import os, cv2, math, numpy as np, subprocess
from PIL import ImageFont, ImageDraw, Image

# ===================== STYLE / FONTS =====================
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

# ===================== Helpers =====================
def _ang(a,b,c):
    ba=np.array([a[0]-b[0], a[1]-b[1]]); bc=np.array([c[0]-b[0], c[1]-b[1]])
    den=(np.linalg.norm(ba)*np.linalg.norm(bc))+1e-9
    cos=float(np.clip(np.dot(ba,bc)/den, -1, 1))
    return float(np.degrees(np.arccos(cos)))

def _ema(prev,new,alpha):
    return float(new) if prev is None else (alpha*float(new) + (1-alpha)*float(prev))

def _half_floor10(x: float) -> float:
    return max(0.0, min(10.0, math.floor(x * 2.0) / 2.0))

def display_half_str(x):
    q=round(float(x)*2)/2.0
    return str(int(round(q))) if abs(q-round(q))<1e-9 else f"{q:.1f}"

def score_label(s):
    s=float(s)
    if s>=9.0: return "Excellent"
    if s>=7.0: return "Good"
    if s>=5.5: return "Fair"
    return "Needs work"

def _wrap_two_lines(draw, text, font, max_width):
    words=text.split()
    if not words: return [""]
    lines, cur = [], ""
    for w in words:
        t=(cur+" "+w).strip()
        if draw.textlength(t, font=font) <= max_width: cur=t
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

# ===================== BODY-ONLY skeleton =====================
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

def _dyn_thickness(h):
    line = max(2, int(round(h * 0.002)))
    dot  = max(3, int(round(h * 0.004)))
    return line, dot

def draw_body_only(frame, landmarks, color=(255,255,255)):
    h, w = frame.shape[:2]
    line_thick, dot_r = _dyn_thickness(h)
    for a, b in _BODY_CONNECTIONS:
        pa, pb = landmarks[a], landmarks[b]
        ax, ay = int(pa.x*w), int(pa.y*h); bx, by = int(pb.x*w), int(pb.y*h)
        cv2.line(frame, (ax, ay), (bx, by), color, line_thick, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = landmarks[i]; x, y = int(p.x*w), int(p.y*h)
        cv2.circle(frame, (x, y), dot_r, color, -1, cv2.LINE_AA)
    return frame

# ===================== Overlay (BGR↔RGB תקין) =====================
def draw_overlay(frame, reps=0, feedback=None, height_pct=0.0):
    h, w, _ = frame.shape
    # Donut (OpenCV) ב-BGR
    ref_h = max(int(h*0.06), int(REPS_FONT_SIZE*1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12; cx = w - margin - radius; cy = max(ref_h + radius//8, radius + thick//2 + 2)
    pct = float(np.clip(height_pct,0,1))
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thick, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, -90 + int(360*pct), DEPTH_COLOR, thick, lineType=cv2.LINE_AA)
    # טקסטים ב-PIL (RGB)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {int(reps)}"
    pad_x, pad_y = 10, 6
    tw = draw.textlength(reps_text, font=REPS_FONT); th = REPS_FONT.size
    base = np.array(pil); over = base.copy()
    cv2.rectangle(over, (0,0), (int(tw + 2*pad_x), int(th + 2*pad_y)), (0,0,0), -1)
    base = cv2.addWeighted(over, BAR_BG_ALPHA, base, 1 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(base); draw = ImageDraw.Draw(pil)
    draw.text((pad_x, pad_y-1), reps_text, font=REPS_FONT, fill=(255,255,255))
    # תוויות דונאט
    gap = max(2, int(radius*0.10))
    base_y = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size)//2
    label_txt = "HEIGHT"; pct_txt = f"{int(pct*100)}%"
    lw = draw.textlength(label_txt, font=DEPTH_LABEL_FONT); pw = draw.textlength(pct_txt, font=DEPTH_PCT_FONT)
    draw.text((cx - int(lw//2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pw//2), base_y + DEPTH_LABEL_FONT.size + gap), pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    # פידבק תחתון
    if feedback:
        max_w = int(w - 2*12 - 20)
        lines = _wrap_two_lines(draw, feedback, FEEDBACK_FONT, max_w)
        line_h = FEEDBACK_FONT.size + 6
        block_h = 2*8 + len(lines)*line_h + (len(lines)-1)*4
        y0 = max(0, h - max(6, int(h*0.02)) - block_h); y1 = h - max(6, int(h*0.02))
        base2 = np.array(pil); over2 = base2.copy()
        cv2.rectangle(over2, (0,y0), (w,y1), (0,0,0), -1)
        base2 = cv2.addWeighted(over2, BAR_BG_ALPHA, base2, 1 - BAR_BG_ALPHA, 0)
        pil = Image.fromarray(base2); draw = ImageDraw.Draw(pil)
        ty = y0 + 8
        for ln in lines:
            tw2 = draw.textlength(ln, font=FEEDBACK_FONT); tx = max(12, (w-int(tw2))//2)
            draw.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255,255,255)); ty += line_h + 4
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ===================== Params =====================
# ריכוך “chin over bar”
CHIN_RELAX = float(os.getenv("CHIN_RELAX", "0.85"))

# ספים (מאוזנים ומוכחים)
ELBOW_TOP_ANGLE = 100.0
HEAD_MIN_ASCENT = 0.06
HEAD_VEL_UP_TINY = -0.0008

VIS_THR_STRICT = 0.55
ELBOW_EMA_ALPHA = 0.35
HEAD_EMA_ALPHA  = 0.35

RESET_ELBOW   = 158.0
RESET_DESCENT = 0.04

WRIST_VIS_THR = 0.45
WRIST_ABOVE_HEAD_MARGIN = 0.04
TORSO_X_THR   = 0.012

ONBAR_MIN_FRAMES  = 2
OFFBAR_MIN_FRAMES = 3

SWING_THR = 0.012
SWING_MIN_STREAK = 6

REFRACTORY_FRAMES = 8

AUTO_STOP_AFTER_EXIT_SEC = 1.5
TAIL_NOPOSE_STOP_SEC     = 1.2

# Bottom extension
BOTTOM_EXT_MIN_ANGLE = 168.0
BOTTOM_HYST_DEG      = 2.0
BOTTOM_FAIL_MIN_REPS = 2

# Feedback cues & weights
FB_CUE_HIGHER = "Try to pull a little higher"
FB_CUE_BOTTOM = "Fully extend arms at the bottom"
FB_CUE_SWING  = "Try to reduce momentum"

FB_WEIGHTS = {FB_CUE_HIGHER:0.5, FB_CUE_BOTTOM:0.7, FB_CUE_SWING:0.7}
FB_DEFAULT_WEIGHT   = 0.5
PENALTY_MIN_IF_ANY  = 0.5

DEBUG_ONBAR = bool(int(os.getenv("DEBUG_ONBAR", "0")))

# ===================== MAIN =====================
def run_pullup_analysis(video_path,
                        frame_skip=3,
                        scale=0.4,
                        output_path="pullup_analyzed.mp4",
                        feedback_path="pullup_feedback.txt",
                        preserve_quality=False,
                        encode_crf=None,
                        return_video=True,
                        fast_mode=None):
    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    # Presets
    model_complexity = 1
    if fast_mode is True:
        return_video = False
        model_complexity = 0
        scale = min(scale, 0.3)
        frame_skip = max(frame_skip, 4)

    if preserve_quality:
        scale = 1.0; frame_skip = 1
        encode_crf = 18 if encode_crf is None else encode_crf
    else:
        encode_crf = 23 if encode_crf is None else encode_crf

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _ret_err("Could not open video", feedback_path)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    sec_to_frames = lambda s: max(1, int(s * effective_fps))

    # התאמות לדילוגים / fast
    FS = max(1, int(frame_skip))
    HEAD_MIN_ASCENT_EFF = HEAD_MIN_ASCENT * CHIN_RELAX * (0.75 if FS >= 3 else 1.0)
    REFR_EFF = max(1, int(round(REFRACTORY_FRAMES / FS)))
    if fast_mode: REFR_EFF = 1
    RESET_DESCENT_EFF = max(RESET_DESCENT * 0.35, RESET_DESCENT / max(1.0, FS*0.8))
    REARM_DESCENT_EFF = max(RESET_DESCENT_EFF * 0.50, 0.015)  # מאפשר ספירה בחזרה הבאה גם עם דיפ קצר

    if fast_mode:
        VIS_EFF   = min(VIS_THR_STRICT, 0.35)
        WRIST_VIS_EFF = 0.25
        WRIST_MARGIN_EFF = WRIST_ABOVE_HEAD_MARGIN * 0.7
        TORSO_X_EFF = TORSO_X_THR * 1.6
    else:
        VIS_EFF   = VIS_THR_STRICT
        WRIST_VIS_EFF = WRIST_VIS_THR
        WRIST_MARGIN_EFF = WRIST_ABOVE_HEAD_MARGIN
        TORSO_X_EFF = TORSO_X_THR

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_idx = 0
    eff_idx = 0

    # Counters
    rep_count=0; good_reps=0; bad_reps=0
    rep_reports=[]; all_scores=[]

    # Dynamic indices
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
    baseline_head_y_global=None

    asc_base_head=None
    allow_new_peak=True
    last_peak_eff=-10**9
    last_top_head=None

    # מחזור לצורך Post-hoc
    cycle_peak_ascent = 0.0
    cycle_min_elbow   = 999.0
    counted_this_cycle = False

    # On/Off-bar
    onbar=False; onbar_streak=0; offbar_streak=0
    prev_torso_cx=None
    offbar_frames_since_any_rep = 0
    nopose_frames_since_any_rep = 0

    # Feedback
    session_feedback=set()
    rt_fb_msg=None; rt_fb_hold=0
    rep_penalty_current = 0.0

    # Swing
    swing_streak=0
    swing_already_reported=False
    bottom_already_reported=False
    bottom_phase_max_elbow = None
    bottom_fail_count = 0

    OFFBAR_STOP_FRAMES = sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES = sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES  = sec_to_frames(0.8)

    with mp_pose.Pose(model_complexity=model_complexity,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            grabbed = cap.grab()
            if not grabbed: break
            frame_idx += 1
            if frame_idx % FS != 0:
                continue
            ret, frame = cap.retrieve()
            if not ret: break
            eff_idx += 1

            if scale != 1.0:
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            h, w = frame.shape[:2]

            if return_video and out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w,h))

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            height_live = 0.0

            if not res.pose_landmarks:
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep + 1) if rep_count>0 else 0
                if rep_count>0 and nopose_frames_since_any_rep >= NOPOSE_STOP_FRAMES: break
                if return_video and out is not None:
                    frame = draw_overlay(frame, reps=rep_count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=0.0)
                    out.write(frame)
                if rt_fb_hold>0: rt_fb_hold-=1
                continue

            nopose_frames_since_any_rep = 0
            lms = res.pose_landmarks.landmark
            side, S,E,W = _pick_side_dyn(lms)

            # Visibility
            min_vis = min(lms[NOSE].visibility, lms[S].visibility, lms[E].visibility, lms[W].visibility)
            vis_strict_ok = (min_vis >= VIS_EFF)

            # Positions/angles
            head_raw = float(lms[NOSE].y)
            raw_elbow_L = _ang((lms[LSH].x, lms[LSH].y), (lms[LE].x, lms[LE].y), (lms[LW].x, lms[LW].y))
            raw_elbow_R = _ang((lms[RSH].x, lms[RSH].y), (lms[RE].x, lms[RE].y), (lms[RW].x, lms[RW].y))
            raw_elbow = raw_elbow_L if side == "LEFT" else raw_elbow_R
            raw_elbow_min = min(raw_elbow_L, raw_elbow_R)

            elbow_ema = _ema(elbow_ema, raw_elbow, ELBOW_EMA_ALPHA)
            head_ema  = _ema(head_ema,  head_raw,  HEAD_EMA_ALPHA)
            head_y = head_ema; elbow_angle = elbow_ema
            if baseline_head_y_global is None: baseline_head_y_global = head_y
            height_live = float(np.clip((baseline_head_y_global - head_y)/max(0.12, HEAD_MIN_ASCENT_EFF*1.2), 0.0, 1.0))

            # Torso drift
            torso_cx = np.mean([lms[LSH].x, lms[RSH].x, lms[LH].x, lms[RH].x]) * w
            torso_dx_norm = 0.0 if prev_torso_cx is None else abs(torso_cx - prev_torso_cx)/max(1.0, w)
            prev_torso_cx = torso_cx

            # On/Off-bar gating
            lw_vis = lms[LW].visibility; rw_vis = lms[RW].visibility
            lw_above = (lw_vis >= WRIST_VIS_EFF) and (lms[LW].y < lms[NOSE].y - WRIST_MARGIN_EFF)
            rw_above = (rw_vis >= WRIST_VIS_EFF) and (lms[RW].y < lms[NOSE].y - WRIST_MARGIN_EFF)
            grip = (lw_above or rw_above)

            if vis_strict_ok and grip and (torso_dx_norm <= TORSO_X_EFF):
                onbar_streak += 1; offbar_streak = 0
            else:
                offbar_streak += 1; onbar_streak = 0

            if DEBUG_ONBAR and eff_idx % 10 == 0:
                print(f"[DBG] eff={eff_idx} onbar={onbar} visMin={min_vis:.2f} lwAbove={lw_above} rwAbove={rw_above} torsoDx={torso_dx_norm:.4f}")

            # כניסה ל-onbar
            if (not onbar) and onbar_streak >= ONBAR_MIN_FRAMES:
                onbar = True
                asc_base_head = None
                allow_new_peak = True
                swing_streak = 0
                rep_penalty_current = 0.0
                # reset cycle
                cycle_peak_ascent = 0.0
                cycle_min_elbow   = 999.0
                counted_this_cycle = False

            # יציאה מ-onbar → Flush פוסט־הוק אם צריך
            if onbar and offbar_streak >= OFFBAR_MIN_FRAMES:
                if (not counted_this_cycle) and (cycle_peak_ascent >= HEAD_MIN_ASCENT_EFF) and (cycle_min_elbow <= ELBOW_TOP_ANGLE):
                    _count_rep(rep_reports, rep_count, rep_penalty_current, cycle_min_elbow,
                               asc_base_head if asc_base_head is not None else head_y,
                               baseline_head_y_global - cycle_peak_ascent if baseline_head_y_global is not None else head_y,
                               all_scores)
                    rep_count += 1
                    good_reps, bad_reps = _mark_good_bad(all_scores[-1], good_reps, bad_reps)
                onbar = False; offbar_frames_since_any_rep = 0
                asc_base_head = None
                cycle_peak_ascent = 0.0; cycle_min_elbow = 999.0; counted_this_cycle = False

            if (not onbar) and rep_count>0:
                offbar_frames_since_any_rep += 1
                if offbar_frames_since_any_rep >= OFFBAR_STOP_FRAMES: break

            # ===== Counting (ON BAR) =====
            head_vel = 0.0 if head_prev is None else (head_y - head_prev)
            cur_rt = None

            if onbar and vis_strict_ok:
                # התחלת עלייה
                if asc_base_head is None:
                    if head_vel < -abs(HEAD_VEL_UP_TINY):
                        asc_base_head = head_y
                        rep_penalty_current = 0.0
                        cycle_peak_ascent = 0.0
                        cycle_min_elbow   = elbow_angle
                        counted_this_cycle = False
                else:
                    # עדכון מחזור
                    cycle_peak_ascent = max(cycle_peak_ascent, (asc_base_head - head_y))
                    cycle_min_elbow   = min(cycle_min_elbow, elbow_angle)

                    # ריסט (ירידה) — לפני איפוס, Flush פוסט־הוק אם צריך
                    if (head_y - asc_base_head) > (RESET_DESCENT_EFF) or (elbow_angle >= RESET_ELBOW):
                        if (not counted_this_cycle) and (cycle_peak_ascent >= HEAD_MIN_ASCENT_EFF) and (cycle_min_elbow <= ELBOW_TOP_ANGLE):
                            _count_rep(rep_reports, rep_count, rep_penalty_current, cycle_min_elbow,
                                       asc_base_head, baseline_head_y_global - cycle_peak_ascent if baseline_head_y_global is not None else head_y,
                                       all_scores)
                            rep_count += 1
                            good_reps, bad_reps = _mark_good_bad(all_scores[-1], good_reps, bad_reps)
                        asc_base_head = head_y
                        rep_penalty_current = 0.0
                        cycle_peak_ascent = 0.0
                        cycle_min_elbow   = elbow_angle
                        counted_this_cycle = False
                        allow_new_peak = True
                        bottom_phase_max_elbow = None
                        bottom_already_reported = False

                ascent_amt = 0.0 if asc_base_head is None else (asc_base_head - head_y)

                # TOP אונליין (עם fallback raw ו-REARM)
                at_top = (elbow_angle <= ELBOW_TOP_ANGLE) and (ascent_amt >= HEAD_MIN_ASCENT_EFF)
                raw_top = (raw_elbow_min <= (ELBOW_TOP_ANGLE + 4.0)) and (ascent_amt >= HEAD_MIN_ASCENT_EFF * 0.92)
                at_top = at_top or raw_top
                can_cnt = (eff_idx - last_peak_eff) >= REFR_EFF

                if at_top and allow_new_peak and can_cnt and (not counted_this_cycle):
                    _count_rep(rep_reports, rep_count, rep_penalty_current, elbow_angle,
                               asc_base_head if asc_base_head is not None else head_y, head_y, all_scores)
                    rep_count += 1
                    good_reps, bad_reps = _mark_good_bad(all_scores[-1], good_reps, bad_reps)
                    last_peak_eff = eff_idx
                    last_top_head = head_y
                    allow_new_peak = False
                    counted_this_cycle = True
                    bottom_already_reported = False
                    bottom_phase_max_elbow = max(raw_elbow_L, raw_elbow_R)

                # Rearm על דיפ קטן מהטופ (לפתור חזרה #9)
                if (allow_new_peak is False) and (last_top_head is not None):
                    if (head_y - last_top_head) >= REARM_DESCENT_EFF:
                        allow_new_peak = True
                        bottom_already_reported = False
                        bottom_phase_max_elbow = None

                # RT: higher
                if (cur_rt is None) and (asc_base_head is not None) and (ascent_amt < HEAD_MIN_ASCENT*0.7) and (head_vel < -abs(HEAD_VEL_UP_TINY)):
                    session_feedback.add(FB_CUE_HIGHER); cur_rt = FB_CUE_HIGHER
                    rep_penalty_current += FB_WEIGHTS.get(FB_CUE_HIGHER, FB_DEFAULT_WEIGHT)

                # RT: swing
                if torso_dx_norm > SWING_THR: swing_streak += 1
                else: swing_streak = max(0, swing_streak-1)
                if (cur_rt is None) and (swing_streak >= SWING_MIN_STREAK) and (not swing_already_reported):
                    session_feedback.add(FB_CUE_SWING); cur_rt = FB_CUE_SWING; swing_already_reported = True
                    rep_penalty_current += FB_WEIGHTS.get(FB_CUE_SWING, FB_DEFAULT_WEIGHT)

            else:
                asc_base_head = None; allow_new_peak = True
                swing_streak = 0
                bottom_phase_max_elbow = None

            # RT hold
            if cur_rt:
                if cur_rt != rt_fb_msg:
                    rt_fb_msg = cur_rt; rt_fb_hold = RT_FB_HOLD_FRAMES
                else:
                    rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold > 0: rt_fb_hold -= 1

            # ציור
            if return_video and out is not None:
                frame = draw_body_only(frame, lms)
                frame = draw_overlay(frame, reps=rep_count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=height_live)
                out.write(frame)

            if head_y is not None: head_prev = head_y

    # ===== EOF Flush =====
    # אם יש מחזור פתוח שלא נספר, נספור אותו כעת.
    if onbar and (not counted_this_cycle) and (cycle_peak_ascent >= HEAD_MIN_ASCENT_EFF) and (cycle_min_elbow <= ELBOW_TOP_ANGLE):
        _count_rep(rep_reports, rep_count, rep_penalty_current, cycle_min_elbow,
                   asc_base_head if asc_base_head is not None else (baseline_head_y_global or 0.0),
                   (baseline_head_y_global - cycle_peak_ascent) if baseline_head_y_global is not None else (baseline_head_y_global or 0.0),
                   all_scores)
        rep_count += 1
        good_reps, bad_reps = _mark_good_bad(all_scores[-1], good_reps, bad_reps)

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    # ===== Technique score =====
    if rep_count == 0:
        technique_score = 0.0
    else:
        penalty = 0.0
        if session_feedback:
            penalty = sum(FB_WEIGHTS.get(msg, FB_DEFAULT_WEIGHT) for msg in session_feedback)
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
        raw_score = max(0.0, 10.0 - penalty)
        technique_score = _half_floor10(raw_score)

    # form_tip (לא יחזור null)
    form_tip = None
    if session_feedback:
        form_tip = max(session_feedback, key=lambda m: FB_WEIGHTS.get(m, FB_DEFAULT_WEIGHT))
    elif 'rt_fb_msg' in locals() and rt_fb_msg:
        form_tip = rt_fb_msg
    fb_list = sorted(session_feedback) if session_feedback else ["Great form! Keep it up 💪"]

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            if fb_list:
                f.write("Feedback:\n")
                for ln in fb_list:
                    f.write(f"- {ln}\n")
    except Exception:
        pass

    # ===== Encode video (אם צריך) =====
    final_path = ""
    if return_video:
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
        "feedback": fb_list,
        "form_tip": form_tip,
        "tips": [],
        "reps": rep_reports,
        "video_path": final_path,
        "feedback_path": feedback_path
    }

# ---------- helpers ----------
def _count_rep(rep_reports, rep_count, rep_penalty_current, top_elbow, ascent_from, peak_head_y, all_scores):
    rep_score = max(0.0, 10.0 - rep_penalty_current)
    all_scores.append(rep_score)
    rep_reports.append({
        "rep_index": int(rep_count + 1),
        "score": float(rep_score),
        "good": bool(rep_score >= 10.0 - 1e-6),
        "top_elbow": float(top_elbow),
        "ascent_from": float(ascent_from),
        "peak_head_y": float(peak_head_y)
    })

def _mark_good_bad(score, good, bad):
    if score >= 10.0 - 1e-6: return good+1, bad
    return good, bad+1

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
        "feedback": [msg], "form_tip": msg,
        "tips": [], "reps": [], "video_path": "", "feedback_path": feedback_path
    }

# תאימות
def run_analysis(*args, **kwargs):
    return run_pullup_analysis(*args, **kwargs)

