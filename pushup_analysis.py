# -*- coding: utf-8 -*-
# pushup_analysis.py — IMPROVED: Better fast detection + FIXED separation + FIXED overlay cache

import os, cv2, math, numpy as np, subprocess
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ============ Styles ============
BAR_BG_ALPHA=0.55; DONUT_RADIUS_SCALE=0.72; DONUT_THICKNESS_FRAC=0.28
DEPTH_COLOR=(40,200,80); DEPTH_RING_BG=(70,70,70)
FONT_PATH="Roboto-VariableFont_wdth,wght.ttf"
_REF_H = 480.0
_REF_REPS_FONT_SIZE = 28
_REF_FEEDBACK_FONT_SIZE = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE = 18

def _load_font(p, s):
    try: return ImageFont.truetype(p, s)
    except: pass
    for fb in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try: return ImageFont.truetype(fb, s)
        except: continue
    try: return ImageFont.load_default(size=s)
    except TypeError: return ImageFont.load_default()

def _scaled_font_size(ref_size, canvas_h):
    return max(10, int(round(ref_size * (canvas_h / _REF_H))))

# ============ MediaPipe ============
try:
    import mediapipe as mp
    mp_pose=mp.solutions.pose
except Exception:
    mp_pose=None

# ============ Helpers ============
def _ang(a,b,c):
    ba=np.array([a[0]-b[0],a[1]-b[1]]); bc=np.array([c[0]-b[0],c[1]-b[1]])
    den=(np.linalg.norm(ba)*np.linalg.norm(bc))+1e-9
    cos=float(np.clip(np.dot(ba,bc)/den,-1,1)); return float(np.degrees(np.arccos(cos)))

def _ema(prev,new,a): return float(new) if prev is None else (a*float(new)+(1-a)*float(prev))
def _half_floor10(x): return max(0.0,min(10.0, math.floor(x*2.0)/2.0))
def display_half_str(x): q=round(float(x)*2)/2.0; return str(int(round(q))) if abs(q-round(q))<1e-9 else f"{q:.1f}"
def score_label(s):
    s=float(s)
    if s>=9.0: return "Excellent"
    if s>=7.0: return "Good"
    if s>=5.5: return "Fair"
    return "Needs work"

def _wrap_two_lines(draw, text, font, max_width):
    words=text.split()
    if not words: return [""]
    lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        if draw.textlength(t, font=font)<=max_width: cur=t
        else:
            if cur: lines.append(cur)
            cur=w
        if len(lines)==2: break
    if cur and len(lines)<2: lines.append(cur)
    if len(lines)>=2 and draw.textlength(lines[-1], font=font)>max_width:
        last=lines[-1]+"\u2026"
        while draw.textlength(last, font=font)>max_width and len(last)>1:
            last=last[:-2]+"\u2026"
        lines[-1]=last
    return lines

def _dyn_thickness(h):
    return max(2,int(round(h*0.003))), max(3,int(round(h*0.005)))

# ============ Body-only skeleton ============
_FACE_LMS=set(); _BODY_CONNECTIONS=tuple(); _BODY_POINTS=tuple()
if mp_pose:
    _FACE_LMS={
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
        mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
    }
    _BODY_CONNECTIONS=tuple((a,b) for (a,b) in mp_pose.POSE_CONNECTIONS if a not in _FACE_LMS and b not in _FACE_LMS)
    _BODY_POINTS=tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

def draw_body_only(frame, lms, color=(255,255,255)):
    h,w=frame.shape[:2]; line, dot=_dyn_thickness(h)
    for a,b in _BODY_CONNECTIONS:
        pa, pb=lms[a], lms[b]
        ax,ay=int(pa.x*w), int(pa.y*h); bx,by=int(pb.x*w), int(pb.y*h)
        cv2.line(frame,(ax,ay),(bx,by),color,line,cv2.LINE_AA)
    for i in _BODY_POINTS:
        p=lms[i]; x,y=int(p.x*w),int(p.y*h)
        cv2.circle(frame,(x,y),dot,color,-1,cv2.LINE_AA)
    return frame

# ============ OVERLAY — HD 1080p render → downscale to frame ============
HD_H = 1080

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Render overlay at 1080p then downscale to frame resolution for sharp text."""
    h, w = frame.shape[:2]
    fh = HD_H
    fw = max(1, int(round(w * fh / h)))
    pct = float(np.clip(depth_pct, 0, 1))

    reps_font_size        = _scaled_font_size(_REF_REPS_FONT_SIZE, fh)
    feedback_font_size    = _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, fh)
    depth_label_font_size = _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, fh)
    depth_pct_font_size   = _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, fh)

    reps_font        = _load_font(FONT_PATH, reps_font_size)
    feedback_font    = _load_font(FONT_PATH, feedback_font_size)
    depth_label_font = _load_font(FONT_PATH, depth_label_font_size)
    depth_pct_font   = _load_font(FONT_PATH, depth_pct_font_size)

    _tmp   = Image.new("RGBA", (1, 1))
    _tdraw = ImageDraw.Draw(_tmp)

    sample_txt = "Reps: 00"
    pad_x = max(6, int(fw * 0.013))
    pad_y = max(4, int(fh * 0.010))
    tw    = _tdraw.textlength(sample_txt, font=reps_font)
    rep_box_w = int(tw + 2 * pad_x)
    rep_box_h = int(reps_font.size + 2 * pad_y)

    ref_h_donut = max(int(fh * 0.06), int(reps_font_size * 1.6))
    radius = int(ref_h_donut * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = max(8, int(fw * 0.016))
    cx = fw - margin - radius
    cy = max(ref_h_donut + radius // 8, radius + thick // 2 + 2)

    safe_margin = max(4, int(fh * 0.012))
    fb_pad_x    = max(8, int(fw * 0.016))
    fb_pad_y    = max(4, int(fh * 0.010))
    line_gap    = max(2, int(fh * 0.006))
    max_text_w  = fw - 2 * fb_pad_x - max(8, int(fw * 0.015))
    line_h      = feedback_font.size + max(4, int(fh * 0.008))
    block_h     = 2 * fb_pad_y + 2 * line_h + line_gap
    fb_y0       = max(0, fh - safe_margin - block_h)
    fb_y1       = fh - safe_margin

    bg_alpha_val = int(round(255 * BAR_BG_ALPHA))

    rep_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.rectangle(rep_bg, (0, 0), (rep_box_w, rep_box_h), (0, 0, 0, bg_alpha_val), -1)

    fb_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.rectangle(fb_bg, (0, fb_y0), (fw, fb_y1), (0, 0, 0, bg_alpha_val), -1)

    donut_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.circle(donut_bg, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)

    label_gap     = max(2, int(radius * 0.10))
    label_block_h = depth_label_font.size + label_gap + depth_pct_font.size
    label_by      = cy - label_block_h // 2

    rep_txt_x = pad_x
    rep_txt_y = pad_y - 1

    # Composite static layers
    canvas = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    canvas.alpha_composite(Image.fromarray(rep_bg, mode="RGBA"))
    if feedback:
        canvas.alpha_composite(Image.fromarray(fb_bg, mode="RGBA"))
    canvas.alpha_composite(Image.fromarray(donut_bg, mode="RGBA"))

    # Dynamic arc
    if pct > 0:
        arc_np = np.zeros((fh, fw, 4), dtype=np.uint8)
        start_ang = -90
        end_ang = start_ang + int(360 * pct)
        cv2.ellipse(arc_np, (cx, cy), (radius, radius), 0,
                    start_ang, end_ang, (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)
        canvas.alpha_composite(Image.fromarray(arc_np, mode="RGBA"))

    draw = ImageDraw.Draw(canvas)

    # Reps counter
    txt = f"Reps: {int(reps)}"
    draw.text((rep_txt_x, rep_txt_y), txt,
              font=reps_font, fill=(255, 255, 255, 255))

    # Donut labels
    label   = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label,   font=depth_label_font)
    pw = draw.textlength(pct_txt, font=depth_pct_font)
    draw.text((cx - int(lw // 2), label_by),
              label, font=depth_label_font, fill=(255, 255, 255, 255))
    draw.text((cx - int(pw // 2), label_by + depth_label_font.size + label_gap),
              pct_txt, font=depth_pct_font, fill=(255, 255, 255, 255))

    # Feedback text
    if feedback:
        fb_lines = _wrap_two_lines(draw, feedback, feedback_font, max_text_w)
        ty = fb_y0 + fb_pad_y
        for ln in fb_lines:
            tw2 = draw.textlength(ln, font=feedback_font)
            tx  = max(fb_pad_x, (fw - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=feedback_font, fill=(255, 255, 255, 255))
            ty += line_h + line_gap

    # Downscale to frame size and alpha blend
    canvas_small = cv2.resize(np.array(canvas), (w, h), interpolation=cv2.INTER_AREA)
    alpha       = canvas_small[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = canvas_small[:, :, [2, 1, 0]].astype(np.float32)
    result = frame.astype(np.float32) * (1.0 - alpha) + overlay_bgr * alpha
    return result.astype(np.uint8)

# ============ IMPROVED Motion Detection ============
BASE_FRAME_SKIP = 2
ACTIVE_FRAME_SKIP = 2
MOTION_DETECTION_WINDOW = 8
MOTION_VEL_THRESHOLD = 0.0010
MOTION_ACCEL_THRESHOLD = 0.0006
ELBOW_CHANGE_THRESHOLD = 5.0
COOLDOWN_FRAMES = 15
MIN_VEL_FOR_MOTION = 0.0004

class MotionDetector:
    def __init__(self):
        self.shoulder_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.elbow_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.velocity_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.raw_elbow_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.is_active = False
        self.cooldown_counter = 0
        self.last_process_frame = -999
        self.activation_count = 0
        self.last_activation_reason = ""
        
    def add_sample(self, shoulder_y, elbow_angle, raw_elbow_min, frame_idx):
        self.shoulder_history.append(shoulder_y)
        self.elbow_history.append(elbow_angle)
        self.raw_elbow_history.append(raw_elbow_min)
        
        motion_detected = False
        reason = ""
        
        if len(self.shoulder_history) >= 2:
            vel = abs(self.shoulder_history[-1] - self.shoulder_history[-2])
            self.velocity_history.append(vel)
            
            if len(self.velocity_history) >= 3:
                max_vel = max(self.velocity_history)
                recent_avg = sum(list(self.velocity_history)[-3:]) / 3
                accel = abs(self.velocity_history[-1] - self.velocity_history[-2])
                
                if max_vel > MOTION_VEL_THRESHOLD:
                    motion_detected = True; reason = f"high_vel({max_vel:.4f})"
                elif accel > MOTION_ACCEL_THRESHOLD:
                    motion_detected = True; reason = f"accel({accel:.4f})"
                elif recent_avg > MOTION_VEL_THRESHOLD * 0.65:
                    motion_detected = True; reason = f"sustained({recent_avg:.4f})"
        
        if len(self.elbow_history) >= 3:
            elbow_change = abs(self.elbow_history[-1] - self.elbow_history[-3])
            elbow_vel = abs(self.elbow_history[-1] - self.elbow_history[-2])
            if elbow_change > ELBOW_CHANGE_THRESHOLD:
                motion_detected = True; reason = f"elbow_change({elbow_change:.1f})"
            elif elbow_vel > ELBOW_CHANGE_THRESHOLD * 0.55:
                motion_detected = True; reason = f"elbow_vel({elbow_vel:.1f})"
        
        if len(self.raw_elbow_history) >= 3:
            raw_change = abs(self.raw_elbow_history[-1] - self.raw_elbow_history[-3])
            raw_vel = abs(self.raw_elbow_history[-1] - self.raw_elbow_history[-2])
            if raw_change > 11.0: motion_detected = True; reason = f"raw_spike({raw_change:.1f})"
            elif raw_vel > 7.0: motion_detected = True; reason = f"raw_vel({raw_vel:.1f})"
        
        if len(self.raw_elbow_history) >= 5:
            elbows = list(self.raw_elbow_history)
            went_down = elbows[-5] - elbows[-3] > 13
            went_up = elbows[-1] - elbows[-3] > 13
            if went_down and went_up: motion_detected = True; reason = "V_pattern"
        
        if len(self.shoulder_history) >= 5:
            diffs = [self.shoulder_history[i+1] - self.shoulder_history[i] 
                    for i in range(len(self.shoulder_history)-1)]
            if len(diffs) >= 4:
                sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
                max_diff = max(abs(d) for d in diffs)
                if sign_changes >= 1 and max_diff > MIN_VEL_FOR_MOTION:
                    motion_detected = True; reason = "direction_change"
        
        if motion_detected: self.activate(reason)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0: self.is_active = False
    
    def activate(self, reason=""):
        if not self.is_active:
            self.is_active = True; self.activation_count += 1; self.last_activation_reason = reason
        self.cooldown_counter = COOLDOWN_FRAMES
    
    def should_process(self, frame_idx):
        skip = ACTIVE_FRAME_SKIP if (self.is_active or self.cooldown_counter > 0) else BASE_FRAME_SKIP
        should = (frame_idx - self.last_process_frame) >= skip
        if should: self.last_process_frame = frame_idx
        return should
    
    def get_stats(self):
        return {"is_active": self.is_active, "cooldown": self.cooldown_counter,
                "activations": self.activation_count, "last_reason": self.last_activation_reason}

# ============ Parameters ============
ELBOW_BENT_ANGLE = 102.0
SHOULDER_MIN_DESCENT = 0.036
RESET_ASCENT = 0.024
RESET_ELBOW = 153.0
REFRACTORY_FRAMES = 1
ELBOW_EMA_ALPHA = 0.72
SHOULDER_EMA_ALPHA = 0.67
VIS_THR_STRICT = 0.29
PLANK_BODY_ANGLE_MAX = 26.0
HANDS_BELOW_SHOULDERS = 0.035
ONPUSHUP_MIN_FRAMES = 2
OFFPUSHUP_MIN_FRAMES = 5
AUTO_STOP_AFTER_EXIT_SEC = 1.5
TAIL_NOPOSE_STOP_SEC = 1.0

# Feedback
FB_ERROR_DEPTH = "Go deeper (chest to floor)"
FB_ERROR_HIPS = "Keep hips level (don't sag or pike)"
FB_ERROR_LOCKOUT = "Fully lockout arms at top"
FB_ERROR_ELBOWS = "Keep elbows at 45\u00b0 (not flared)"
PERF_TIP_SLOW_DOWN = "Lower slowly for better control"
PERF_TIP_TEMPO = "Try 2-1-2 tempo (down-pause-up)"
PERF_TIP_BREATHING = "Breathe: inhale down, exhale up"
PERF_TIP_CORE = "Engage core throughout movement"

FB_W_DEPTH = 1.2; FB_W_HIPS = 1.0; FB_W_LOCKOUT = 0.9; FB_W_ELBOWS = 0.7
FB_WEIGHTS = {FB_ERROR_DEPTH: FB_W_DEPTH, FB_ERROR_HIPS: FB_W_HIPS, FB_ERROR_LOCKOUT: FB_W_LOCKOUT, FB_ERROR_ELBOWS: FB_W_ELBOWS}
FB_DEFAULT_WEIGHT = 0.5; PENALTY_MIN_IF_ANY = 0.5

FORM_ERROR_PRIORITY = [FB_ERROR_DEPTH, FB_ERROR_LOCKOUT, FB_ERROR_HIPS, FB_ERROR_ELBOWS]
PERF_TIP_PRIORITY = [PERF_TIP_SLOW_DOWN, PERF_TIP_TEMPO, PERF_TIP_BREATHING, PERF_TIP_CORE]

DEPTH_EXCELLENT_ANGLE = 85.0; DEPTH_GOOD_ANGLE = 95.0; DEPTH_FAIR_ANGLE = 105.0; DEPTH_POOR_ANGLE = 115.0
HIP_EXCELLENT = 8.0; HIP_GOOD = 15.0; HIP_FAIR = 22.0; HIP_POOR = 30.0
LOCKOUT_EXCELLENT = 175.0; LOCKOUT_GOOD = 170.0; LOCKOUT_FAIR = 165.0; LOCKOUT_POOR = 160.0
FLARE_EXCELLENT = 45.0; FLARE_GOOD = 55.0; FLARE_FAIR = 65.0; FLARE_POOR = 75.0
DESCENT_SPEED_IDEAL = 0.0010; DESCENT_SPEED_FAST = 0.0012

DEPTH_FAIL_MIN_REPS = 2; HIPS_FAIL_MIN_REPS = 2; LOCKOUT_FAIL_MIN_REPS = 2
FLARE_FAIL_MIN_REPS = 2; TEMPO_CHECK_MIN_REPS = 1
DEPTH_ERROR_ANGLE = 110.0; LOCKOUT_ERROR_ANGLE = 165.0

BURST_FRAMES = 4
INFLECT_VEL_THR = 0.0027

DEBUG_ONPUSHUP = bool(int(os.getenv("DEBUG_ONPUSHUP", "0")))
DEBUG_MOTION = bool(int(os.getenv("DEBUG_MOTION", "0")))
DEBUG_GRADING = bool(int(os.getenv("DEBUG_GRADING", "1")))

MIN_CYCLE_ELBOW_SAMPLES = 4
ROBUST_BOTTOM_PERCENTILE = 25; ROBUST_TOP_PERCENTILE = 75; ROBUST_CONFIRMED_PERCENTILE = 10


def run_pushup_analysis(video_path,
                        frame_skip=None,
                        scale=0.4,
                        output_path="pushup_analyzed.mp4",
                        feedback_path="pushup_feedback.txt",
                        preserve_quality=False,
                        encode_crf=None,
                        return_video=True,
                        fast_mode=None):
    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    model_complexity = 0 if fast_mode else 1
    if fast_mode:
        return_video = False
        scale = min(scale, 0.35)

    if preserve_quality:
        scale=1.0; encode_crf=18 if encode_crf is None else encode_crf
    else:
        encode_crf=23 if encode_crf is None else encode_crf

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): return _ret_err("Could not open video", feedback_path)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    # effective_fps used for timing calculations only; output video uses fps_in for correct playback speed
    effective_fps = max(1.0, fps_in / max(1, BASE_FRAME_SKIP))
    sec_to_frames = lambda s: max(1, int(s * effective_fps))

    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=None; frame_idx=0

    rep_count=0; good_reps=0; bad_reps=0; rep_reports=[]; all_scores=[]

    motion_detector = MotionDetector()
    frames_processed = 0
    frames_skipped = 0

    LSH=mp_pose.PoseLandmark.LEFT_SHOULDER.value;  RSH=mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LE =mp_pose.PoseLandmark.LEFT_ELBOW.value;     RE =mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW =mp_pose.PoseLandmark.LEFT_WRIST.value;     RW =mp_pose.PoseLandmark.RIGHT_WRIST.value
    LH =mp_pose.PoseLandmark.LEFT_HIP.value;       RH =mp_pose.PoseLandmark.RIGHT_HIP.value
    LA =mp_pose.PoseLandmark.LEFT_ANKLE.value;     RA =mp_pose.PoseLandmark.RIGHT_ANKLE.value

    def _pick_side_dyn(lms):
        vL=lms[LSH].visibility+lms[LE].visibility+lms[LW].visibility
        vR=lms[RSH].visibility+lms[RE].visibility+lms[RW].visibility
        return ("LEFT",LSH,LE,LW) if vL>=vR else ("RIGHT",RSH,RE,RW)

    elbow_ema=None; shoulder_ema=None; shoulder_prev=None; shoulder_vel_prev=None
    baseline_shoulder_y=None
    desc_base_shoulder=None; allow_new_bottom=True; last_bottom_frame=-10**9
    cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False

    onpushup=False; onpushup_streak=0; offpushup_streak=0
    offpushup_frames_since_any_rep=0; nopose_frames_since_any_rep=0

    session_form_errors=set()
    session_perf_tips=set()
    rt_fb_msg=None; rt_fb_hold=0

    cycle_tip_deeper=False; cycle_tip_hips=False; cycle_tip_lockout=False; cycle_tip_elbows=False
    cycle_bottom_samples=[]; cycle_top_samples=[]; confirmed_bottom_samples=[]
    depth_fail_count=0; hips_fail_count=0; lockout_fail_count=0; flare_fail_count=0
    depth_already_reported=False; hips_already_reported=False
    lockout_already_reported=False; flare_already_reported=False
    
    fast_descent_count=0
    tempo_already_reported=False

    bottom_phase_min_elbow=None
    top_phase_max_elbow=None
    cycle_max_hip_misalign=None
    cycle_max_flare=None
    cycle_max_descent_vel=0.0
    in_descent_phase=False

    OFFPUSHUP_STOP_FRAMES=sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES=sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES=sec_to_frames(0.8)
    REARM_ASCENT_EFF=max(RESET_ASCENT*0.58, 0.014)

    burst_cntr=0

    with mp_pose.Pose(model_complexity=model_complexity, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            process_now = False
            if burst_cntr > 0:
                process_now = True; burst_cntr -= 1
            elif motion_detector.should_process(frame_idx):
                process_now = True
            
            if not process_now:
                frames_skipped += 1; continue
            
            frames_processed += 1

            if scale != 1.0:
                frame=cv2.resize(frame,(0,0),fx=scale,fy=scale)
            h,w=frame.shape[:2]

            if return_video and out is None:
                out=cv2.VideoWriter(output_path, fourcc, fps_in, (orig_w, orig_h))

            res=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            depth_live=0.0

            if not res.pose_landmarks:
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep+1) if rep_count>0 else 0
                if rep_count>0 and nopose_frames_since_any_rep>=NOPOSE_STOP_FRAMES: break
                if return_video and out is not None:
                    _fo = cv2.resize(frame, (orig_w, orig_h))
                    _fo = draw_overlay(_fo, reps=rep_count,
                                       feedback=(rt_fb_msg if rt_fb_hold>0 else None),
                                       depth_pct=0.0)
                    out.write(_fo)
                if rt_fb_hold>0: rt_fb_hold-=1
                continue

            nopose_frames_since_any_rep=0
            lms=res.pose_landmarks.landmark
            side,S,E,W=_pick_side_dyn(lms)

            min_vis=min(lms[S].visibility,lms[E].visibility,lms[W].visibility,lms[LH].visibility,lms[RH].visibility)
            vis_strict_ok=(min_vis>=VIS_THR_STRICT)

            shoulder_raw=float(lms[S].y)
            raw_elbow_L=_ang((lms[LSH].x,lms[LSH].y),(lms[LE].x,lms[LE].y),(lms[LW].x,lms[LW].y))
            raw_elbow_R=_ang((lms[RSH].x,lms[RSH].y),(lms[RE].x,lms[RE].y),(lms[RW].x,lms[RW].y))
            raw_elbow=raw_elbow_L if side=="LEFT" else raw_elbow_R
            raw_elbow_min=min(raw_elbow_L,raw_elbow_R)

            elbow_ema=_ema(elbow_ema,raw_elbow,ELBOW_EMA_ALPHA)
            shoulder_ema=_ema(shoulder_ema,shoulder_raw,SHOULDER_EMA_ALPHA)
            shoulder_y=shoulder_ema; elbow_angle=elbow_ema
            
            motion_detector.add_sample(shoulder_y, elbow_angle, raw_elbow_min, frame_idx)
            
            if baseline_shoulder_y is None: baseline_shoulder_y=shoulder_y
            depth_live=float(np.clip((shoulder_y-baseline_shoulder_y)/max(0.10,SHOULDER_MIN_DESCENT*1.2),0.0,1.0))

            body_angle = _calculate_body_angle(lms, LSH, RSH, LH, RH, LA, RA)
            hands_position = (lms[LW].y > lms[LSH].y - HANDS_BELOW_SHOULDERS) and (lms[RW].y > lms[RSH].y - HANDS_BELOW_SHOULDERS)
            in_plank = (body_angle <= PLANK_BODY_ANGLE_MAX) and hands_position

            if vis_strict_ok and in_plank:
                onpushup_streak+=1; offpushup_streak=0
            else:
                offpushup_streak+=1; onpushup_streak=0

            if (not onpushup) and onpushup_streak>=ONPUSHUP_MIN_FRAMES:
                onpushup=True
                desc_base_shoulder=None; allow_new_bottom=True
                cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_tip_deeper=False; cycle_tip_hips=False; cycle_tip_lockout=False; cycle_tip_elbows=False
                cycle_bottom_samples=[]; cycle_top_samples=[]; confirmed_bottom_samples=[]
                bottom_phase_min_elbow=None; top_phase_max_elbow=None
                cycle_max_hip_misalign=None; cycle_max_flare=None
                cycle_max_descent_vel=0.0
                in_descent_phase=False
                motion_detector.activate("enter_plank")

            if onpushup and offpushup_streak>=OFFPUSHUP_MIN_FRAMES:
                robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow, confirmed_bottom=confirmed_bottom_samples)
                cycle_has_issues = _evaluate_cycle_form(lms, robust_bottom_elbow, robust_top_elbow,
                                   cycle_max_hip_misalign, cycle_max_flare, cycle_max_descent_vel,
                                   depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                                   fast_descent_count, depth_already_reported, hips_already_reported,
                                   lockout_already_reported, flare_already_reported, tempo_already_reported,
                                   session_form_errors, session_perf_tips, rep_count, locals())

                if (not counted_this_cycle) and (cycle_max_descent>=SHOULDER_MIN_DESCENT) and (cycle_min_elbow<=ELBOW_BENT_ANGLE):
                    _count_rep(rep_reports,rep_count,cycle_min_elbow,
                               desc_base_shoulder if desc_base_shoulder is not None else shoulder_y,
                               baseline_shoulder_y+cycle_max_descent if baseline_shoulder_y is not None else shoulder_y,
                               all_scores, cycle_has_issues,
                               robust_bottom_elbow, robust_top_elbow, cycle_max_hip_misalign, cycle_max_flare)
                    rep_count+=1
                    if cycle_has_issues: bad_reps+=1
                    else: good_reps+=1

                onpushup=False; offpushup_frames_since_any_rep=0
                desc_base_shoulder=None; cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_tip_deeper=False; cycle_tip_hips=False; cycle_tip_lockout=False; cycle_tip_elbows=False
                cycle_bottom_samples=[]; cycle_top_samples=[]; confirmed_bottom_samples=[]
                bottom_phase_min_elbow=None; top_phase_max_elbow=None
                cycle_max_hip_misalign=None; cycle_max_flare=None
                cycle_max_descent_vel=0.0

            if (not onpushup) and rep_count>0:
                offpushup_frames_since_any_rep+=1
                if offpushup_frames_since_any_rep>=OFFPUSHUP_STOP_FRAMES: break

            shoulder_vel=0.0 if shoulder_prev is None else (shoulder_y-shoulder_prev)
            cur_rt=None

            if onpushup and (desc_base_shoulder is not None):
                near_inflect = (abs(shoulder_vel) <= INFLECT_VEL_THR)
                sign_flip = (shoulder_vel_prev is not None) and ((shoulder_vel_prev < 0 and shoulder_vel >= 0) or (shoulder_vel_prev > 0 and shoulder_vel <= 0))
                if near_inflect or sign_flip:
                    burst_cntr = max(burst_cntr, BURST_FRAMES)
                    motion_detector.activate("inflection")
            shoulder_vel_prev = shoulder_vel

            if onpushup and vis_strict_ok:
                if desc_base_shoulder is None:
                    if shoulder_vel>abs(INFLECT_VEL_THR):
                        desc_base_shoulder=shoulder_y
                        cycle_max_descent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                        cycle_bottom_samples=[]; cycle_top_samples=[]; confirmed_bottom_samples=[]
                        bottom_phase_min_elbow=None; top_phase_max_elbow=None
                        cycle_max_hip_misalign=None; cycle_max_flare=None
                        cycle_max_descent_vel=0.0
                        in_descent_phase=True
                        motion_detector.activate("start_descent")
                else:
                    cycle_max_descent=max(cycle_max_descent,(shoulder_y-desc_base_shoulder))
                    cycle_min_elbow=min(cycle_min_elbow,elbow_angle)
                    
                    vel_abs = abs(shoulder_vel)
                    if shoulder_vel > 0 and in_descent_phase:
                        cycle_max_descent_vel = max(cycle_max_descent_vel, vel_abs)
                    if vel_abs > 0.0005:
                        cycle_max_descent_vel = max(cycle_max_descent_vel, vel_abs)

                    min_elb_now = min(raw_elbow_L, raw_elbow_R)
                    cycle_bottom_samples.append(min_elb_now)
                    if len(cycle_bottom_samples) > 40: cycle_bottom_samples.pop(0)
                    if bottom_phase_min_elbow is None: bottom_phase_min_elbow = min_elb_now
                    else: bottom_phase_min_elbow = min(bottom_phase_min_elbow, min_elb_now)

                    max_elb_now = max(raw_elbow_L, raw_elbow_R)
                    cycle_top_samples.append(max_elb_now)
                    if len(cycle_top_samples) > 40: cycle_top_samples.pop(0)
                    if top_phase_max_elbow is None: top_phase_max_elbow = max_elb_now
                    else: top_phase_max_elbow = max(top_phase_max_elbow, max_elb_now)

                    hip_misalign = _calculate_hip_misalignment(lms, LSH, RSH, LH, RH, LA, RA)
                    if cycle_max_hip_misalign is None: cycle_max_hip_misalign = hip_misalign
                    else: cycle_max_hip_misalign = max(cycle_max_hip_misalign, hip_misalign)

                    elbow_flare = _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW)
                    if cycle_max_flare is None: cycle_max_flare = elbow_flare
                    else: cycle_max_flare = max(cycle_max_flare, elbow_flare)

                    reset_by_asc=(desc_base_shoulder is not None) and ((desc_base_shoulder-shoulder_y)>=RESET_ASCENT)
                    reset_by_elb=(elbow_angle>=RESET_ELBOW)
                    
                    if shoulder_vel < 0 and in_descent_phase: in_descent_phase = False
                    
                    if reset_by_asc or reset_by_elb:
                        robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(
                            cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow,
                            confirmed_bottom=confirmed_bottom_samples)
                        cycle_has_issues = _evaluate_cycle_form(lms, robust_bottom_elbow, robust_top_elbow,
                                           cycle_max_hip_misalign, cycle_max_flare, cycle_max_descent_vel,
                                           depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                                           fast_descent_count, depth_already_reported, hips_already_reported,
                                           lockout_already_reported, flare_already_reported, tempo_already_reported,
                                           session_form_errors, session_perf_tips, rep_count, locals())

                        if (not counted_this_cycle) and (cycle_max_descent>=SHOULDER_MIN_DESCENT) and (cycle_min_elbow<=ELBOW_BENT_ANGLE):
                            _count_rep(rep_reports,rep_count,cycle_min_elbow, desc_base_shoulder,
                                       baseline_shoulder_y+cycle_max_descent if baseline_shoulder_y is not None else shoulder_y,
                                       all_scores, cycle_has_issues,
                                       robust_bottom_elbow, robust_top_elbow, cycle_max_hip_misalign, cycle_max_flare)
                            rep_count+=1
                            if cycle_has_issues: bad_reps+=1
                            else: good_reps+=1

                        desc_base_shoulder=shoulder_y
                        cycle_max_descent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                        allow_new_bottom=True
                        cycle_bottom_samples=[]; cycle_top_samples=[]; confirmed_bottom_samples=[]
                        bottom_phase_min_elbow=None; top_phase_max_elbow=None
                        cycle_max_hip_misalign=None; cycle_max_flare=None
                        cycle_max_descent_vel=0.0
                        cycle_tip_deeper=False; cycle_tip_hips=False; cycle_tip_lockout=False; cycle_tip_elbows=False
                        in_descent_phase=True
                        motion_detector.activate("reset")

                descent_amt=0.0 if desc_base_shoulder is None else (shoulder_y-desc_base_shoulder)

                at_bottom=(elbow_angle<=ELBOW_BENT_ANGLE) and (descent_amt>=SHOULDER_MIN_DESCENT)
                raw_bottom=(raw_elbow_min<=(ELBOW_BENT_ANGLE+9.0)) and (descent_amt>=SHOULDER_MIN_DESCENT*0.87)
                at_bottom=at_bottom or raw_bottom
                can_cnt=(frame_idx - last_bottom_frame) >= REFRACTORY_FRAMES

                if at_bottom:
                    confirmed_bottom_samples.append(raw_elbow_min)
                    if len(confirmed_bottom_samples) > 15: confirmed_bottom_samples.pop(0)

                if at_bottom and allow_new_bottom and can_cnt and (not counted_this_cycle):
                    rep_has_issues = False
                    robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow, confirmed_bottom=confirmed_bottom_samples)
                    if robust_bottom_elbow and robust_bottom_elbow > DEPTH_ERROR_ANGLE: rep_has_issues = True
                    if robust_top_elbow and robust_top_elbow < LOCKOUT_ERROR_ANGLE: rep_has_issues = True
                    if cycle_max_hip_misalign and cycle_max_hip_misalign > HIP_FAIR: rep_has_issues = True
                    if cycle_max_flare and cycle_max_flare > FLARE_FAIR: rep_has_issues = True
                    
                    _count_rep(rep_reports,rep_count,elbow_angle,
                               desc_base_shoulder if desc_base_shoulder is not None else shoulder_y, shoulder_y,
                               all_scores, rep_has_issues,
                               robust_bottom_elbow if robust_bottom_elbow else raw_elbow_min,
                               robust_top_elbow if robust_top_elbow else max(raw_elbow_L, raw_elbow_R),
                               cycle_max_hip_misalign if cycle_max_hip_misalign else 0.0,
                               cycle_max_flare if cycle_max_flare else 0.0)
                    rep_count+=1
                    if rep_has_issues: bad_reps+=1
                    else: good_reps+=1
                    last_bottom_frame=frame_idx; allow_new_bottom=False; counted_this_cycle=True
                    top_phase_max_elbow = max(raw_elbow_L, raw_elbow_R)
                    in_descent_phase=False
                    motion_detector.activate("count_rep")

                if (allow_new_bottom is False) and (last_bottom_frame>0):
                    if shoulder_prev is not None and (shoulder_prev - shoulder_y) > 0 and (desc_base_shoulder is not None):
                        if ((desc_base_shoulder + cycle_max_descent) - shoulder_y) >= REARM_ASCENT_EFF:
                            allow_new_bottom=True

                if at_bottom and not cycle_tip_deeper:
                    robust_bottom_elbow, _ = _robust_cycle_elbows(cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow, confirmed_bottom=confirmed_bottom_samples)
                    if robust_bottom_elbow and robust_bottom_elbow > DEPTH_ERROR_ANGLE:
                        cycle_tip_deeper = True
                        depth_fail_count += 1
                        if depth_fail_count >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                            session_form_errors.add(FB_ERROR_DEPTH)
                            depth_already_reported = True
                            cur_rt = FB_ERROR_DEPTH

            else:
                desc_base_shoulder=None; allow_new_bottom=True

            if cur_rt:
                if cur_rt!=rt_fb_msg: rt_fb_msg=cur_rt; rt_fb_hold=RT_FB_HOLD_FRAMES
                else: rt_fb_hold=max(rt_fb_hold,RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold>0: rt_fb_hold-=1

            if return_video and out is not None:
                _fo = cv2.resize(frame, (orig_w, orig_h))
                draw_body_only(_fo, lms)
                _fo = draw_overlay(_fo, reps=rep_count,
                                   feedback=(rt_fb_msg if rt_fb_hold>0 else None),
                                   depth_pct=depth_live)
                out.write(_fo)

            if shoulder_y is not None: shoulder_prev=shoulder_y

    if onpushup and (not counted_this_cycle) and (cycle_max_descent>=SHOULDER_MIN_DESCENT) and (cycle_min_elbow<=ELBOW_BENT_ANGLE):
        robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow, confirmed_bottom=confirmed_bottom_samples)
        cycle_has_issues = _evaluate_cycle_form(lms, robust_bottom_elbow, robust_top_elbow,
                           cycle_max_hip_misalign, cycle_max_flare, cycle_max_descent_vel,
                           depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                           fast_descent_count, depth_already_reported, hips_already_reported,
                           lockout_already_reported, flare_already_reported, tempo_already_reported,
                           session_form_errors, session_perf_tips, rep_count, locals())
        _count_rep(rep_reports,rep_count,cycle_min_elbow,
                   desc_base_shoulder if desc_base_shoulder is not None else (baseline_shoulder_y or 0.0),
                   (baseline_shoulder_y + cycle_max_descent) if baseline_shoulder_y is not None else (baseline_shoulder_y or 0.0),
                   all_scores, cycle_has_issues,
                   robust_bottom_elbow, robust_top_elbow, cycle_max_hip_misalign, cycle_max_flare)
        rep_count+=1
        if cycle_has_issues: bad_reps+=1
        else: good_reps+=1

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    if rep_count==0: 
        technique_score=0.0
    else:
        if session_form_errors:
            penalty = sum(FB_WEIGHTS.get(m,FB_DEFAULT_WEIGHT) for m in set(session_form_errors))
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
        else:
            penalty = 0.0
        technique_score=_half_floor10(max(0.0,10.0-penalty))

    if technique_score == 10.0 and rep_count > 0:
        good_reps = rep_count; bad_reps = 0

    form_errors_list = [err for err in FORM_ERROR_PRIORITY if err in session_form_errors]
    perf_tips_list = [tip for tip in PERF_TIP_PRIORITY if tip in session_perf_tips]
    primary_form_error = form_errors_list[0] if form_errors_list else None
    primary_perf_tip = perf_tips_list[0] if perf_tips_list else None

    total_frames = frames_processed + frames_skipped
    efficiency = (frames_skipped / total_frames * 100) if total_frames > 0 else 0

    try:
        with open(feedback_path,"w",encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Good Reps: {int(good_reps)} | Bad Reps: {int(bad_reps)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            if form_errors_list:
                f.write("\n Form Corrections (affecting score):\n")
                for ln in form_errors_list: f.write(f"- {ln}\n")
            if perf_tips_list:
                f.write("\n Performance Tips (not affecting score):\n")
                for ln in perf_tips_list: f.write(f"- {ln}\n")
    except Exception: pass

    final_path=""
    if return_video and os.path.exists(output_path):
        encoded_path=output_path.replace(".mp4","_encoded.mp4")
        try:
            subprocess.run(["ffmpeg","-y","-i",output_path,"-c:v","libx264","-preset","medium",
                            "-crf",str(int(encode_crf)),"-movflags","+faststart","-pix_fmt","yuv420p",
                            encoded_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            final_path=encoded_path if os.path.exists(encoded_path) else output_path
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                try: os.remove(output_path)
                except: pass
        except Exception:
            final_path=output_path if os.path.exists(output_path) else ""

    result = {
        "squat_count": int(rep_count),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": form_errors_list if form_errors_list else (["Great form! Keep it up \U0001f4aa"] if technique_score == 10.0 and rep_count > 0 else []),
        "tips": perf_tips_list,
        "reps": rep_reports,
        "video_path": final_path if return_video else "",
        "feedback_path": feedback_path,
        "processing_stats": {
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "efficiency_percent": round(efficiency, 1),
            "motion_activations": motion_detector.activation_count
        }
    }
    
    if primary_perf_tip: result["form_tip"] = primary_perf_tip
    if primary_form_error: result["primary_form_error"] = primary_form_error
    if primary_perf_tip: result["primary_perf_tip"] = primary_perf_tip

    return result

# ============ Helper Functions ============

def _robust_cycle_elbows(bottom_samples, top_samples, fallback_bottom=None, fallback_top=None, confirmed_bottom=None):
    robust_bottom = fallback_bottom
    robust_top = fallback_top

    if confirmed_bottom and len(confirmed_bottom) >= 1:
        arr = np.array(confirmed_bottom, dtype=np.float32)
        robust_bottom = float(np.percentile(arr, ROBUST_CONFIRMED_PERCENTILE))
    elif bottom_samples:
        arr = np.array(bottom_samples, dtype=np.float32)
        robust_bottom = float(np.percentile(arr, ROBUST_BOTTOM_PERCENTILE))

    if fallback_bottom is not None and robust_bottom is not None:
        robust_bottom = min(robust_bottom, fallback_bottom)

    if top_samples:
        arr = np.array(top_samples, dtype=np.float32)
        robust_top = float(np.percentile(arr, ROBUST_TOP_PERCENTILE))

    return robust_bottom, robust_top

def _calculate_body_angle(lms, LSH, RSH, LH, RH, LA, RA):
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    mid_ank = ((lms[LA].x + lms[RA].x)/2.0, (lms[LA].y + lms[RA].y)/2.0)
    dx = mid_sh[0] - mid_ank[0]; dy = mid_sh[1] - mid_ank[1]
    return abs(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9)))

def _calculate_hip_misalignment(lms, LSH, RSH, LH, RH, LA, RA):
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    mid_hp = ((lms[LH].x + lms[RH].x)/2.0, (lms[LH].y + lms[RH].y)/2.0)
    mid_ank = ((lms[LA].x + lms[RA].x)/2.0, (lms[LA].y + lms[RA].y)/2.0)
    return abs(180.0 - _ang(mid_sh, mid_hp, mid_ank))

def _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW):
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    
    left_vec_sh = (mid_sh[0] - lms[LSH].x, mid_sh[1] - lms[LSH].y)
    left_vec_elb = (lms[LE].x - lms[LSH].x, lms[LE].y - lms[LSH].y)
    left_angle = abs(math.degrees(math.atan2(
        left_vec_sh[0]*left_vec_elb[1] - left_vec_sh[1]*left_vec_elb[0],
        left_vec_sh[0]*left_vec_elb[0] + left_vec_sh[1]*left_vec_elb[1])))
    
    right_vec_sh = (mid_sh[0] - lms[RSH].x, mid_sh[1] - lms[RSH].y)
    right_vec_elb = (lms[RE].x - lms[RSH].x, lms[RE].y - lms[RSH].y)
    right_angle = abs(math.degrees(math.atan2(
        right_vec_sh[0]*right_vec_elb[1] - right_vec_sh[1]*right_vec_elb[0],
        right_vec_sh[0]*right_vec_elb[0] + right_vec_sh[1]*right_vec_elb[1])))
    
    return max(left_angle, right_angle)

def _evaluate_cycle_form(lms, bottom_phase_min_elbow, top_phase_max_elbow,
                        cycle_max_hip_misalign, cycle_max_flare, cycle_max_descent_vel,
                        depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                        fast_descent_count, depth_already_reported, hips_already_reported,
                        lockout_already_reported, flare_already_reported, tempo_already_reported,
                        session_form_errors, session_perf_tips, rep_count, local_vars):
    
    has_depth_issue = has_lockout_issue = has_hips_issue = has_flare_issue = False
    
    if bottom_phase_min_elbow is not None and local_vars.get("cycle_bottom_samples") and len(local_vars["cycle_bottom_samples"]) >= MIN_CYCLE_ELBOW_SAMPLES:
        if bottom_phase_min_elbow > DEPTH_ERROR_ANGLE:
            has_depth_issue = True
            local_vars['cycle_tip_deeper'] = True
            local_vars['depth_fail_count'] += 1
            if local_vars['depth_fail_count'] >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                session_form_errors.add(FB_ERROR_DEPTH)
                local_vars['depth_already_reported'] = True

    if top_phase_max_elbow is not None and local_vars.get("cycle_top_samples") and len(local_vars["cycle_top_samples"]) >= MIN_CYCLE_ELBOW_SAMPLES:
        if top_phase_max_elbow < LOCKOUT_ERROR_ANGLE:
            has_lockout_issue = True
            local_vars['cycle_tip_lockout'] = True
            local_vars['lockout_fail_count'] += 1
            if local_vars['lockout_fail_count'] >= LOCKOUT_FAIL_MIN_REPS and not lockout_already_reported:
                session_form_errors.add(FB_ERROR_LOCKOUT)
                local_vars['lockout_already_reported'] = True

    if cycle_max_hip_misalign is not None:
        if cycle_max_hip_misalign > HIP_FAIR:
            has_hips_issue = True
            local_vars['cycle_tip_hips'] = True
            local_vars['hips_fail_count'] += 1
            if local_vars['hips_fail_count'] >= HIPS_FAIL_MIN_REPS and not hips_already_reported:
                session_form_errors.add(FB_ERROR_HIPS)
                local_vars['hips_already_reported'] = True

    if cycle_max_flare is not None:
        if cycle_max_flare > FLARE_FAIR:
            has_flare_issue = True
            local_vars['cycle_tip_elbows'] = True
            local_vars['flare_fail_count'] += 1
            if local_vars['flare_fail_count'] >= FLARE_FAIL_MIN_REPS and not flare_already_reported:
                session_form_errors.add(FB_ERROR_ELBOWS)
                local_vars['flare_already_reported'] = True
    
    if rep_count >= TEMPO_CHECK_MIN_REPS and not tempo_already_reported:
        if cycle_max_descent_vel > DESCENT_SPEED_FAST:
            local_vars['fast_descent_count'] += 1
            if local_vars['fast_descent_count'] >= 1:
                session_perf_tips.add(PERF_TIP_SLOW_DOWN)
                session_perf_tips.add(PERF_TIP_TEMPO)
                local_vars['tempo_already_reported'] = True
    
    return has_depth_issue or has_lockout_issue or has_hips_issue or has_flare_issue

def _count_rep(rep_reports, rep_count, bottom_elbow, descent_from, bottom_shoulder_y, all_scores, rep_has_tip,
               bottom_phase_min_elbow, top_phase_max_elbow, cycle_max_hip_misalign, cycle_max_flare):
    
    depth_score = lockout_score = hips_score = flare_score = 10.0
    
    if bottom_phase_min_elbow:
        if bottom_phase_min_elbow <= DEPTH_EXCELLENT_ANGLE: depth_score = 10.0
        elif bottom_phase_min_elbow <= DEPTH_GOOD_ANGLE: depth_score = 9.0
        elif bottom_phase_min_elbow <= DEPTH_FAIR_ANGLE: depth_score = 7.5
        elif bottom_phase_min_elbow <= DEPTH_POOR_ANGLE: depth_score = 5.0
        else: depth_score = 3.0
    
    if top_phase_max_elbow:
        if top_phase_max_elbow >= LOCKOUT_EXCELLENT: lockout_score = 10.0
        elif top_phase_max_elbow >= LOCKOUT_GOOD: lockout_score = 9.0
        elif top_phase_max_elbow >= LOCKOUT_FAIR: lockout_score = 7.5
        elif top_phase_max_elbow >= LOCKOUT_POOR: lockout_score = 5.0
        else: lockout_score = 3.0
    
    if cycle_max_hip_misalign is not None:
        if cycle_max_hip_misalign <= HIP_EXCELLENT: hips_score = 10.0
        elif cycle_max_hip_misalign <= HIP_GOOD: hips_score = 9.0
        elif cycle_max_hip_misalign <= HIP_FAIR: hips_score = 7.5
        elif cycle_max_hip_misalign <= HIP_POOR: hips_score = 5.0
        else: hips_score = 3.0
    
    if cycle_max_flare is not None:
        if cycle_max_flare <= FLARE_EXCELLENT: flare_score = 10.0
        elif cycle_max_flare <= FLARE_GOOD: flare_score = 9.0
        elif cycle_max_flare <= FLARE_FAIR: flare_score = 7.5
        elif cycle_max_flare <= FLARE_POOR: flare_score = 5.0
        else: flare_score = 3.0
    
    rep_score = (depth_score * 0.35 + lockout_score * 0.25 + hips_score * 0.25 + flare_score * 0.15)
    rep_score = round(rep_score * 2) / 2
    all_scores.append(rep_score)
    
    rep_reports.append({
        "rep_index": int(rep_count+1),
        "score": float(rep_score),
        "good": bool(rep_score >= 9.0),
        "bottom_elbow": float(bottom_elbow),
        "descent_from": float(descent_from),
        "bottom_shoulder_y": float(bottom_shoulder_y),
        "detailed_scores": {"depth": float(depth_score), "lockout": float(lockout_score),
                           "hips": float(hips_score), "flare": float(flare_score)},
        "measurements": {
            "bottom_elbow_angle": float(bottom_phase_min_elbow) if bottom_phase_min_elbow else None,
            "top_elbow_angle": float(top_phase_max_elbow) if top_phase_max_elbow else None,
            "hip_misalignment": float(cycle_max_hip_misalign) if cycle_max_hip_misalign else None,
            "elbow_flare": float(cycle_max_flare) if cycle_max_flare else None
        }
    })

def _ret_err(msg, feedback_path):
    try:
        with open(feedback_path,"w",encoding="utf-8") as f: f.write(msg+"\n")
    except Exception: pass
    return {
        "squat_count": 0, "technique_score": 0.0,
        "technique_score_display": display_half_str(0.0),
        "technique_label": score_label(0.0),
        "good_reps": 0, "bad_reps": 0,
        "feedback": [], "tips": [],
        "reps": [], "video_path": "", "feedback_path": feedback_path
    }

def run_analysis(*args, **kwargs):
    return run_pushup_analysis(*args, **kwargs)
