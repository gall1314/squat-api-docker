# -*- coding: utf-8 -*-
# dips_analysis.py — complete dips counter with form feedback

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

def _load_font(p,s):
    try: return ImageFont.truetype(p,s)
    except: return ImageFont.load_default()

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
        last=lines[-1]+"…"
        while draw.textlength(last, font=font)>max_width and len(last)>1:
            last=last[:-2]+"…"
        lines[-1]=last
    return lines

def _dyn_thickness(h):
    return max(2,int(round(h*0.002))), max(3,int(round(h*0.004)))

def _dist_xy(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

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

HD_H = 1080
_OVERLAY_CACHE = {}


def _build_overlay_cache(w, h):
    """Build and cache fonts + static 1080p layers keyed on frame resolution."""
    fh = HD_H
    fw = max(1, int(round(w * fh / h)))

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

    label_gap     = max(2, int(radius * 0.10))
    label_block_h = depth_label_font.size + label_gap + depth_pct_font.size
    label_by      = cy - label_block_h // 2

    rep_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.rectangle(rep_bg, (0, 0), (rep_box_w, rep_box_h), (0, 0, 0, bg_alpha_val), -1)

    fb_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.rectangle(fb_bg, (0, fb_y0), (fw, fb_y1), (0, 0, 0, bg_alpha_val), -1)

    donut_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.circle(donut_bg, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)

    return {
        "fw": fw, "fh": fh,
        "reps_font": reps_font, "feedback_font": feedback_font,
        "depth_label_font": depth_label_font, "depth_pct_font": depth_pct_font,
        "radius": radius, "thick": thick, "cx": cx, "cy": cy,
        "rep_txt_x": pad_x, "rep_txt_y": pad_y - 1,
        "fb_pad_x": fb_pad_x, "fb_pad_y": fb_pad_y,
        "fb_y0": fb_y0, "fb_y1": fb_y1,
        "line_h": line_h, "line_gap": line_gap,
        "max_text_w": max_text_w,
        "label_by": label_by, "label_gap": label_gap,
        "rep_bg_pil":   Image.fromarray(rep_bg,   mode="RGBA"),
        "fb_bg_pil":    Image.fromarray(fb_bg,    mode="RGBA"),
        "donut_bg_pil": Image.fromarray(donut_bg, mode="RGBA"),
    }


def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Render overlay at 1080p (cached) then downscale to frame resolution for sharp text."""
    h, w = frame.shape[:2]
    key = (w, h)
    if key not in _OVERLAY_CACHE:
        _OVERLAY_CACHE[key] = _build_overlay_cache(w, h)
    c = _OVERLAY_CACHE[key]

    pct = float(np.clip(depth_pct, 0, 1))
    fw, fh = c["fw"], c["fh"]
    cx, cy, radius, thick = c["cx"], c["cy"], c["radius"], c["thick"]

    canvas = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    canvas.alpha_composite(c["rep_bg_pil"])
    if feedback:
        canvas.alpha_composite(c["fb_bg_pil"])
    canvas.alpha_composite(c["donut_bg_pil"])

    if pct > 0:
        arc_np = np.zeros((fh, fw, 4), dtype=np.uint8)
        start_ang = -90
        end_ang = start_ang + int(360 * pct)
        cv2.ellipse(arc_np, (cx, cy), (radius, radius), 0,
                    start_ang, end_ang, (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)
        canvas.alpha_composite(Image.fromarray(arc_np, mode="RGBA"))

    draw = ImageDraw.Draw(canvas)

    txt = f"Reps: {int(reps)}"
    draw.text((c["rep_txt_x"], c["rep_txt_y"]), txt,
              font=c["reps_font"], fill=(255, 255, 255, 255))

    label   = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label,   font=c["depth_label_font"])
    pw = draw.textlength(pct_txt, font=c["depth_pct_font"])
    draw.text((cx - int(lw // 2), c["label_by"]),
              label, font=c["depth_label_font"], fill=(255, 255, 255, 255))
    draw.text((cx - int(pw // 2), c["label_by"] + c["depth_label_font"].size + c["label_gap"]),
              pct_txt, font=c["depth_pct_font"], fill=(255, 255, 255, 255))

    if feedback:
        fb_lines = _wrap_two_lines(draw, feedback, c["feedback_font"], c["max_text_w"])
        ty = c["fb_y0"] + c["fb_pad_y"]
        for ln in fb_lines:
            tw2 = draw.textlength(ln, font=c["feedback_font"])
            tx  = max(c["fb_pad_x"], (fw - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=c["feedback_font"], fill=(255, 255, 255, 255))
            ty += c["line_h"] + c["line_gap"]

    canvas_small = cv2.resize(np.array(canvas), (w, h), interpolation=cv2.INTER_AREA)
    alpha       = canvas_small[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = canvas_small[:, :, [2, 1, 0]].astype(np.float32)
    result = frame.astype(np.float32) * (1.0 - alpha) + overlay_bgr * alpha
    return result.astype(np.uint8)

# ============ Dips Parameters ============
# ספירת חזרות
ELBOW_BENT_ANGLE = 90.0          # זווית מרפק מקסימלית למטה (90° = עומק מלא)
SHOULDER_MIN_DESCENT = 0.045     # כתף צריכה לרדת לפחות 4.5% מגובה המסך
RESET_ASCENT = 0.025             # כתף צריכה לעלות חזרה 2.5% כדי לאפס
RESET_ELBOW = 150.0              # זווית מרפק מינימלית בחזרה למעלה
REFRACTORY_FRAMES = 4            # מסגרות המתנה בין חזרות

# EMA smoothing (higher α = faster response, less lag)
ELBOW_EMA_ALPHA = 0.45
SHOULDER_EMA_ALPHA = 0.38

# Minimum descent duration before allowing a count
MIN_DESC_FRAMES = 2

# Rearm requirements (after counting, before next rep allowed)
REARM_ASCENT_RATIO = 0.40   # must recover 40 % of peak descent
REARM_ELBOW_ABOVE = 130.0   # elbow must open past this angle

# Velocity smoothing window (frames)
VEL_WINDOW = 3

# זיהוי תנוחת דיפס (ידיים מתחת לכתפיים)
VIS_THR_STRICT = 0.30
WRIST_BELOW_SHOULDER_MARGIN = 0.02  # שורשי ידיים צריכים להיות מתחת לכתפיים
TORSO_STABILITY_THR = 0.012         # תנועת טורסו מקסימלית
ONDIPS_MIN_FRAMES = 3
OFFDIPS_MIN_FRAMES = 6
AUTO_STOP_AFTER_EXIT_SEC = 1.2
TAIL_NOPOSE_STOP_SEC = 1.0

# ============ Feedback Cues & Weights ============
FB_CUE_DEEPER = "Go deeper (elbows to 90°)"
FB_CUE_LEAN = "Reduce forward lean (keep torso upright)"
FB_CUE_LOCKOUT = "Fully lockout elbows at top"
FB_CUE_ELBOWS_IN = "Keep elbows closer to body"

FB_W_DEEPER = float(os.getenv("FB_W_DEEPER", "1.0"))
FB_W_LEAN = float(os.getenv("FB_W_LEAN", "0.7"))
FB_W_LOCKOUT = float(os.getenv("FB_W_LOCKOUT", "0.8"))
FB_W_ELBOWS_IN = float(os.getenv("FB_W_ELBOWS_IN", "0.6"))

FB_WEIGHTS = {
    FB_CUE_DEEPER: FB_W_DEEPER,
    FB_CUE_LEAN: FB_W_LEAN,
    FB_CUE_LOCKOUT: FB_W_LOCKOUT,
    FB_CUE_ELBOWS_IN: FB_W_ELBOWS_IN,
}
FB_DEFAULT_WEIGHT = 0.5
PENALTY_MIN_IF_ANY = 0.5
FORM_TIP_PRIORITY = [FB_CUE_DEEPER, FB_CUE_LOCKOUT, FB_CUE_ELBOWS_IN, FB_CUE_LEAN]

# ============ Form Detection Thresholds ============
# Depth
DEPTH_MIN_ANGLE = 95.0           # זווית מרפק מינימלית לעומק טוב
DEPTH_NEAR_DEG = 8.0             # טווח "קרוב מספיק"

# Lean (torso angle)
TORSO_MAX_LEAN = 25.0            # זווית טורסו מקסימלית מאנכי
LEAN_NEAR_DEG = 5.0              # טווח "קרוב מספיק"

# Lockout
LOCKOUT_MIN_ANGLE = 165.0        # זווית מרפק מינימלית בנעילה
LOCKOUT_NEAR_DEG = 8.0           # טווח "קרוב מספיק"

# Elbow flare
ELBOW_FLARE_MAX = 45.0           # זווית מקסימלית של מרפקים מהגוף
FLARE_NEAR_DEG = 8.0             # טווח "קרוב מספיק"

# Minimum reps before session feedback
DEPTH_FAIL_MIN_REPS = 2
LEAN_FAIL_MIN_REPS = 2
LOCKOUT_FAIL_MIN_REPS = 2
FLARE_FAIL_MIN_REPS = 2

# ============ Micro-burst ============
BURST_FRAMES = int(os.getenv("BURST_FRAMES", "2"))
INFLECT_VEL_THR = float(os.getenv("INFLECT_VEL_THR", "0.0006"))

DEBUG_ONDIPS = bool(int(os.getenv("DEBUG_ONDIPS", "0")))

def run_dips_analysis(video_path,
                      frame_skip=3,
                      scale=0.4,
                      output_path="dips_analyzed.mp4",
                      feedback_path="dips_feedback.txt",
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
        scale=1.0; frame_skip=1; encode_crf=18 if encode_crf is None else encode_crf
    else:
        encode_crf=23 if encode_crf is None else encode_crf

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): return _ret_err("Could not open video", feedback_path)

    fps_in=cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps=max(1.0, fps_in/max(1,frame_skip))
    sec_to_frames=lambda s: max(1,int(s*effective_fps))

    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=None; frame_idx=0

    # Counters
    rep_count=0; good_reps=0; bad_reps=0; rep_reports=[]; all_scores=[]

    # Landmarks
    LSH=mp_pose.PoseLandmark.LEFT_SHOULDER.value;  RSH=mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LE =mp_pose.PoseLandmark.LEFT_ELBOW.value;     RE =mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW =mp_pose.PoseLandmark.LEFT_WRIST.value;     RW =mp_pose.PoseLandmark.RIGHT_WRIST.value
    LH =mp_pose.PoseLandmark.LEFT_HIP.value;       RH =mp_pose.PoseLandmark.RIGHT_HIP.value

    def _pick_side_dyn(lms):
        vL=lms[LSH].visibility+lms[LE].visibility+lms[LW].visibility
        vR=lms[RSH].visibility+lms[RE].visibility+lms[RW].visibility
        return ("LEFT",LSH,LE,LW) if vL>=vR else ("RIGHT",RSH,RE,RW)

    # State
    elbow_ema=None; shoulder_ema=None; shoulder_prev=None; shoulder_vel_prev=None
    baseline_shoulder_y=None
    desc_base_shoulder=None; allow_new_bottom=True; last_bottom_frame=-10**9
    cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
    cycle_min_elbow_raw=999.0       # raw (unsmoothed) elbow minimum
    desc_frame_count=0              # frames spent in current descent
    cycle_bottom_shoulder=None      # shoulder y at counted bottom
    vel_buf=deque(maxlen=VEL_WINDOW)  # velocity smoothing buffer

    ondips=False; ondips_streak=0; offdips_streak=0; prev_torso_cx=None
    offdips_frames_since_any_rep=0; nopose_frames_since_any_rep=0

    # Feedback state
    session_feedback=set()
    rt_fb_msg=None; rt_fb_hold=0

    # Per-cycle trackers
    cycle_tip_deeper=False; cycle_tip_lean=False; cycle_tip_lockout=False; cycle_tip_elbows=False
    depth_fail_count=0; lean_fail_count=0; lockout_fail_count=0; flare_fail_count=0
    depth_already_reported=False; lean_already_reported=False
    lockout_already_reported=False; flare_already_reported=False

    # Phase trackers
    bottom_phase_min_elbow=None
    top_phase_max_elbow=None
    cycle_max_lean=None
    cycle_max_flare=None

    OFFDIPS_STOP_FRAMES=sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES=sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES=sec_to_frames(0.8)

    REARM_ASCENT_EFF=max(RESET_ASCENT*0.60, 0.012)  # legacy fallback

    # Micro-burst
    burst_cntr=0

    with mp_pose.Pose(model_complexity=model_complexity, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            process_now = (burst_cntr > 0) or (frame_idx % max(1, frame_skip) == 0)
            if not process_now:
                continue
            if burst_cntr > 0:
                burst_cntr -= 1

            if scale != 1.0:
                frame=cv2.resize(frame,(0,0),fx=scale,fy=scale)
            h,w=frame.shape[:2]

            if return_video and out is None:
                out=cv2.VideoWriter(output_path,fourcc,effective_fps,(w,h))

            res=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            depth_live=0.0

            if not res.pose_landmarks:
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep+1) if rep_count>0 else 0
                if rep_count>0 and nopose_frames_since_any_rep>=NOPOSE_STOP_FRAMES: break
                if return_video and out is not None:
                    out.write(draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),depth_pct=0.0))
                if rt_fb_hold>0: rt_fb_hold-=1
                continue

            nopose_frames_since_any_rep=0
            lms=res.pose_landmarks.landmark
            side,S,E,W=_pick_side_dyn(lms)

            min_vis=min(lms[S].visibility,lms[E].visibility,lms[W].visibility)
            vis_strict_ok=(min_vis>=VIS_THR_STRICT)

            shoulder_raw=float(lms[S].y)
            raw_elbow_L=_ang((lms[LSH].x,lms[LSH].y),(lms[LE].x,lms[LE].y),(lms[LW].x,lms[LW].y))
            raw_elbow_R=_ang((lms[RSH].x,lms[RSH].y),(lms[RE].x,lms[RE].y),(lms[RW].x,lms[RW].y))
            raw_elbow=raw_elbow_L if side=="LEFT" else raw_elbow_R
            raw_elbow_min=min(raw_elbow_L,raw_elbow_R)

            elbow_ema=_ema(elbow_ema,raw_elbow,ELBOW_EMA_ALPHA)
            shoulder_ema=_ema(shoulder_ema,shoulder_raw,SHOULDER_EMA_ALPHA)
            shoulder_y=shoulder_ema; elbow_angle=elbow_ema
            
            if baseline_shoulder_y is None: baseline_shoulder_y=shoulder_y
            depth_live=float(np.clip((shoulder_y-baseline_shoulder_y)/max(0.08,SHOULDER_MIN_DESCENT*1.2),0.0,1.0))

            # Torso stability
            torso_cx=np.mean([lms[LSH].x,lms[RSH].x,lms[LH].x,lms[RH].x])*w
            torso_dx_norm=0.0 if prev_torso_cx is None else abs(torso_cx-prev_torso_cx)/max(1.0,w)
            prev_torso_cx=torso_cx

            # Check dips position (wrists below shoulders)
            lw_below=(lms[LW].visibility>=VIS_THR_STRICT) and (lms[LW].y>lms[LSH].y+WRIST_BELOW_SHOULDER_MARGIN)
            rw_below=(lms[RW].visibility>=VIS_THR_STRICT) and (lms[RW].y>lms[RSH].y+WRIST_BELOW_SHOULDER_MARGIN)
            in_position=(lw_below or rw_below)

            if vis_strict_ok and in_position and (torso_dx_norm<=TORSO_STABILITY_THR):
                ondips_streak+=1; offdips_streak=0
            else:
                offdips_streak+=1; ondips_streak=0

            if DEBUG_ONDIPS and frame_idx % 10 == 0:
                print(f"[DBG] f={frame_idx} ondips={ondips} vis={min_vis:.2f} lwBelow={lw_below} rwBelow={rw_below} torsoDx={torso_dx_norm:.4f}")

            # Enter dips position
            if (not ondips) and ondips_streak>=ONDIPS_MIN_FRAMES:
                ondips=True
                desc_base_shoulder=None; allow_new_bottom=True
                cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_min_elbow_raw=999.0; desc_frame_count=0; cycle_bottom_shoulder=None
                vel_buf.clear()
                cycle_tip_deeper=False; cycle_tip_lean=False; cycle_tip_lockout=False; cycle_tip_elbows=False
                bottom_phase_min_elbow=None
                top_phase_max_elbow=None
                cycle_max_lean=None
                cycle_max_flare=None

            # Exit dips position → post-hoc
            if ondips and offdips_streak>=OFFDIPS_MIN_FRAMES:
                # Evaluate form at cycle end
                _evaluate_cycle_form(lms, bottom_phase_min_elbow, top_phase_max_elbow, 
                                   cycle_max_lean, cycle_max_flare,
                                   depth_fail_count, lean_fail_count, lockout_fail_count, flare_fail_count,
                                   depth_already_reported, lean_already_reported, 
                                   lockout_already_reported, flare_already_reported,
                                   session_feedback, locals())

                # Count rep if valid (accept raw OR smoothed elbow)
                _elbow_ok = (cycle_min_elbow <= ELBOW_BENT_ANGLE) or (cycle_min_elbow_raw <= ELBOW_BENT_ANGLE + 5)
                if (not counted_this_cycle) and (cycle_max_descent>=SHOULDER_MIN_DESCENT) and _elbow_ok and (desc_frame_count >= MIN_DESC_FRAMES):
                    rep_has_tip = cycle_tip_deeper or cycle_tip_lean or cycle_tip_lockout or cycle_tip_elbows
                    _count_rep(rep_reports,rep_count,cycle_min_elbow,
                               desc_base_shoulder if desc_base_shoulder is not None else shoulder_y,
                               baseline_shoulder_y+cycle_max_descent if baseline_shoulder_y is not None else shoulder_y,
                               all_scores, rep_has_tip)
                    rep_count+=1
                    if rep_has_tip: bad_reps+=1
                    else: good_reps+=1

                ondips=False; offdips_frames_since_any_rep=0
                desc_base_shoulder=None; cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_min_elbow_raw=999.0; desc_frame_count=0; cycle_bottom_shoulder=None
                vel_buf.clear()
                cycle_tip_deeper=False; cycle_tip_lean=False; cycle_tip_lockout=False; cycle_tip_elbows=False
                bottom_phase_min_elbow=None
                top_phase_max_elbow=None
                cycle_max_lean=None
                cycle_max_flare=None

            if (not ondips) and rep_count>0:
                offdips_frames_since_any_rep+=1
                if offdips_frames_since_any_rep>=OFFDIPS_STOP_FRAMES: break

            raw_vel=0.0 if shoulder_prev is None else (shoulder_y-shoulder_prev)
            vel_buf.append(raw_vel)
            shoulder_vel=sum(vel_buf)/len(vel_buf) if vel_buf else raw_vel  # smoothed
            cur_rt=None

            # Micro-burst near inflection
            if ondips and (desc_base_shoulder is not None):
                near_inflect = (abs(shoulder_vel) <= INFLECT_VEL_THR)
                sign_flip = (shoulder_vel_prev is not None) and ((shoulder_vel_prev < 0 and shoulder_vel >= 0) or (shoulder_vel_prev > 0 and shoulder_vel <= 0))
                if near_inflect or sign_flip:
                    burst_cntr = max(burst_cntr, BURST_FRAMES)
            shoulder_vel_prev = shoulder_vel

            if ondips and vis_strict_ok:
                # REP COUNTING
                if desc_base_shoulder is None:
                    if shoulder_vel>abs(INFLECT_VEL_THR):  # Moving down
                        desc_base_shoulder=shoulder_y
                        cycle_max_descent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                        cycle_min_elbow_raw=raw_elbow_min; desc_frame_count=1; cycle_bottom_shoulder=None
                        bottom_phase_min_elbow=None
                        top_phase_max_elbow=None
                        cycle_max_lean=None
                        cycle_max_flare=None
                else:
                    desc_frame_count+=1
                    cycle_max_descent=max(cycle_max_descent,(shoulder_y-desc_base_shoulder))
                    cycle_min_elbow=min(cycle_min_elbow,elbow_angle)
                    cycle_min_elbow_raw=min(cycle_min_elbow_raw,raw_elbow_min)

                    # Track bottom phase (deepest point)
                    min_elb_now = min(raw_elbow_L, raw_elbow_R)
                    if bottom_phase_min_elbow is None: bottom_phase_min_elbow = min_elb_now
                    else: bottom_phase_min_elbow = min(bottom_phase_min_elbow, min_elb_now)

                    # Track top phase (highest point)
                    max_elb_now = max(raw_elbow_L, raw_elbow_R)
                    if top_phase_max_elbow is None: top_phase_max_elbow = max_elb_now
                    else: top_phase_max_elbow = max(top_phase_max_elbow, max_elb_now)

                    # Track lean (torso angle)
                    torso_angle = _calculate_torso_lean(lms, LSH, RSH, LH, RH)
                    if cycle_max_lean is None: cycle_max_lean = torso_angle
                    else: cycle_max_lean = max(cycle_max_lean, torso_angle)

                    # Track elbow flare
                    elbow_flare = _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW)
                    if cycle_max_flare is None: cycle_max_flare = elbow_flare
                    else: cycle_max_flare = max(cycle_max_flare, elbow_flare)

                    reset_by_asc=(desc_base_shoulder is not None) and ((desc_base_shoulder-shoulder_y)>=RESET_ASCENT)
                    reset_by_elb=(elbow_angle>=RESET_ELBOW)

                    # Require BOTH shoulder ascent AND elbow extension for reset
                    # (fallback: ascent alone if enough frames elapsed)
                    do_reset = (reset_by_asc and reset_by_elb) or (reset_by_asc and desc_frame_count >= MIN_DESC_FRAMES * 3)
                    if do_reset:
                        # Evaluate form
                        _evaluate_cycle_form(lms, bottom_phase_min_elbow, top_phase_max_elbow,
                                           cycle_max_lean, cycle_max_flare,
                                           depth_fail_count, lean_fail_count, lockout_fail_count, flare_fail_count,
                                           depth_already_reported, lean_already_reported,
                                           lockout_already_reported, flare_already_reported,
                                           session_feedback, locals())

                        # Count rep (accept raw OR smoothed elbow)
                        _elbow_ok2 = (cycle_min_elbow <= ELBOW_BENT_ANGLE) or (cycle_min_elbow_raw <= ELBOW_BENT_ANGLE + 5)
                        if (not counted_this_cycle) and (cycle_max_descent>=SHOULDER_MIN_DESCENT) and _elbow_ok2 and (desc_frame_count >= MIN_DESC_FRAMES):
                            rep_has_tip = cycle_tip_deeper or cycle_tip_lean or cycle_tip_lockout or cycle_tip_elbows
                            _count_rep(rep_reports,rep_count,cycle_min_elbow,
                                       desc_base_shoulder,
                                       baseline_shoulder_y+cycle_max_descent if baseline_shoulder_y is not None else shoulder_y,
                                       all_scores, rep_has_tip)
                            rep_count+=1
                            if rep_has_tip: bad_reps+=1
                            else: good_reps+=1

                        # Reset for next cycle
                        desc_base_shoulder=shoulder_y
                        cycle_max_descent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                        cycle_min_elbow_raw=raw_elbow_min; desc_frame_count=0; cycle_bottom_shoulder=None
                        allow_new_bottom=True
                        bottom_phase_min_elbow=None
                        top_phase_max_elbow=None
                        cycle_max_lean=None
                        cycle_max_flare=None
                        cycle_tip_deeper=False; cycle_tip_lean=False; cycle_tip_lockout=False; cycle_tip_elbows=False

                descent_amt=0.0 if desc_base_shoulder is None else (shoulder_y-desc_base_shoulder)

                at_bottom=(elbow_angle<=ELBOW_BENT_ANGLE) and (descent_amt>=SHOULDER_MIN_DESCENT)
                raw_bottom=(raw_elbow_min<=(ELBOW_BENT_ANGLE+5.0)) and (descent_amt>=SHOULDER_MIN_DESCENT*0.92)
                at_bottom=at_bottom or raw_bottom
                enough_frames=(desc_frame_count >= MIN_DESC_FRAMES)
                can_cnt=(frame_idx - last_bottom_frame) >= REFRACTORY_FRAMES

                if at_bottom and allow_new_bottom and can_cnt and enough_frames and (not counted_this_cycle):
                    rep_has_tip = cycle_tip_deeper or cycle_tip_lean or cycle_tip_lockout or cycle_tip_elbows
                    _count_rep(rep_reports,rep_count,elbow_angle,
                               desc_base_shoulder if desc_base_shoulder is not None else shoulder_y, shoulder_y,
                               all_scores, rep_has_tip)
                    rep_count+=1
                    if rep_has_tip: bad_reps+=1
                    else: good_reps+=1
                    last_bottom_frame=frame_idx; allow_new_bottom=False; counted_this_cycle=True
                    cycle_bottom_shoulder=shoulder_y
                    top_phase_max_elbow = max(raw_elbow_L, raw_elbow_R)

                # Ratio-based rearm: must recover a fraction of descent AND extend elbows
                if (allow_new_bottom is False) and (last_bottom_frame>0):
                    if shoulder_prev is not None and (shoulder_prev - shoulder_y) > 0 and (desc_base_shoulder is not None):
                        ascent_from_bottom = (cycle_bottom_shoulder - shoulder_y) if cycle_bottom_shoulder is not None else ((desc_base_shoulder + cycle_max_descent) - shoulder_y)
                        ascent_ratio = ascent_from_bottom / max(cycle_max_descent, 0.001)
                        elbow_open = (elbow_angle >= REARM_ELBOW_ABOVE)
                        # Primary: ratio + elbow. Fallback: old absolute threshold if ratio almost met
                        if (ascent_ratio >= REARM_ASCENT_RATIO and elbow_open) or (ascent_from_bottom >= REARM_ASCENT_EFF and ascent_ratio >= REARM_ASCENT_RATIO * 0.6):
                            allow_new_bottom=True

                # Real-time feedback at bottom
                if at_bottom and not cycle_tip_deeper:
                    if bottom_phase_min_elbow and bottom_phase_min_elbow > DEPTH_MIN_ANGLE:
                        cycle_tip_deeper = True
                        depth_fail_count += 1
                        if depth_fail_count >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                            session_feedback.add(FB_CUE_DEEPER)
                            depth_already_reported = True
                            cur_rt = FB_CUE_DEEPER

            else:
                desc_base_shoulder=None; allow_new_bottom=True
                desc_frame_count=0; cycle_min_elbow_raw=999.0; cycle_bottom_shoulder=None

            # RT hold
            if cur_rt:
                if cur_rt!=rt_fb_msg: rt_fb_msg=cur_rt; rt_fb_hold=RT_FB_HOLD_FRAMES
                else: rt_fb_hold=max(rt_fb_hold,RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold>0: rt_fb_hold-=1

            # Draw
            if return_video and out is not None:
                frame=draw_body_only(frame,lms)
                frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),depth_pct=depth_live)
                out.write(frame)

            if shoulder_y is not None: shoulder_prev=shoulder_y

    # EOF post-hoc (accept raw OR smoothed elbow)
    _eof_elbow_ok = (cycle_min_elbow <= ELBOW_BENT_ANGLE) or (cycle_min_elbow_raw <= ELBOW_BENT_ANGLE + 5)
    if ondips and (not counted_this_cycle) and (cycle_max_descent>=SHOULDER_MIN_DESCENT) and _eof_elbow_ok and (desc_frame_count >= MIN_DESC_FRAMES):
        _evaluate_cycle_form(lms, bottom_phase_min_elbow, top_phase_max_elbow,
                           cycle_max_lean, cycle_max_flare,
                           depth_fail_count, lean_fail_count, lockout_fail_count, flare_fail_count,
                           depth_already_reported, lean_already_reported,
                           lockout_already_reported, flare_already_reported,
                           session_feedback, locals())

        rep_has_tip = cycle_tip_deeper or cycle_tip_lean or cycle_tip_lockout or cycle_tip_elbows
        _count_rep(rep_reports,rep_count,cycle_min_elbow,
                   desc_base_shoulder if desc_base_shoulder is not None else (baseline_shoulder_y or 0.0),
                   (baseline_shoulder_y + cycle_max_descent) if baseline_shoulder_y is not None else (baseline_shoulder_y or 0.0),
                   all_scores, rep_has_tip)
        rep_count+=1
        if rep_has_tip: bad_reps+=1
        else: good_reps+=1

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    # Session score
    if rep_count==0: technique_score=0.0
    else:
        if session_feedback:
            penalty = sum(FB_WEIGHTS.get(m,FB_DEFAULT_WEIGHT) for m in set(session_feedback))
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
        else:
            penalty = 0.0
        technique_score=_half_floor10(max(0.0,10.0-penalty))

    # Build feedback
    all_fb = set(session_feedback) if session_feedback else set()
    fb_list = [cue for cue in FORM_TIP_PRIORITY if cue in all_fb]
    if not fb_list and technique_score >= 10.0 - 1e-6:
        fb_list = ["Great form! Keep it up 💪"]

    form_tip = None
    if all_fb:
        form_tip = max(all_fb, key=lambda m: (FB_WEIGHTS.get(m, FB_DEFAULT_WEIGHT),
                                              -FORM_TIP_PRIORITY.index(m) if m in FORM_TIP_PRIORITY else -999))

    # Write feedback file
    try:
        with open(feedback_path,"w",encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            if fb_list:
                f.write("Feedback:\n")
                for ln in fb_list: f.write(f"- {ln}\n")
    except Exception:
        pass

    # Encode video
    final_path=""
    if return_video and os.path.exists(output_path):
        encoded_path=output_path.replace(".mp4","_encoded.mp4")
        try:
            subprocess.run(["ffmpeg","-y","-i",output_path,"-c:v","libx264","-preset","medium",
                            "-crf",str(int(encode_crf if encode_crf is not None else 23)),"-movflags","+faststart","-pix_fmt","yuv420p",
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
        "feedback": fb_list,
        "tips": [],
        "reps": rep_reports,
        "video_path": final_path if return_video else "",
        "feedback_path": feedback_path
    }
    if form_tip is not None:
        result["form_tip"] = form_tip

    return result

# ============ Helper Functions ============
def _calculate_torso_lean(lms, LSH, RSH, LH, RH):
    """חישוב זווית הטיה של הטורסו מהאנכי"""
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    mid_hp = ((lms[LH].x + lms[RH].x)/2.0, (lms[LH].y + lms[RH].y)/2.0)
    
    # Vector from hip to shoulder
    dx = mid_sh[0] - mid_hp[0]
    dy = mid_sh[1] - mid_hp[1]
    
    # Angle from vertical (0° = perfectly upright)
    angle = abs(math.degrees(math.atan2(abs(dx), abs(dy))))
    return angle

def _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW):
    """חישוב זווית התרחקות מרפקים מהגוף"""
    # Calculate body centerline
    mid_sh_x = (lms[LSH].x + lms[RSH].x) / 2.0
    
    # Left elbow angle from centerline
    left_dx = abs(lms[LE].x - mid_sh_x)
    left_dy = abs(lms[LE].y - lms[LSH].y)
    left_angle = math.degrees(math.atan2(left_dx, left_dy + 1e-9))
    
    # Right elbow angle from centerline
    right_dx = abs(lms[RE].x - mid_sh_x)
    right_dy = abs(lms[RE].y - lms[RSH].y)
    right_angle = math.degrees(math.atan2(right_dx, right_dy + 1e-9))
    
    return max(left_angle, right_angle)

def _evaluate_cycle_form(lms, bottom_phase_min_elbow, top_phase_max_elbow,
                        cycle_max_lean, cycle_max_flare,
                        depth_fail_count, lean_fail_count, lockout_fail_count, flare_fail_count,
                        depth_already_reported, lean_already_reported,
                        lockout_already_reported, flare_already_reported,
                        session_feedback, local_vars):
    """הערכת פורם בסוף מחזור"""
    
    # Depth check
    depth_ok = (bottom_phase_min_elbow is not None) and (bottom_phase_min_elbow <= DEPTH_MIN_ANGLE)
    depth_near = (bottom_phase_min_elbow is not None) and (bottom_phase_min_elbow <= (DEPTH_MIN_ANGLE + DEPTH_NEAR_DEG))
    
    if not depth_ok and not depth_near:
        local_vars['cycle_tip_deeper'] = True
        local_vars['depth_fail_count'] += 1
        if local_vars['depth_fail_count'] >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
            session_feedback.add(FB_CUE_DEEPER)
            local_vars['depth_already_reported'] = True

    # Lockout check
    lockout_ok = (top_phase_max_elbow is not None) and (top_phase_max_elbow >= LOCKOUT_MIN_ANGLE)
    lockout_near = (top_phase_max_elbow is not None) and (top_phase_max_elbow >= (LOCKOUT_MIN_ANGLE - LOCKOUT_NEAR_DEG))
    
    if not lockout_ok and not lockout_near:
        local_vars['cycle_tip_lockout'] = True
        local_vars['lockout_fail_count'] += 1
        if local_vars['lockout_fail_count'] >= LOCKOUT_FAIL_MIN_REPS and not lockout_already_reported:
            session_feedback.add(FB_CUE_LOCKOUT)
            local_vars['lockout_already_reported'] = True

    # Lean check
    lean_ok = (cycle_max_lean is not None) and (cycle_max_lean <= TORSO_MAX_LEAN)
    lean_near = (cycle_max_lean is not None) and (cycle_max_lean <= (TORSO_MAX_LEAN + LEAN_NEAR_DEG))
    
    if not lean_ok and not lean_near:
        local_vars['cycle_tip_lean'] = True
        local_vars['lean_fail_count'] += 1
        if local_vars['lean_fail_count'] >= LEAN_FAIL_MIN_REPS and not lean_already_reported:
            session_feedback.add(FB_CUE_LEAN)
            local_vars['lean_already_reported'] = True

    # Elbow flare check
    flare_ok = (cycle_max_flare is not None) and (cycle_max_flare <= ELBOW_FLARE_MAX)
    flare_near = (cycle_max_flare is not None) and (cycle_max_flare <= (ELBOW_FLARE_MAX + FLARE_NEAR_DEG))
    
    if not flare_ok and not flare_near:
        local_vars['cycle_tip_elbows'] = True
        local_vars['flare_fail_count'] += 1
        if local_vars['flare_fail_count'] >= FLARE_FAIL_MIN_REPS and not flare_already_reported:
            session_feedback.add(FB_CUE_ELBOWS_IN)
            local_vars['flare_already_reported'] = True

def _count_rep(rep_reports, rep_count, bottom_elbow, descent_from, bottom_shoulder_y, all_scores, rep_has_tip):
    rep_score = 10.0 if not rep_has_tip else 9.5
    all_scores.append(rep_score)
    rep_reports.append({
        "rep_index": int(rep_count+1),
        "score": float(rep_score),
        "good": bool(rep_score >= 10.0 - 1e-6),
        "bottom_elbow": float(bottom_elbow),
        "descent_from": float(descent_from),
        "bottom_shoulder_y": float(bottom_shoulder_y)
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
    return run_dips_analysis(*args, **kwargs)
