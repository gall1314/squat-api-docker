# -*- coding: utf-8 -*-
# pullup_analysis.py — SMART FORM_TIP edition with FAST/SLOW path optimization
# form_tip = optimization advice (doesn't affect score, only helps improve)

import os, cv2, math, numpy as np, subprocess
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ============ Styles ============
BAR_BG_ALPHA=0.55; DONUT_RADIUS_SCALE=0.72; DONUT_THICKNESS_FRAC=0.28
DEPTH_COLOR=(40,200,80); DEPTH_RING_BG=(70,70,70)
FONT_PATH="Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE=28; FEEDBACK_FONT_SIZE=22; DEPTH_LABEL_FONT_SIZE=14; DEPTH_PCT_FONT_SIZE=18

def _load_font(p,s):
    try: return ImageFont.truetype(p,s)
    except: return ImageFont.load_default()

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

# ============ Body-only skeleton (SLOW PATH ONLY) ============
_FACE_LMS=set(); _BODY_CONNECTIONS=tuple(); _BODY_POINTS=tuple()

def _init_skeleton_data():
    global _FACE_LMS, _BODY_CONNECTIONS, _BODY_POINTS
    if mp_pose and not _BODY_CONNECTIONS:
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

def draw_overlay(frame, reps=0, feedback=None, height_pct=0.0):
    h,w,_=frame.shape
    ref_h=max(int(h*0.06),int(REPS_FONT_SIZE*1.6))
    r=int(ref_h*DONUT_RADIUS_SCALE); th=max(3,int(r*DONUT_THICKNESS_FRAC))
    m=12; cx=w-m-r; cy=max(ref_h+r//8, r+th//2+2)
    pct=float(np.clip(height_pct,0,1))
    cv2.circle(frame,(cx,cy),r,DEPTH_RING_BG,th,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(r,r),0,-90,-90+int(360*pct),DEPTH_COLOR,th,cv2.LINE_AA)
    
    REPS_FONT=_load_font(FONT_PATH,REPS_FONT_SIZE)
    FEEDBACK_FONT=_load_font(FONT_PATH,FEEDBACK_FONT_SIZE)
    DEPTH_LABEL_FONT=_load_font(FONT_PATH,DEPTH_LABEL_FONT_SIZE)
    DEPTH_PCT_FONT=_load_font(FONT_PATH,DEPTH_PCT_FONT_SIZE)
    
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)

    txt=f"Reps: {int(reps)}"; pad_x,pad_y=10,6
    tw=draw.textlength(txt,font=REPS_FONT); thh=REPS_FONT.size
    base=np.array(pil); over=base.copy()
    cv2.rectangle(over,(0,0),(int(tw+2*pad_x),int(thh+2*pad_y)),(0,0,0),-1)
    base=cv2.addWeighted(over,BAR_BG_ALPHA,base,1-BAR_BG_ALPHA,0)
    pil=Image.fromarray(base); draw=ImageDraw.Draw(pil)
    draw.text((pad_x,pad_y-1),txt,font=REPS_FONT,fill=(255,255,255))

    gap=max(2,int(r*0.10)); by=cy-(DEPTH_LABEL_FONT.size+gap+DEPTH_PCT_FONT.size)//2
    label="HEIGHT"; pct_txt=f"{int(pct*100)}%"
    lw=draw.textlength(label,font=DEPTH_LABEL_FONT); pw=draw.textlength(pct_txt,font=DEPTH_PCT_FONT)
    draw.text((cx-int(lw//2),by),label,font=DEPTH_LABEL_FONT,fill=(255,255,255))
    draw.text((cx-int(pw//2),by+DEPTH_LABEL_FONT.size+gap),pct_txt,font=DEPTH_PCT_FONT,fill=(255,255,255))

    if feedback:
        max_w=int(w-2*12-20); lines=_wrap_two_lines(draw,feedback,FEEDBACK_FONT,max_w)
        line_h=FEEDBACK_FONT.size+6
        block_h=2*8+len(lines)*line_h+(len(lines)-1)*4
        y0=max(0,h-max(6,int(h*0.02))-block_h); y1=h-max(6,int(h*0.02))
        base2=np.array(pil); over2=base2.copy()
        cv2.rectangle(over2,(0,y0),(w,y1),(0,0,0),-1)
        base2=cv2.addWeighted(over2,BAR_BG_ALPHA,base2,1-BAR_BG_ALPHA,0)
        pil=Image.fromarray(base2); draw=ImageDraw.Draw(pil)
        ty=y0+8
        for ln in lines:
            tw2=draw.textlength(ln,font=FEEDBACK_FONT); tx=max(12,(w-int(tw2))//2)
            draw.text((tx,ty),ln,font=FEEDBACK_FONT,fill=(255,255,255)); ty+=line_h+4
    return cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)

# ============ Params ============
ELBOW_TOP_ANGLE=100.0
HEAD_MIN_ASCENT=0.0075
RESET_DESCENT=0.0045
RESET_ELBOW=135.0
REFRACTORY_FRAMES=2
HEAD_VEL_UP_TINY=0.0002
ELBOW_EMA_ALPHA=0.35; HEAD_EMA_ALPHA=0.30

VIS_THR_STRICT=0.30; WRIST_VIS_THR=0.20
WRIST_ABOVE_HEAD_MARGIN=0.02; TORSO_X_THR=0.010
ONBAR_MIN_FRAMES=2; OFFBAR_MIN_FRAMES=6
AUTO_STOP_AFTER_EXIT_SEC=1.2; TAIL_NOPOSE_STOP_SEC=1.0

# ============ FEEDBACK (issues that affect score) ============
FB_CUE_HIGHER = "Go a bit higher (chin over bar)"
FB_CUE_SWING  = "Reduce body swing (no kipping)"
FB_CUE_BOTTOM = "Fully extend arms at bottom"

FB_W_HIGHER = float(os.getenv("FB_W_HIGHER", "1.0"))
FB_W_SWING  = float(os.getenv("FB_W_SWING",  "0.5"))
FB_W_BOTTOM = float(os.getenv("FB_W_BOTTOM", "1.0"))

FB_WEIGHTS = {
    FB_CUE_HIGHER: FB_W_HIGHER,
    FB_CUE_SWING:  FB_W_SWING,
    FB_CUE_BOTTOM: FB_W_BOTTOM,
}
FB_DEFAULT_WEIGHT=0.5
PENALTY_MIN_IF_ANY=0.5

FEEDBACK_ORDER = [FB_CUE_HIGHER, FB_CUE_BOTTOM, FB_CUE_SWING]

# ============ FORM_TIP (optimization advice - doesn't affect score!) ============
FORM_TIP_SLOW_ECCENTRIC = "slow_eccentric"
FORM_TIP_EXPLOSIVE_PULL = "explosive_pull" 
FORM_TIP_TOP_PAUSE = "top_pause"
FORM_TIP_TEMPO_CONSISTENCY = "tempo_consistency"

FORM_TIP_MESSAGES = {
    FORM_TIP_SLOW_ECCENTRIC: "Try lowering slower (3-4 sec) for better muscle growth",
    FORM_TIP_EXPLOSIVE_PULL: "Pull up faster to build explosive strength",
    FORM_TIP_TOP_PAUSE: "Hold at the top for 1-2 seconds to maximize strength",
    FORM_TIP_TEMPO_CONSISTENCY: "Keep a steady rhythm between reps for endurance",
}

ECCENTRIC_FAST_THR = float(os.getenv("ECCENTRIC_FAST_THR", "0.8"))
CONCENTRIC_SLOW_THR = float(os.getenv("CONCENTRIC_SLOW_THR", "2.0"))
TOP_HOLD_SHORT_THR = float(os.getenv("TOP_HOLD_SHORT_THR", "0.3"))
TEMPO_VARIANCE_THR = float(os.getenv("TEMPO_VARIANCE_THR", "0.35"))

SWING_THR=0.012; SWING_MIN_STREAK=3
BOTTOM_EXT_MIN_ANGLE=155.0; BOTTOM_NEAR_DEG=6.0
BOTTOM_FAIL_MIN_REPS=2
DEBUG_ONBAR=bool(int(os.getenv("DEBUG_ONBAR","0")))

FACE_CHECK_INTERVAL = int(os.getenv("FACE_CHECK_INTERVAL", "3"))

CHIN_BAR_MARGIN   = float(os.getenv("CHIN_BAR_MARGIN", "0.006"))
EYE_BAR_MARGIN    = float(os.getenv("EYE_BAR_MARGIN",  "0.012"))
CHIN_NEAR_EXTRA   = float(os.getenv("CHIN_NEAR_EXTRA","0.008"))
EYE_NEAR_RELAX    = float(os.getenv("EYE_NEAR_RELAX","0.005"))

BAR_EMA_ALPHA     = float(os.getenv("BAR_EMA_ALPHA",   "0.60"))
MOUTH_VIS_THR     = float(os.getenv("MOUTH_VIS_THR",   "0.40"))
EYE_VIS_THR       = float(os.getenv("EYE_VIS_THR",     "0.40"))
FACE_PASS_WIN     = int(os.getenv("FACE_PASS_WIN",     "3"))
FACE_NEAR_WIN     = int(os.getenv("FACE_NEAR_WIN",     "5"))

BURST_ENABLED     = bool(int(os.getenv("BURST_ENABLED", "0")))
BURST_FRAMES      = int(os.getenv("BURST_FRAMES",    "2"))
INFLECT_VEL_THR   = float(os.getenv("INFLECT_VEL_THR","0.0006"))

LOCKOUT_DIST_MIN    = float(os.getenv("LOCKOUT_DIST_MIN","0.36"))
LOCKOUT_NEAR_RELAX  = float(os.getenv("LOCKOUT_NEAR_RELAX","0.02"))

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

    # ============ PATH DECISION ============
    fast_path = (fast_mode is True)
    slow_path = not fast_path
    
    if fast_path:
        return_video = False  # Never produce video in fast path
    
    # Model complexity
    model_complexity = 1

    # Encoding params (only relevant for slow path)
    if preserve_quality:
        scale=1.0; frame_skip=1; encode_crf=18 if encode_crf is None else encode_crf
    else:
        encode_crf=23 if encode_crf is None else encode_crf

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): return _ret_err("Could not open video", feedback_path)

    fps_in=cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps=max(1.0, fps_in/max(1,frame_skip))
    sec_to_frames=lambda s: max(1,int(s*effective_fps))
    frames_to_sec=lambda f: float(f)/effective_fps

    # Video writer (SLOW PATH ONLY)
    out=None
    if slow_path:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        _init_skeleton_data()  # Initialize skeleton drawing data
    
    frame_idx=0

    # Counters
    rep_count=0; good_reps=0; bad_reps=0; rep_reports=[]; all_scores=[]

    # Landmarks
    LSH=mp_pose.PoseLandmark.LEFT_SHOULDER.value;  RSH=mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LE =mp_pose.PoseLandmark.LEFT_ELBOW.value;     RE =mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW =mp_pose.PoseLandmark.LEFT_WRIST.value;     RW =mp_pose.PoseLandmark.RIGHT_WRIST.value
    LH =mp_pose.PoseLandmark.LEFT_HIP.value;       RH =mp_pose.PoseLandmark.RIGHT_HIP.value
    NOSE=mp_pose.PoseLandmark.NOSE.value
    MOUTH_L=mp_pose.PoseLandmark.MOUTH_LEFT.value; MOUTH_R=mp_pose.PoseLandmark.MOUTH_RIGHT.value
    LEFT_EYE=mp_pose.PoseLandmark.LEFT_EYE.value;  RIGHT_EYE=mp_pose.PoseLandmark.RIGHT_EYE.value

    def _pick_side_dyn(lms):
        vL=lms[LSH].visibility+lms[LE].visibility+lms[LW].visibility
        vR=lms[RSH].visibility+lms[RE].visibility+lms[RW].visibility
        return ("LEFT",LSH,LE,LW) if vL>=vR else ("RIGHT",RSH,RE,RW)

    # State
    elbow_ema=None; head_ema=None; head_prev=None; head_vel_prev=None
    baseline_head_y_global=None
    asc_base_head=None; allow_new_peak=True; last_peak_frame=-10**9
    cycle_peak_ascent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False

    onbar=False; onbar_streak=0; offbar_streak=0; prev_torso_cx=None
    offbar_frames_since_any_rep=0; nopose_frames_since_any_rep=0

    # Feedback state
    session_feedback=set()
    rt_fb_msg=None; rt_fb_hold=0
    swing_streak=0
    bottom_phase_max_elbow=None
    bottom_fail_count=0
    bottom_lockout_max=None
    bottom_already_reported=False

    # Per-cycle tips
    cycle_tip_higher=False; cycle_tip_swing=False; cycle_tip_bottom=False

    # Tempo tracking for form_tip
    rep_phase_timings = []
    phase_start_frame = None
    phase_type = None
    last_phase_transition_frame = None

    OFFBAR_STOP_FRAMES=sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES=sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES=sec_to_frames(0.8)

    REARM_DESCENT_EFF=max(RESET_DESCENT*0.60, 0.012)

    # Face-vs-bar
    bar_y_ema=None
    cycle_face_pass=False
    cycle_face_near=False
    face_pass_q = deque(maxlen=FACE_PASS_WIN)
    face_near_q = deque(maxlen=FACE_NEAR_WIN)
    face_check_counter = 0

    burst_cntr=0
    processed_frame_count = 0

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
            
            processed_frame_count += 1

            # Resize frame (for both paths, affects pose detection)
            if scale != 1.0:
                frame=cv2.resize(frame,(0,0),fx=scale,fy=scale)
            h,w=frame.shape[:2]

            # Initialize video writer (SLOW PATH ONLY)
            if slow_path and out is None:
                out=cv2.VideoWriter(output_path,fourcc,effective_fps,(w,h))

            res=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            height_live=0.0

            if not res.pose_landmarks:
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep+1) if rep_count>0 else 0
                if rep_count>0 and nopose_frames_since_any_rep>=NOPOSE_STOP_FRAMES: break
                
                # SLOW PATH: Write frame with overlay
                if slow_path and out is not None:
                    out.write(draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),height_pct=0.0))
                
                if rt_fb_hold>0: rt_fb_hold-=1
                continue

            nopose_frames_since_any_rep=0
            lms=res.pose_landmarks.landmark
            side,S,E,W=_pick_side_dyn(lms)

            min_vis=min(lms[NOSE].visibility,lms[S].visibility,lms[E].visibility,lms[W].visibility)
            vis_strict_ok=(min_vis>=VIS_THR_STRICT)

            head_raw=float(lms[NOSE].y)
            raw_elbow_L=_ang((lms[LSH].x,lms[LSH].y),(lms[LE].x,lms[LE].y),(lms[LW].x,lms[LW].y))
            raw_elbow_R=_ang((lms[RSH].x,lms[RSH].y),(lms[RE].x,lms[RE].y),(lms[RW].x,lms[RW].y))
            raw_elbow=raw_elbow_L if side=="LEFT" else raw_elbow_R
            raw_elbow_min=min(raw_elbow_L,raw_elbow_R)

            elbow_ema=_ema(elbow_ema,raw_elbow,ELBOW_EMA_ALPHA)
            head_ema=_ema(head_ema,head_raw,HEAD_EMA_ALPHA)
            head_y=head_ema; elbow_angle=elbow_ema
            if baseline_head_y_global is None: baseline_head_y_global=head_y
            height_live=float(np.clip((baseline_head_y_global-head_y)/max(0.12,HEAD_MIN_ASCENT*1.2),0.0,1.0))

            torso_cx=np.mean([lms[LSH].x,lms[RSH].x,lms[LH].x,lms[RH].x])*w
            torso_dx_norm=0.0 if prev_torso_cx is None else abs(torso_cx-prev_torso_cx)/max(1.0,w)
            prev_torso_cx=torso_cx

            lw_vis=lms[LW].visibility; rw_vis=lms[RW].visibility
            lw_above=(lw_vis>=WRIST_VIS_THR) and (lms[LW].y<lms[NOSE].y-WRIST_ABOVE_HEAD_MARGIN)
            rw_above=(rw_vis>=WRIST_VIS_THR) and (lms[RW].y<lms[NOSE].y-WRIST_ABOVE_HEAD_MARGIN)
            grip=(lw_above or rw_above)

            if vis_strict_ok and grip and (torso_dx_norm<=TORSO_X_THR):
                onbar_streak+=1; offbar_streak=0
            else:
                offbar_streak+=1; onbar_streak=0

            # Enter onbar
            if (not onbar) and onbar_streak>=ONBAR_MIN_FRAMES:
                onbar=True
                asc_base_head=None; allow_new_peak=True; swing_streak=0
                cycle_peak_ascent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_tip_higher=False; cycle_tip_swing=False; cycle_tip_bottom=False
                cycle_face_pass=False; cycle_face_near=False
                face_pass_q.clear(); face_near_q.clear()
                bottom_phase_max_elbow=None
                bottom_lockout_max=None
                phase_start_frame=None
                phase_type=None

            # Exit onbar
            if onbar and offbar_streak>=OFFBAR_MIN_FRAMES:
                if _check_bottom_extension(bottom_phase_max_elbow, bottom_lockout_max):
                    cycle_tip_bottom = True
                    bottom_fail_count += 1
                    if bottom_fail_count >= BOTTOM_FAIL_MIN_REPS and not bottom_already_reported:
                        session_feedback.add(FB_CUE_BOTTOM)
                        bottom_already_reported = True

                if (not counted_this_cycle) and (cycle_peak_ascent>=HEAD_MIN_ASCENT) and (cycle_min_elbow<=ELBOW_TOP_ANGLE):
                    rep_has_tip = cycle_tip_higher or cycle_tip_swing or cycle_tip_bottom
                    _count_rep(rep_reports,rep_count,0.0,cycle_min_elbow,
                               asc_base_head if asc_base_head is not None else head_y,
                               baseline_head_y_global-cycle_peak_ascent if baseline_head_y_global is not None else head_y,
                               all_scores, rep_has_tip)
                    rep_count+=1
                    if rep_has_tip: bad_reps+=1
                    else: good_reps+=1

                onbar=False; offbar_frames_since_any_rep=0
                asc_base_head=None; cycle_peak_ascent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_tip_higher=False; cycle_tip_swing=False; cycle_tip_bottom=False
                cycle_face_pass=False; cycle_face_near=False
                face_pass_q.clear(); face_near_q.clear()
                bottom_phase_max_elbow=None
                bottom_lockout_max=None
                phase_start_frame=None
                phase_type=None

            if (not onbar) and rep_count>0:
                offbar_frames_since_any_rep+=1
                if offbar_frames_since_any_rep>=OFFBAR_STOP_FRAMES: break

            head_vel=0.0 if head_prev is None else (head_y-head_prev)
            cur_rt=None

            # Micro-burst
            if BURST_ENABLED and onbar and (asc_base_head is not None):
                near_inflect = (abs(head_vel) <= INFLECT_VEL_THR)
                sign_flip = (head_vel_prev is not None) and ((head_vel_prev > 0 and head_vel <= 0) or (head_vel_prev < 0 and head_vel >= 0))
                if near_inflect or sign_flip:
                    if burst_cntr == 0:
                        burst_cntr = BURST_FRAMES
            head_vel_prev = head_vel

            if onbar and vis_strict_ok:
                # REP COUNTING
                if asc_base_head is None:
                    if head_vel<-abs(HEAD_VEL_UP_TINY):
                        asc_base_head=head_y
                        cycle_peak_ascent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                        bottom_phase_max_elbow=None
                        bottom_lockout_max=None
                        phase_start_frame=processed_frame_count
                        phase_type="concentric"
                else:
                    cycle_peak_ascent=max(cycle_peak_ascent,(asc_base_head-head_y))
                    cycle_min_elbow=min(cycle_min_elbow,elbow_angle)

                    # Track bottom metrics
                    max_elb_now = max(raw_elbow_L, raw_elbow_R)
                    if bottom_phase_max_elbow is None: bottom_phase_max_elbow = max_elb_now
                    else: bottom_phase_max_elbow = max(bottom_phase_max_elbow, max_elb_now)

                    # Lockout distance
                    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
                    mid_hp = ((lms[LH].x + lms[RH].x)/2.0, (lms[LH].y + lms[RH].y)/2.0)
                    torso_len = _dist_xy(mid_sh, mid_hp) + 1e-6
                    left_lock  = _dist_xy((lms[LSH].x, lms[LSH].y), (lms[LW].x, lms[LW].y)) / torso_len
                    right_lock = _dist_xy((lms[RSH].x, lms[RSH].y), (lms[RW].x, lms[RW].y)) / torso_len
                    lockout_now = max(left_lock, right_lock)
                    if bottom_lockout_max is None: bottom_lockout_max = lockout_now
                    else: bottom_lockout_max = max(bottom_lockout_max, lockout_now)

                    reset_by_desc=(asc_base_head is not None) and ((head_y-asc_base_head)>=RESET_DESCENT)
                    reset_by_elb =(elbow_angle>=RESET_ELBOW)
                    
                    if reset_by_desc or reset_by_elb:
                        # Eccentric phase ended → record timing
                        if phase_type == "eccentric" and phase_start_frame is not None:
                            eccentric_frames = processed_frame_count - phase_start_frame
                            eccentric_sec = frames_to_sec(eccentric_frames)
                            if len(rep_phase_timings) > 0:
                                rep_phase_timings[-1]["eccentric_sec"] = eccentric_sec
                        
                        if _check_bottom_extension(bottom_phase_max_elbow, bottom_lockout_max):
                            cycle_tip_bottom = True
                            bottom_fail_count += 1
                            if bottom_fail_count >= BOTTOM_FAIL_MIN_REPS and not bottom_already_reported:
                                session_feedback.add(FB_CUE_BOTTOM)
                                bottom_already_reported = True

                        if (not counted_this_cycle) and (cycle_peak_ascent>=HEAD_MIN_ASCENT) and (cycle_min_elbow<=ELBOW_TOP_ANGLE):
                            rep_has_tip = cycle_tip_higher or cycle_tip_swing or cycle_tip_bottom
                            _count_rep(rep_reports,rep_count,0.0,cycle_min_elbow,
                                       asc_base_head,
                                       baseline_head_y_global-cycle_peak_ascent if baseline_head_y_global is not None else head_y,
                                       all_scores, rep_has_tip)
                            rep_count+=1
                            if rep_has_tip: bad_reps+=1
                            else: good_reps+=1

                        asc_base_head=head_y
                        cycle_peak_ascent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                        allow_new_peak=True
                        bottom_phase_max_elbow=None
                        bottom_lockout_max=None
                        cycle_tip_higher=False; cycle_tip_swing=False; cycle_tip_bottom=False
                        cycle_face_pass=False; cycle_face_near=False
                        face_pass_q.clear(); face_near_q.clear()
                        phase_start_frame=processed_frame_count
                        phase_type="concentric"

                ascent_amt=0.0 if asc_base_head is None else (asc_base_head-head_y)

                at_top=(elbow_angle<=ELBOW_TOP_ANGLE) and (ascent_amt>=HEAD_MIN_ASCENT)
                raw_top=(raw_elbow_min<=(ELBOW_TOP_ANGLE+4.0)) and (ascent_amt>=HEAD_MIN_ASCENT*0.92)
                at_top=at_top or raw_top
                can_cnt=(frame_idx - last_peak_frame) >= REFRACTORY_FRAMES

                if at_top and allow_new_peak and can_cnt and (not counted_this_cycle):
                    # Concentric ended → record timing
                    if phase_type == "concentric" and phase_start_frame is not None:
                        concentric_frames = processed_frame_count - phase_start_frame
                        concentric_sec = frames_to_sec(concentric_frames)
                        rep_phase_timings.append({
                            "concentric_sec": concentric_sec,
                            "top_hold_sec": 0.0,
                            "eccentric_sec": 0.0
                        })
                        phase_start_frame = processed_frame_count
                        phase_type = "top"
                    
                    rep_has_tip = cycle_tip_higher or cycle_tip_swing or cycle_tip_bottom
                    _count_rep(rep_reports,rep_count,0.0,elbow_angle,
                               asc_base_head if asc_base_head is not None else head_y, head_y,
                               all_scores, rep_has_tip)
                    rep_count+=1
                    if rep_has_tip: bad_reps+=1
                    else: good_reps+=1
                    last_peak_frame=frame_idx; allow_new_peak=False; counted_this_cycle=True
                    bottom_phase_max_elbow = max(raw_elbow_L, raw_elbow_R)

                if (allow_new_peak is False) and (last_peak_frame>0):
                    if head_prev is not None and (head_y - head_prev) > 0 and (asc_base_head is not None):
                        if (head_y - (asc_base_head - cycle_peak_ascent)) >= REARM_DESCENT_EFF:
                            # Top hold ended → record timing
                            if phase_type == "top" and phase_start_frame is not None:
                                top_hold_frames = processed_frame_count - phase_start_frame
                                top_hold_sec = frames_to_sec(top_hold_frames)
                                if len(rep_phase_timings) > 0:
                                    rep_phase_timings[-1]["top_hold_sec"] = top_hold_sec
                                phase_start_frame = processed_frame_count
                                phase_type = "eccentric"
                            
                            allow_new_peak=True

                # Face detection (throttled)
                face_check_counter += 1
                if face_check_counter >= FACE_CHECK_INTERVAL:
                    face_check_counter = 0
                    
                    cur_bar_y = min(lms[LW].y, lms[RW].y)
                    bar_y_ema = cur_bar_y if bar_y_ema is None else (BAR_EMA_ALPHA*cur_bar_y + (1 - BAR_EMA_ALPHA)*bar_y_ema)
                    bar_y_eff = bar_y_ema if bar_y_ema is not None else cur_bar_y

                    mouth_ok = (lms[MOUTH_L].visibility>=MOUTH_VIS_THR and lms[MOUTH_R].visibility>=MOUTH_VIS_THR)
                    eyes_ok  = (lms[LEFT_EYE].visibility>=EYE_VIS_THR and lms[RIGHT_EYE].visibility>=EYE_VIS_THR)
                    mouth_y = (lms[MOUTH_L].y + lms[MOUTH_R].y)*0.5 if mouth_ok else None
                    eye_min_y = min(lms[LEFT_EYE].y, lms[RIGHT_EYE].y) if eyes_ok else None

                    eye_margin = max(EYE_BAR_MARGIN, CHIN_BAR_MARGIN + 0.002)
                    passed_by_chin = (mouth_y is not None) and (mouth_y <= bar_y_eff + CHIN_BAR_MARGIN)
                    passed_by_eyes = (mouth_y is None) and (eye_min_y is not None) and (eye_min_y <= bar_y_eff - eye_margin)
                    near_by_chin  = (mouth_y is not None) and (mouth_y <= bar_y_eff + (CHIN_BAR_MARGIN + CHIN_NEAR_EXTRA))
                    near_by_eyes  = (mouth_y is None) and (eye_min_y is not None) and (eye_min_y <= bar_y_eff - max(0.001, eye_margin - EYE_NEAR_RELAX))

                    face_pass_q.append(passed_by_chin or passed_by_eyes)
                    face_near_q.append(near_by_chin or near_by_eyes)
                    pass_hits = sum(face_pass_q)
                    near_hits = sum(face_near_q)
                    if pass_hits >= FACE_PASS_WIN:
                        cycle_face_pass = True
                    if near_hits >= FACE_NEAR_WIN:
                        cycle_face_near = True

                    near_peak = (
                        asc_base_head is not None and
                        ascent_amt >= HEAD_MIN_ASCENT * 0.85
                    )

                    if near_peak and (head_vel < -abs(HEAD_VEL_UP_TINY)) and allow_new_peak:
                        face_failed = (
                            len(face_near_q) >= FACE_NEAR_WIN
                            and (not cycle_face_pass)
                            and (not cycle_face_near)
                        )

                        if face_failed and not cycle_tip_higher:
                            cycle_tip_higher = True
                            session_feedback.add(FB_CUE_HIGHER)
                            cur_rt = FB_CUE_HIGHER
                            allow_new_peak = False

                # Swing cue
                if torso_dx_norm>SWING_THR: swing_streak+=1
                else: swing_streak=max(0,swing_streak-1)
                if (cur_rt is None) and (swing_streak>=SWING_MIN_STREAK) and (not cycle_tip_swing):
                    session_feedback.add(FB_CUE_SWING); cur_rt=FB_CUE_SWING; cycle_tip_swing=True

            else:
                asc_base_head=None; allow_new_peak=True
                swing_streak=0

            # RT feedback hold
            if cur_rt:
                if cur_rt!=rt_fb_msg: rt_fb_msg=cur_rt; rt_fb_hold=RT_FB_HOLD_FRAMES
                else: rt_fb_hold=max(rt_fb_hold,RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold>0: rt_fb_hold-=1

            # ============ SLOW PATH: Draw and write frame ============
            if slow_path and out is not None:
                frame=draw_body_only(frame,lms)
                frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),height_pct=height_live)
                out.write(frame)

            if head_y is not None: head_prev=head_y

    # EOF post-hoc
    if onbar and (not counted_this_cycle) and (cycle_peak_ascent>=HEAD_MIN_ASCENT) and (cycle_min_elbow<=ELBOW_TOP_ANGLE):
        if _check_bottom_extension(bottom_phase_max_elbow, bottom_lockout_max):
            cycle_tip_bottom = True
            bottom_fail_count += 1
            if bottom_fail_count >= BOTTOM_FAIL_MIN_REPS and not bottom_already_reported:
                session_feedback.add(FB_CUE_BOTTOM)
                bottom_already_reported = True

        rep_has_tip = cycle_tip_higher or cycle_tip_swing or cycle_tip_bottom
        _count_rep(rep_reports,rep_count,0.0,cycle_min_elbow,
                   asc_base_head if asc_base_head is not None else (baseline_head_y_global or 0.0),
                   (baseline_head_y_global - cycle_peak_ascent) if baseline_head_y_global is not None else (baseline_head_y_global or 0.0),
                   all_scores, rep_has_tip)
        rep_count+=1
        if rep_has_tip: bad_reps+=1
        else: good_reps+=1

    cap.release()
    if slow_path and out: 
        out.release()
    cv2.destroyAllWindows()

    # ============ SESSION SCORE (feedback-based) ============
    if rep_count==0:
        technique_score=0.0
    elif all_scores:
        technique_score=_half_floor10(np.mean(all_scores))
    else:
        if session_feedback:
            penalty = sum(FB_WEIGHTS.get(m,FB_DEFAULT_WEIGHT) for m in set(session_feedback))
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
        else:
            penalty = 0.0
        technique_score=_half_floor10(max(0.0,10.0-penalty))

    # ============ FORM_TIP LOGIC (optimization, not score-based) ============
    form_tip = None
    
    if rep_count >= 2 and len(rep_phase_timings) >= 2:
        concentric_times = [t["concentric_sec"] for t in rep_phase_timings if t["concentric_sec"] > 0]
        top_hold_times = [t["top_hold_sec"] for t in rep_phase_timings if t["top_hold_sec"] > 0]
        eccentric_times = [t["eccentric_sec"] for t in rep_phase_timings if t["eccentric_sec"] > 0]
        
        tip_scores = {}
        
        if len(eccentric_times) >= 2:
            avg_eccentric = np.mean(eccentric_times)
            if avg_eccentric < ECCENTRIC_FAST_THR:
                tip_scores[FORM_TIP_SLOW_ECCENTRIC] = 100
        
        if len(top_hold_times) >= 2:
            avg_top_hold = np.mean(top_hold_times)
            if avg_top_hold < TOP_HOLD_SHORT_THR:
                tip_scores[FORM_TIP_TOP_PAUSE] = 80
        
        if len(concentric_times) >= 2:
            avg_concentric = np.mean(concentric_times)
            if avg_concentric > CONCENTRIC_SLOW_THR:
                tip_scores[FORM_TIP_EXPLOSIVE_PULL] = 70
        
        if len(concentric_times) >= 3:
            total_times = [c + t + e for c, t, e in zip(
                concentric_times[:min(len(concentric_times), len(top_hold_times), len(eccentric_times))],
                top_hold_times[:min(len(concentric_times), len(top_hold_times), len(eccentric_times))],
                eccentric_times[:min(len(concentric_times), len(top_hold_times), len(eccentric_times))]
            )]
            if len(total_times) >= 3:
                cv = np.std(total_times) / (np.mean(total_times) + 1e-9)
                if cv > TEMPO_VARIANCE_THR:
                    tip_scores[FORM_TIP_TEMPO_CONSISTENCY] = 60
        
        if tip_scores:
            best_tip = max(tip_scores, key=tip_scores.get)
            form_tip = FORM_TIP_MESSAGES.get(best_tip)

    # Build feedback list
    all_fb = set(session_feedback) if session_feedback else set()
    fb_list = [cue for cue in FEEDBACK_ORDER if cue in all_fb]
    if not fb_list and rep_count > 0 and bad_reps == 0:
        fb_list = ["Great form! Keep it up 💪"]

    # Write feedback file
    try:
        with open(feedback_path,"w",encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            if fb_list:
                f.write("Feedback:\n")
                for ln in fb_list: f.write(f"- {ln}\n")
            if form_tip:
                f.write(f"\nForm Tip: {form_tip}\n")
    except Exception:
        pass

    # ============ SLOW PATH: Encode video ============
    final_path=""
    if slow_path and os.path.exists(output_path):
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
        "video_path": final_path if slow_path else "",
        "feedback_path": feedback_path
    }
    if form_tip is not None:
        result["form_tip"] = form_tip

    return result

def _check_bottom_extension(bottom_phase_max_elbow, bottom_lockout_max):
    """Returns True if bottom extension FAILED (needs tip)"""
    elbow_ok   = (bottom_phase_max_elbow is not None) and (bottom_phase_max_elbow >= BOTTOM_EXT_MIN_ANGLE)
    elbow_near = (bottom_phase_max_elbow is not None) and (bottom_phase_max_elbow >= (BOTTOM_EXT_MIN_ANGLE - BOTTOM_NEAR_DEG))
    lock_ok    = (bottom_lockout_max is not None) and (bottom_lockout_max >= LOCKOUT_DIST_MIN)
    lock_near  = (bottom_lockout_max is not None) and (bottom_lockout_max >= (LOCKOUT_DIST_MIN - LOCKOUT_NEAR_RELAX))
    extended   = elbow_ok or lock_ok
    near_enough= elbow_near or lock_near
    return not (extended or near_enough)

def _count_rep(rep_reports, rep_count, _rep_penalty_unused, top_elbow, ascent_from, peak_head_y, all_scores, rep_has_tip):
    rep_score = 10.0 if not rep_has_tip else 9.5
    all_scores.append(rep_score)
    rep_reports.append({
        "rep_index": int(rep_count+1),
        "score": float(rep_score),
        "good": bool(rep_score >= 10.0 - 1e-6),
        "top_elbow": float(top_elbow),
        "ascent_from": float(ascent_from),
        "peak_head_y": float(peak_head_y)
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
    return run_pullup_analysis(*args, **kwargs)
