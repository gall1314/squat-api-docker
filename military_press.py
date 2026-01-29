# -*- coding: utf-8 -*-
# overhead_press_analysis.py — complete overhead press counter with form feedback

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
REPS_FONT=_load_font(FONT_PATH,REPS_FONT_SIZE)
FEEDBACK_FONT=_load_font(FONT_PATH,FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT=_load_font(FONT_PATH,DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT=_load_font(FONT_PATH,DEPTH_PCT_FONT_SIZE)

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
    _BODY_CONNECTIONS=tuple((a,b) in mp_pose.POSE_CONNECTIONS if a not in _FACE_LMS and b not in _FACE_LMS)
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

# ============ Overhead Press Parameters ============
# ספירת חזרות
ELBOW_LOCKOUT_ANGLE = 165.0      # זווית מרפק מינימלית למעלה (ננעול מלא)
WRIST_MIN_ASCENT = 0.08          # שורש יד צריך לעלות לפחות 8% מגובה המסך
RESET_DESCENT = 0.04             # שורש יד צריך לרדת חזרה 4% כדי לאפס
RESET_ELBOW = 100.0              # זווית מרפק מקסימלית בחזרה למטה (כתפיים)
REFRACTORY_FRAMES = 3            # מסגרות המתנה בין חזרות

# EMA smoothing
ELBOW_EMA_ALPHA = 0.35
WRIST_EMA_ALPHA = 0.30

# זיהוי תנוחת overhead press (עומד, שורשי ידיים מעל כתפיים)
VIS_THR_STRICT = 0.30
WRIST_ABOVE_SHOULDER_MARGIN = 0.03  # שורשי ידיים צריכים להיות מעל כתפיים
HIP_STABILITY_THR = 0.010            # תנועת ירכיים מקסימלית (לא כיפוף)
ONPRESS_MIN_FRAMES = 3
OFFPRESS_MIN_FRAMES = 6
AUTO_STOP_AFTER_EXIT_SEC = 1.5
TAIL_NOPOSE_STOP_SEC = 1.0

# ============ Feedback Cues & Weights ============
FB_CUE_LOCKOUT = "Fully lockout elbows overhead"
FB_CUE_ARCH = "Don't hyperextend lower back (brace core)"
FB_CUE_PATH = "Press straight up (keep bar close to face)"
FB_CUE_LEGS = "Don't use leg drive (strict press only)"

FB_W_LOCKOUT = float(os.getenv("FB_W_LOCKOUT", "1.0"))
FB_W_ARCH = float(os.getenv("FB_W_ARCH", "0.9"))
FB_W_PATH = float(os.getenv("FB_W_PATH", "0.7"))
FB_W_LEGS = float(os.getenv("FB_W_LEGS", "0.8"))

FB_WEIGHTS = {
    FB_CUE_LOCKOUT: FB_W_LOCKOUT,
    FB_CUE_ARCH: FB_W_ARCH,
    FB_CUE_PATH: FB_W_PATH,
    FB_CUE_LEGS: FB_W_LEGS,
}
FB_DEFAULT_WEIGHT = 0.5
PENALTY_MIN_IF_ANY = 0.5
FORM_TIP_PRIORITY = [FB_CUE_LOCKOUT, FB_CUE_ARCH, FB_CUE_LEGS, FB_CUE_PATH]

# ============ Form Detection Thresholds ============
# Lockout
LOCKOUT_MIN_ANGLE = 165.0        # זווית מרפק מינימלית לננעול
LOCKOUT_NEAR_DEG = 8.0           # טווח "קרוב מספיק"

# Back arch (hyperextension)
TORSO_MAX_ARCH = 15.0            # זווית מקסימלית של קמירת גב מאחור
ARCH_NEAR_DEG = 4.0              # טווח "קרוב מספיק"

# Bar path (horizontal deviation)
PATH_MAX_DEVIATION = 0.08        # סטייה אופקית מקסימלית של שורש היד
PATH_NEAR_RELAX = 0.02           # טווח "קרוב מספיק"

# Leg drive detection
HIP_DROP_MAX = 0.025             # ירידת ירכיים מקסימלית (סימן ל-leg drive)
HIP_DROP_NEAR = 0.010            # טווח "קרוב מספיק"

# Minimum reps before session feedback
LOCKOUT_FAIL_MIN_REPS = 2
ARCH_FAIL_MIN_REPS = 2
PATH_FAIL_MIN_REPS = 2
LEGS_FAIL_MIN_REPS = 2

# ============ Micro-burst ============
BURST_FRAMES = int(os.getenv("BURST_FRAMES", "2"))
INFLECT_VEL_THR = float(os.getenv("INFLECT_VEL_THR", "0.0008"))

DEBUG_ONPRESS = bool(int(os.getenv("DEBUG_ONPRESS", "0")))

def run_overhead_press_analysis(video_path,
                                 frame_skip=3,
                                 scale=0.4,
                                 output_path="overhead_press_analyzed.mp4",
                                 feedback_path="overhead_press_feedback.txt",
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
    elbow_ema=None; wrist_ema=None; wrist_prev=None; wrist_vel_prev=None
    baseline_wrist_y=None; baseline_wrist_x=None
    asc_base_wrist=None; allow_new_peak=True; last_peak_frame=-10**9
    cycle_max_ascent=0.0; cycle_max_elbow=0.0; counted_this_cycle=False

    onpress=False; onpress_streak=0; offpress_streak=0; prev_hip_y=None
    offpress_frames_since_any_rep=0; nopose_frames_since_any_rep=0

    # Feedback state
    session_feedback=set()
    rt_fb_msg=None; rt_fb_hold=0

    # Per-cycle trackers
    cycle_tip_lockout=False; cycle_tip_arch=False; cycle_tip_path=False; cycle_tip_legs=False
    lockout_fail_count=0; arch_fail_count=0; path_fail_count=0; legs_fail_count=0
    lockout_already_reported=False; arch_already_reported=False
    path_already_reported=False; legs_already_reported=False

    # Phase trackers
    top_phase_max_elbow=None
    cycle_max_arch=None
    cycle_max_path_dev=None
    cycle_min_hip_y=None
    cycle_start_hip_y=None

    OFFPRESS_STOP_FRAMES=sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES=sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES=sec_to_frames(0.8)

    REARM_DESCENT_EFF=max(RESET_DESCENT*0.60, 0.020)

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
            height_live=0.0

            if not res.pose_landmarks:
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep+1) if rep_count>0 else 0
                if rep_count>0 and nopose_frames_since_any_rep>=NOPOSE_STOP_FRAMES: break
                if return_video and out is not None:
                    out.write(draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),height_pct=0.0))
                if rt_fb_hold>0: rt_fb_hold-=1
                continue

            nopose_frames_since_any_rep=0
            lms=res.pose_landmarks.landmark
            side,S,E,W=_pick_side_dyn(lms)

            min_vis=min(lms[S].visibility,lms[E].visibility,lms[W].visibility)
            vis_strict_ok=(min_vis>=VIS_THR_STRICT)

            wrist_raw_y=float(lms[W].y)
            wrist_raw_x=float(lms[W].x)
            raw_elbow_L=_ang((lms[LSH].x,lms[LSH].y),(lms[LE].x,lms[LE].y),(lms[LW].x,lms[LW].y))
            raw_elbow_R=_ang((lms[RSH].x,lms[RSH].y),(lms[RE].x,lms[RE].y),(lms[RW].x,lms[RW].y))
            raw_elbow=raw_elbow_L if side=="LEFT" else raw_elbow_R
            raw_elbow_max=max(raw_elbow_L,raw_elbow_R)

            elbow_ema=_ema(elbow_ema,raw_elbow,ELBOW_EMA_ALPHA)
            wrist_ema=_ema(wrist_ema,wrist_raw_y,WRIST_EMA_ALPHA)
            wrist_y=wrist_ema; elbow_angle=elbow_ema
            
            if baseline_wrist_y is None: 
                baseline_wrist_y=wrist_y
                baseline_wrist_x=wrist_raw_x
            height_live=float(np.clip((baseline_wrist_y-wrist_y)/max(0.12,WRIST_MIN_ASCENT*1.2),0.0,1.0))

            # Hip stability (detect leg drive)
            hip_y=(lms[LH].y + lms[RH].y) / 2.0

            # Check press position (wrists above shoulders, standing)
            lw_above=(lms[LW].visibility>=VIS_THR_STRICT) and (lms[LW].y<lms[LSH].y-WRIST_ABOVE_SHOULDER_MARGIN)
            rw_above=(lms[RW].visibility>=VIS_THR_STRICT) and (lms[RW].y<lms[RSH].y-WRIST_ABOVE_SHOULDER_MARGIN)
            in_position=(lw_above or rw_above)

            hip_stable=True
            if prev_hip_y is not None:
                hip_movement=abs(hip_y-prev_hip_y)
                hip_stable=(hip_movement<=HIP_STABILITY_THR)
            prev_hip_y=hip_y

            if vis_strict_ok and in_position and hip_stable:
                onpress_streak+=1; offpress_streak=0
            else:
                offpress_streak+=1; onpress_streak=0

            if DEBUG_ONPRESS and frame_idx % 10 == 0:
                print(f"[DBG] f={frame_idx} onpress={onpress} vis={min_vis:.2f} lwAbove={lw_above} rwAbove={rw_above} hipStable={hip_stable}")

            # Enter press position
            if (not onpress) and onpress_streak>=ONPRESS_MIN_FRAMES:
                onpress=True
                asc_base_wrist=None; allow_new_peak=True
                cycle_max_ascent=0.0; cycle_max_elbow=0.0; counted_this_cycle=False
                cycle_tip_lockout=False; cycle_tip_arch=False; cycle_tip_path=False; cycle_tip_legs=False
                top_phase_max_elbow=None
                cycle_max_arch=None
                cycle_max_path_dev=None
                cycle_min_hip_y=None
                cycle_start_hip_y=hip_y

            # Exit press position → post-hoc
            if onpress and offpress_streak>=OFFPRESS_MIN_FRAMES:
                # Evaluate form at cycle end
                _evaluate_cycle_form(lms, top_phase_max_elbow, cycle_max_arch, 
                                   cycle_max_path_dev, cycle_min_hip_y, cycle_start_hip_y,
                                   lockout_fail_count, arch_fail_count, path_fail_count, legs_fail_count,
                                   lockout_already_reported, arch_already_reported, 
                                   path_already_reported, legs_already_reported,
                                   session_feedback, locals())

                # Count rep if valid
                if (not counted_this_cycle) and (cycle_max_ascent>=WRIST_MIN_ASCENT) and (cycle_max_elbow>=ELBOW_LOCKOUT_ANGLE):
                    rep_has_tip = cycle_tip_lockout or cycle_tip_arch or cycle_tip_path or cycle_tip_legs
                    _count_rep(rep_reports,rep_count,cycle_max_elbow,
                               asc_base_wrist if asc_base_wrist is not None else wrist_y,
                               baseline_wrist_y-cycle_max_ascent if baseline_wrist_y is not None else wrist_y,
                               all_scores, rep_has_tip)
                    rep_count+=1
                    if rep_has_tip: bad_reps+=1
                    else: good_reps+=1

                onpress=False; offpress_frames_since_any_rep=0
                asc_base_wrist=None; cycle_max_ascent=0.0; cycle_max_elbow=0.0; counted_this_cycle=False
                cycle_tip_lockout=False; cycle_tip_arch=False; cycle_tip_path=False; cycle_tip_legs=False
                top_phase_max_elbow=None
                cycle_max_arch=None
                cycle_max_path_dev=None
                cycle_min_hip_y=None
                cycle_start_hip_y=None

            if (not onpress) and rep_count>0:
                offpress_frames_since_any_rep+=1
                if offpress_frames_since_any_rep>=OFFPRESS_STOP_FRAMES: break

            wrist_vel=0.0 if wrist_prev is None else (wrist_y-wrist_prev)
            cur_rt=None

            # Micro-burst near inflection
            if onpress and (asc_base_wrist is not None):
                near_inflect = (abs(wrist_vel) <= INFLECT_VEL_THR)
                sign_flip = (wrist_vel_prev is not None) and ((wrist_vel_prev < 0 and wrist_vel >= 0) or (wrist_vel_prev > 0 and wrist_vel <= 0))
                if near_inflect or sign_flip:
                    burst_cntr = max(burst_cntr, BURST_FRAMES)
            wrist_vel_prev = wrist_vel

            if onpress and vis_strict_ok:
                # REP COUNTING
                if asc_base_wrist is None:
                    if wrist_vel<-abs(INFLECT_VEL_THR):  # Moving up
                        asc_base_wrist=wrist_y
                        cycle_max_ascent=0.0; cycle_max_elbow=elbow_angle; counted_this_cycle=False
                        top_phase_max_elbow=None
                        cycle_max_arch=None
                        cycle_max_path_dev=None
                        cycle_min_hip_y=hip_y
                        cycle_start_hip_y=hip_y
                        baseline_wrist_x=wrist_raw_x
                else:
                    cycle_max_ascent=max(cycle_max_ascent,(asc_base_wrist-wrist_y))
                    cycle_max_elbow=max(cycle_max_elbow,elbow_angle)

                    # Track top phase (highest point - lockout)
                    max_elb_now = max(raw_elbow_L, raw_elbow_R)
                    if top_phase_max_elbow is None: top_phase_max_elbow = max_elb_now
                    else: top_phase_max_elbow = max(top_phase_max_elbow, max_elb_now)

                    # Track back arch
                    arch_angle = _calculate_back_arch(lms, LSH, RSH, LH, RH)
                    if cycle_max_arch is None: cycle_max_arch = arch_angle
                    else: cycle_max_arch = max(cycle_max_arch, arch_angle)

                    # Track bar path deviation
                    path_dev = abs(wrist_raw_x - baseline_wrist_x)
                    if cycle_max_path_dev is None: cycle_max_path_dev = path_dev
                    else: cycle_max_path_dev = max(cycle_max_path_dev, path_dev)

                    # Track hip drop (leg drive)
                    if cycle_min_hip_y is None: cycle_min_hip_y = hip_y
                    else: cycle_min_hip_y = min(cycle_min_hip_y, hip_y)

                    reset_by_desc=(asc_base_wrist is not None) and ((wrist_y-asc_base_wrist)>=RESET_DESCENT)
                    reset_by_elb =(elbow_angle<=RESET_ELBOW)
                    
                    if reset_by_desc or reset_by_elb:
                        # Evaluate form
                        _evaluate_cycle_form(lms, top_phase_max_elbow, cycle_max_arch,
                                           cycle_max_path_dev, cycle_min_hip_y, cycle_start_hip_y,
                                           lockout_fail_count, arch_fail_count, path_fail_count, legs_fail_count,
                                           lockout_already_reported, arch_already_reported,
                                           path_already_reported, legs_already_reported,
                                           session_feedback, locals())

                        # Count rep
                        if (not counted_this_cycle) and (cycle_max_ascent>=WRIST_MIN_ASCENT) and (cycle_max_elbow>=ELBOW_LOCKOUT_ANGLE):
                            rep_has_tip = cycle_tip_lockout or cycle_tip_arch or cycle_tip_path or cycle_tip_legs
                            _count_rep(rep_reports,rep_count,cycle_max_elbow,
                                       asc_base_wrist,
                                       baseline_wrist_y-cycle_max_ascent if baseline_wrist_y is not None else wrist_y,
                                       all_scores, rep_has_tip)
                            rep_count+=1
                            if rep_has_tip: bad_reps+=1
                            else: good_reps+=1

                        # Reset for next cycle
                        asc_base_wrist=wrist_y
                        cycle_max_ascent=0.0; cycle_max_elbow=elbow_angle; counted_this_cycle=False
                        allow_new_peak=True
                        top_phase_max_elbow=None
                        cycle_max_arch=None
                        cycle_max_path_dev=None
                        cycle_min_hip_y=hip_y
                        cycle_start_hip_y=hip_y
                        baseline_wrist_x=wrist_raw_x
                        cycle_tip_lockout=False; cycle_tip_arch=False; cycle_tip_path=False; cycle_tip_legs=False

                ascent_amt=0.0 if asc_base_wrist is None else (asc_base_wrist-wrist_y)

                at_top=(elbow_angle>=ELBOW_LOCKOUT_ANGLE) and (ascent_amt>=WRIST_MIN_ASCENT)
                raw_top=(raw_elbow_max>=(ELBOW_LOCKOUT_ANGLE-5.0)) and (ascent_amt>=WRIST_MIN_ASCENT*0.92)
                at_top=at_top or raw_top
                can_cnt=(frame_idx - last_peak_frame) >= REFRACTORY_FRAMES

                if at_top and allow_new_peak and can_cnt and (not counted_this_cycle):
                    rep_has_tip = cycle_tip_lockout or cycle_tip_arch or cycle_tip_path or cycle_tip_legs
                    _count_rep(rep_reports,rep_count,elbow_angle,
                               asc_base_wrist if asc_base_wrist is not None else wrist_y, wrist_y,
                               all_scores, rep_has_tip)
                    rep_count+=1
                    if rep_has_tip: bad_reps+=1
                    else: good_reps+=1
                    last_peak_frame=frame_idx; allow_new_peak=False; counted_this_cycle=True

                if (allow_new_peak is False) and (last_peak_frame>0):
                    if wrist_prev is not None and (wrist_y - wrist_prev) > 0 and (asc_base_wrist is not None):
                        if (wrist_y - (asc_base_wrist - cycle_max_ascent)) >= REARM_DESCENT_EFF:
                            allow_new_peak=True

                # Real-time feedback at top
                if at_top and not cycle_tip_lockout:
                    if top_phase_max_elbow and top_phase_max_elbow < LOCKOUT_MIN_ANGLE:
                        cycle_tip_lockout = True
                        lockout_fail_count += 1
                        if lockout_fail_count >= LOCKOUT_FAIL_MIN_REPS and not lockout_already_reported:
                            session_feedback.add(FB_CUE_LOCKOUT)
                            lockout_already_reported = True
                            cur_rt = FB_CUE_LOCKOUT

            else:
                asc_base_wrist=None; allow_new_peak=True

            # RT hold
            if cur_rt:
                if cur_rt!=rt_fb_msg: rt_fb_msg=cur_rt; rt_fb_hold=RT_FB_HOLD_FRAMES
                else: rt_fb_hold=max(rt_fb_hold,RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold>0: rt_fb_hold-=1

            # Draw
            if return_video and out is not None:
                frame=draw_body_only(frame,lms)
                frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),height_pct=height_live)
                out.write(frame)

            if wrist_y is not None: wrist_prev=wrist_y

    # EOF post-hoc
    if onpress and (not counted_this_cycle) and (cycle_max_ascent>=WRIST_MIN_ASCENT) and (cycle_max_elbow>=ELBOW_LOCKOUT_ANGLE):
        _evaluate_cycle_form(lms, top_phase_max_elbow, cycle_max_arch,
                           cycle_max_path_dev, cycle_min_hip_y, cycle_start_hip_y,
                           lockout_fail_count, arch_fail_count, path_fail_count, legs_fail_count,
                           lockout_already_reported, arch_already_reported,
                           path_already_reported, legs_already_reported,
                           session_feedback, locals())

        rep_has_tip = cycle_tip_lockout or cycle_tip_arch or cycle_tip_path or cycle_tip_legs
        _count_rep(rep_reports,rep_count,cycle_max_elbow,
                   asc_base_wrist if asc_base_wrist is not None else (baseline_wrist_y or 0.0),
                   (baseline_wrist_y - cycle_max_ascent) if baseline_wrist_y is not None else (baseline_wrist_y or 0.0),
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
def _calculate_back_arch(lms, LSH, RSH, LH, RH):
    """חישוב קמירת גב מאחור (hyperextension)"""
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    mid_hp = ((lms[LH].x + lms[RH].x)/2.0, (lms[LH].y + lms[RH].y)/2.0)
    
    # Vector from hip to shoulder
    dx = mid_sh[0] - mid_hp[0]
    dy = mid_sh[1] - mid_hp[1]
    
    # Negative dx means leaning back (arch)
    # Angle from vertical (0° = perfectly upright, positive = arch back)
    if dx < 0:  # Leaning back
        angle = abs(math.degrees(math.atan2(abs(dx), abs(dy))))
        return angle
    return 0.0

def _evaluate_cycle_form(lms, top_phase_max_elbow, cycle_max_arch,
                        cycle_max_path_dev, cycle_min_hip_y, cycle_start_hip_y,
                        lockout_fail_count, arch_fail_count, path_fail_count, legs_fail_count,
                        lockout_already_reported, arch_already_reported,
                        path_already_reported, legs_already_reported,
                        session_feedback, local_vars):
    """הערכת פורם בסוף מחזור"""
    
    # Lockout check
    lockout_ok = (top_phase_max_elbow is not None) and (top_phase_max_elbow >= LOCKOUT_MIN_ANGLE)
    lockout_near = (top_phase_max_elbow is not None) and (top_phase_max_elbow >= (LOCKOUT_MIN_ANGLE - LOCKOUT_NEAR_DEG))
    
    if not lockout_ok and not lockout_near:
        local_vars['cycle_tip_lockout'] = True
        local_vars['lockout_fail_count'] += 1
        if local_vars['lockout_fail_count'] >= LOCKOUT_FAIL_MIN_REPS and not lockout_already_reported:
            session_feedback.add(FB_CUE_LOCKOUT)
            local_vars['lockout_already_reported'] = True

    # Back arch check
    arch_ok = (cycle_max_arch is not None) and (cycle_max_arch <= TORSO_MAX_ARCH)
    arch_near = (cycle_max_arch is not None) and (cycle_max_arch <= (TORSO_MAX_ARCH + ARCH_NEAR_DEG))
    
    if not arch_ok and not arch_near:
        local_vars['cycle_tip_arch'] = True
        local_vars['arch_fail_count'] += 1
        if local_vars['arch_fail_count'] >= ARCH_FAIL_MIN_REPS and not arch_already_reported:
            session_feedback.add(FB_CUE_ARCH)
            local_vars['arch_already_reported'] = True

    # Bar path check
    path_ok = (cycle_max_path_dev is not None) and (cycle_max_path_dev <= PATH_MAX_DEVIATION)
    path_near = (cycle_max_path_dev is not None) and (cycle_max_path_dev <= (PATH_MAX_DEVIATION + PATH_NEAR_RELAX))
    
    if not path_ok and not path_near:
        local_vars['cycle_tip_path'] = True
        local_vars['path_fail_count'] += 1
        if local_vars['path_fail_count'] >= PATH_FAIL_MIN_REPS and not path_already_reported:
            session_feedback.add(FB_CUE_PATH)
            local_vars['path_already_reported'] = True

    # Leg drive check
    if cycle_min_hip_y is not None and cycle_start_hip_y is not None:
        hip_drop = cycle_start_hip_y - cycle_min_hip_y  # Positive = dropped
        legs_ok = hip_drop <= HIP_DROP_MAX
        legs_near = hip_drop <= (HIP_DROP_MAX + HIP_DROP_NEAR)
        
        if not legs_ok and not legs_near:
            local_vars['cycle_tip_legs'] = True
            local_vars['legs_fail_count'] += 1
            if local_vars['legs_fail_count'] >= LEGS_FAIL_MIN_REPS and not legs_already_reported:
                session_feedback.add(FB_CUE_LEGS)
                local_vars['legs_already_reported'] = True

def _count_rep(rep_reports, rep_count, top_elbow, ascent_from, peak_wrist_y, all_scores, rep_has_tip):
    rep_score = 10.0 if not rep_has_tip else 9.5
    all_scores.append(rep_score)
    rep_reports.append({
        "rep_index": int(rep_count+1),
        "score": float(rep_score),
        "good": bool(rep_score >= 10.0 - 1e-6),
        "top_elbow": float(top_elbow),
        "ascent_from": float(ascent_from),
        "peak_wrist_y": float(peak_wrist_y)
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
    return run_overhead_press_analysis(*args, **kwargs)
