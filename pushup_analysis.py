# -*- coding: utf-8 -*-
# pushup_analysis.py — IMPROVED: Better fast detection + FIXED separation

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

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    h,w,_=frame.shape
    ref_h=max(int(h*0.06),int(REPS_FONT_SIZE*1.6))
    r=int(ref_h*DONUT_RADIUS_SCALE); th=max(3,int(r*DONUT_THICKNESS_FRAC))
    m=12; cx=w-m-r; cy=max(ref_h+r//8, r+th//2+2)
    pct=float(np.clip(depth_pct,0,1))
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
    label="DEPTH"; pct_txt=f"{int(pct*100)}%"
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

# ============ IMPROVED Motion Detection - better for fast speeds ============
BASE_FRAME_SKIP = 1
ACTIVE_FRAME_SKIP = 1
MOTION_DETECTION_WINDOW = 8
MOTION_VEL_THRESHOLD = 0.0010       # יותר רגיש לתנועה מהירה
MOTION_ACCEL_THRESHOLD = 0.0006     # יותר רגיש לשינויי קצב
ELBOW_CHANGE_THRESHOLD = 5.0        # יותר רגיש לשינוי בזווית מרפק
COOLDOWN_FRAMES = 26                # שומר חלון פעילות מעט ארוך יותר
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
        
        # בדיקה 1: מהירות כתף
        if len(self.shoulder_history) >= 2:
            vel = abs(self.shoulder_history[-1] - self.shoulder_history[-2])
            self.velocity_history.append(vel)
            
            if len(self.velocity_history) >= 3:
                max_vel = max(self.velocity_history)
                recent_avg = sum(list(self.velocity_history)[-3:]) / 3
                accel = abs(self.velocity_history[-1] - self.velocity_history[-2])
                
                if max_vel > MOTION_VEL_THRESHOLD:
                    motion_detected = True
                    reason = f"high_vel({max_vel:.4f})"
                elif accel > MOTION_ACCEL_THRESHOLD:
                    motion_detected = True
                    reason = f"accel({accel:.4f})"
                elif recent_avg > MOTION_VEL_THRESHOLD * 0.65:
                    motion_detected = True
                    reason = f"sustained({recent_avg:.4f})"
        
        # בדיקה 2: שינוי במרפק EMA
        if len(self.elbow_history) >= 3:
            elbow_change = abs(self.elbow_history[-1] - self.elbow_history[-3])
            elbow_vel = abs(self.elbow_history[-1] - self.elbow_history[-2])
            if elbow_change > ELBOW_CHANGE_THRESHOLD:
                motion_detected = True
                reason = f"elbow_change({elbow_change:.1f}°)"
            elif elbow_vel > ELBOW_CHANGE_THRESHOLD * 0.55:
                motion_detected = True
                reason = f"elbow_vel({elbow_vel:.1f}°)"
        
        # בדיקה 3: שינוי גולמי במרפק (חשוב למהירות!)
        if len(self.raw_elbow_history) >= 3:
            raw_change = abs(self.raw_elbow_history[-1] - self.raw_elbow_history[-3])
            raw_vel = abs(self.raw_elbow_history[-1] - self.raw_elbow_history[-2])
            if raw_change > 11.0:  # מעט יותר רגיש
                motion_detected = True
                reason = f"raw_spike({raw_change:.1f}°)"
            elif raw_vel > 7.0:  # מעט יותר רגיש
                motion_detected = True
                reason = f"raw_vel({raw_vel:.1f}°)"
        
        # בדיקה 4: תבנית V
        if len(self.raw_elbow_history) >= 5:
            elbows = list(self.raw_elbow_history)
            went_down = elbows[-5] - elbows[-3] > 13  # יותר רגיש
            went_up = elbows[-1] - elbows[-3] > 13
            if went_down and went_up:
                motion_detected = True
                reason = f"V_pattern"
        
        # בדיקה 5: שינוי כיוון
        if len(self.shoulder_history) >= 5:
            diffs = [self.shoulder_history[i+1] - self.shoulder_history[i] 
                    for i in range(len(self.shoulder_history)-1)]
            if len(diffs) >= 4:
                sign_changes = sum(1 for i in range(len(diffs)-1) 
                                 if diffs[i] * diffs[i+1] < 0)
                max_diff = max(abs(d) for d in diffs)
                
                if sign_changes >= 1 and max_diff > MIN_VEL_FOR_MOTION:
                    motion_detected = True
                    reason = f"direction_change"
        
        if motion_detected:
            self.activate(reason)
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0:
                self.is_active = False
    
    def activate(self, reason=""):
        if not self.is_active:
            self.is_active = True
            self.activation_count += 1
            self.last_activation_reason = reason
        self.cooldown_counter = COOLDOWN_FRAMES
    
    def should_process(self, frame_idx):
        if self.is_active or self.cooldown_counter > 0:
            skip = ACTIVE_FRAME_SKIP
        else:
            skip = BASE_FRAME_SKIP
        
        should = (frame_idx - self.last_process_frame) >= skip
        if should:
            self.last_process_frame = frame_idx
        return should
    
    def get_stats(self):
        return {
            "is_active": self.is_active,
            "cooldown": self.cooldown_counter,
            "activations": self.activation_count,
            "last_reason": self.last_activation_reason
        }

# ============ Slightly More Forgiving Parameters - BUT STRICTER DEPTH ============
ELBOW_BENT_ANGLE = 102.0            # מעט סלחני יותר כדי לא לפספס חזרות מהירות
SHOULDER_MIN_DESCENT = 0.036        # סף ירידה מעט נמוך יותר לחזרות מהירות
RESET_ASCENT = 0.024                # קצת מהיר יותר
RESET_ELBOW = 153.0                 # קצת מהיר יותר
REFRACTORY_FRAMES = 1               # מינימום

ELBOW_EMA_ALPHA = 0.72              # קצת יותר מהיר
SHOULDER_EMA_ALPHA = 0.67           # קצת יותר מהיר

VIS_THR_STRICT = 0.29
PLANK_BODY_ANGLE_MAX = 26.0
HANDS_BELOW_SHOULDERS = 0.035
ONPUSHUP_MIN_FRAMES = 1
OFFPUSHUP_MIN_FRAMES = 5
AUTO_STOP_AFTER_EXIT_SEC = 1.5
TAIL_NOPOSE_STOP_SEC = 1.0

# ============ FIXED Feedback System ============
# Form Errors - מורידים נקודות (מופיעים ב-"feedback" עם ⚠️)
FB_ERROR_DEPTH = "Go deeper (chest to floor)"        # ✅ עודכן
FB_ERROR_HIPS = "Keep hips level (don't sag or pike)"
FB_ERROR_LOCKOUT = "Fully lockout arms at top"       # ✅ הדגשה על lockout
FB_ERROR_ELBOWS = "Keep elbows at 45° (not flared)"

# Performance Tips - לא מורידים נקודות (מופיעים ב-"form_tip" עם 💡)
PERF_TIP_SLOW_DOWN = "Lower slowly for better control"
PERF_TIP_TEMPO = "Try 2-1-2 tempo (down-pause-up)"
PERF_TIP_BREATHING = "Breathe: inhale down, exhale up"
PERF_TIP_CORE = "Engage core throughout movement"

FB_W_DEPTH = 1.2
FB_W_HIPS = 1.0
FB_W_LOCKOUT = 0.9
FB_W_ELBOWS = 0.7

FB_WEIGHTS = {
    FB_ERROR_DEPTH: FB_W_DEPTH,
    FB_ERROR_HIPS: FB_W_HIPS,
    FB_ERROR_LOCKOUT: FB_W_LOCKOUT,
    FB_ERROR_ELBOWS: FB_W_ELBOWS,
}
FB_DEFAULT_WEIGHT = 0.5
PENALTY_MIN_IF_ANY = 0.5

FORM_ERROR_PRIORITY = [FB_ERROR_DEPTH, FB_ERROR_LOCKOUT, FB_ERROR_HIPS, FB_ERROR_ELBOWS]
PERF_TIP_PRIORITY = [PERF_TIP_SLOW_DOWN, PERF_TIP_TEMPO, PERF_TIP_BREATHING, PERF_TIP_CORE]

DEPTH_EXCELLENT_ANGLE = 85.0    # ✅ מעולה: עמוק מאוד (היה 95)
DEPTH_GOOD_ANGLE = 95.0          # ✅ טוב: 90-95° (היה 105)
DEPTH_FAIR_ANGLE = 105.0         # בינוני: 100-105°
DEPTH_POOR_ANGLE = 115.0         # גרוע: מעל 105°

HIP_EXCELLENT = 8.0
HIP_GOOD = 15.0
HIP_FAIR = 22.0
HIP_POOR = 30.0

LOCKOUT_EXCELLENT = 175.0        # ✅ מעולה: כמעט ישור מלא (היה 170)
LOCKOUT_GOOD = 170.0             # ✅ טוב: ישור סביר (היה 165)
LOCKOUT_FAIR = 165.0             # ✅ בינוני: חסר קצת (היה 157)
LOCKOUT_POOR = 160.0             # ✅ גרוע: לא מיישר (היה 150)

FLARE_EXCELLENT = 45.0
FLARE_GOOD = 55.0
FLARE_FAIR = 65.0
FLARE_POOR = 75.0

DESCENT_SPEED_IDEAL = 0.0010
DESCENT_SPEED_FAST = 0.0012          # ✅ אולטרה רגיש (היה 0.0015)

DEPTH_FAIL_MIN_REPS = 2              # דורש עקביות לפני הערה
HIPS_FAIL_MIN_REPS = 2               # ירכיים - אפשר 2
LOCKOUT_FAIL_MIN_REPS = 2            # דורש עקביות לפני הערה
FLARE_FAIL_MIN_REPS = 2              # מרפקים - אפשר 2
TEMPO_CHECK_MIN_REPS = 1             # ✅ מיידי (היה 2)

# Error-report thresholds are intentionally more forgiving than scoring buckets
# to reduce false positives from camera angle / landmark noise.
DEPTH_ERROR_ANGLE = 110.0
LOCKOUT_ERROR_ANGLE = 165.0

BURST_FRAMES = 8                    # קצת יותר אגרסיבי
INFLECT_VEL_THR = 0.0027

DEBUG_ONPUSHUP = bool(int(os.getenv("DEBUG_ONPUSHUP", "0")))
DEBUG_MOTION = bool(int(os.getenv("DEBUG_MOTION", "0")))
DEBUG_GRADING = bool(int(os.getenv("DEBUG_GRADING", "1")))

MIN_CYCLE_ELBOW_SAMPLES = 4
ROBUST_BOTTOM_PERCENTILE = 15
ROBUST_TOP_PERCENTILE = 85
BOTTOM_SAMPLE_DESCENT_RATIO = 0.55
TOP_SAMPLE_ASCENT_RATIO = 0.40


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

    fps_in=cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps=max(1.0, fps_in/max(1, BASE_FRAME_SKIP))
    sec_to_frames=lambda s: max(1,int(s*effective_fps))

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

    # ✅ הפרדה נכונה
    session_form_errors=set()       # שגיאות פורם (⚠️)
    session_perf_tips=set()         # טיפים לשיפור (💡)
    rt_fb_msg=None; rt_fb_hold=0

    cycle_tip_deeper=False; cycle_tip_hips=False; cycle_tip_lockout=False; cycle_tip_elbows=False
    cycle_bottom_samples=[]; cycle_top_samples=[]
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
                process_now = True
                burst_cntr -= 1
            elif motion_detector.should_process(frame_idx):
                process_now = True
            
            if not process_now:
                frames_skipped += 1
                continue
            
            frames_processed += 1

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
            
            if DEBUG_MOTION and frame_idx % 30 == 0:
                stats = motion_detector.get_stats()
                print(f"[MOTION] f={frame_idx} active={stats['is_active']} "
                      f"cooldown={stats['cooldown']} acts={stats['activations']} "
                      f"reason={stats['last_reason']}")
            
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
                cycle_bottom_samples=[]; cycle_top_samples=[]
                bottom_phase_min_elbow=None; top_phase_max_elbow=None
                cycle_max_hip_misalign=None; cycle_max_flare=None
                cycle_max_descent_vel=0.0
                in_descent_phase=False
                motion_detector.activate("enter_plank")

            if onpushup and offpushup_streak>=OFFPUSHUP_MIN_FRAMES:
                robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow)
                cycle_has_issues = _evaluate_cycle_form(lms, robust_bottom_elbow, robust_top_elbow,
                                   cycle_max_hip_misalign, cycle_max_flare,
                                   cycle_max_descent_vel,
                                   depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                                   fast_descent_count,
                                   depth_already_reported, hips_already_reported,
                                   lockout_already_reported, flare_already_reported, tempo_already_reported,
                                   session_form_errors, session_perf_tips, rep_count, locals())

                if (not counted_this_cycle) and (cycle_max_descent>=SHOULDER_MIN_DESCENT) and (cycle_min_elbow<=ELBOW_BENT_ANGLE):
                    _count_rep(rep_reports,rep_count,cycle_min_elbow,
                               desc_base_shoulder if desc_base_shoulder is not None else shoulder_y,
                               baseline_shoulder_y+cycle_max_descent if baseline_shoulder_y is not None else shoulder_y,
                               all_scores, cycle_has_issues,  # ✅ Use return value
                               robust_bottom_elbow, robust_top_elbow, 
                               cycle_max_hip_misalign, cycle_max_flare)
                    rep_count+=1
                    if cycle_has_issues: bad_reps+=1  # ✅ Use return value
                    else: good_reps+=1

                onpushup=False; offpushup_frames_since_any_rep=0
                desc_base_shoulder=None; cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_tip_deeper=False; cycle_tip_hips=False; cycle_tip_lockout=False; cycle_tip_elbows=False
                cycle_bottom_samples=[]; cycle_top_samples=[]
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
                        cycle_bottom_samples=[]; cycle_top_samples=[]
                        bottom_phase_min_elbow=None; top_phase_max_elbow=None
                        cycle_max_hip_misalign=None; cycle_max_flare=None
                        cycle_max_descent_vel=0.0
                        in_descent_phase=True
                        motion_detector.activate("start_descent")
                else:
                    cycle_max_descent=max(cycle_max_descent,(shoulder_y-desc_base_shoulder))
                    cycle_min_elbow=min(cycle_min_elbow,elbow_angle)
                    
                    vel_abs = abs(shoulder_vel)
                    # ✅ Track velocity during descent (both EMA and raw)
                    if shoulder_vel > 0 and in_descent_phase:
                        cycle_max_descent_vel = max(cycle_max_descent_vel, vel_abs)
                    # ✅ Also track ANY significant movement
                    if vel_abs > 0.0005:  # Any movement
                        cycle_max_descent_vel = max(cycle_max_descent_vel, vel_abs)

                    min_elb_now = min(raw_elbow_L, raw_elbow_R)
                    cycle_bottom_samples.append(min_elb_now)
                    if len(cycle_bottom_samples) > 40:
                        cycle_bottom_samples.pop(0)
                    if bottom_phase_min_elbow is None: bottom_phase_min_elbow = min_elb_now
                    else: bottom_phase_min_elbow = min(bottom_phase_min_elbow, min_elb_now)

                    max_elb_now = max(raw_elbow_L, raw_elbow_R)
                    cycle_top_samples.append(max_elb_now)
                    if len(cycle_top_samples) > 40:
                        cycle_top_samples.pop(0)
                    if top_phase_max_elbow is None: top_phase_max_elbow = max_elb_now
                    else: top_phase_max_elbow = max(top_phase_max_elbow, max_elb_now)

                    # Collect depth samples only near the lower portion of the rep
                    descent_ratio = 0.0 if SHOULDER_MIN_DESCENT <= 1e-9 else (cycle_max_descent / SHOULDER_MIN_DESCENT)
                    if in_descent_phase and (descent_ratio >= BOTTOM_SAMPLE_DESCENT_RATIO or min_elb_now <= (ELBOW_BENT_ANGLE + 18.0)):
                        cycle_bottom_samples.append(min_elb_now)
                        if len(cycle_bottom_samples) > 30:
                            cycle_bottom_samples.pop(0)

                    # Collect lockout samples near the top (during ascent / reset)
                    ascent_progress = max(0.0, (desc_base_shoulder + cycle_max_descent) - shoulder_y)
                    ascent_ratio = 0.0 if SHOULDER_MIN_DESCENT <= 1e-9 else (ascent_progress / SHOULDER_MIN_DESCENT)
                    if (not in_descent_phase) and (ascent_ratio >= TOP_SAMPLE_ASCENT_RATIO or max_elb_now >= (LOCKOUT_POOR - 3.0)):
                        cycle_top_samples.append(max_elb_now)
                        if len(cycle_top_samples) > 30:
                            cycle_top_samples.pop(0)

                    hip_misalign = _calculate_hip_misalignment(lms, LSH, RSH, LH, RH, LA, RA)
                    if cycle_max_hip_misalign is None: cycle_max_hip_misalign = hip_misalign
                    else: cycle_max_hip_misalign = max(cycle_max_hip_misalign, hip_misalign)

                    elbow_flare = _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW)
                    if cycle_max_flare is None: cycle_max_flare = elbow_flare
                    else: cycle_max_flare = max(cycle_max_flare, elbow_flare)

                    reset_by_asc=(desc_base_shoulder is not None) and ((desc_base_shoulder-shoulder_y)>=RESET_ASCENT)
                    reset_by_elb =(elbow_angle>=RESET_ELBOW)
                    
                    if shoulder_vel < 0 and in_descent_phase:
                        in_descent_phase = False
                    
                    if reset_by_asc or reset_by_elb:
                        robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(
                            cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow
                        )
                        cycle_has_issues = _evaluate_cycle_form(lms, robust_bottom_elbow, robust_top_elbow,
                                           cycle_max_hip_misalign, cycle_max_flare,
                                           cycle_max_descent_vel,
                                           depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                                           fast_descent_count,
                                           depth_already_reported, hips_already_reported,
                                           lockout_already_reported, flare_already_reported, tempo_already_reported,
                                           session_form_errors, session_perf_tips, rep_count, locals())

                        if (not counted_this_cycle) and (cycle_max_descent>=SHOULDER_MIN_DESCENT) and (cycle_min_elbow<=ELBOW_BENT_ANGLE):
                            _count_rep(rep_reports,rep_count,cycle_min_elbow,
                                       desc_base_shoulder,
                                       baseline_shoulder_y+cycle_max_descent if baseline_shoulder_y is not None else shoulder_y,
                                       all_scores, cycle_has_issues,
                                       robust_bottom_elbow, robust_top_elbow,
                                       cycle_max_hip_misalign, cycle_max_flare)
                            rep_count+=1
                            if cycle_has_issues: bad_reps+=1
                            else: good_reps+=1

                        desc_base_shoulder=shoulder_y
                        cycle_max_descent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                        allow_new_bottom=True
                        cycle_bottom_samples=[]; cycle_top_samples=[]
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

                if at_bottom and allow_new_bottom and can_cnt and (not counted_this_cycle):
                    # ✅ Calculate if has issues based on current measurements
                    rep_has_issues = False
                    robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow)
                    if robust_bottom_elbow and robust_bottom_elbow > DEPTH_ERROR_ANGLE:
                        rep_has_issues = True
                    if robust_top_elbow and robust_top_elbow < LOCKOUT_ERROR_ANGLE:
                        rep_has_issues = True
                    if cycle_max_hip_misalign and cycle_max_hip_misalign > HIP_FAIR:
                        rep_has_issues = True
                    if cycle_max_flare and cycle_max_flare > FLARE_FAIR:
                        rep_has_issues = True
                    
                    _count_rep(rep_reports,rep_count,elbow_angle,
                               desc_base_shoulder if desc_base_shoulder is not None else shoulder_y, shoulder_y,
                               all_scores, rep_has_issues,  # ✅
                               robust_bottom_elbow if robust_bottom_elbow else raw_elbow_min,
                               robust_top_elbow if robust_top_elbow else max(raw_elbow_L, raw_elbow_R),
                               cycle_max_hip_misalign if cycle_max_hip_misalign else 0.0,
                               cycle_max_flare if cycle_max_flare else 0.0)
                    rep_count+=1
                    if rep_has_issues: bad_reps+=1  # ✅
                    else: good_reps+=1
                    last_bottom_frame=frame_idx; allow_new_bottom=False; counted_this_cycle=True
                    in_descent_phase=False
                    motion_detector.activate("count_rep")

                if (allow_new_bottom is False) and (last_bottom_frame>0):
                    if shoulder_prev is not None and (shoulder_prev - shoulder_y) > 0 and (desc_base_shoulder is not None):
                        if ((desc_base_shoulder + cycle_max_descent) - shoulder_y) >= REARM_ASCENT_EFF:
                            allow_new_bottom=True
                    max_elb_now = max(raw_elbow_L, raw_elbow_R)
                    if max_elb_now >= (LOCKOUT_POOR - 3.0):
                        cycle_top_samples.append(max_elb_now)
                        if len(cycle_top_samples) > 30:
                            cycle_top_samples.pop(0)

                if at_bottom and not cycle_tip_deeper:
                    robust_bottom_elbow, _ = _robust_cycle_elbows(cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow)
                    if robust_bottom_elbow and robust_bottom_elbow > DEPTH_ERROR_ANGLE:
                        cycle_tip_deeper = True
                        depth_fail_count += 1
                        if depth_fail_count >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                            session_form_errors.add(FB_ERROR_DEPTH)  # ⚠️
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
                frame=draw_body_only(frame,lms)
                frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),depth_pct=depth_live)
                out.write(frame)

            if shoulder_y is not None: shoulder_prev=shoulder_y

    if onpushup and (not counted_this_cycle) and (cycle_max_descent>=SHOULDER_MIN_DESCENT) and (cycle_min_elbow<=ELBOW_BENT_ANGLE):
        robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(cycle_bottom_samples, cycle_top_samples, bottom_phase_min_elbow, top_phase_max_elbow)
        cycle_has_issues = _evaluate_cycle_form(lms, robust_bottom_elbow, robust_top_elbow,
                           cycle_max_hip_misalign, cycle_max_flare,
                           cycle_max_descent_vel,
                           depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                           fast_descent_count,
                           depth_already_reported, hips_already_reported,
                           lockout_already_reported, flare_already_reported, tempo_already_reported,
                           session_form_errors, session_perf_tips, rep_count, locals())

        _count_rep(rep_reports,rep_count,cycle_min_elbow,
                   desc_base_shoulder if desc_base_shoulder is not None else (baseline_shoulder_y or 0.0),
                   (baseline_shoulder_y + cycle_max_descent) if baseline_shoulder_y is not None else (baseline_shoulder_y or 0.0),
                   all_scores, cycle_has_issues,  # ✅
                   robust_bottom_elbow, robust_top_elbow,
                   cycle_max_hip_misalign, cycle_max_flare)
        rep_count+=1
        if cycle_has_issues: bad_reps+=1  # ✅
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

    form_errors_list = [err for err in FORM_ERROR_PRIORITY if err in session_form_errors]
    perf_tips_list = [tip for tip in PERF_TIP_PRIORITY if tip in session_perf_tips]

    primary_form_error = form_errors_list[0] if form_errors_list else None
    primary_perf_tip = perf_tips_list[0] if perf_tips_list else None

    total_frames = frames_processed + frames_skipped
    efficiency = (frames_skipped / total_frames * 100) if total_frames > 0 else 0
    print(f"\n[EFFICIENCY] Processed: {frames_processed}, Skipped: {frames_skipped}, "
          f"Total: {total_frames}, Saved: {efficiency:.1f}%")
    print(f"[EFFICIENCY] Motion activations: {motion_detector.activation_count}")
    
    if DEBUG_GRADING:
        print(f"\n[GRADING] Total reps: {rep_count}, Good: {good_reps}, Bad: {bad_reps}")
        if all_scores:
            print(f"[GRADING] Average score: {sum(all_scores)/len(all_scores):.1f}")
        print(f"[GRADING] Technique score: {technique_score:.1f}")
        print(f"[GRADING] Form errors: {form_errors_list}")
        print(f"[GRADING] Performance tips: {perf_tips_list}")

    try:
        with open(feedback_path,"w",encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Good Reps: {int(good_reps)} | Bad Reps: {int(bad_reps)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            f.write(f"\nProcessing Efficiency: {efficiency:.1f}% frames skipped\n")
            f.write(f"Motion Detection Activations: {motion_detector.activation_count}\n")
            if form_errors_list:
                f.write("\n⚠️ Form Corrections (affecting score):\n")
                for ln in form_errors_list: f.write(f"- {ln}\n")
            if perf_tips_list:
                f.write("\n💡 Performance Tips (not affecting score):\n")
                for ln in perf_tips_list: f.write(f"• {ln}\n")
    except Exception:
        pass

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
        "feedback": form_errors_list,      # ⚠️ Form Errors בלבד
        "tips": perf_tips_list,            # רשימה מלאה של tips
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
    
    # ✅ Form Tip = רק Performance Tips (לא Form Errors!)
    if primary_perf_tip:
        result["form_tip"] = primary_perf_tip    # 💡 רק טיפים לביצוע!
    
    # שמירת ההפרדה המלאה ב-JSON
    if primary_form_error:
        result["primary_form_error"] = primary_form_error  # ⚠️ בנפרד
    if primary_perf_tip:
        result["primary_perf_tip"] = primary_perf_tip      # 💡 בנפרד

    return result

# ============ Helper Functions ============

def _robust_cycle_elbows(bottom_samples, top_samples, fallback_bottom=None, fallback_top=None):
    robust_bottom = fallback_bottom
    robust_top = fallback_top

    if bottom_samples:
        arr = np.array(bottom_samples, dtype=np.float32)
        robust_bottom = float(np.percentile(arr, ROBUST_BOTTOM_PERCENTILE))
    if top_samples:
        arr = np.array(top_samples, dtype=np.float32)
        robust_top = float(np.percentile(arr, ROBUST_TOP_PERCENTILE))

    # Guardrail: avoid impossible overlap from noisy short windows
    if robust_bottom is not None and robust_top is not None and robust_top < robust_bottom:
        robust_bottom = min(float(robust_bottom), float(fallback_bottom) if fallback_bottom is not None else float(robust_bottom))
        robust_top = max(float(robust_top), float(fallback_top) if fallback_top is not None else float(robust_top))

    return robust_bottom, robust_top

def _calculate_body_angle(lms, LSH, RSH, LH, RH, LA, RA):
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    mid_ank = ((lms[LA].x + lms[RA].x)/2.0, (lms[LA].y + lms[RA].y)/2.0)
    dx = mid_sh[0] - mid_ank[0]
    dy = mid_sh[1] - mid_ank[1]
    angle = abs(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9)))
    return angle

def _calculate_hip_misalignment(lms, LSH, RSH, LH, RH, LA, RA):
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    mid_hp = ((lms[LH].x + lms[RH].x)/2.0, (lms[LH].y + lms[RH].y)/2.0)
    mid_ank = ((lms[LA].x + lms[RA].x)/2.0, (lms[LA].y + lms[RA].y)/2.0)
    angle = _ang(mid_sh, mid_hp, mid_ank)
    deviation = abs(180.0 - angle)
    return deviation

def _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW):
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    
    left_vec_sh = (mid_sh[0] - lms[LSH].x, mid_sh[1] - lms[LSH].y)
    left_vec_elb = (lms[LE].x - lms[LSH].x, lms[LE].y - lms[LSH].y)
    left_angle = abs(math.degrees(math.atan2(
        left_vec_sh[0]*left_vec_elb[1] - left_vec_sh[1]*left_vec_elb[0],
        left_vec_sh[0]*left_vec_elb[0] + left_vec_sh[1]*left_vec_elb[1]
    )))
    
    right_vec_sh = (mid_sh[0] - lms[RSH].x, mid_sh[1] - lms[RSH].y)
    right_vec_elb = (lms[RE].x - lms[RSH].x, lms[RE].y - lms[RSH].y)
    right_angle = abs(math.degrees(math.atan2(
        right_vec_sh[0]*right_vec_elb[1] - right_vec_sh[1]*right_vec_elb[0],
        right_vec_sh[0]*right_vec_elb[0] + right_vec_sh[1]*right_vec_elb[1]
    )))
    
    return max(left_angle, right_angle)

def _evaluate_cycle_form(lms, bottom_phase_min_elbow, top_phase_max_elbow,
                        cycle_max_hip_misalign, cycle_max_flare,
                        cycle_max_descent_vel,
                        depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                        fast_descent_count,
                        depth_already_reported, hips_already_reported,
                        lockout_already_reported, flare_already_reported, tempo_already_reported,
                        session_form_errors, session_perf_tips, rep_count, local_vars):
    
    # Track tips for this cycle
    has_depth_issue = False
    has_lockout_issue = False
    has_hips_issue = False
    has_flare_issue = False
    
    # Form errors - מורידים נקודות
    if bottom_phase_min_elbow is not None and local_vars.get("cycle_bottom_samples") and len(local_vars["cycle_bottom_samples"]) >= MIN_CYCLE_ELBOW_SAMPLES:
        if bottom_phase_min_elbow > DEPTH_ERROR_ANGLE:
            has_depth_issue = True  # ✅
            local_vars['cycle_tip_deeper'] = True
            local_vars['depth_fail_count'] += 1
            if local_vars['depth_fail_count'] >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                session_form_errors.add(FB_ERROR_DEPTH)
                local_vars['depth_already_reported'] = True

    if top_phase_max_elbow is not None and local_vars.get("cycle_top_samples") and len(local_vars["cycle_top_samples"]) >= MIN_CYCLE_ELBOW_SAMPLES:
        if top_phase_max_elbow < LOCKOUT_ERROR_ANGLE:
            has_lockout_issue = True  # ✅
            local_vars['cycle_tip_lockout'] = True
            local_vars['lockout_fail_count'] += 1
            if local_vars['lockout_fail_count'] >= LOCKOUT_FAIL_MIN_REPS and not lockout_already_reported:
                session_form_errors.add(FB_ERROR_LOCKOUT)
                local_vars['lockout_already_reported'] = True

    if cycle_max_hip_misalign is not None:
        if cycle_max_hip_misalign > HIP_FAIR:
            has_hips_issue = True  # ✅
            local_vars['cycle_tip_hips'] = True
            local_vars['hips_fail_count'] += 1
            if local_vars['hips_fail_count'] >= HIPS_FAIL_MIN_REPS and not hips_already_reported:
                session_form_errors.add(FB_ERROR_HIPS)
                local_vars['hips_already_reported'] = True

    if cycle_max_flare is not None:
        if cycle_max_flare > FLARE_FAIR:
            has_flare_issue = True  # ✅
            local_vars['cycle_tip_elbows'] = True
            local_vars['flare_fail_count'] += 1
            if local_vars['flare_fail_count'] >= FLARE_FAIL_MIN_REPS and not flare_already_reported:
                session_form_errors.add(FB_ERROR_ELBOWS)
                local_vars['flare_already_reported'] = True
    
    # ✅ Performance tips - לא מורידים נקודות
    if rep_count >= TEMPO_CHECK_MIN_REPS and not tempo_already_reported:
        if cycle_max_descent_vel > DESCENT_SPEED_FAST:
            local_vars['fast_descent_count'] += 1
            if DEBUG_GRADING:
                print(f"[TEMPO] Rep {rep_count+1}: descent_vel={cycle_max_descent_vel:.5f}, "
                      f"count={local_vars['fast_descent_count']}")
            if local_vars['fast_descent_count'] >= 1:  # ✅ מיידי! (היה 2)
                session_perf_tips.add(PERF_TIP_SLOW_DOWN)
                session_perf_tips.add(PERF_TIP_TEMPO)
                local_vars['tempo_already_reported'] = True
                if DEBUG_GRADING:
                    print(f"[TEMPO] ✅ Tempo tip added!")
        elif DEBUG_GRADING and rep_count < 5:
            print(f"[TEMPO] Rep {rep_count+1}: descent_vel={cycle_max_descent_vel:.5f} "
                  f"< threshold {DESCENT_SPEED_FAST:.5f}")
    
    # ✅ Return whether this cycle had issues
    return has_depth_issue or has_lockout_issue or has_hips_issue or has_flare_issue

def _count_rep(rep_reports, rep_count, bottom_elbow, descent_from, bottom_shoulder_y, all_scores, rep_has_tip,
               bottom_phase_min_elbow, top_phase_max_elbow, cycle_max_hip_misalign, cycle_max_flare):
    
    depth_score = 10.0
    lockout_score = 10.0
    hips_score = 10.0
    flare_score = 10.0
    
    if bottom_phase_min_elbow:
        if bottom_phase_min_elbow <= DEPTH_EXCELLENT_ANGLE:
            depth_score = 10.0
        elif bottom_phase_min_elbow <= DEPTH_GOOD_ANGLE:
            depth_score = 9.0
        elif bottom_phase_min_elbow <= DEPTH_FAIR_ANGLE:
            depth_score = 7.5
        elif bottom_phase_min_elbow <= DEPTH_POOR_ANGLE:
            depth_score = 5.0
        else:
            depth_score = 3.0
    
    if top_phase_max_elbow:
        if top_phase_max_elbow >= LOCKOUT_EXCELLENT:
            lockout_score = 10.0
        elif top_phase_max_elbow >= LOCKOUT_GOOD:
            lockout_score = 9.0
        elif top_phase_max_elbow >= LOCKOUT_FAIR:
            lockout_score = 7.5
        elif top_phase_max_elbow >= LOCKOUT_POOR:
            lockout_score = 5.0
        else:
            lockout_score = 3.0
    
    if cycle_max_hip_misalign is not None:
        if cycle_max_hip_misalign <= HIP_EXCELLENT:
            hips_score = 10.0
        elif cycle_max_hip_misalign <= HIP_GOOD:
            hips_score = 9.0
        elif cycle_max_hip_misalign <= HIP_FAIR:
            hips_score = 7.5
        elif cycle_max_hip_misalign <= HIP_POOR:
            hips_score = 5.0
        else:
            hips_score = 3.0
    
    if cycle_max_flare is not None:
        if cycle_max_flare <= FLARE_EXCELLENT:
            flare_score = 10.0
        elif cycle_max_flare <= FLARE_GOOD:
            flare_score = 9.0
        elif cycle_max_flare <= FLARE_FAIR:
            flare_score = 7.5
        elif cycle_max_flare <= FLARE_POOR:
            flare_score = 5.0
        else:
            flare_score = 3.0
    
    rep_score = (depth_score * 0.35 + lockout_score * 0.25 + hips_score * 0.25 + flare_score * 0.15)
    rep_score = round(rep_score * 2) / 2
    
    all_scores.append(rep_score)
    
    if DEBUG_GRADING:
        print(f"[REP {rep_count+1}] Total: {rep_score:.1f} | "
              f"Depth: {depth_score:.1f} ({bottom_phase_min_elbow:.1f}°) | "
              f"Lockout: {lockout_score:.1f} ({top_phase_max_elbow:.1f}°) | "
              f"Hips: {hips_score:.1f} ({cycle_max_hip_misalign:.1f}°) | "
              f"Flare: {flare_score:.1f} ({cycle_max_flare:.1f}°)")
    
    rep_reports.append({
        "rep_index": int(rep_count+1),
        "score": float(rep_score),
        "good": bool(rep_score >= 9.0),
        "bottom_elbow": float(bottom_elbow),
        "descent_from": float(descent_from),
        "bottom_shoulder_y": float(bottom_shoulder_y),
        "detailed_scores": {
            "depth": float(depth_score),
            "lockout": float(lockout_score),
            "hips": float(hips_score),
            "flare": float(flare_score)
        },
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
