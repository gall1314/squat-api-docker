# -*- coding: utf-8 -*-
# overhead_press_analysis.py — military/overhead press rep counter with form feedback

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
# Rep counting
BOTTOM_ELBOW_MAX = 115.0
TOP_ELBOW_MIN = 165.0
WRIST_ABOVE_HEAD_MARGIN = 0.02
BOTTOM_WRIST_MARGIN = 0.02
REFRACTORY_FRAMES = 3

# EMA smoothing
ELBOW_EMA_ALPHA = 0.35
WRIST_EMA_ALPHA = 0.30

# Visibility thresholds
VIS_THR_STRICT = 0.30

# Form thresholds
DEPTH_MIN_ANGLE = 105.0
DEPTH_NEAR_DEG = 8.0
LOCKOUT_MIN_ANGLE = 170.0
LOCKOUT_NEAR_DEG = 6.0
TORSO_MAX_LEAN = 25.0
LEAN_NEAR_DEG = 5.0
WRIST_STACK_MAX = 0.06
STACK_NEAR_THR = 0.015

# Minimum reps before session feedback
DEPTH_FAIL_MIN_REPS = 2
LOCKOUT_FAIL_MIN_REPS = 2
LEAN_FAIL_MIN_REPS = 2
STACK_FAIL_MIN_REPS = 2

# ============ Feedback Cues & Weights ============
FB_CUE_LOCKOUT = "Press to full lockout overhead"
FB_CUE_DEPTH = "Lower to shoulder level each rep"
FB_CUE_LEAN = "Keep torso upright (avoid leaning back)"
FB_CUE_STACK = "Keep wrists stacked over shoulders"

FB_W_LOCKOUT = float(os.getenv("FB_W_LOCKOUT", "1.0"))
FB_W_DEPTH = float(os.getenv("FB_W_DEPTH", "0.9"))
FB_W_LEAN = float(os.getenv("FB_W_LEAN", "0.7"))
FB_W_STACK = float(os.getenv("FB_W_STACK", "0.6"))

FB_WEIGHTS = {
    FB_CUE_LOCKOUT: FB_W_LOCKOUT,
    FB_CUE_DEPTH: FB_W_DEPTH,
    FB_CUE_LEAN: FB_W_LEAN,
    FB_CUE_STACK: FB_W_STACK,
}
FB_DEFAULT_WEIGHT = 0.5
PENALTY_MIN_IF_ANY = 0.5
FORM_TIP_PRIORITY = [FB_CUE_LOCKOUT, FB_CUE_DEPTH, FB_CUE_STACK, FB_CUE_LEAN]

DEBUG_ONPRESS = bool(int(os.getenv("DEBUG_ONPRESS", "0")))

# ============ Helper Functions ============
def _calculate_torso_lean(lms, LSH, RSH, LH, RH):
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    mid_hp = ((lms[LH].x + lms[RH].x)/2.0, (lms[LH].y + lms[RH].y)/2.0)
    dx = mid_sh[0] - mid_hp[0]
    dy = mid_sh[1] - mid_hp[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9)))

# ============ Main Analyzer ============
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
    if not cap.isOpened():
        return _ret_err("Could not open video", feedback_path)

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
    NOSE=mp_pose.PoseLandmark.NOSE.value

    def _pick_side_dyn(lms):
        vL=lms[LSH].visibility+lms[LE].visibility+lms[LW].visibility
        vR=lms[RSH].visibility+lms[RE].visibility+lms[RW].visibility
        return ("LEFT",LSH,LE,LW) if vL>=vR else ("RIGHT",RSH,RE,RW)

    # State
    elbow_ema=None; wrist_ema=None
    bottom_ready=False; top_reached=False; last_rep_frame=-10**9

    # Per-cycle metrics
    min_elbow=None; max_elbow=None; max_lean=None; max_stack=None
    min_bottom_wrist=None

    # Feedback state
    session_feedback=set()
    depth_fail_count=0; lockout_fail_count=0; lean_fail_count=0; stack_fail_count=0
    depth_already_reported=False; lockout_already_reported=False
    lean_already_reported=False; stack_already_reported=False

    # RT feedback
    rt_fb_msg=None; rt_fb_hold=0

    with mp_pose.Pose(model_complexity=model_complexity, enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_skip>1 and (frame_idx % frame_skip)!=0: continue

            if scale and scale!=1.0:
                frame=cv2.resize(frame,None,fx=scale,fy=scale)

            h,w=frame.shape[:2]
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=pose.process(rgb)

            height_pct=0.0
            if not res.pose_landmarks:
                if return_video and out:
                    frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),height_pct=height_pct)
                    out.write(frame)
                continue

            lms=res.pose_landmarks.landmark

            if out is None and return_video:
                out=cv2.VideoWriter(output_path,fourcc,max(1.0, fps_in/max(1,frame_skip)),(w,h))

            side,SH,EL,WR = _pick_side_dyn(lms)
            vis_ok = (lms[SH].visibility>VIS_THR_STRICT and lms[EL].visibility>VIS_THR_STRICT and lms[WR].visibility>VIS_THR_STRICT)

            if not vis_ok:
                if return_video and out:
                    frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),height_pct=height_pct)
                    out.write(frame)
                continue

            sh=(lms[SH].x, lms[SH].y); el=(lms[EL].x, lms[EL].y); wr=(lms[WR].x, lms[WR].y)
            elbow_angle=_ang(sh, el, wr)
            elbow_ema=_ema(elbow_ema, elbow_angle, ELBOW_EMA_ALPHA)

            wrist_y=wr[1]
            wrist_ema=_ema(wrist_ema, wrist_y, WRIST_EMA_ALPHA)

            head_y=lms[NOSE].y if lms[NOSE].visibility>VIS_THR_STRICT else (sh[1] - 0.12)
            head_y=min(head_y, sh[1]-0.02)
            height_denom=max(0.05, (sh[1]-head_y))
            height_pct=float(np.clip((sh[1]-wrist_y)/height_denom, 0.0, 1.0))

            torso_lean=_calculate_torso_lean(lms, LSH, RSH, LH, RH)
            stack_offset=abs(wr[0]-sh[0])

            bottom_cond = (elbow_ema is not None and elbow_ema <= BOTTOM_ELBOW_MAX and wrist_ema >= (sh[1]-BOTTOM_WRIST_MARGIN))
            top_cond = (elbow_ema is not None and elbow_ema >= TOP_ELBOW_MIN and wrist_ema <= (head_y - WRIST_ABOVE_HEAD_MARGIN))

            if vis_ok:
                min_elbow = elbow_ema if min_elbow is None else min(min_elbow, elbow_ema)
                max_elbow = elbow_ema if max_elbow is None else max(max_elbow, elbow_ema)
                max_lean = torso_lean if max_lean is None else max(max_lean, torso_lean)
                max_stack = stack_offset if max_stack is None else max(max_stack, stack_offset)
                min_bottom_wrist = wrist_ema if min_bottom_wrist is None else max(min_bottom_wrist, wrist_ema)

            if not top_reached:
                if bottom_cond:
                    bottom_ready = True
                if bottom_ready and top_cond and (frame_idx - last_rep_frame) >= REFRACTORY_FRAMES:
                    cycle_tip_depth = False
                    cycle_tip_lockout = False
                    cycle_tip_lean = False
                    cycle_tip_stack = False

                    depth_ok = (min_elbow is not None) and (min_elbow <= DEPTH_MIN_ANGLE)
                    depth_near = (min_elbow is not None) and (min_elbow <= (DEPTH_MIN_ANGLE + DEPTH_NEAR_DEG))
                    if not depth_ok and not depth_near:
                        cycle_tip_depth = True
                        depth_fail_count += 1
                        if depth_fail_count >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                            session_feedback.add(FB_CUE_DEPTH)
                            depth_already_reported = True

                    lockout_ok = (max_elbow is not None) and (max_elbow >= LOCKOUT_MIN_ANGLE)
                    lockout_near = (max_elbow is not None) and (max_elbow >= (LOCKOUT_MIN_ANGLE - LOCKOUT_NEAR_DEG))
                    if not lockout_ok and not lockout_near:
                        cycle_tip_lockout = True
                        lockout_fail_count += 1
                        if lockout_fail_count >= LOCKOUT_FAIL_MIN_REPS and not lockout_already_reported:
                            session_feedback.add(FB_CUE_LOCKOUT)
                            lockout_already_reported = True

                    lean_ok = (max_lean is not None) and (max_lean <= TORSO_MAX_LEAN)
                    lean_near = (max_lean is not None) and (max_lean <= (TORSO_MAX_LEAN + LEAN_NEAR_DEG))
                    if not lean_ok and not lean_near:
                        cycle_tip_lean = True
                        lean_fail_count += 1
                        if lean_fail_count >= LEAN_FAIL_MIN_REPS and not lean_already_reported:
                            session_feedback.add(FB_CUE_LEAN)
                            lean_already_reported = True

                    stack_ok = (max_stack is not None) and (max_stack <= WRIST_STACK_MAX)
                    stack_near = (max_stack is not None) and (max_stack <= (WRIST_STACK_MAX + STACK_NEAR_THR))
                    if not stack_ok and not stack_near:
                        cycle_tip_stack = True
                        stack_fail_count += 1
                        if stack_fail_count >= STACK_FAIL_MIN_REPS and not stack_already_reported:
                            session_feedback.add(FB_CUE_STACK)
                            stack_already_reported = True

                    rep_has_tip = cycle_tip_depth or cycle_tip_lockout or cycle_tip_lean or cycle_tip_stack
                    rep_score = 10.0 if not rep_has_tip else 9.5
                    all_scores.append(rep_score)
                    rep_reports.append({
                        "rep_index": int(rep_count+1),
                        "score": float(rep_score),
                        "good": bool(rep_score >= 10.0 - 1e-6),
                        "bottom_elbow": float(min_elbow if min_elbow is not None else 0.0),
                        "top_elbow": float(max_elbow if max_elbow is not None else 0.0),
                        "max_lean": float(max_lean if max_lean is not None else 0.0),
                        "max_stack_offset": float(max_stack if max_stack is not None else 0.0),
                    })

                    rep_count += 1
                    if rep_has_tip: bad_reps += 1
                    else: good_reps += 1

                    last_rep_frame = frame_idx
                    top_reached = True
                    bottom_ready = False

                    if cycle_tip_lockout: rt_fb_msg = FB_CUE_LOCKOUT
                    elif cycle_tip_depth: rt_fb_msg = FB_CUE_DEPTH
                    elif cycle_tip_stack: rt_fb_msg = FB_CUE_STACK
                    elif cycle_tip_lean: rt_fb_msg = FB_CUE_LEAN
                    rt_fb_hold = sec_to_frames(1.0)

                    min_elbow=None; max_elbow=None; max_lean=None; max_stack=None; min_bottom_wrist=None
            else:
                if bottom_cond:
                    top_reached = False
                    bottom_ready = True
                    min_elbow=None; max_elbow=None; max_lean=None; max_stack=None; min_bottom_wrist=None

            if rt_fb_hold>0:
                rt_fb_hold -= 1

            if return_video and out:
                frame=draw_body_only(frame, lms)
                frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),height_pct=height_pct)
                out.write(frame)

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
