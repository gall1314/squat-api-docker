# -*- coding: utf-8 -*-
# pushup_analysis.py — simple push-up rep counter with basic form feedback

import os, cv2, math, numpy as np, subprocess
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

# ============ Push-up Parameters ============
BOTTOM_ELBOW_MAX = 95.0
TOP_ELBOW_MIN = 160.0
REFRACTORY_FRAMES = 3

ELBOW_EMA_ALPHA = 0.35
BODY_EMA_ALPHA = 0.30

VIS_THR_STRICT = 0.30

DEPTH_MIN_ANGLE = 105.0
DEPTH_NEAR_DEG = 8.0
LOCKOUT_MIN_ANGLE = 165.0
LOCKOUT_NEAR_DEG = 6.0
BODY_LINE_MIN = 160.0
BODY_LINE_NEAR_DEG = 6.0

DEPTH_FAIL_MIN_REPS = 2
LOCKOUT_FAIL_MIN_REPS = 2
BODY_LINE_FAIL_MIN_REPS = 2

# ============ Feedback Cues & Weights ============
FB_CUE_DEPTH = "Lower your chest closer to the floor"
FB_CUE_LOCKOUT = "Press to full elbow lockout"
FB_CUE_BODY = "Keep your body in a straight line"

FB_W_DEPTH = float(os.getenv("FB_W_DEPTH", "1.0"))
FB_W_LOCKOUT = float(os.getenv("FB_W_LOCKOUT", "0.8"))
FB_W_BODY = float(os.getenv("FB_W_BODY", "0.7"))

FB_WEIGHTS = {
    FB_CUE_DEPTH: FB_W_DEPTH,
    FB_CUE_LOCKOUT: FB_W_LOCKOUT,
    FB_CUE_BODY: FB_W_BODY,
}
FB_DEFAULT_WEIGHT = 0.5
PENALTY_MIN_IF_ANY = 0.5
FORM_TIP_PRIORITY = [FB_CUE_DEPTH, FB_CUE_LOCKOUT, FB_CUE_BODY]

def _pick_side_dyn(lms):
    LSH=mp_pose.PoseLandmark.LEFT_SHOULDER.value;  RSH=mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LE =mp_pose.PoseLandmark.LEFT_ELBOW.value;     RE =mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW =mp_pose.PoseLandmark.LEFT_WRIST.value;     RW =mp_pose.PoseLandmark.RIGHT_WRIST.value
    vL=lms[LSH].visibility+lms[LE].visibility+lms[LW].visibility
    vR=lms[RSH].visibility+lms[RE].visibility+lms[RW].visibility
    return ("LEFT",LSH,LE,LW) if vL>=vR else ("RIGHT",RSH,RE,RW)

# ============ Main Analyzer ============
def run_pushup_analysis(video_path,
                        frame_skip=3,
                        scale=0.4,
                        output_path="pushup_analyzed.mp4",
                        feedback_path="pushup_feedback.txt",
                        preserve_quality=False,
                        encode_crf=None,
                        return_video=True,
                        fast_mode=None):
    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    fast_path = (fast_mode is True)
    slow_path = not fast_path
    if fast_path:
        return_video = False

    effective_frame_skip = frame_skip
    effective_scale = scale
    model_complexity = 0 if fast_path else 1

    if preserve_quality:
        effective_scale=1.0; effective_frame_skip=1; encode_crf=18 if encode_crf is None else encode_crf
    else:
        encode_crf=23 if encode_crf is None else encode_crf

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): return _ret_err("Could not open video", feedback_path)

    fps_in=cap.get(cv2.CAP_PROP_FPS) or 25

    out=None
    if slow_path:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')

    frame_idx=0
    rep_count=0; good_reps=0; bad_reps=0; rep_reports=[]; all_scores=[]

    LSH=mp_pose.PoseLandmark.LEFT_SHOULDER.value;  RSH=mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LH =mp_pose.PoseLandmark.LEFT_HIP.value;       RH =mp_pose.PoseLandmark.RIGHT_HIP.value
    LK =mp_pose.PoseLandmark.LEFT_KNEE.value;      RK =mp_pose.PoseLandmark.RIGHT_KNEE.value

    elbow_ema=None; body_ema=None
    in_bottom=False; refractory=0
    min_elbow=None; max_elbow=None; min_body=None

    session_feedback=set()
    depth_fail_count=0; lockout_fail_count=0; body_fail_count=0
    depth_already_reported=False; lockout_already_reported=False; body_already_reported=False

    rt_fb_msg=None; rt_fb_hold=0

    with mp_pose.Pose(model_complexity=model_complexity, enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if effective_frame_skip>1 and (frame_idx % effective_frame_skip)!=0: continue

            if effective_scale and effective_scale!=1.0:
                frame=cv2.resize(frame,None,fx=effective_scale,fy=effective_scale)

            h,w=frame.shape[:2]
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=pose.process(rgb)

            depth_pct=0.0
            if not res.pose_landmarks:
                if return_video and out:
                    frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),depth_pct=depth_pct)
                    out.write(frame)
                continue

            lms=res.pose_landmarks.landmark

            if out is None and return_video:
                out=cv2.VideoWriter(output_path,fourcc,max(1.0, fps_in/max(1,effective_frame_skip)),(w,h))

            side,SH,EL,WR = _pick_side_dyn(lms)
            vis_ok = (lms[SH].visibility>VIS_THR_STRICT and lms[EL].visibility>VIS_THR_STRICT and lms[WR].visibility>VIS_THR_STRICT)

            if not vis_ok:
                if return_video and out:
                    frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),depth_pct=depth_pct)
                    out.write(frame)
                continue

            sh=(lms[SH].x, lms[SH].y); el=(lms[EL].x, lms[EL].y); wr=(lms[WR].x, lms[WR].y)
            elbow_angle=_ang(sh, el, wr)
            elbow_ema=_ema(elbow_ema, elbow_angle, ELBOW_EMA_ALPHA)

            hip_idx = LH if side == "LEFT" else RH
            knee_idx = LK if side == "LEFT" else RK
            vis_body = (lms[hip_idx].visibility>VIS_THR_STRICT and lms[knee_idx].visibility>VIS_THR_STRICT)
            body_angle=None
            if vis_body:
                hip=(lms[hip_idx].x, lms[hip_idx].y); knee=(lms[knee_idx].x, lms[knee_idx].y)
                body_angle=_ang(sh, hip, knee)
                body_ema=_ema(body_ema, body_angle, BODY_EMA_ALPHA)

            if elbow_ema is not None:
                depth_pct=float(np.clip((TOP_ELBOW_MIN - elbow_ema)/(TOP_ELBOW_MIN - BOTTOM_ELBOW_MAX), 0.0, 1.0))

            if elbow_ema is not None:
                min_elbow = elbow_ema if min_elbow is None else min(min_elbow, elbow_ema)
                max_elbow = elbow_ema if max_elbow is None else max(max_elbow, elbow_ema)
            if body_ema is not None:
                min_body = body_ema if min_body is None else min(min_body, body_ema)

            bottom_cond = (elbow_ema is not None and elbow_ema <= BOTTOM_ELBOW_MAX)
            top_cond = (elbow_ema is not None and elbow_ema >= TOP_ELBOW_MIN)

            if not in_bottom:
                if bottom_cond:
                    in_bottom = True
            else:
                if top_cond and refractory==0:
                    rep_count += 1
                    rep_feedback=set()
                    rep_penalty=0.0

                    depth_ok = (min_elbow is not None and min_elbow <= DEPTH_MIN_ANGLE)
                    depth_near = (min_elbow is not None and min_elbow <= (DEPTH_MIN_ANGLE + DEPTH_NEAR_DEG))
                    if not (depth_ok or depth_near):
                        rep_feedback.add(FB_CUE_DEPTH)
                        rep_penalty += FB_WEIGHTS.get(FB_CUE_DEPTH, FB_DEFAULT_WEIGHT)
                        depth_fail_count += 1
                        if depth_fail_count>=DEPTH_FAIL_MIN_REPS: depth_already_reported=True

                    lockout_ok = (max_elbow is not None and max_elbow >= LOCKOUT_MIN_ANGLE)
                    lockout_near = (max_elbow is not None and max_elbow >= (LOCKOUT_MIN_ANGLE - LOCKOUT_NEAR_DEG))
                    if not (lockout_ok or lockout_near):
                        rep_feedback.add(FB_CUE_LOCKOUT)
                        rep_penalty += FB_WEIGHTS.get(FB_CUE_LOCKOUT, FB_DEFAULT_WEIGHT)
                        lockout_fail_count += 1
                        if lockout_fail_count>=LOCKOUT_FAIL_MIN_REPS: lockout_already_reported=True

                    body_ok = (min_body is not None and min_body >= BODY_LINE_MIN)
                    body_near = (min_body is not None and min_body >= (BODY_LINE_MIN - BODY_LINE_NEAR_DEG))
                    if not (body_ok or body_near):
                        rep_feedback.add(FB_CUE_BODY)
                        rep_penalty += FB_WEIGHTS.get(FB_CUE_BODY, FB_DEFAULT_WEIGHT)
                        body_fail_count += 1
                        if body_fail_count>=BODY_LINE_FAIL_MIN_REPS: body_already_reported=True

                    if depth_already_reported: session_feedback.add(FB_CUE_DEPTH)
                    if lockout_already_reported: session_feedback.add(FB_CUE_LOCKOUT)
                    if body_already_reported: session_feedback.add(FB_CUE_BODY)

                    rep_score=max(0.0, 10.0-rep_penalty)
                    rep_good = rep_score >= 9.0
                    good_reps += 1 if rep_good else 0
                    bad_reps += 0 if rep_good else 1
                    all_scores.append(rep_score)
                    rep_reports.append({
                        "rep_index": int(rep_count),
                        "score": float(rep_score),
                        "good": bool(rep_good),
                        "min_elbow": float(min_elbow) if min_elbow is not None else 0.0,
                        "max_elbow": float(max_elbow) if max_elbow is not None else 0.0,
                        "min_body": float(min_body) if min_body is not None else 0.0
                    })

                    if rep_feedback:
                        rt_fb_msg = next((m for m in FORM_TIP_PRIORITY if m in rep_feedback), None)
                        rt_fb_hold = int(max(1.0, fps_in/max(1,effective_frame_skip)) * 0.8)

                    in_bottom=False
                    refractory=REFRACTORY_FRAMES
                    min_elbow=None; max_elbow=None; min_body=None

            if refractory>0:
                refractory -= 1
            if rt_fb_hold>0:
                rt_fb_hold -= 1

            if return_video and out:
                frame=draw_body_only(frame, lms)
                frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),depth_pct=depth_pct)
                out.write(frame)

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    if rep_count==0: technique_score=0.0
    else:
        if session_feedback:
            penalty = sum(FB_WEIGHTS.get(m,FB_DEFAULT_WEIGHT) for m in set(session_feedback))
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
        else:
            penalty = 0.0
        technique_score=_half_floor10(max(0.0,10.0-penalty))

    all_fb = set(session_feedback) if session_feedback else set()
    fb_list = [cue for cue in FORM_TIP_PRIORITY if cue in all_fb]

    form_tip = None
    if all_fb:
        form_tip = max(all_fb, key=lambda m: (FB_WEIGHTS.get(m, FB_DEFAULT_WEIGHT),
                                              -FORM_TIP_PRIORITY.index(m) if m in FORM_TIP_PRIORITY else -999))

    try:
        with open(feedback_path,"w",encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            if fb_list:
                f.write("Feedback:\n")
                for ln in fb_list: f.write(f"- {ln}\n")
    except Exception:
        pass

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
        "video_path": final_path if slow_path else "",
        "feedback_path": feedback_path
    }
    if form_tip is not None:
        result["form_tip"] = form_tip

    return result

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
