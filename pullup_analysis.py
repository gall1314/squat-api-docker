# -*- coding: utf-8 -*-
# pullup_analysis.py — Pull-ups counter (ascents-only) using ELBOW angle + small head ascent.
# FIX: pre-set "walking" gate uses *horizontal torso* motion only, and only before the first counted rep.
# After first rep (or timeout), gate is OFF so reps aren't blocked by normal pull-up movement.

import os, math, subprocess
import cv2, numpy as np
from PIL import ImageFont, ImageDraw, Image

# ===================== UI (מיושר לסקוואט) =====================
BAR_BG_ALPHA=0.55
REPS_FONT_SIZE=28; FEEDBACK_FONT_SIZE=22
DEPTH_LABEL_FONT_SIZE=14; DEPTH_PCT_FONT_SIZE=18
FONT_PATH="Roboto-VariableFont_wdth,wght.ttf"
DONUT_RADIUS_SCALE=0.72; DONUT_THICKNESS_FRAC=0.28
ASCENT_COLOR=(40,200,80); DONUT_RING_BG=(70,70,70)
RT_FB_HOLD_SEC=0.8

def _load_font(p,s):
    try: return ImageFont.truetype(p,s)
    except: return ImageFont.load_default()
REPS_FONT=_load_font(FONT_PATH,REPS_FONT_SIZE)
FEEDBACK_FONT=_load_font(FONT_PATH,FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT=_load_font(FONT_PATH,DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT=_load_font(FONT_PATH,DEPTH_PCT_FONT_SIZE)

# ===================== MediaPipe =====================
try:
    import mediapipe as mp
    mp_pose=mp.solutions.pose
    MP_OK=True
except:
    MP_OK=False

def score_label(s):
    s=float(s)
    if s>=9.5:return "Excellent"
    if s>=8.5:return "Very good"
    if s>=7.0:return "Good"
    if s>=5.5:return "Fair"
    return "Needs work"

def display_half_str(x):
    q=round(float(x)*2)/2.0
    return str(int(round(q))) if abs(q-round(q))<1e-9 else f"{q:.1f}"

# ===================== ציור שלד גוף בלבד =====================
_FACE={getattr(mp_pose.PoseLandmark,n).value for n in [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER","RIGHT_EYE","RIGHT_EYE_OUTER",
    "LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT"
]} if MP_OK else set()
_BODY_CONNS=tuple((a,b) for (a,b) in (mp_pose.POSE_CONNECTIONS if MP_OK else []) if a not in _FACE and b not in _FACE)
_BODY_POINTS=tuple(sorted({i for c in _BODY_CONNS for i in c})) if MP_OK else tuple()

def draw_body_only(frame,lms,color=(255,255,255)):
    h,w=frame.shape[:2]
    for a,b in _BODY_CONNS:
        pa, pb = lms[a], lms[b]
        ax,ay=int(pa.x*w),int(pa.y*h); bx,by=int(pb.x*w),int(pb.y*h)
        cv2.line(frame,(ax,ay),(bx,by),color,2,cv2.LINE_AA)
    for i in _BODY_POINTS:
        p=lms[i]; x,y=int(p.x*w),int(p.y*h)
        cv2.circle(frame,(x,y),3,color,-1,cv2.LINE_AA)
    return frame

def _wrap_two_lines(draw,text,font,max_w):
    words=text.split(); lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        if draw.textlength(t,font=font)<=max_w: cur=t
        else:
            if cur: lines.append(cur); cur=w
        if len(lines)==2: break
    if cur and len(lines)<2: lines.append(cur)
    if len(lines)>=2 and draw.textlength(lines[-1],font=font)>max_w:
        last=lines[-1]+"…"
        while draw.textlength(last,font=font)>max_w and len(last)>1:
            last=last[:-2]+"…"
        lines[-1]=last
    return lines

def _donut(frame,c,r,t,pct):
    pct=float(np.clip(pct,0,1)); cx,cy=int(c[0]),int(c[1]); r=int(r); t=int(t)
    cv2.circle(frame,(cx,cy),r,DONUT_RING_BG,t,cv2.LINE_AA)
    start=-90; end=start+int(360*pct)
    cv2.ellipse(frame,(cx,cy),(r,r),0,start,end,ASCENT_COLOR,t,cv2.LINE_AA)
    return frame

def draw_overlay(frame,reps=0,feedback=None,height_pct=0.0):
    h,w,_=frame.shape
    # Reps box
    pil=Image.fromarray(frame); draw=ImageDraw.Draw(pil)
    txt=f"Reps: {reps}"; pad_x,pad_y=10,6
    tw=draw.textlength(txt,font=REPS_FONT); th=REPS_FONT.size
    top=frame.copy(); cv2.rectangle(top,(0,0),(int(tw+2*pad_x),int(th+2*pad_y)),(0,0,0),-1)
    frame=cv2.addWeighted(top,BAR_BG_ALPHA,frame,1-BAR_BG_ALPHA,0)
    pil=Image.fromarray(frame); ImageDraw.Draw(pil).text((pad_x,pad_y-1),txt,font=REPS_FONT,fill=(255,255,255))
    frame=np.array(pil)
    # Donut
    ref_h=max(int(h*0.06),int(REPS_FONT_SIZE*1.6)); radius=int(ref_h*DONUT_RADIUS_SCALE)
    thick=max(3,int(radius*DONUT_THICKNESS_FRAC)); m=12
    cx=w-m-radius; cy=max(ref_h+radius//8, radius+thick//2+2)
    frame=_donut(frame,(cx,cy),radius,thick,float(np.clip(height_pct,0,1)))
    pil=Image.fromarray(frame); draw=ImageDraw.Draw(pil)
    label="HEIGHT"; pct=f"{int(float(np.clip(height_pct,0,1))*100)}%"; gap=max(2,int(radius*0.10))
    base_y=cy-(DEPTH_LABEL_FONT_SIZE+gap+DEPTH_PCT_FONT_SIZE)//2
    lw=draw.textlength(label,font=DEPTH_LABEL_FONT); pw=draw.textlength(pct,font=DEPTH_PCT_FONT)
    draw.text((cx-int(lw//2),base_y),label,font=DEPTH_LABEL_FONT,fill=(255,255,255))
    draw.text((cx-int(pw//2),base_y+DEPTH_LABEL_FONT_SIZE+gap),pct,font=DEPTH_PCT_FONT,fill=(255,255,255))
    frame=np.array(pil)
    # Feedback
    if feedback:
        pil=Image.fromarray(frame); d=ImageDraw.Draw(pil)
        safe=max(6,int(h*0.02)); padx,pady,gap=12,8,4; max_w=int(w-2*padx-20)
        lines=_wrap_two_lines(d,feedback,FEEDBACK_FONT,max_w)
        line_h=FEEDBACK_FONT.size+6; block=2*pady+len(lines)*line_h+(len(lines)-1)*gap
        y0=max(0,h-safe-block); y1=h-safe
        over=frame.copy(); cv2.rectangle(over,(0,y0),(w,y1),(0,0,0),-1)
        frame=cv2.addWeighted(over,BAR_BG_ALPHA,frame,1-BAR_BG_ALPHA,0)
        pil=Image.fromarray(frame); d=ImageDraw.Draw(pil); ty=y0+pady
        for ln in lines:
            tw=d.textlength(ln,font=FEEDBACK_FONT); tx=max(padx,(w-int(tw))//2)
            d.text((tx,ty),ln,font=FEEDBACK_FONT,fill=(255,255,255)); ty+=line_h+gap
        frame=np.array(pil)
    return frame

# ===================== חישובים =====================
def _ang(a,b,c):
    ba=np.array([a[0]-b[0],a[1]-b[1]]); bc=np.array([c[0]-b[0],c[1]-b[1]])
    den=(np.linalg.norm(ba)*np.linalg.norm(bc))+1e-9
    cos=float(np.clip(np.dot(ba,bc)/den,-1,1))
    return float(np.degrees(np.arccos(cos)))
def _ema(prev,new,alpha): return float(new) if prev is None else (alpha*float(new)+(1-alpha)*float(prev))

# ===================== פרמטרים =====================
# ספירה על בסיס מרפק + עליית ראש
ELBOW_TOP_THRESHOLD=100.0      # סגירת מרפק מספיקה
ELBOW_RESET_THRESHOLD=140.0    # פתיחת מרפק לשחרור ספירה הבאה
HEAD_MIN_ASCENT=0.012          # ~1.2% גובה פריים
RESET_DESCENT=0.008            # ירידה קטנה לשחרור ספירה
REFRACTORY_FRAMES=3            # דיבאונס בין פסגות
HEAD_VEL_UP_TINY=0.00025
ELBOW_EMA_ALPHA=0.35; HEAD_EMA_ALPHA=0.30

# זיהוי שלד
VIS_THR_RELAX=0.22             # לעיבוד
VIS_THR_STRICT=0.30            # לספירה (ודאות)

# PRE-SET WALK GATE — אופקי בלבד + לפני חזרה ראשונה
PRESET_GATE=True
PRESET_GATE_TIMEOUT_SEC=1.2
TORSO_WALK_X_THR=0.008         # שינוי במרכז טורסו באחוז מרוחב פריים
TORSO_WALK_MIN_FRAMES=4
TORSO_X_EMA_ALPHA=0.6

def run_pullup_analysis(video_path, frame_skip=1, scale=1.0,
                        output_path="pullup_analyzed.mp4",
                        feedback_path="pullup_feedback.txt"):
    if not MP_OK:
        return {"squat_count":0,"technique_score":0.0,"technique_score_display":display_half_str(0.0),
                "technique_label":score_label(0.0),"good_reps":0,"bad_reps":0,"feedback":["Mediapipe not available"],
                "tips":[],"reps":[],"video_path":"","feedback_path":feedback_path}

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"squat_count":0,"technique_score":0.0,"technique_score_display":display_half_str(0.0),
                "technique_label":score_label(0.0),"good_reps":0,"bad_reps":0,"feedback":["Could not open video"],
                "tips":[],"reps":[],"video_path":"","feedback_path":feedback_path}

    fourcc=cv2.VideoWriter_fourcc(*'mp4v'); out=None; frame_no=0
    rep_count=0; rep_reports=[]; good_reps=0; bad_reps=0; all_scores=[]

    def _pick_side_dyn(lms):
        def vis(i): 
            try: return float(lms[i].visibility or 0.0)
            except: return 0.0
        vL=vis(mp_pose.PoseLandmark.LEFT_SHOULDER.value)+vis(mp_pose.PoseLandmark.LEFT_ELBOW.value)+vis(mp_pose.PoseLandmark.LEFT_WRIST.value)
        vR=vis(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)+vis(mp_pose.PoseLandmark.RIGHT_ELBOW.value)+vis(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        return "LEFT" if vL>=vR else "RIGHT"

    allow_new_peak=True; last_peak_frame=-999999
    asc_base_head=None; baseline_head_y_global=None; ascent_live=0.0
    elbow_ema=None; head_ema=None; head_prev=None

    fps_in=cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps=max(1.0, fps_in/max(1,frame_skip))
    dt=1.0/effective_fps
    RT_FB_HOLD_FRAMES=max(2,int(RT_FB_HOLD_SEC/dt))
    rt_fb_msg=None; rt_fb_hold=0

    # PRE-SET gate state
    preset_gate_active=PRESET_GATE
    preset_gate_timeout_frames=max(1,int(PRESET_GATE_TIMEOUT_SEC*effective_fps))
    preset_timer=preset_gate_timeout_frames
    torso_cx_ema=None; walk_x_streak=0

    with mp_pose.Pose(model_complexity=1,min_detection_confidence=0.6,min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ok,frame=cap.read()
            if not ok: break
            frame_no+=1
            if frame_skip>1 and (frame_no%frame_skip)!=0: continue
            if scale!=1.0: frame=cv2.resize(frame,(0,0),fx=scale,fy=scale)

            h,w=frame.shape[:2]
            if out is None: out=cv2.VideoWriter(output_path,fourcc,effective_fps,(w,h))

            res=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            elbow_angle=None; head_y=None; lms=None
            vis_relaxed_ok=False; vis_strict_ok=False

            if res.pose_landmarks:
                lms=res.pose_landmarks.landmark
                side=_pick_side_dyn(lms)
                S=getattr(mp_pose.PoseLandmark,f"{side}_SHOULDER").value
                E=getattr(mp_pose.PoseLandmark,f"{side}_ELBOW").value
                W=getattr(mp_pose.PoseLandmark,f"{side}_WRIST").value
                NOSE=mp_pose.PoseLandmark.NOSE.value

                min_vis=min(lms[NOSE].visibility,lms[S].visibility,lms[E].visibility,lms[W].visibility)
                vis_relaxed_ok=(min_vis>=VIS_THR_RELAX)
                vis_strict_ok =(min_vis>=VIS_THR_STRICT)

                if vis_relaxed_ok:
                    head_raw=float(lms[NOSE].y)
                    raw_elbow=_ang((lms[S].x,lms[S].y),(lms[E].x,lms[E].y),(lms[W].x,lms[W].y))
                    elbow_ema=_ema(elbow_ema,raw_elbow,ELBOW_EMA_ALPHA)
                    head_ema=_ema(head_ema,head_raw,HEAD_EMA_ALPHA)
                    head_y=head_ema; elbow_angle=elbow_ema
                    if baseline_head_y_global is None: baseline_head_y_global=head_y

                # -------- PRE-SET WALK GATE (אופקי) --------
                # משתמש במרכז טורסו (כתפיים+ירכיים) בציר X בלבד
                LSH=mp_pose.PoseLandmark.LEFT_SHOULDER.value; RSH=mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                LHIP=mp_pose.PoseLandmark.LEFT_HIP.value; RHIP=mp_pose.PoseLandmark.RIGHT_HIP.value
                torso_cx = np.mean([lms[LSH].x,lms[RSH].x,lms[LHIP].x,lms[RHIP].x])*w
                torso_cx_ema=_ema(torso_cx_ema,torso_cx,TORSO_X_EMA_ALPHA)
                if preset_gate_active:
                    dx=0.0 if torso_cx_ema is None else abs(torso_cx - torso_cx_ema)/max(1.0,w)
                    if dx > TORSO_WALK_X_THR:
                        walk_x_streak=min(TORSO_WALK_MIN_FRAMES, walk_x_streak+1)
                    else:
                        walk_x_streak=max(0, walk_x_streak-1)
                    # מורידים טיימר רק כשיש שלד (שלא ייגמר בזמן מסך שחור)
                    preset_timer=max(0, preset_timer-1)
                    # אם עבר הזמן או נספרה חזרה — נבטל את ה-gate
                    if preset_timer==0 or rep_count>0:
                        preset_gate_active=False
            # ---- תנאי gate לספירה ----
            if preset_gate_active:
                # בזמן PRE-SET – חייב שלד ודאי + לא "הליכה" אופקית רציפה
                count_gate_ok = (vis_strict_ok and walk_x_streak < TORSO_WALK_MIN_FRAMES)
            else:
                # אחרי PRE-SET – אין gate הליכה בכלל
                count_gate_ok = vis_strict_ok

            # מהירות ראש
            head_vel=0.0
            if head_y is not None and head_prev is not None:
                head_vel=head_y - head_prev

            # Baseline לעלייה
            if head_y is not None:
                if asc_base_head is None:
                    # בתחילת סט: אם gate מאפשר – קבע baseline מיד
                    if count_gate_ok: asc_base_head=head_y
                else:
                    if (head_y - asc_base_head) > RESET_DESCENT*2:
                        asc_base_head=head_y

            # -------- ספירת פסגות (עליות בלבד) --------
            if count_gate_ok and (head_y is not None) and (elbow_angle is not None):
                top_by_elbow = (elbow_angle <= ELBOW_TOP_THRESHOLD)
                top_by_vel = False
                if asc_base_head is not None and head_prev is not None:
                    asc_ok = ((asc_base_head - head_y) >= HEAD_MIN_ASCENT)
                    top_by_vel = (head_vel >= 0) and asc_ok
                top_ok = top_by_elbow or top_by_vel
                can_count = (frame_no - last_peak_frame) >= REFRACTORY_FRAMES
                if top_ok and allow_new_peak and can_count:
                    rep_count+=1; good_reps+=1; all_scores.append(10.0)
                    rep_reports.append({
                        "rep_index":rep_count,
                        "top_elbow":float(elbow_angle),
                        "peak_head_y":float(head_y),
                        "asc_from":float(asc_base_head if asc_base_head is not None else head_y),
                        "preset_gate_active":bool(preset_gate_active)
                    })
                    last_peak_frame=frame_no; allow_new_peak=False
                    # ברגע שסופרים – מבטלים gate להבא
                    preset_gate_active=False
                # reset קל בין פסגות
                reset_by_descent=(asc_base_head is not None) and ((head_y - asc_base_head) >= RESET_DESCENT)
                reset_by_elbow=(elbow_angle >= ELBOW_RESET_THRESHOLD)
                if reset_by_descent or reset_by_elbow:
                    allow_new_peak=True
                    asc_base_head=head_y

                # RT feedback עדין
                cur_rt=None
                if asc_base_head is not None:
                    asc_live = asc_base_head - head_y
                    if asc_live < HEAD_MIN_ASCENT*0.7 and head_vel < -HEAD_VEL_UP_TINY:
                        cur_rt="Go a bit higher (chin over bar)"
                if cur_rt:
                    if cur_rt!=rt_fb_msg: rt_fb_msg=cur_rt; rt_fb_hold=RT_FB_HOLD_FRAMES
                    else: rt_fb_hold=max(rt_fb_hold,RT_FB_HOLD_FRAMES)
                else:
                    if rt_fb_hold>0: rt_fb_hold-=1
            else:
                if rt_fb_hold>0: rt_fb_hold-=1

            # דונאט HEIGHT לייב
            ascent_live=0.0 if (baseline_head_y_global is None or head_y is None) \
                else float(np.clip((baseline_head_y_global - head_y)/max(0.12,HEAD_MIN_ASCENT*1.2),0,1))

            # ציור
            if res.pose_landmarks:
                frame=draw_body_only(frame,res.pose_landmarks.landmark)
            frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),height_pct=ascent_live)
            out.write(frame)

            if head_y is not None: head_prev=head_y

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    avg=np.mean(all_scores) if all_scores else 0.0
    technique_score=round(round(avg*2)/2,2)
    session_tip="Slow down the lowering phase to maximize hypertrophy"
    try:
        with open(feedback_path,"w",encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            f.write(f"Tip: {session_tip}\n")
    except: pass

    # faststart
    encoded_path=output_path.replace(".mp4","_encoded.mp4")
    try:
        subprocess.run(['ffmpeg','-y','-i',output_path,'-c:v','libx264','-preset','fast','-movflags','+faststart','-pix_fmt','yuv420p',encoded_path],
                       check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        final_path=encoded_path if os.path.isfile(encoded_path) else output_path
    except: final_path=output_path
    if not os.path.isfile(final_path) and os.path.isfile(output_path): final_path=output_path

    return {
        "squat_count":int(rep_count),
        "technique_score":float(technique_score),
        "technique_score_display":display_half_str(technique_score),
        "technique_label":score_label(technique_score),
        "good_reps":int(good_reps),
        "bad_reps":int(bad_reps),
        "feedback":["Great form! Keep it up 💪"],
        "tips":[session_tip],
        "reps":rep_reports,
        "video_path":final_path,
        "feedback_path":feedback_path
    }

# תאימות
def run_analysis(*args,**kwargs): return run_pullup_analysis(*args,**kwargs)



