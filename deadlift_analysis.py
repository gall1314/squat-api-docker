# -*- coding: utf-8 -*-
# Deadlift — רספונסיבי + Hold-Last רק לרגליים כשלא מזוהה
import os, cv2, math, numpy as np, subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===== STYLE / FONTS (כמו בסקוואט) =====
BAR_BG_ALPHA=0.55; DONUT_RADIUS_SCALE=0.72; DONUT_THICKNESS_FRAC=0.28
DEPTH_COLOR=(40,200,80); DEPTH_RING_BG=(70,70,70)
FONT_PATH="Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE=28; FEEDBACK_FONT_SIZE=22; DEPTH_LABEL_FONT_SIZE=14; DEPTH_PCT_FONT_SIZE=18
def _f(path,size):
    try: return ImageFont.truetype(path,size)
    except: return ImageFont.load_default()
REPS_FONT=_f(FONT_PATH,REPS_FONT_SIZE); FEEDBACK_FONT=_f(FONT_PATH,FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT=_f(FONT_PATH,DEPTH_LABEL_FONT_SIZE); DEPTH_PCT_FONT=_f(FONT_PATH,DEPTH_PCT_FONT_SIZE)

mp_pose=mp.solutions.pose

# ===== ציוני תצוגה (סטנדרט) =====
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

# ===== BODY-ONLY =====
_FACE={mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_EYE_INNER.value,
       mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
       mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value,
       mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value, mp_pose.PoseLandmark.LEFT_EAR.value,
       mp_pose.PoseLandmark.RIGHT_EAR.value, mp_pose.PoseLandmark.MOUTH_LEFT.value,
       mp_pose.PoseLandmark.MOUTH_RIGHT.value}
_BODY_CONNS=tuple((a,b) for (a,b) in mp_pose.POSE_CONNECTIONS if a not in _FACE and b not in _FACE)
_BODY_POINTS=tuple(sorted({i for (a,b) in _BODY_CONNS for i in (a,b)}))
def draw_body_only_from_dict(frame, pts):
    h,w=frame.shape[:2]
    for a,b in _BODY_CONNS:
        if a in pts and b in pts:
            ax,ay=int(pts[a][0]*w),int(pts[a][1]*h); bx,by=int(pts[b][0]*w),int(pts[b][1]*h)
            cv2.line(frame,(ax,ay),(bx,by),(255,255,255),2,cv2.LINE_AA)
    for i in _BODY_POINTS:
        if i in pts:
            x,y=int(pts[i][0]*w),int(pts[i][1]*h)
            cv2.circle(frame,(x,y),3,(255,255,255),-1,cv2.LINE_AA)
    return frame

# ===== גיאומטריה + גב =====
def analyze_back_curvature(shoulder, hip, head_like, threshold=0.04):
    line_vec=hip-shoulder; nrm=np.linalg.norm(line_vec)+1e-9
    u=line_vec/nrm; proj_len=np.dot(head_like-shoulder,u); proj=shoulder+proj_len*u
    off=head_like-proj; signed=float(np.sign(off[1])*-1*np.linalg.norm(off))
    return signed, signed<-threshold

# ===== פרמטרים =====
HINGE_START_THRESH=0.08; STAND_DELTA_TARGET=0.025; END_THRESH=0.035; MIN_FRAMES_BETWEEN_REPS=10
PROG_ALPHA=0.35  # החלקת הדונאט בלבד

# ===== Overlay (כמו בסקוואט) =====
def _wrap2(draw,text,font,maxw):
    words=text.split(); 
    if not words:return [""]
    lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        if draw.textlength(t,font=font)<=maxw: cur=t
        else:
            if cur: lines.append(cur)
            cur=w
        if len(lines)==2: break
    if cur and len(lines)<2: lines.append(cur)
    leftover=len(words)-sum(len(l.split()) for l in lines)
    if leftover>0 and len(lines)>=2:
        last=lines[-1]+"…"
        while draw.textlength(last,font=font)>maxw and len(last)>1:
            last=last[:-2]+"…"
        lines[-1]=last
    return lines

def _donut(frame,c,r,t,p):
    p=float(np.clip(p,0,1)); cx,cy=int(c[0]),int(c[1]); r=int(r); t=int(t)
    cv2.circle(frame,(cx,cy),r,DEPTH_RING_BG,t,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(r,r),0,-90,-90+int(360*p),DEPTH_COLOR,t,cv2.LINE_AA)
    return frame

def draw_overlay(frame,reps=0,feedback=None,progress_pct=0.0):
    h,w,_=frame.shape
    # Reps box
    pil=Image.fromarray(frame); d=ImageDraw.Draw(pil)
    txt=f"Reps: {reps}"; padx,pady=10,6
    tw=d.textlength(txt,font=REPS_FONT); th=REPS_FONT.size
    x0,y0=0,0; x1=int(tw+2*padx); y1=int(th+2*pady)
    top=frame.copy(); cv2.rectangle(top,(x0,y0),(x1,y1),(0,0,0),-1)
    frame=cv2.addWeighted(top,BAR_BG_ALPHA,frame,1-BAR_BG_ALPHA,0)
    pil=Image.fromarray(frame); ImageDraw.Draw(pil).text((x0+padx,y0+pady-1),txt,font=REPS_FONT,fill=(255,255,255))
    frame=np.array(pil)
    # Donut
    ref_h=max(int(h*0.06),int(REPS_FONT_SIZE*1.6)); r=int(ref_h*DONUT_RADIUS_SCALE)
    thick=max(3,int(r*DONUT_THICKNESS_FRAC)); m=12; cx=w-m-r; cy=max(ref_h+r//8,r+thick//2+2)
    frame=_donut(frame,(cx,cy),r,thick,float(np.clip(progress_pct,0,1)))
    pil=Image.fromarray(frame); d=ImageDraw.Draw(pil)
    label="DEPTH"; pct=f"{int(float(np.clip(progress_pct,0,1))*100)}%"
    lw=d.textlength(label,font=DEPTH_LABEL_FONT); pw=d.textlength(pct,font=DEPTH_PCT_FONT)
    gap=max(2,int(r*0.10)); base=cy-(DEPTH_LABEL_FONT.size+gap+DEPTH_PCT_FONT.size)//2
    d.text((cx-int(lw//2),base),label,font=DEPTH_LABEL_FONT,fill=(255,255,255))
    d.text((cx-int(pw//2),base+DEPTH_LABEL_FONT.size+gap),pct,font=DEPTH_PCT_FONT,fill=(255,255,255))
    frame=np.array(pil)
    # Feedback
    if feedback:
        pil_fb=Image.fromarray(frame); dfb=ImageDraw.Draw(pil_fb)
        safe=max(6,int(h*0.02)); padx,pady,lg=12,8,4; maxw=int(w-2*padx-20)
        lines=_wrap2(dfb,feedback,FEEDBACK_FONT,maxw); lh=FEEDBACK_FONT.size+6
        block=(2*pady)+len(lines)*lh+(len(lines)-1)*lg; y0=max(0,h-safe-block); y1=h-safe
        over=frame.copy(); cv2.rectangle(over,(0,y0),(w,y1),(0,0,0),-1)
        frame=cv2.addWeighted(over,BAR_BG_ALPHA,frame,1-BAR_BG_ALPHA,0)
        pil_fb=Image.fromarray(frame); dfb=ImageDraw.Draw(pil_fb); ty=y0+pady
        for ln in lines:
            tw=dfb.textlength(ln,font=FEEDBACK_FONT); tx=max(padx,(w-int(tw))//2)
            dfb.text((tx,ty),ln,font=FEEDBACK_FONT,fill=(255,255,255)); ty+=lh+lg
        frame=np.array(pil_fb)
    return frame

# ===== MAIN =====
def run_deadlift_analysis(video_path, frame_skip=3, scale=0.4,
                          output_path="deadlift_analyzed.mp4",
                          feedback_path="deadlift_feedback.txt"):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"squat_count":0,"technique_score":0.0,"technique_score_display":display_half_str(0.0),
                "technique_label":score_label(0.0),"good_reps":0,"bad_reps":0,
                "feedback":["Could not open video"],"reps":[],"video_path":"","feedback_path":feedback_path}

    mp_pose_mod=mp.solutions.pose
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'); out=None
    fps_in=cap.get(cv2.CAP_PROP_FPS) or 25; eff_fps=max(1.0,fps_in/max(1,frame_skip))
    dt=1.0/float(eff_fps)

    counter=good=bad=0; all_scores=[]; reps_report=[]; overall_fb=[]
    rep=False; last_rep_f=-999; i=0

    top_ref=STAND_DELTA_TARGET; bottom_est=None; prog=1.0
    BACK_MIN_FRAMES=max(2,int(0.25/dt)); back_frames=0

    # hold-last לרגליים בלבד
    leg_prev={}
    R=mp_pose_mod.PoseLandmark
    LEG_IDXS=[R.RIGHT_ANKLE.value,R.RIGHT_KNEE.value,R.RIGHT_HEEL.value,R.RIGHT_FOOT_INDEX.value,
              R.LEFT_ANKLE.value,R.LEFT_KNEE.value,R.LEFT_HEEL.value,R.LEFT_FOOT_INDEX.value]
    VIS_THR=0.5

    with mp_pose_mod.Pose(model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret: break
            i+=1
            if i%frame_skip!=0: continue
            if scale!=1.0: frame=cv2.resize(frame,(0,0),fx=scale,fy=scale)

            h,w=frame.shape[:2]
            if out is None: out=cv2.VideoWriter(output_path,fourcc,eff_fps,(w,h))

            img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=pose.process(img)

            rt_fb=None
            if not res.pose_landmarks:
                frame=draw_overlay(frame,reps=counter,feedback=None,progress_pct=prog); out.write(frame); continue

            try:
                lm=res.pose_landmarks.landmark

                # נקודות חיות (לשמור רספונסיביות)
                shoulder=np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y],dtype=float)
                hip     =np.array([lm[R.RIGHT_HIP.value].x,      lm[R.RIGHT_HIP.value].y],     dtype=float)

                # ראש (חי)
                head=None
                for idx in (R.RIGHT_EAR.value,R.LEFT_EAR.value,R.NOSE.value):
                    if lm[idx].visibility>0.5:
                        head=np.array([lm[idx].x,lm[idx].y],dtype=float); break
                if head is None:
                    frame=draw_overlay(frame,reps=counter,feedback=None,progress_pct=prog); out.write(frame); continue

                # רגליים — hold-last אם visibility נמוך; אחרת ערך חי (בלי EMA)
                leg_pts={}
                for idx in LEG_IDXS:
                    p=lm[idx]
                    if p.visibility>=VIS_THR:
                        leg_prev[idx]=(float(p.x),float(p.y))
                        leg_pts[idx]=leg_prev[idx]
                    elif idx in leg_prev:
                        leg_pts[idx]=leg_prev[idx]  # החזק ערך אחרון
                    # אם אין prev — לא נוסיף נקודה (נמנע קפיצות)

                # delta_x — עומק ההינג’
                delta_x=abs(hip[0]-shoulder[0])

                # עדכון top_ref כשאנכי (שמירה זריזה)
                if delta_x < (STAND_DELTA_TARGET*1.4):
                    top_ref=0.9*top_ref+0.1*delta_x

                # גב
                mid=(shoulder+hip)*0.5*0.4 + head*0.6
                _,rounded=analyze_back_curvature(shoulder,hip,mid)
                back_frames = back_frames+1 if rounded else max(0,back_frames-1)

                # התחלת חזרה
                if (not rep) and (delta_x>HINGE_START_THRESH) and (i-last_rep_f>MIN_FRAMES_BETWEEN_REPS):
                    rep=True; bottom_est=delta_x; back_frames=0

                # דונאט דו-כיווני
                if rep:
                    bottom_est=max(bottom_est or delta_x, delta_x)
                    denom=max(1e-4,(bottom_est-top_ref))
                    pr=1.0 - ((delta_x-top_ref)/denom); pr=float(np.clip(pr,0,1))
                    prog=PROG_ALPHA*pr+(1-PROG_ALPHA)*prog
                    if rounded: rt_fb="Keep your back straighter"
                else:
                    prog=PROG_ALPHA*1.0+(1-PROG_ALPHA)*prog

                # סיום חזרה
                if rep and (delta_x<END_THRESH):
                    if i-last_rep_f>MIN_FRAMES_BETWEEN_REPS:
                        fb=[]; pen=0.0
                        if delta_x>(top_ref+0.02): fb.append("Try to finish more upright"); pen+=1.0
                        if back_frames>=BACK_MIN_FRAMES: fb.append("Try to keep your back a bit straighter"); pen+=1.5
                        score=round(max(4,10-pen)*2)/2
                        for f in fb:
                            if f not in overall_fb: overall_fb.append(f)
                        moved=(bottom_est-delta_x)>0.05
                        if moved:
                            counter+=1; (good:=good)+(0); 
                            if score>=9.5: good+=1
                            else: bad+=1
                            all_scores.append(score)
                            reps_report.append({"rep_index":counter,"score":float(score),
                                                "score_display":display_half_str(score),"feedback":fb,"tip":None})
                        last_rep_f=i
                    rep=False; bottom_est=None; back_frames=0

                # ציור שלד: גוף + רגליים מ־hold-last
                pts_draw={}; 
                # shoulder/hip/head חיות
                pts_draw[R.RIGHT_SHOULDER.value]=tuple(shoulder); pts_draw[R.RIGHT_HIP.value]=tuple(hip)
                if R.RIGHT_EAR.value in LEG_IDXS: pass
                # הוסף את הרגליים המחוזקות
                for idx,pt in leg_pts.items(): pts_draw[idx]=pt

                frame=draw_body_only_from_dict(frame,pts_draw)
                frame=draw_overlay(frame,reps=counter,feedback=rt_fb,progress_pct=prog)
                out.write(frame)

            except Exception:
                frame=draw_overlay(frame,reps=counter,feedback=None,progress_pct=prog)
                if out is not None: out.write(frame)
                continue

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    avg=np.mean(all_scores) if all_scores else 0.0
    technique_score=round(round(avg*2)/2,2)
    feedback_list=overall_fb[:] if overall_fb else ["Great form! Keep your spine neutral and hinge smoothly. 💪"]

    try:
        with open(feedback_path,"w",encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list: f.write(f"- {fb}\n")
    except: pass

    encoded=output_path.replace(".mp4","_encoded.mp4")
    try:
        subprocess.run(["ffmpeg","-y","-i",output_path,"-c:v","libx264","-preset","fast",
                        "-movflags","+faststart","-pix_fmt","yuv420p",encoded],check=False)
        if os.path.exists(output_path) and os.path.exists(encoded): os.remove(output_path)
    except: pass
    final=encoded if os.path.exists(encoded) else (output_path if os.path.exists(output_path) else "")

    return {
        "squat_count": counter,
        "technique_score": technique_score,
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": good, "bad_reps": bad,
        "feedback": feedback_list, "reps": reps_report,
        "video_path": final, "feedback_path": feedback_path
    }
