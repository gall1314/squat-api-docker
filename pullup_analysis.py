# -*- coding: utf-8 -*-
# pullup_analysis.py — ספירה מקורית עם "דילוג אדפטיבי" ללא cap.set (קל ומהיר),
# כך שבזמן תנועה מעבדים כל פריים (step=1) כדי לא לאבד שני פריימים רצופים בטופ.
# אוברליי/טמפלייט/שמות שדות תואם לסקוואט. שם הספירה: squat_count.

import os, cv2, math, time, json, numpy as np, subprocess
from PIL import ImageFont, ImageDraw, Image

# ====== סטייל ======
BAR_BG_ALPHA=0.55
FONT_PATH="Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE=28; FEEDBACK_FONT_SIZE=22; DEPTH_LABEL_FONT_SIZE=14; DEPTH_PCT_FONT_SIZE=18
def _load_font(p,s):
    try: return ImageFont.truetype(p,s)
    except Exception: return ImageFont.load_default()
REPS_FONT=_load_font(FONT_PATH,REPS_FONT_SIZE)
FEEDBACK_FONT=_load_font(FONT_PATH,FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT=_load_font(FONT_PATH,DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT=_load_font(FONT_PATH,DEPTH_PCT_FONT_SIZE)
DONUT_RADIUS_SCALE=0.72; DONUT_THICKNESS_FRAC=0.28
DEPTH_COLOR=(40,200,80); DEPTH_RING_BG=(70,70,70)

# ====== ספים ======
ELBOW_START_THRESHOLD=150.0
ELBOW_TOP_THRESHOLD=65.0
ELBOW_BOTTOM_THRESHOLD=160.0
HEAD_MIN_ASCENT=0.06
HEAD_VEL_UP_THRESH=0.0025
HEAD_TOP_STICK_FRAMES=2
FPS_FALLBACK=25.0
BASE_FRAME_SKIP_IDLE=3
BASE_FRAME_SKIP_MOVE=1

# ====== עזר ======
def _angle(a,b,c):
    try:
        ba=(a[0]-b[0],a[1]-b[1]); bc=(c[0]-b[0],c[1]-b[1])
        den=((ba[0]**2+ba[1]**2)**0.5*(bc[0]**2+bc[1]**2)**0.5)+1e-9
        cosang=(ba[0]*bc[0]+ba[1]*bc[1])/den
        cosang=max(-1.0,min(1.0,cosang))
        return math.degrees(math.acos(cosang))
    except Exception:
        return 180.0

def _round_score_half(x): return round(x*2)/2.0
def display_half_str(x):
    q=round(float(x)*2)/2.0
    return str(int(round(q))) if abs(q-round(q))<1e-9 else f"{q:.1f}"

def score_label(s):
    s=float(s)
    if s>=9.5:return "Excellent"
    if s>=8.5:return "Very good"
    if s>=7.0:return "Good"
    if s>=5.5:return "Fair"
    return "Needs work"

# ====== MediaPipe ======
try:
    import mediapipe as mp
    MP_AVAILABLE=True
except Exception:
    MP_AVAILABLE=False

if MP_AVAILABLE:
    mp_pose=mp.solutions.pose
    _FACE_LMS={mp_pose.PoseLandmark.NOSE.value,
               mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
               mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
               mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
               mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value}
    _BODY_CONNECTIONS=tuple((a,b) for (a,b) in mp_pose.POSE_CONNECTIONS if a not in _FACE_LMS and b not in _FACE_LMS)
    _BODY_POINTS=tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

POSE_IDXS={"nose":0,"left_shoulder":11,"right_shoulder":12,"left_elbow":13,"right_elbow":14,"left_wrist":15,"right_wrist":16}

def _pick_side(lms):
    rs=lms[POSE_IDXS["right_shoulder"]].visibility; ls=lms[POSE_IDXS["left_shoulder"]].visibility
    re=lms[POSE_IDXS["right_elbow"]].visibility; le=lms[POSE_IDXS["left_elbow"]].visibility
    return "right" if (rs>0.4)+(re>0.4) >= (ls>0.4)+(le>0.4) else "left"

def _get_xy(landmarks, idx, w, h):
    lm=landmarks[idx]
    return (lm.x*w, lm.y*h, lm.visibility)

def _safe_vis(*vals, min_v=0.4): return all(v>=min_v for v in vals)

def draw_body_only(frame, landmarks, color=(255,255,255)):
    if not MP_AVAILABLE: return frame
    h,w=frame.shape[:2]
    for a,b in _BODY_CONNECTIONS:
        pa=landmarks[a]; pb=landmarks[b]
        cv2.line(frame,(int(pa.x*w),int(pa.y*h)),(int(pb.x*w),int(pb.y*h)),color,2,cv2.LINE_AA)
    for i in _BODY_POINTS:
        p=landmarks[i]; cv2.circle(frame,(int(p.x*w),int(p.y*h)),3,color,-1,cv2.LINE_AA)
    return frame

# ====== Overlay ======
def draw_ascent_donut(frame, center, radius, thickness, pct):
    pct=float(np.clip(pct,0.0,1.0)); cx,cy=int(center[0]),int(center[1])
    radius=int(radius); thickness=int(thickness)
    cv2.circle(frame,(cx,cy),radius,DEPTH_RING_BG,thickness,cv2.LINE_AA)
    start_ang=-90; end_ang=start_ang+int(360*pct)
    cv2.ellipse(frame,(cx,cy),(radius,radius),0,start_ang,end_ang,DEPTH_COLOR,thickness,cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, ascent_pct=0.0):
    h,w=frame.shape[:2]
    # Reps
    rep_text=f"Reps: {int(reps)}"
    (tw,th),_=cv2.getTextSize(rep_text,cv2.FONT_HERSHEY_SIMPLEX,0.9,2); pad=8
    bg=frame.copy(); cv2.rectangle(bg,(0,0),(tw+2*pad,th+int(1.8*pad)),(0,0,0),-1)
    frame=cv2.addWeighted(bg,BAR_BG_ALPHA,frame,1.0-BAR_BG_ALPHA,0)
    cv2.putText(frame,rep_text,(pad,th+pad),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2,cv2.LINE_AA)

    # Donut
    cx=w-70; cy=60
    radius=int(min(80,w*0.09)*DONUT_RADIUS_SCALE); thick=max(8,int(radius*DONUT_THICKNESS_FRAC))
    frame=draw_ascent_donut(frame,(cx,cy),radius,thick,ascent_pct)

    # Center labels
    depth_pct=int(round(ascent_pct*100)); pil=Image.fromarray(frame); d=ImageDraw.Draw(pil)
    t1="ASCENT"; t2=f"{depth_pct}%"
    d.text((cx-d.textlength(t1,DEPTH_LABEL_FONT)/2, cy-20), t1, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    d.text((cx-d.textlength(t2,DEPTH_PCT_FONT)/2,  cy+2),  t2, font=DEPTH_PCT_FONT,   fill=(255,255,255))
    frame=np.array(pil)

    # Bottom feedback (2 lines max)
    if feedback:
        pil_fb=Image.fromarray(frame); draw_fb=ImageDraw.Draw(pil_fb)
        safe_m=max(6,int(h*0.02)); bx,by,gap=12,8,4; max_w=int(w-2*bx-20)
        def wrap_two_lines(draw,text,font,max_w):
            words=text.split(); lines=[]; cur=""
            for w_ in words:
                trial=(cur+" "+w_).strip()
                if draw.textlength(trial,font=FEEDBACK_FONT)<=max_w: cur=trial
                else:
                    if cur: lines.append(cur); cur=w_
                if len(lines)==2: break
            if cur and len(lines)<2: lines.append(cur)
            leftover=len(words)-sum(len(x.split()) for x in lines)
            if leftover>0 and len(lines)>=2:
                last=lines[-1]+"…"
                while draw.textlength(last,font=FEEDBACK_FONT)>max_w and len(last)>1:
                    last=last[:-2]+"…"
                lines[-1]=last
            return lines
        lines=wrap_two_lines(draw_fb,feedback,FEEDBACK_FONT,max_w)
        line_h=FEEDBACK_FONT.size+6; block_h=(2*by)+len(lines)*line_h+(len(lines)-1)*gap
        y0=max(0,h-safe_m-block_h); y1=h-safe_m
        over=frame.copy(); cv2.rectangle(over,(0,y0),(w,y1),(0,0,0),-1)
        frame=cv2.addWeighted(over,BAR_BG_ALPHA,frame,1.0-BAR_BG_ALPHA,0)
        pil_fb=Image.fromarray(frame); draw_fb=ImageDraw.Draw(pil_fb)
        ty=y0+by
        for ln in lines:
            tw=draw_fb.textlength(ln,font=FEEDBACK_FONT); tx=max(bx,(w-int(tw))//2)
            draw_fb.text((tx,ty),ln,font=FEEDBACK_FONT,fill=(255,255,255)); ty+=line_h+gap
        frame=np.array(pil_fb)
    return frame

# ====== Analyzer ======
class PullUpAnalyzer:
    def __init__(self): self.reset()
    def reset(self):
        self.rep_count=0; self.in_rep=False; self.seen_top_frames=0
        self.min_head_y_in_rep=None; self.rep_started_elbow=None
        self.last_head_y=None; self.last_elbow=None
        self.reps_meta=[]; self.all_scores=[]; self.session_best_feedback=""
        self.ascent_live=0.0; self.baseline_head_y_global=None
        self.rt_fb_msg=None; self.rt_fb_hold=0; self.RT_FB_HOLD_FRAMES=20
        self.side_locked=None

    def _confirm_top(self, elbow_angle, head_y, rep_baseline_head_y):
        elbow_ok=(elbow_angle is not None and elbow_angle<=ELBOW_TOP_THRESHOLD)
        head_ok=(rep_baseline_head_y is not None and head_y is not None and
                 (rep_baseline_head_y-head_y)>=HEAD_MIN_ASCENT)
        return elbow_ok and head_ok

    def _confirm_start(self, elbow_angle, head_vel):
        elbow_ok=(elbow_angle is not None and elbow_angle<ELBOW_START_THRESHOLD)
        head_speed_ok=(head_vel < -HEAD_VEL_UP_THRESH)
        return elbow_ok and head_speed_ok

    def _movement_detected(self, elbow_angle, head_y):
        mov=False
        if self.last_elbow is not None and elbow_angle is not None and abs(elbow_angle-self.last_elbow)>1.5: mov=True
        if self.last_head_y is not None and head_y is not None and abs(head_y-self.last_head_y)>0.002: mov=True
        return mov

    def process(self, input_path, output_path=None, frame_skip=3, scale=0.4, overlay_enabled=True):
        if not MP_AVAILABLE: raise RuntimeError("mediapipe not available")
        try: cv2.setUseOptimized(True); cv2.setNumThreads(1)
        except Exception: pass

        cap=cv2.VideoCapture(input_path)
        if not cap.isOpened(): return self._empty_result("Could not open video", output_path)

        fps_in=cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        if fps_in<=1: fps_in=FPS_FALLBACK
        # ה־VideoWriter ירוץ ב-fps אפקטיבי שמבוסס על frame_skip (ערך הבסיס במנוחה)
        effective_fps=max(1.0, fps_in/max(1,frame_skip))
        self.RT_FB_HOLD_FRAMES=max(2, int(0.8 * effective_fps / (effective_fps if effective_fps>0 else 1)))

        print(f"[PULLUP] open | fps_in={fps_in:.2f} | skip_idle={frame_skip} | skip_move=1 | eff_fps≈{effective_fps:.2f}", flush=True)

        ok, first=cap.read()
        if not ok:
            cap.release()
            return self._empty_result("Empty video", output_path)
        h0,w0=first.shape[:2]
        out_w=int(w0*(scale if scale>0 else 1.0)); out_h=int(h0*(scale if scale>0 else 1.0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out=None
        if output_path:
            fourcc=cv2.VideoWriter_fourcc(*'mp4v')
            out=cv2.VideoWriter(output_path,fourcc,effective_fps,(out_w,out_h))

        frame_idx=0
        next_process_frame=1  # נחליט דינמית כמה לדלג
        rep_baseline_head_y=None
        last_log=time.time()

        with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ret,frame_orig=cap.read()
                if not ret: break
                frame_idx+=1

                if frame_idx < next_process_frame:
                    continue

                frame=cv2.resize(frame_orig,(out_w,out_h),interpolation=cv2.INTER_LINEAR)
                image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=pose.process(image)

                elbow_angle=None; head_y=None

                if not results.pose_landmarks:
                    self.ascent_live=0.0
                    step_now=BASE_FRAME_SKIP_IDLE
                    if self.rt_fb_hold>0: self.rt_fb_hold-=1
                    if overlay_enabled:
                        frame=draw_overlay(frame,reps=self.rep_count,feedback=(self.rt_fb_msg if self.rt_fb_hold>0 else None),ascent_pct=self.ascent_live)
                    if out is not None: out.write(frame)
                    next_process_frame=frame_idx+step_now
                    if time.time()-last_log>0.3:
                        print(f"[PULLUP] f{frame_idx} reps={self.rep_count} (no landmarks) step={step_now}", flush=True)
                        last_log=time.time()
                    continue

                lms=results.pose_landmarks.landmark
                if self.side_locked is None:
                    self.side_locked=_pick_side(lms)
                    print(f"[PULLUP] side locked = {self.side_locked}", flush=True)
                side=self.side_locked

                sh_idx=POSE_IDXS["right_shoulder"] if side=="right" else POSE_IDXS["left_shoulder"]
                el_idx=POSE_IDXS["right_elbow"]    if side=="right" else POSE_IDXS["left_elbow"]
                wr_idx=POSE_IDXS["right_wrist"]    if side=="right" else POSE_IDXS["left_wrist"]

                nose=_get_xy(lms,POSE_IDXS["nose"],out_w,out_h)
                shld=_get_xy(lms,sh_idx,out_w,out_h)
                elbw=_get_xy(lms,el_idx,out_w,out_h)
                wrst=_get_xy(lms,wr_idx,out_w,out_h)

                if _safe_vis(nose[2],shld[2],elbw[2],wrst[2],min_v=0.4):
                    head_y=nose[1]/out_h
                    elbow_angle=_angle(shld[:2],elbw[:2],wrst[:2])
                    if self.baseline_head_y_global is None:
                        self.baseline_head_y_global=head_y

                self.ascent_live=float(np.clip((self.baseline_head_y_global - head_y) if (self.baseline_head_y_global is not None and head_y is not None) else 0.0, 0.0, 1.0))

                head_vel=0.0
                if self.last_head_y is not None and head_y is not None:
                    head_vel=head_y - self.last_head_y  # שלילי = עולה

                # ========= ספירה =========
                if elbow_angle is not None and head_y is not None:
                    if not self.in_rep and rep_baseline_head_y is None:
                        rep_baseline_head_y=head_y

                    if not self.in_rep:
                        if self._confirm_start(elbow_angle, head_vel):
                            self.in_rep=True; self.min_head_y_in_rep=head_y
                            self.seen_top_frames=0; self.rep_started_elbow=elbow_angle
                            print(f"[PULLUP] rep START f{frame_idx} | elbow={elbow_angle:.1f} | head_vel={head_vel:.4f}", flush=True)
                    else:
                        self.min_head_y_in_rep = head_y if self.min_head_y_in_rep is None else min(self.min_head_y_in_rep, head_y)

                        if self._confirm_top(elbow_angle, head_y, rep_baseline_head_y):
                            self.seen_top_frames+=1
                        else:
                            self.seen_top_frames=0

                        if self.seen_top_frames>=HEAD_TOP_STICK_FRAMES:
                            ascent_peak=max(0.0,(self.baseline_head_y_global or head_y)-(self.min_head_y_in_rep or head_y))
                            penalty=0.0; fb=None
                            if ascent_peak<0.08: fb="Aim for chin over the bar"; penalty+=2.5
                            elif ascent_peak<HEAD_MIN_ASCENT: fb="Aim for chin over the bar"; penalty+=2.0
                            if self.rep_started_elbow is not None and self.rep_started_elbow < ELBOW_BOTTOM_THRESHOLD-5:
                                fb = fb or "Try to fully extend at the bottom"; penalty+=1.0
                            score=_round_score_half(10.0 if penalty==0.0 else max(4.0,10.0-min(penalty,6.0)))
                            self.rep_count+=1
                            print(f"[PULLUP] REP {self.rep_count} | peak={ascent_peak:.3f} | top_elbow={elbow_angle:.1f}", flush=True)
                            self.reps_meta.append({"rep_index":self.rep_count,"score":float(score),"score_display":display_half_str(score),
                                                   "feedback":([fb] if fb else []),"ascent_peak":float(ascent_peak),
                                                   "start_elbow":float(self.rep_started_elbow) if self.rep_started_elbow is not None else None,
                                                   "top_elbow":float(elbow_angle)})
                            self.all_scores.append(score)
                            if fb and not self.session_best_feedback: self.session_best_feedback=fb
                            self.in_rep=False; rep_baseline_head_y=None
                            self.min_head_y_in_rep=None; self.rep_started_elbow=None; self.seen_top_frames=0

                # ====== RT-feedback ======
                if self.in_rep:
                    if self.ascent_live<0.03:
                        if self.rt_fb_msg!="Aim for chin over the bar":
                            self.rt_fb_msg="Aim for chin over the bar"; self.rt_fb_hold=self.RT_FB_HOLD_FRAMES
                        else:
                            self.rt_fb_hold=max(self.rt_fb_hold,self.RT_FB_HOLD_FRAMES)
                    else:
                        if self.rt_fb_hold>0: self.rt_fb_hold-=1
                else:
                    self.rt_fb_msg=None; self.rt_fb_hold=0

                # ציור/כתיבה
                if results.pose_landmarks is not None:
                    frame=draw_body_only(frame,results.pose_landmarks.landmark)
                if overlay_enabled:
                    frame=draw_overlay(frame,reps=self.rep_count,feedback=(self.rt_fb_msg if self.rt_fb_hold>0 else None),ascent_pct=self.ascent_live)
                if out is not None: out.write(frame)

                # עדכונים/לוג
                if elbow_angle is not None: self.last_elbow=elbow_angle
                if head_y is not None: self.last_head_y=head_y

                moving=self._movement_detected(elbow_angle, head_y) or self.in_rep
                step_now=BASE_FRAME_SKIP_MOVE if moving else BASE_FRAME_SKIP_IDLE
                next_process_frame=frame_idx + step_now

                if time.time()-last_log>0.3:
                    eh=f"{elbow_angle:.1f}" if elbow_angle is not None else "NA"
                    hy=f"{head_y:.3f}" if head_y is not None else "NA"
                    print(f"[PULLUP] f{frame_idx} reps={self.rep_count} in_rep={self.in_rep} elbow={eh} head_y={hy} step={step_now}", flush=True)
                    last_log=time.time()

        cap.release(); 
        if out is not None: out.release()

        # faststart
        final_video_path=""
        if output_path:
            encoded=output_path.replace(".mp4","_encoded.mp4")
            try:
                subprocess.run(["ffmpeg","-y","-i",output_path,"-c:v","libx264","-preset","fast","-movflags","+faststart","-pix_fmt","yuv420p",encoded],
                               check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(output_path) and os.path.exists(encoded): os.remove(output_path)
                final_video_path = encoded if os.path.exists(encoded) else (output_path if os.path.exists(output_path) else "")
            except Exception:
                final_video_path = output_path if os.path.exists(output_path) else ""

        avg=float(np.mean(self.all_scores)) if self.all_scores else 0.0
        technique_score=round(round(avg*2)/2,2)
        good_reps=sum(1 for s in self.all_scores if s>=9.5); bad_reps=max(0,self.rep_count-good_reps)
        feedback_list=[self.session_best_feedback] if self.session_best_feedback else ["Great form! Keep it up 💪"]
        tips=["Slow down the lowering phase to maximize hypertrophy"]

        return {"squat_count":int(self.rep_count),
                "technique_score":float(technique_score),
                "technique_score_display":display_half_str(technique_score),
                "technique_label":score_label(technique_score),
                "good_reps":int(good_reps),"bad_reps":int(bad_reps),
                "feedback":feedback_list,"tips":tips,"reps":self.reps_meta,
                "video_path":final_video_path,"feedback_path":"pullup_feedback.txt"}

    def _empty_result(self,msg,output_path):
        return {"squat_count":0,"technique_score":0.0,
                "technique_score_display":display_half_str(0.0),
                "technique_label":score_label(0.0),
                "good_reps":0,"bad_reps":0,"feedback":[msg],"tips":[],
                "reps":[], "video_path":output_path or "","feedback_path":"pullup_feedback.txt"}

# ====== API ======
def run_pullup_analysis(input_path, frame_skip=3, scale=0.4, output_path=None, overlay_enabled=True):
    try: cv2.setNumThreads(1)
    except Exception: pass
    t0=time.time(); print(f"[PULLUP] start | input={input_path} | skip_idle={frame_skip} | scale={scale}", flush=True)
    analyzer=PullUpAnalyzer()
    res=analyzer.process(input_path=input_path, output_path=output_path, frame_skip=frame_skip, scale=scale, overlay_enabled=overlay_enabled)
    res["elapsed_sec"]=round(time.time()-t0,3)
    if output_path and "video_path" not in res: res["video_path"]=output_path
    print(f"[PULLUP] result: reps={res.get('squat_count')} | score={res.get('technique_score_display')} | elapsed={res['elapsed_sec']}s", flush=True)
    return res

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--input","-i",required=True)
    ap.add_argument("--output","-o",default="")
    ap.add_argument("--scale",type=float,default=0.4)
    ap.add_argument("--skip",type=int,default=3)
    ap.add_argument("--no-overlay",action="store_true")
    args=ap.parse_args()
    result=run_pullup_analysis(input_path=args.input, frame_skip=args.skip, scale=args.scale, output_path=(args.output if args.output else None), overlay_enabled=(not args.no_overlay))
    print(json.dumps(result, ensure_ascii=False, indent=2))
