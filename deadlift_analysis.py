# -*- coding: utf-8 -*-
# deadlift_analysis.py — Full fast-first pipeline with artifact + original-quality render and full overlay
#
# Public entry (what app.py imports):
#   run_deadlift_analysis(
#       video_path, frame_skip=6, scale=0.40,
#       output_path=None, return_video=True, model_complexity=1, yolo_onnx=None
#   ) -> dict with keys your FE expects:
#       exercise, squat_count, rep_count, technique_score, technique_score_display,
#       technique_label, good_reps, bad_reps, reps, feedback, video_path
#
# Internal:
#   analyze_deadlift_fast(...) -> artifact (timeline+per-rep report)
#   render_deadlift_video(artifact, out_path, preset="medium", crf=18) -> original-quality mp4 with overlay

import os, cv2, math, uuid, subprocess
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional

# ---------- perf hygiene ----------
try:
    cv2.setNumThreads(1)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------- optional onnxruntime (YOLO) ----------
try:
    import onnxruntime as ort
    _ORT = True
except Exception:
    _ORT = False

mp_pose = mp.solutions.pose

# ============================== helpers ==============================
def _score_label(s: float) -> str:
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def _half_str(x: float) -> str:
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

class EMA:
    def __init__(self, alpha=0.25, init=None):
        self.a=float(alpha); self.v=init; self.ready=init is not None
    def update(self, x):
        x=float(x)
        if not self.ready: self.v=x; self.ready=True
        else: self.v=self.a*x+(1-self.a)*self.v
        return self.v

# ====================== geometry / back curvature ====================
def analyze_back_curvature(shoulder, hip, head_like, threshold=0.04):
    # signed distance of head-like point from the shoulder-hip line (negative=inward/rounded)
    line_vec = hip - shoulder
    nrm = np.linalg.norm(line_vec) + 1e-9
    u = line_vec / nrm
    proj_len = np.dot(head_like - shoulder, u)
    proj = shoulder + proj_len * u
    off = head_like - proj
    signed = float(np.sign(off[1]) * -1 * np.linalg.norm(off))  # inward (rounded) => negative
    return signed, signed < -threshold

# ============================== Kalman ===============================
class _KF:
    def __init__(self, q=1e-3, r=6e-3):
        self.x = np.zeros((4,1), dtype=float) # x,y,vx,vy
        self.P = np.eye(4)*1.0
        self.Q = np.eye(4)*q
        self.R = np.eye(2)*r
        self.F = np.eye(4)
        self.H = np.zeros((2,4)); self.H[0,0]=1; self.H[1,1]=1
        self.inited=False
    def predict(self, dt):
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        z = np.array(z, dtype=float).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        maha2 = float(y.T @ np.linalg.inv(S) @ y)
        return maha2

class KalmanLegTracker:
    LEG_VIS_THR    = 0.55
    LEG_MAX_MAH    = 6.0
    LEG_MISS_FLOOR = 6
    LEG_SIDE_XSPAN = 0.25
    def __init__(self, side="right"):
        PL = mp_pose.PoseLandmark
        self.side=side
        self.hip  = PL.RIGHT_HIP.value   if side=="right" else PL.LEFT_HIP.value
        self.knee = PL.RIGHT_KNEE.value  if side=="right" else PL.LEFT_KNEE.value
        self.ankle= PL.RIGHT_ANKLE.value if side=="right" else PL.LEFT_ANKLE.value
        self.heel = PL.RIGHT_HEEL.value  if side=="right" else PL.LEFT_HEEL.value
        self.foot = PL.RIGHT_FOOT_INDEX.value if side=="right" else PL.LEFT_FOOT_INDEX.value
        self.idxs=[self.knee,self.ankle,self.heel,self.foot]
        self.kf  ={i:_KF() for i in self.idxs}
        self.miss={i:0 for i in self.idxs}
        self.floor_y_med=None; self.floor_hist=[]
    @staticmethod
    def _near_hands(lm, p):
        PL = mp_pose.PoseLandmark
        for h in (PL.RIGHT_ELBOW.value, PL.RIGHT_WRIST.value, PL.LEFT_ELBOW.value, PL.LEFT_WRIST.value):
            if lm[h].visibility>0.5:
                if math.hypot(p[0]-lm[h].x, p[1]-lm[h].y) < 0.06: return True
        return False
    def _update_floor(self, lm):
        cand=[]
        for idx in (self.ankle,self.heel,self.foot):
            if lm[idx].visibility>0.6: cand.append(lm[idx].y)
        if cand:
            y=sorted(cand)[-1]
            self.floor_hist.append(y)
            if len(self.floor_hist)>25: self.floor_hist.pop(0)
            self.floor_y_med=float(np.median(self.floor_hist))
    def update(self, lm, dt, occl=None):
        self._update_floor(lm)
        out={}
        hip_x = lm[self.hip].x
        def inside(p):
            if not occl: return False
            x,y=p
            for (x1,y1,x2,y2) in occl:
                dx=(x2-x1)*0.08; dy=(y2-y1)*0.08
                if (x1-dx)<=x<=(x2+dx) and (y1-dy)<=y<=(y2+dy): return True
            return False
        for idx in self.idxs:
            kf=self.kf[idx]
            if not kf.inited:
                if lm[idx].visibility>=self.LEG_VIS_THR:
                    kf.x[:2,0]=[lm[idx].x,lm[idx].y]; kf.x[2:,0]=[0.0,0.0]; kf.inited=True
                else: continue
            kf.predict(dt)
            meas_ok=False
            if lm[idx].visibility>=self.LEG_VIS_THR:
                z=(lm[idx].x,lm[idx].y)
                if (not self._near_hands(lm,z)) and (abs(z[0]-hip_x)<=self.LEG_SIDE_XSPAN) and (not inside(z)):
                    maha2=kf.update(z)
                    if maha2 <= (self.LEG_MAX_MAH**2): meas_ok=True
            if meas_ok: self.miss[idx]=0
            else:
                self.miss[idx]+=1
                if self.miss[idx]>=self.LEG_MISS_FLOOR and self.floor_y_med is not None:
                    kf.x[1,0]=0.8*kf.x[1,0]+0.2*self.floor_y_med
            out[idx]=(float(kf.x[0,0]), float(kf.x[1,0]))
        return out

# ========================= bar proxy via wrists ======================
class BarYTracker:
    WR_THR=0.55; MIN_W=0.10; MAX_W=0.55; DRIFT=0.12
    def __init__(self):
        self.kf=_KF(q=8e-4, r=9e-4); self.ema_y=EMA(0.20); self.valid=False
        self.top_ema=EMA(0.05); self.w0=None
    def reset_rep(self): self.w0=None
    def update(self, lm):
        PL=mp_pose.PoseLandmark; L=lm[PL.LEFT_WRIST.value]; R=lm[PL.RIGHT_WRIST.value]
        if L.visibility<self.WR_THR or R.visibility<self.WR_THR: self.valid=False; return None, False
        w=abs(R.x-L.x)
        if not (self.MIN_W<=w<=self.MAX_W): self.valid=False; return None, False
        if self.w0 is None: self.w0=w
        elif abs(w-self.w0) > self.DRIFT*max(1e-3,self.w0): self.valid=False; return None, False
        cx=0.5*(L.x+R.x); y=0.5*(L.y+R.y)
        if not self.kf.inited:
            self.kf.x[:2,0]=[cx,y]; self.kf.x[2:,0]=[0.0,0.0]; self.kf.inited=True
        else:
            self.kf.predict(1/25.0); self.kf.update([cx,y])
        y_f=self.ema_y.update(float(self.kf.x[1,0])); self.top_ema.update(y_f)
        self.valid=True; return y_f, True
    def top_ref(self): return self.top_ema.v if self.top_ema.ready else None

# ============================== YOLO opt =============================
class YoloOccluder:
    def __init__(self, onnx_path, providers=None, input_size=640, conf=0.25, iou=0.45, allow=None):
        if not _ORT: raise RuntimeError("onnxruntime not available")
        if not os.path.exists(onnx_path): raise FileNotFoundError(onnx_path)
        self.sess=ort.InferenceSession(onnx_path, providers=providers or ort.get_available_providers())
        self.inp=self.sess.get_inputs()[0].name; self.out=self.sess.get_outputs()[0].name
        self.imgsz=int(input_size); self.conf=float(conf); self.iou=float(iou); self.allow=allow
    @staticmethod
    def _lb(img, new=640, color=(114,114,114)):
        h,w=img.shape[:2]
        if isinstance(new,int): new=(new,new)
        r=min(new[0]/h, new[1]/w)
        new_unpad=(int(round(w*r)), int(round(h*r)))
        dw,dh = new[1]-new_unpad[0], new[0]-new_unpad[1]
        dw,dh = dw//2, dh//2
        img=cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img=cv2.copyMakeBorder(img, dh,dh,dw,dw, cv2.BORDER_CONSTANT, value=color)
        return img,r,(dw,dh)
    @staticmethod
    def _nms(boxes, scores, iou):
        if len(boxes)==0: return []
        b=boxes.astype(np.float32); x1,y1,x2,y2=b[:,0],b[:,1],b[:,2],b[:,3]
        area=(x2-x1+1)*(y2-y1+1); order=scores.argsort()[::-1]; keep=[]
        while order.size>0:
            i=order[0]; keep.append(i)
            xx1=np.maximum(x1[i], x1[order[1:]])
            yy1=np.maximum(y1[i], y1[order[1:]])
            xx2=np.minimum(x2[i], x2[order[1:]])
            yy2=np.minimum(y2[i], y2[order[1:]])
            w=np.maximum(0, xx2-xx1+1); h=np.maximum(0, yy2-yy1+1)
            inter=w*h; ovr=inter/(area[i]+area[order[1:]]-inter+1e-9)
            inds=np.where(ovr<=iou)[0]; order=order[inds+1]
        return keep
    def infer(self, bgr):
        h0,w0=bgr.shape[:2]
        img,r,(dw,dh)=self._lb(bgr, self.imgsz)
        rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        rgb=np.transpose(rgb,(2,0,1))[None]
        out=self.sess.run([self.out],{self.inp:rgb})[0]
        o=out[0] if out.ndim==3 and out.shape[0]==1 else out
        if o.shape[-1] < 6: return []
        if o.shape[0] in (84,85):
            xywh=o[0:4,:].T; cls=o[5:,:].T if o.shape[0]>=85 else o[4:,:].T
        else:
            xywh=o[:,0:4];   cls=o[:,5:] if o.shape[1]>=85 else o[:,4:]
        conf=cls.max(axis=1); cid=np.argmax(cls,axis=1)
        m=conf>=self.conf
        xywh,conf,cid=xywh[m],conf[m],cid[m]
        if self.allow is not None:
            sel=np.isin(cid, list(self.allow)); xywh,conf=xywh[sel],conf[sel]
        if len(xywh)==0: return []
        xyxy=np.zeros_like(xywh); xyxy[:,0]=xywh[:,0]-xywh[:,2]/2; xyxy[:,1]=xywh[:,1]-xywh[:,3]/2
        xyxy[:,2]=xywh[:,0]+xywh[:,2]/2; xyxy[:,3]=xywh[:,1]+xywh[:,3]/2
        keep=self._nms(xyxy,conf,self.iou); xyxy=xyxy[keep]
        xyxy[:,[0,2]]-=dw; xyxy[:,[1,3]]-=dh; xyxy/=r
        xyxy[:,0]=np.clip(xyxy[:,0],0,w0-1); xyxy[:,2]=np.clip(xyxy[:,2],0,w0-1)
        xyxy[:,1]=np.clip(xyxy[:,1],0,h0-1); xyxy[:,3]=np.clip(xyxy[:,3],0,h0-1)
        boxes=[]
        for x1,y1,x2,y2 in xyxy:
            boxes.append((float(x1)/w0, float(y1)/h0, float(x2)/w0, float(y2)/h0))
        return boxes

# ============================== overlay ==============================
def _draw_donut(frame, cx, cy, radius, thick, pct):
    pct=float(np.clip(pct,0.0,1.0))
    ring_bg=(70,70,70); color=(40,200,80)
    cv2.circle(frame,(cx,cy),radius,ring_bg,thick,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(radius,radius),0,-90,-90+int(360*pct),color,thick,cv2.LINE_AA)

def draw_overlay_like_now(frame, reps_done, pct, feedback_text=None):
    # Reps (top-left)
    txt=f"Reps: {reps_done}"
    (tw,th),_=cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,1.0,2)
    pad=10; cv2.rectangle(frame,(0,0),(tw+2*pad, th+2*pad),(0,0,0),-1)
    cv2.putText(frame,txt,(pad, th+pad-2), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2, cv2.LINE_AA)
    # Donut (top-right)
    h,w=frame.shape[:2]; ref_h=max(int(h*0.06), int(28*1.6))
    r=int(ref_h*0.72); thick=max(3,int(r*0.28)); m=12
    cx=w-m-r; cy=max(ref_h+r//8, r+thick//2+2)
    _draw_donut(frame, cx, cy, r, thick, pct)
    cv2.putText(frame,"DEPTH",(cx-30, cy-(thick+10)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1, cv2.LINE_AA)
    cv2.putText(frame,f"{int(100*np.clip(pct,0,1))}%",(cx-18, cy+(thick+16)), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1, cv2.LINE_AA)
    # Feedback bottom
    if feedback_text:
        maxw=int(w*0.9); words=feedback_text.split(); lines=[]; cur=""
        for wd in words:
            test=(cur+" "+wd).strip(); (tw2,_),_=cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            if tw2<=maxw: cur=test
            else:
                if cur: lines.append(cur); cur=wd
                if len(lines)==2: break
        if cur and len(lines)<2: lines.append(cur)
        line_h=24; block_h=20+line_h*len(lines); y0=h-block_h-10
        cv2.rectangle(frame,(0,y0),(w,y0+block_h),(0,0,0),-1)
        y=y0+20
        for ln in lines:
            (tw2,th2),_=cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            x=max(10,(w-tw2)//2)
            cv2.putText(frame,ln,(x,y), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2, cv2.LINE_AA)
            y+=line_h

# ============================== ffmpeg ===============================
def _open_ffmpeg_sink(w,h,fps,out_path,preset="medium",crf=18):
    cmd=[
        "ffmpeg","-y",
        "-f","rawvideo","-pix_fmt","bgr24","-s",f"{w}x{h}","-r",str(fps),"-i","pipe:0",
        "-an","-c:v","libx264","-preset",preset,"-crf",str(crf),
        "-movflags","+faststart","-vf","format=yuv420p",
        out_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# ========================== fast analysis ============================
def analyze_deadlift_fast(
    video_path: str,
    work_scale: float = 0.40,
    frame_skip: int = 6,
    model_complexity: int = 1,
    yolo_onnx: Optional[str] = None,
    yolo_input: int = 640,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    yolo_allow: Optional[set] = None
) -> Dict[str, Any]:
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"analysis_id": uuid.uuid4().hex,"video_path":video_path,"summary":{"reps":0,"score":0.0,"display":"0","label":"Needs work"},"reps_report":[],"timeline":[],"meta":{"ok":False,"reason":"Could not open video"}}
    fps_in=cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (n_frames/fps_in) if n_frames>0 else None

    # FSM/thresholds
    HINGE_START=0.08; HINGE_END=0.035
    MIN_BW=10; MIN_TOTAL=12
    ROM_MIN=0.055

    PL=mp_pose.PoseLandmark
    state_TOP, state_DOWN, state_UP = 0,1,2
    state=state_TOP
    last_rep_f = -10**9; fidx=0

    reps=0; reps_report=[]; timeline=[]; scores=[]
    top_ref=None; bottom_est=None
    eff_fps = max(1.0, fps_in/max(1,frame_skip))
    down_frames=up_frames=0
    hinge_min=hinge_max=None

    right_leg=left_leg=None
    bar=BarYTracker()

    det=None
    if yolo_onnx and _ORT and os.path.exists(yolo_onnx):
        try:
            det=YoloOccluder(yolo_onnx, input_size=yolo_input, conf=yolo_conf, iou=yolo_iou, allow=yolo_allow)
        except Exception:
            det=None

    with mp_pose.Pose(model_complexity=model_complexity, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            fidx+=1
            if fidx % frame_skip != 0: continue

            small = cv2.resize(frame,(0,0), fx=work_scale, fy=work_scale) if work_scale!=1.0 else frame
            occl=[]
            if det is not None:
                try: occl=det.infer(small)
                except Exception: occl=[]

            rgb=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res=pose.process(rgb)
            if not res.pose_landmarks: continue
            lm=res.pose_landmarks.landmark

            if right_leg is None or left_leg is None:
                L=lm[PL.LEFT_ANKLE.value].visibility; R=lm[PL.RIGHT_ANKLE.value].visibility
                prefer="left" if L>R else "right"
                right_leg=KalmanLegTracker("right" if prefer=="right" else "left")
                left_leg =KalmanLegTracker("left"  if prefer=="right" else "right")

            shoulder=np.array([lm[PL.RIGHT_SHOULDER.value].x, lm[PL.RIGHT_SHOULDER.value].y], dtype=float)
            hip     =np.array([lm[PL.RIGHT_HIP.value].x,      lm[PL.RIGHT_HIP.value].y],      dtype=float)

            # back curvature quick proxy (head/ear or nose)
            head=None
            for idx in (PL.RIGHT_EAR.value, PL.LEFT_EAR.value, PL.NOSE.value):
                if lm[idx].visibility>0.5:
                    head=np.array([lm[idx].x, lm[idx].y], dtype=float); break
            if head is None: continue
            mid_spine=(shoulder+hip)*0.5*0.4 + head*0.6
            _, rounded = analyze_back_curvature(shoulder, hip, mid_spine)

            delta_x = abs(hip[0]-shoulder[0])
            if top_ref is None: top_ref=float(delta_x)
            else:
                if delta_x < (top_ref*1.4): top_ref=0.9*top_ref+0.1*float(delta_x)

            # bar proxy and legs (for robustness; not used directly for count decisions except end/start heuristics)
            bar_y, bar_ok = bar.update(lm)
            right_leg.update(lm, 1.0/eff_fps, occl)
            left_leg.update (lm, 1.0/eff_fps, occl)

            t_sec = fidx / fps_in

            # FSM
            if state==state_TOP:
                start = (delta_x - top_ref) > HINGE_START
                if (not start) and bar_ok and bar.top_ref() is not None:
                    start = (bar_y - bar.top_ref()) > (ROM_MIN*0.5)
                if start and (fidx - last_rep_f > MIN_BW):
                    state=state_DOWN
                    bottom_est=delta_x
                    hinge_min=hinge_max=delta_x
                    down_frames=up_frames=0
                    bar.reset_rep()
                    rep_start = t_sec

            elif state==state_DOWN:
                bottom_est = max(bottom_est or delta_x, delta_x)
                hinge_min = min(hinge_min, delta_x) if hinge_min is not None else delta_x
                hinge_max = max(hinge_max, delta_x) if hinge_max is not None else delta_x
                down_frames += 1
                if down_frames>=4 and (hinge_max - delta_x) > 0.01:
                    state=state_UP

            elif state==state_UP:
                hinge_min = min(hinge_min, delta_x) if hinge_min is not None else delta_x
                hinge_max = max(hinge_max, delta_x) if hinge_max is not None else delta_x
                up_frames += 1
                end_hinge = (delta_x - top_ref) < HINGE_END
                end_bar   = (bar_ok and bar.top_ref() is not None and abs(bar_y - bar.top_ref()) < (ROM_MIN*0.5))
                if end_hinge or end_bar:
                    long_enough=(down_frames+up_frames)>=MIN_TOTAL
                    rom = hinge_max - hinge_min if (hinge_max is not None and hinge_min is not None) else 0.0
                    enough_rom = rom >= ROM_MIN
                    if long_enough and enough_rom and (fidx - last_rep_f > MIN_BW):
                        reps+=1
                        penalty=0.0; fb=[]
                        if rounded: fb.append("Try to keep your back a bit straighter"); penalty+=1.5
                        if (delta_x - top_ref) > 0.02:
                            fb.append("Try to finish more upright"); penalty+=1.0
                        score = round(max(4.0, 10.0 - penalty) * 2)/2
                        tip = "Keep the bar close and move smoothly"
                        reps_report.append({"rep_index":reps,"score":float(score),"score_display":_half_str(score),"feedback":fb,"tip":tip})
                        timeline.append({"start":rep_start, "end":t_sec})
                        scores.append(score)
                        last_rep_f=fidx
                    state=state_TOP
                    bottom_est=None
                    down_frames=up_frames=0
                    hinge_min=hinge_max=None
                    bar.reset_rep()

    cap.release()
    avg=float(np.mean(scores)) if scores else 0.0
    technique_score=round(round(avg*2)/2,2)
    return {
        "analysis_id": uuid.uuid4().hex,
        "video_path": video_path,
        "summary": {
            "reps": reps,
            "score": technique_score,
            "display": _half_str(technique_score),
            "label": _score_label(technique_score),
        },
        "reps_report": reps_report,
        "timeline": timeline,
        "meta": {
            "ok": True,
            "version": "deadlift_fsm_full_2.0",
            "fps_in": fps_in,
            "work_scale": work_scale,
            "frame_skip": frame_skip,
            "model_complexity": model_complexity,
            "yolo_enabled": bool(det is not None)
        }
    }

# ============================ video render ===========================
def render_deadlift_video(
    artifact: Dict[str,Any],
    out_path: str,
    preset: str = "medium",
    crf: int = 18
) -> Dict[str,Any]:
    video_path=artifact["video_path"]
    timeline=sorted(artifact.get("timeline",[]), key=lambda x:x["start"])
    reps_report=artifact.get("reps_report",[])
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): return {"ok":False,"reason":"Could not open source video"}
    fps=cap.get(cv2.CAP_PROP_FPS) or artifact.get("meta",{}).get("fps_in",25.0)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    sink=_open_ffmpeg_sink(W,H,fps,out_path,preset= preset, crf=crf)

    ends=np.array([r["end"] for r in timeline], dtype=np.float32)
    starts=np.array([r["start"] for r in timeline], dtype=np.float32)
    total=len(timeline)

    last_key=None; last_frame=None
    fidx=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t=fidx/float(fps)

        done=int(np.searchsorted(ends, t, side="right"))
        pct=0.0; fb=None
        if done < total:
            s=starts[done]; e=ends[done]
            if s<=t<=e:
                pct=float((t - s)/max(1e-6, e - s))
                if done < len(reps_report):
                    fbl=reps_report[done].get("feedback",[])
                    tip=reps_report[done].get("tip")
                    fb=" • ".join(fbl[:2]) if fbl else (tip or None)

        key=(done, int(pct*20))
        if key!=last_key or last_frame is None:
            ov=frame.copy()
            draw_overlay_like_now(ov, reps_done=done, pct=pct, feedback_text=fb)
            last_frame=ov; last_key=key

        sink.stdin.write(last_frame.tobytes())
        fidx+=1

    cap.release()
    try:
        sink.stdin.close(); sink.wait()
    except Exception:
        pass
    return {"ok":True, "video_path": out_path, "reps": total}

# ============================== public API ===========================
def run_deadlift_analysis(
    video_path: str,
    frame_skip: int = 6,
    scale: float = 0.40,
    output_path: Optional[str] = None,
    return_video: bool = True,
    model_complexity: int = 1,
    yolo_onnx: Optional[str] = None
) -> Dict[str,Any]:
    """
    Public entry used by app.py.
    - return_video=False → fast analysis only (no video)
    - return_video=True  → render original-quality video with overlay (no re-analysis)
    """
    artifact = analyze_deadlift_fast(
        video_path=video_path,
        work_scale=scale,
        frame_skip=frame_skip,
        model_complexity=model_complexity,
        yolo_onnx=yolo_onnx
    )

    # defensive fetch
    summary = artifact.get("summary") or {}
    reps_report = artifact.get("reps_report") or []
    reps_total = int(summary.get("reps") or 0)
    score_avg = float(summary.get("score") or 0.0)
    score_display = str(summary.get("display") or "0")
    score_label = str(summary.get("label") or "Needs work")

    good_reps = sum(1 for r in reps_report if float(r.get("score") or 0.0) >= 9.5)
    bad_reps  = sum(1 for r in reps_report if float(r.get("score") or 0.0)  < 9.5)

    result = {
        "exercise": "deadlift",
        "squat_count": reps_total,                 # FE expects this key
        "rep_count": reps_total,                   # also keep rep_count for safety
        "technique_score": score_avg,
        "technique_score_display": score_display,
        "technique_label": score_label,
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "reps": reps_report,                       # [{rep_index, score, score_display, feedback[], tip}]
        "feedback": ["Analysis complete"],         # always a non-empty list
        "video_path": ""                           # filled only when rendering
    }

    if not return_video:
        return result

    # produce original-quality video + overlay (no re-analysis)
    if not output_path:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_analyzed.mp4"

    rr = render_deadlift_video(artifact, out_path=output_path, preset="medium", crf=18)
    if rr.get("ok"):
        result["video_path"] = rr.get("video_path") or output_path
    else:
        result["video_path"] = ""
        result["feedback"].append(f"Render failed: {rr.get('reason','unknown')}")

    return result

# Old name compatibility if someone imports analyze_deadlift:
analyze_deadlift = run_deadlift_analysis
