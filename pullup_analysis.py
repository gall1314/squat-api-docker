# -*- coding: utf-8 -*-
# pullup_analysis.py — keep original rep counting (27/27) + softened "higher" feedback
# ספירה נשארת 1:1 כמו אצלך; רק הפידבק "Go a bit higher" רוכך.
# "חזרה טובה" = חזרה בלי שום טיפ במחזור.

import os, cv2, math, numpy as np, subprocess
from PIL import ImageFont, ImageDraw, Image

# ---------- Styles ----------
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

# ---------- MediaPipe ----------
try:
    import mediapipe as mp
    mp_pose=mp.solutions.pose
except Exception:
    mp_pose=None

# ---------- Helpers ----------
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

# ---------- Draw body-only (ללא פנים) ----------
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

def _dyn_thickness(h):
    return max(2,int(round(h*0.002))), max(3,int(round(h*0.004)))

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

# ---------- Params (ספירה מקורית) ----------
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

# ---- Feedback cues & weights ----
FB_CUE_HIGHER="Go a bit higher (chin over bar)"
FB_CUE_SWING ="Reduce body swing (no kipping)"
FB_CUE_BOTTOM="Fully extend arms at bottom"
FB_WEIGHTS={FB_CUE_HIGHER:0.5, FB_CUE_SWING:0.5, FB_CUE_BOTTOM:0.5}
FB_DEFAULT_WEIGHT=0.5; PENALTY_MIN_IF_ANY=0.5

SWING_THR=0.012; SWING_MIN_STREAK=3
BOTTOM_EXT_MIN_ANGLE=155.0; BOTTOM_HYST_DEG=3.0; BOTTOM_FAIL_MIN_REPS=2
DEBUG_ONBAR=bool(int(os.getenv("DEBUG_ONBAR","0")))

# ---- Face-vs-bar ONLY for the HIGHER feedback (does NOT affect counting) ----
CHIN_BAR_MARGIN   = float(os.getenv("CHIN_BAR_MARGIN", "0.006"))  # סף "עבר" לסנטר/פה
EYE_BAR_MARGIN    = float(os.getenv("EYE_BAR_MARGIN",  "0.012"))  # סף "עבר" לעיניים (קשוח יותר)
# "קרוב מספיק" כדי לא לתת טיפ (מרכך):
CHIN_NEAR_EXTRA   = float(os.getenv("CHIN_NEAR_EXTRA","0.004"))   # מרווח נוסף לסנטר לא לקבל טיפ
EYE_NEAR_RELAX    = float(os.getenv("EYE_NEAR_RELAX","0.003"))    # הקלה לעיניים לא לקבל טיפ
BAR_EMA_ALPHA     = float(os.getenv("BAR_EMA_ALPHA",   "0.30"))
MOUTH_VIS_THR     = float(os.getenv("MOUTH_VIS_THR",   "0.50"))
EYE_VIS_THR       = float(os.getenv("EYE_VIS_THR",     "0.50"))

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
    rep_count=0; good_reps=0; bad_reps=0; rep_reports=[]; all_scores=[]

    # landmarks
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

    # state
    elbow_ema=None; head_ema=None; head_prev=None; baseline_head_y_global=None
    asc_base_head=None; allow_new_peak=True; last_peak_frame=-10**9
    cycle_peak_ascent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False

    onbar=False; onbar_streak=0; offbar_streak=0; prev_torso_cx=None
    offbar_frames_since_any_rep=0; nopose_frames_since_any_rep=0

    session_feedback=set(); rt_fb_msg=None; rt_fb_hold=0
    swing_streak=0; swing_already_reported=False
    bottom_already_reported=False; bottom_phase_max_elbow=None; bottom_fail_count=0

    # per-cycle tips
    cycle_tip_higher=False; cycle_tip_swing=False; cycle_tip_bottom=False

    OFFBAR_STOP_FRAMES=sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES=sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES=sec_to_frames(0.8)

    REARM_DESCENT_EFF=max(RESET_DESCENT*0.60, 0.012)

    # face vs bar (feedback only)
    bar_y_ema=None
    cycle_face_pass=False        # עבר קו־מוט (מחמיר)
    cycle_face_near=False        # קרוב מספיק → לא נותנים טיפ

    with mp_pose.Pose(model_complexity=model_complexity, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % max(1,frame_skip) != 0: continue

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

            # כניסה/יציאה מ-onbar (כמו במקור)
            if (not onbar) and onbar_streak>=ONBAR_MIN_FRAMES:
                onbar=True
                asc_base_head=None; allow_new_peak=True; swing_streak=0
                # reset cycle
                cycle_peak_ascent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_tip_higher=False; cycle_tip_swing=False; cycle_tip_bottom=False
                cycle_face_pass=False;  cycle_face_near=False

            if onbar and offbar_streak>=OFFBAR_MIN_FRAMES:
                # פוסט-הוק ספירה (מקורי)
                if (not counted_this_cycle) and (cycle_peak_ascent>=HEAD_MIN_ASCENT) and (cycle_min_elbow<=ELBOW_TOP_ANGLE):
                    # קובעים האם היה טיפ במחזור (חזרה טובה = בלי טיפ)
                    rep_has_tip = cycle_tip_higher or cycle_tip_swing or cycle_tip_bottom
                    rep_penalty = 0.5 if rep_has_tip else 0.0
                    _count_rep(rep_reports,rep_count,rep_penalty,cycle_min_elbow,
                               asc_base_head if asc_base_head is not None else head_y,
                               baseline_head_y_global-cycle_peak_ascent if baseline_head_y_global is not None else head_y,
                               all_scores, rep_has_tip)
                    rep_count+=1
                    if rep_has_tip: bad_reps+=1
                    else: good_reps+=1
                onbar=False; offbar_frames_since_any_rep=0
                asc_base_head=None; cycle_peak_ascent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False

            if (not onbar) and rep_count>0:
                offbar_frames_since_any_rep+=1
                if offbar_frames_since_any_rep>=OFFBAR_STOP_FRAMES: break

            head_vel=0.0 if head_prev is None else (head_y-head_prev)
            cur_rt=None

            if onbar and vis_strict_ok:
                # --------- REP COUNTING (מקורי) ----------
                if asc_base_head is None:
                    if head_vel<-abs(HEAD_VEL_UP_TINY):
                        asc_base_head=head_y
                        cycle_peak_ascent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                else:
                    cycle_peak_ascent=max(cycle_peak_ascent,(asc_base_head-head_y))
                    cycle_min_elbow=min(cycle_min_elbow,elbow_angle)

                    reset_by_desc=(asc_base_head is not None) and ((head_y-asc_base_head)>=RESET_DESCENT)
                    reset_by_elb =(elbow_angle>=RESET_ELBOW)
                    if reset_by_desc or reset_by_elb:
                        if (not counted_this_cycle) and (cycle_peak_ascent>=HEAD_MIN_ASCENT) and (cycle_min_elbow<=ELBOW_TOP_ANGLE):
                            rep_has_tip = cycle_tip_higher or cycle_tip_swing or cycle_tip_bottom
                            rep_penalty = 0.5 if rep_has_tip else 0.0
                            _count_rep(rep_reports,rep_count,rep_penalty,cycle_min_elbow,
                                       asc_base_head,
                                       baseline_head_y_global-cycle_peak_ascent if baseline_head_y_global is not None else head_y,
                                       all_scores, rep_has_tip)
                            rep_count+=1
                            if rep_has_tip: bad_reps+=1
                            else: good_reps+=1
                        asc_base_head=head_y
                        cycle_peak_ascent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                        allow_new_peak=True; bottom_already_reported=False; bottom_phase_max_elbow=None
                        # reset tips per תנועה חדשה
                        cycle_tip_higher=False; cycle_tip_swing=False; cycle_tip_bottom=False
                        cycle_face_pass=False;  cycle_face_near=False

                ascent_amt=0.0 if asc_base_head is None else (asc_base_head-head_y)

                at_top=(elbow_angle<=ELBOW_TOP_ANGLE) and (ascent_amt>=HEAD_MIN_ASCENT)
                raw_top=(raw_elbow_min<=(ELBOW_TOP_ANGLE+4.0)) and (ascent_amt>=HEAD_MIN_ASCENT*0.92)
                at_top=at_top or raw_top
                can_cnt=(frame_idx - last_peak_frame) >= REFRACTORY_FRAMES

                if at_top and allow_new_peak and can_cnt and (not counted_this_cycle):
                    rep_has_tip = cycle_tip_higher or cycle_tip_swing or cycle_tip_bottom
                    rep_penalty = 0.5 if rep_has_tip else 0.0
                    _count_rep(rep_reports,rep_count,rep_penalty,elbow_angle,
                               asc_base_head if asc_base_head is not None else head_y, head_y,
                               all_scores, rep_has_tip)
                    rep_count+=1
                    if rep_has_tip: bad_reps+=1
                    else: good_reps+=1
                    last_peak_frame=frame_idx; allow_new_peak=False; counted_this_cycle=True
                    bottom_already_reported=False; bottom_phase_max_elbow=max(raw_elbow_L,raw_elbow_R)

                if (allow_new_peak is False) and (last_peak_frame>0):
                    if head_prev is not None and (head_y - head_prev) > 0 and (asc_base_head is not None):
                        if (head_y - (asc_base_head - cycle_peak_ascent)) >= REARM_DESCENT_EFF:
                            allow_new_peak=True
                            bottom_already_reported=False
                            bottom_phase_max_elbow=None

                # --------- FEEDBACK ONLY: פנים מול קו-מוט (מרוכך) ----------
                # קו-המוט מהפרקים (y קטן = גבוה)
                cur_bar_y = min(lms[LW].y, lms[RW].y)
                bar_y_ema = cur_bar_y if bar_y_ema is None else (BAR_EMA_ALPHA*cur_bar_y + (1 - BAR_EMA_ALPHA)*bar_y_ema)
                bar_y_eff = bar_y_ema if bar_y_ema is not None else cur_bar_y

                # פה/סנטר או עיניים
                mouth_ok = (lms[MOUTH_L].visibility>=MOUTH_VIS_THR and lms[MOUTH_R].visibility>=MOUTH_VIS_THR)
                eyes_ok  = (lms[LEFT_EYE].visibility>=EYE_VIS_THR and lms[RIGHT_EYE].visibility>=EYE_VIS_THR)
                mouth_y = (lms[MOUTH_L].y + lms[MOUTH_R].y)*0.5 if mouth_ok else None
                eye_min_y = min(lms[LEFT_EYE].y, lms[RIGHT_EYE].y) if eyes_ok else None

                eye_margin = max(EYE_BAR_MARGIN, CHIN_BAR_MARGIN + 0.002)

                passed_by_chin = (mouth_y is not None) and (mouth_y <= bar_y_eff + CHIN_BAR_MARGIN)
                passed_by_eyes = (mouth_y is None) and (eye_min_y is not None) and (eye_min_y <= bar_y_eff - eye_margin)
                # "קרוב מספיק" → לא נותנים טיפ:
                near_by_chin  = (mouth_y is not None) and (mouth_y <= bar_y_eff + (CHIN_BAR_MARGIN + CHIN_NEAR_EXTRA))
                # לעיניים מקלים קצת:
                near_by_eyes  = (mouth_y is None) and (eye_min_y is not None) and (eye_min_y <= bar_y_eff - max(0.001, eye_margin - EYE_NEAR_RELAX))

                if passed_by_chin or passed_by_eyes:
                    cycle_face_pass = True
                if near_by_chin or near_by_eyes:
                    cycle_face_near = True

                # טיפ "higher" מרוכך: מציגים/מסמנים רק אם לא עברנו וגם לא היינו "קרובים מספיק"
                if (asc_base_head is not None) and (ascent_amt >= HEAD_MIN_ASCENT*0.50) and (head_vel < -abs(HEAD_VEL_UP_TINY)):
                    if (not cycle_face_pass) and (not cycle_face_near):
                        if not cycle_tip_higher:
                            session_feedback.add(FB_CUE_HIGHER)
                            cycle_tip_higher=True
                            cur_rt = FB_CUE_HIGHER  # להצגה מידית

                # Swing כרגיל
                if torso_dx_norm>SWING_THR: swing_streak+=1
                else: swing_streak=max(0,swing_streak-1)
                if (cur_rt is None) and (swing_streak>=SWING_MIN_STREAK) and (not cycle_tip_swing):
                    session_feedback.add(FB_CUE_SWING); cur_rt=FB_CUE_SWING; cycle_tip_swing=True

            else:
                asc_base_head=None; allow_new_peak=True
                swing_streak=0; bottom_phase_max_elbow=None

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

            if head_y is not None: head_prev=head_y

    # EOF post-hoc (כמו במקור, עם דירוג per-rep טוב/לא-טוב)
    if onbar and (not counted_this_cycle) and (cycle_peak_ascent>=HEAD_MIN_ASCENT) and (cycle_min_elbow<=ELBOW_TOP_ANGLE):
        rep_has_tip = cycle_tip_higher or cycle_tip_swing or cycle_tip_bottom
        rep_penalty = 0.5 if rep_has_tip else 0.0
        _count_rep(rep_reports,rep_count,rep_penalty,cycle_min_elbow,
                   asc_base_head if asc_base_head is not None else (baseline_head_y_global or 0.0),
                   (baseline_head_y_global - cycle_peak_ascent) if baseline_head_y_global is not None else (baseline_head_y_global or 0.0),
                   all_scores, rep_has_tip)
        rep_count+=1
        if rep_has_tip: bad_reps+=1
        else: good_reps+=1

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    # Session score כמו קודם
    if rep_count==0: technique_score=0.0
    else:
        penalty = max(PENALTY_MIN_IF_ANY, sum(FB_WEIGHTS.get(m,FB_DEFAULT_WEIGHT) for m in session_feedback)) if session_feedback else 0.0
        technique_score=_half_floor10(max(0.0,10.0-penalty))

    fb_list=sorted(session_feedback) if session_feedback else ["Great form! Keep it up 💪"]
    try:
        with open(feedback_path,"w",encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            if fb_list:
                f.write("Feedback:\n"); [f.write(f"- {ln}\n") for ln in fb_list]
    except Exception: pass

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

    return {
        "squat_count": int(rep_count),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": fb_list,
        "form_tip": (fb_list[0] if fb_list else "Great form! Keep it up 💪"),
        "tips": [],
        "reps": rep_reports,
        "video_path": final_path,
        "feedback_path": feedback_path
    }

# ---------- helpers ----------
def _count_rep(rep_reports, rep_count, rep_penalty, top_elbow, ascent_from, peak_head_y, all_scores, rep_has_tip):
    # ציון חזרה: 10 אם אין טיפ במחזור, אחרת 9.5 (חצי נקודה)
    rep_score = 10.0 if not rep_has_tip else 9.5
    all_scores.append(rep_score)
    rep_reports.append({
        "rep_index": int(rep_count+1),
        "score": float(rep_score),
        "good": bool(not rep_has_tip),
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
        "feedback": [msg], "form_tip": msg,
        "tips": [], "reps": [], "video_path": "", "feedback_path": feedback_path
    }

def run_analysis(*args, **kwargs):
    return run_pullup_analysis(*args, **kwargs)

