def run_pullup_analysis(video_path,
                        frame_skip=3,                # כמו עכשיו
                        scale=0.4,                   # כמו עכשיו
                        output_path="pullup_analyzed.mp4",
                        feedback_path="pullup_feedback.txt",
                        preserve_quality=False,      # כמו עכשיו
                        encode_crf=None,             # כמו עכשיו
                        # NEW: תאימות לדדליפט
                        return_video=True,
                        fast_mode=None):
    """
    preserve_quality=True  =>  scale=1.0, frame_skip=1, קידוד CRF=18 (אם encode_crf לא סופק).
    fast_mode=True         =>  זהה ל-return_video=False (אין החזרת וידאו; ניתוח מהיר בלבד).
    return_video=False     =>  אין יצירת VideoWriter/אוברליי — רק ניתוח ותוצאות JSON.
    """
    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    # תאימות לשם/אופן הקריאה בדדליפט
    if fast_mode is True:
        return_video = False

    if preserve_quality:
        scale = 1.0
        frame_skip = 1
        if encode_crf is None:
            encode_crf = 18
    else:
        if encode_crf is None:
            encode_crf = 23

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _ret_err("Could not open video", feedback_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_idx = 0

    # Counters
    rep_count=0; good_reps=0; bad_reps=0
    rep_reports=[]; all_scores=[]

    # Dynamic side indices (pre-calc)
    LSH=mp_pose.PoseLandmark.LEFT_SHOULDER.value;  RSH=mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LE =mp_pose.PoseLandmark.LEFT_ELBOW.value;     RE =mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW =mp_pose.PoseLandmark.LEFT_WRIST.value;     RW =mp_pose.PoseLandmark.RIGHT_WRIST.value
    LH =mp_pose.PoseLandmark.LEFT_HIP.value;       RH =mp_pose.PoseLandmark.RIGHT_HIP.value
    NOSE=mp_pose.PoseLandmark.NOSE.value

    def _pick_side_dyn(lms):
        vL=lms[LSH].visibility + lms[LE].visibility + lms[LW].visibility
        vR=lms[RSH].visibility + lms[RE].visibility + lms[RW].visibility
        return ("LEFT", LSH,LE,LW) if vL>=vR else ("RIGHT", RSH,RE,RW)

    # State
    elbow_ema=None; head_ema=None; head_prev=None
    asc_base_head=None; baseline_head_y_global=None
    allow_new_peak=True; last_peak_frame=-99999

    # On/Off-bar
    onbar=False; onbar_streak=0; offbar_streak=0
    prev_torso_cx=None
    offbar_frames_since_any_rep = 0
    nopose_frames_since_any_rep = 0

    # Feedback session collection (מסך ← API/קובץ) וגם לציון
    session_feedback=set()
    rt_fb_msg=None; rt_fb_hold=0

    # Swing state
    swing_streak=0
    swing_already_reported=False
    bottom_already_reported=False

    # ---- NEW: track bottom-phase maximum elbow extension (raw, both arms) ----
    bottom_phase_max_elbow = None
    bottom_fail_count = 0

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    sec_to_frames = lambda s: max(1, int(s * effective_fps))
    OFFBAR_STOP_FRAMES = sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES = sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES  = sec_to_frames(0.8)

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue
            if scale != 1.0:
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            h, w = frame.shape[:2]

            # נפתח כותב רק אם מחזירים וידאו
            if return_video and out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w,h))

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            height_live = 0.0

            if not res.pose_landmarks:
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep + 1) if rep_count>0 else 0
                if rep_count>0 and nopose_frames_since_any_rep >= NOPOSE_STOP_FRAMES:
                    break
                if return_video and out is not None:
                    frame_draw = draw_overlay(frame, reps=rep_count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=0.0)
                    out.write(frame_draw)
                if rt_fb_hold>0: rt_fb_hold-=1
                continue

            nopose_frames_since_any_rep = 0
            lms = res.pose_landmarks.landmark
            side, S,E,W = _pick_side_dyn(lms)

            # Visibility (מחמיר)
            min_vis = min(lms[NOSE].visibility, lms[S].visibility, lms[E].visibility, lms[W].visibility)
            vis_strict_ok = (min_vis >= VIS_THR_STRICT)

            # זוויות/ראשים (לוגיקה בדיוק כפי שהיה)
            head_raw = float(lms[NOSE].y)
            raw_elbow_L = _ang((lms[LSH].x, lms[LSH].y), (lms[LE].x, lms[LE].y), (lms[LW].x, lms[LW].y))
            raw_elbow_R = _ang((lms[RSH].x, lms[RSH].y), (lms[RE].x, lms[RE].y), (lms[RW].x, lms[RW].y))
            raw_elbow = raw_elbow_L if side == "LEFT" else raw_elbow_R

            elbow_ema = _ema(elbow_ema, raw_elbow, ELBOW_EMA_ALPHA)
            head_ema  = _ema(head_ema,  head_raw,  HEAD_EMA_ALPHA)
            head_y = head_ema; elbow_angle = elbow_ema
            if baseline_head_y_global is None: baseline_head_y_global = head_y
            height_live = float(np.clip((baseline_head_y_global - head_y)/max(0.12, HEAD_MIN_ASCENT*1.2), 0.0, 1.0))

            # Torso X (walking/swing)
            torso_cx = np.mean([lms[LSH].x, lms[RSH].x, lms[LH].x, lms[RH].x]) * w
            torso_dx_norm = 0.0 if prev_torso_cx is None else abs(torso_cx - prev_torso_cx)/max(1.0, w)
            prev_torso_cx = torso_cx

            # On/Off-bar gating
            lw_vis = lms[LW].visibility; rw_vis = lms[RW].visibility
            lw_above = (lw_vis >= WRIST_VIS_THR) and (lms[LW].y < lms[NOSE].y - WRIST_ABOVE_HEAD_MARGIN)
            rw_above = (rw_vis >= WRIST_VIS_THR) and (lms[RW].y < lms[NOSE].y - WRIST_ABOVE_HEAD_MARGIN)
            grip = (lw_above or rw_above)

            if vis_strict_ok and grip and (torso_dx_norm <= TORSO_X_THR):
                onbar_streak += 1; offbar_streak = 0
            else:
                offbar_streak += 1; onbar_streak = 0

            if not onbar and onbar_streak >= ONBAR_MIN_FRAMES:
                onbar = True; asc_base_head = None; allow_new_peak = True
                swing_streak = 0

            if onbar and offbar_streak >= OFFBAR_MIN_FRAMES:
                onbar = False; offbar_frames_since_any_rep = 0

            if not onbar and rep_count>0:
                offbar_frames_since_any_rep += 1
                if offbar_frames_since_any_rep >= OFFBAR_STOP_FRAMES:
                    break

            # Counting (ON-BAR בלבד) — זהה ללוגיקה הקיימת
            head_vel = 0.0 if head_prev is None else (head_y - head_prev)
            cur_rt = None

            if onbar and vis_strict_ok:
                if asc_base_head is None:
                    if head_vel < -HEAD_VEL_UP_TINY:
                        asc_base_head = head_y
                else:
                    if (head_y - asc_base_head) > (RESET_DESCENT * 2):
                        asc_base_head = head_y

                ascent_amt = 0.0 if asc_base_head is None else (asc_base_head - head_y)
                at_top   = (elbow_angle <= ELBOW_TOP_ANGLE) and (ascent_amt >= HEAD_MIN_ASCENT)
                can_cnt  = (frame_idx - last_peak_frame) >= REFRACTORY_FRAMES

                if at_top and allow_new_peak and can_cnt:
                    rep_count += 1; good_reps += 1; all_scores.append(10.0)
                    rep_reports.append({
                        "rep_index": rep_count,
                        "top_elbow": float(elbow_angle),
                        "ascent_from": float(asc_base_head if asc_base_head is not None else head_y),
                        "peak_head_y": float(head_y)
                    })
                    last_peak_frame = frame_idx
                    allow_new_peak = False
                    bottom_already_reported = False
                    bottom_phase_max_elbow = max(raw_elbow_L, raw_elbow_R)

                if (allow_new_peak is False):
                    cand = max(raw_elbow_L, raw_elbow_R)
                    bottom_phase_max_elbow = cand if bottom_phase_max_elbow is None else max(bottom_phase_max_elbow, cand)

                reset_by_desc = (asc_base_head is not None) and ((head_y - asc_base_head) >= RESET_DESCENT)
                reset_by_elb  = (elbow_angle >= RESET_ELBOW)
                if reset_by_desc or reset_by_elb:
                    if not bottom_already_reported:
                        effective_max = bottom_phase_max_elbow if bottom_phase_max_elbow is not None else max(raw_elbow_L, raw_elbow_R)
                        if effective_max < (BOTTOM_EXT_MIN_ANGLE - BOTTOM_HYST_DEG):
                            bottom_fail_count += 1
                            if bottom_fail_count >= BOTTOM_FAIL_MIN_REPS:
                                session_feedback.add(FB_CUE_BOTTOM)
                                cur_rt = cur_rt or FB_CUE_BOTTOM
                        else:
                            bottom_fail_count = max(0, bottom_fail_count - 1)
                        bottom_already_reported = True

                    allow_new_peak = True
                    asc_base_head = head_y
                    bottom_phase_max_elbow = None

                if (cur_rt is None) and (asc_base_head is not None) and (ascent_amt < HEAD_MIN_ASCENT*0.7) and (head_vel < -HEAD_VEL_UP_TINY):
                    session_feedback.add(FB_CUE_HIGHER); cur_rt = FB_CUE_HIGHER

                if torso_dx_norm > SWING_THR:
                    swing_streak += 1
                else:
                    swing_streak = max(0, swing_streak-1)
                if (cur_rt is None) and (swing_streak >= SWING_MIN_STREAK) and (not swing_already_reported):
                    session_feedback.add(FB_CUE_SWING); cur_rt = FB_CUE_SWING; swing_already_reported = True
            else:
                asc_base_head = None; allow_new_peak = True
                swing_streak = 0
                bottom_phase_max_elbow = None

            if cur_rt:
                if cur_rt != rt_fb_msg:
                    rt_fb_msg = cur_rt; rt_fb_hold = RT_FB_HOLD_FRAMES
                else:
                    rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold > 0: rt_fb_hold -= 1

            # ציור וכתיבה — רק במסלול שמחזיר וידאו
            if return_video and out is not None:
                frame_draw = draw_body_only(frame, lms)
                frame_draw = draw_overlay(frame_draw, reps=rep_count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=height_live)
                out.write(frame_draw)

            if head_y is not None: head_prev = head_y

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    # ===== TECHNIQUE SCORE ===== (ללא שינוי)
    if rep_count == 0:
        technique_score = 0.0
    else:
        penalty = 0.0
        if session_feedback:
            penalty = sum(FB_WEIGHTS.get(msg, FB_DEFAULT_WEIGHT) for msg in session_feedback)
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
        raw_score = max(0.0, 10.0 - penalty)
        technique_score = _half_floor(raw_score)

    feedback_list = sorted(session_feedback) if session_feedback else ["Great form! Keep it up 💪"]

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            if feedback_list:
                f.write("Feedback:\n")
                for ln in feedback_list:
                    f.write(f"- {ln}\n")
    except Exception:
        pass

    final_path = ""
    if return_video:
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run([
                "ffmpeg","-y","-i", output_path,
                "-c:v","libx264","-preset","medium",
                "-crf", str(int(encode_crf)),
                "-movflags","+faststart","-pix_fmt","yuv420p",
                encoded_path
            ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            final_path = encoded_path if os.path.exists(encoded_path) else output_path
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                try: os.remove(output_path)
                except: pass
        except Exception:
            final_path = output_path if os.path.exists(output_path) else ""

    return {
        "squat_count": int(rep_count),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": feedback_list,
        "tips": [],
        "reps": rep_reports,
        "video_path": final_path,        # במסלול מהיר יהיה ""
        "feedback_path": feedback_path
    }

