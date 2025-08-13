def run_deadlift_analysis(
    video_path,
    frame_skip=3,
    scale=0.4,
    output_path="deadlift_analyzed.mp4",
    feedback_path="deadlift_feedback.txt"
):
    """
    Deadlift analysis aligned with Bulgarian overlay:
    - Skeleton via drawing_utils (מהיר)
    - Stable rep logic with debounce
    - Live depth donut (hinge progress)
    - Bottom feedback only when issues persist
    - H.264 faststart encode
    """
    mp_pose_mod = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    # -------- Metrics --------
    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []

    # -------- Rep state --------
    rep_in_progress = False
    frame_index = 0
    last_rep_frame = -999
    MIN_FRAMES_BETWEEN_REPS = 10  # debounce כמו בבולגרי

    # -------- In-rep trackers --------
    max_delta_x = 0.0           # hinge depth proxy (hip-shoulder X distance)
    min_knee_angle = 999.0
    spine_flagged = False

    # -------- Live depth (donut) --------
    # טווח הנורמליזציה מכויל להרגיש "טבעי" כמו בבולגרי
    HINGE_START_THRESH = 0.08   # מעבר סף לתחילת חזרה
    HINGE_GOOD_TARGET  = 0.22   # יעד עומק “טוב”
    live_depth_pct = 0.0
    # החלקה קלה לעומק כדי למנוע קפיצות
    depth_smooth = 0.0
    DEPTH_ALPHA = 0.35          # EMA

    # -------- Video writer --------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))

    with mp_pose_mod.Pose(
        model_complexity=1,                 # איזון מהירות/דיוק
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_index % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # ברירת מחדל: אין פידבק שורה תחתונה
            rt_feedback = ""

            if not results.pose_landmarks:
                # שלד לא זמין — מציגים overlay עם עומק 0
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=0.0)
                out.write(frame)
                continue

            try:
                lm = results.pose_landmarks.landmark

                # שלד מהיר
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose_mod.POSE_CONNECTIONS
                )

                # נקודות עיקריות (ימין כברירת מחדל)
                R = mp_pose_mod.PoseLandmark
                shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])
                hip      = np.array([lm[R.RIGHT_HIP.value].x,      lm[R.RIGHT_HIP.value].y])
                knee     = np.array([lm[R.RIGHT_KNEE.value].x,     lm[R.RIGHT_KNEE.value].y])
                ankle    = np.array([lm[R.RIGHT_ANKLE.value].x,    lm[R.RIGHT_ANKLE.value].y])

                # ראש/אוזן/אף – לבחירת נק’ ייחוס לעקמומיות הגב
                head_candidates = [
                    lm[R.RIGHT_EAR.value],
                    lm[R.LEFT_EAR.value],
                    lm[R.NOSE.value]
                ]
                head_point = None
                for c in head_candidates:
                    if c.visibility > 0.5:
                        head_point = np.array([c.x, c.y])
                        break
                if head_point is None:
                    frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=live_depth_pct)
                    out.write(frame)
                    continue

                # מדדים לפריים
                delta_x = abs(hip[0] - shoulder[0])                 # hinge proxy
                knee_angle = calculate_angle(hip, knee, ankle)      # לניטור “ברכיים נעולות"
                mid_spine = (shoulder + hip) * 0.5 * 0.4 + head_point * 0.6
                curvature, is_rounded = analyze_back_curvature(shoulder, hip, mid_spine)

                # ---------- In-rep ----------
                if rep_in_progress:
                    max_delta_x = max(max_delta_x, delta_x)
                    min_knee_angle = min(min_knee_angle, knee_angle)
                    spine_flagged = spine_flagged or is_rounded

                    # דונאט עומק – נרמול והחלקה
                    raw_depth = (max_delta_x - HINGE_START_THRESH) / max(1e-6, (HINGE_GOOD_TARGET - HINGE_START_THRESH))
                    live_depth_pct = float(np.clip(raw_depth, 0.0, 1.0))
                    depth_smooth = DEPTH_ALPHA * live_depth_pct + (1 - DEPTH_ALPHA) * depth_smooth

                    if is_rounded:
                        rt_feedback = "Keep your back straighter"

                # תחילת חזרה: חציית סף
                if not rep_in_progress and delta_x > HINGE_START_THRESH:
                    # debounce
                    if frame_index - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        rep_in_progress = True
                        max_delta_x = delta_x
                        min_knee_angle = knee_angle
                        spine_flagged = is_rounded
                        live_depth_pct = 0.0
                        depth_smooth = 0.0

                # סוף חזרה: חזרה לעמידה יחסית
                elif rep_in_progress and delta_x < 0.035:
                    if frame_index - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        # בדיקת תנועה מספקת + ברכיים לא “נעולות” כל הזמן
                        moved_enough = (max_delta_x - delta_x) > 0.05
                        knee_softened = min_knee_angle < 170

                        if moved_enough and knee_softened:
                            feedbacks = []
                            penalty = 0.0

                            # סיום יותר זקוף
                            if delta_x > 0.05:
                                feedbacks.append("Try to finish more upright")
                                penalty += 1.0
                            # טורסו עמוק עם ברכיים נעולות
                            if max_delta_x > 0.18 and min_knee_angle > 170:
                                feedbacks.append("Try to bend your knees as you lean forward")
                                penalty += 1.0
                            # חוסר סינכרון ירך/חזה
                            if min_knee_angle > 165 and max_delta_x > 0.2:
                                feedbacks.append("Try to lift your chest and hips together")
                                penalty += 1.0
                            # עיגול גב
                            if spine_flagged:
                                feedbacks.append("Your spine is rounding inward too much")
                                penalty += 2.5

                            score = round(max(4, 10 - penalty) * 2) / 2
                            for f in feedbacks:
                                if f not in overall_feedback:
                                    overall_feedback.append(f)

                            counter += 1
                            last_rep_frame = frame_index
                            if score >= 9.5:
                                good_reps += 1
                            else:
                                bad_reps += 1
                                problem_reps.append(counter)
                            all_scores.append(score)

                    # reset rep
                    rep_in_progress = False
                    max_delta_x = 0.0
                    min_knee_angle = 999.0
                    spine_flagged = False
                    live_depth_pct = 0.0
                    depth_smooth = 0.0
                    rt_feedback = ""

                # שלד + Overlay
                frame = draw_overlay(
                    frame,
                    reps=counter,
                    feedback=rt_feedback if rt_feedback else None,
                    depth_pct=depth_smooth
                )
                out.write(frame)

            except Exception:
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=0.0)
                if out is not None:
                    out.write(frame)
                continue

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    technique_score = round((np.mean(all_scores) if all_scores else 0) * 2) / 2
    if not overall_feedback:
        overall_feedback.append("Great form! Keep your spine neutral and hinge smoothly. 💪")

    # סיכום לטקסט
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if overall_feedback:
                f.write("Feedback:\n")
                for fb in overall_feedback:
                    f.write(f"- {fb}\n")
    except Exception:
        pass

    # Encode to H.264 faststart
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            encoded_path
        ], check=False)
        if os.path.exists(output_path) and os.path.exists(encoded_path):
            os.remove(output_path)
    except Exception:
        pass

    final_video_path = encoded_path if os.path.exists(encoded_path) else (
        output_path if os.path.exists(output_path) else ""
    )

    return {
        "technique_score": technique_score,
        "squat_count": counter,          # נשאר בשם הזה בשביל התאמה ל-UI שלך
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": overall_feedback,
        "problem_reps": problem_reps,
        "video_path": final_video_path,
        "feedback_path": feedback_path,
    }

