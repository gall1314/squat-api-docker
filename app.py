@app.route('/deadlift/analyze-fast', methods=['POST'])
def deadlift_analyze_fast():
    """
    מקבל וידאו, מריץ אנליזה קנונית מהירה לדדליפט (ללא וידאו),
    שומר ארטיפקט JSON ב-MEDIA_DIR ומחזיר פרטי תוצאה + analysis_id.
    """
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400

    # שמירת קלט
    unique_id = str(uuid.uuid4())[:8]
    base_filename = f"deadlift_{unique_id}"
    raw_video_path = os.path.join(MEDIA_DIR, base_filename + ".mp4")
    video_file.save(raw_video_path)

    # ניתוח מהיר
    artifact = analyze_deadlift_fast(
        video_path=raw_video_path,
        work_scale=0.40,
        frame_skip=6,
        model_complexity=1
    )
    # שייך את ה-ID גם לשם הקובץ כדי לאתר בקלות
    analysis_id = artifact.get("analysis_id") or str(uuid.uuid4()).replace("-", "")
    artifact_filename = f"{base_filename}_analysis.json"  # שומר קישור בין הוידאו לארטיפקט
    artifact_path = os.path.join(MEDIA_DIR, artifact_filename)

    # נשמור את הארטיפקט לקובץ
    try:
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return jsonify({"error": f"Failed to save artifact: {e}"}), 500

    # נבנה URL נגיש
    artifact_url = request.host_url.rstrip('/') + '/media/' + os.path.basename(artifact_path)

    # נחזיר סיכום קצר + לינק לארטיפקט המלא
    resp = {
        "analysis_id": analysis_id,
        "artifact_url": artifact_url,
        "summary": artifact.get("summary", {}),
        "reps_report": artifact.get("reps_report", []),
        "duration_sec": artifact.get("duration_sec"),
        "video_path": "",         # אין וידאו בשלב זה
        "message": "Deadlift fast analysis complete"
    }
    return jsonify(resp), 200


@app.route('/deadlift/render-video', methods=['POST'])
def deadlift_render_video():
    """
    מקבל analysis_id (או artifact_url/filename), מייצר וידאו מקור עם אוברליי זהה (ללא אנליזה מחדש),
    ושולח לינק לקובץ.
    """
    analysis_id = request.form.get('analysis_id')
    artifact_filename = request.form.get('artifact_filename')  # אופציונלי: שם קובץ JSON ישיר בתוך MEDIA_DIR

    if not analysis_id and not artifact_filename:
        return jsonify({"error": "Missing analysis_id or artifact_filename"}), 400

    # אתור קובץ הארטיפקט: אם קיבלנו שם מפורש – נשתמש בו; אחרת ננסה להצליב לפי analysis_id
    artifact_path = None
    if artifact_filename:
        candidate = os.path.join(MEDIA_DIR, artifact_filename)
        if os.path.isfile(candidate):
            artifact_path = candidate
    else:
        # חיפוש פשוט: קבצים שמתחילים ב-deadlift_<id>_analysis.json או מכילים את ה-id
        for fn in os.listdir(MEDIA_DIR):
            if fn.endswith("_analysis.json") and analysis_id in fn:
                artifact_path = os.path.join(MEDIA_DIR, fn)
                break
        # אם לא נמצא – חפש כל קובץ analysis.json (נאתר לפי תוכן)
        if not artifact_path:
            for fn in os.listdir(MEDIA_DIR):
                if fn.endswith("_analysis.json"):
                    p = os.path.join(MEDIA_DIR, fn)
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            art = json.load(f)
                        if art.get("analysis_id") == analysis_id:
                            artifact_path = p
                            break
                    except Exception:
                        continue

    if not artifact_path or not os.path.isfile(artifact_path):
        return jsonify({"error": "Artifact JSON not found"}), 404

    # טען את הארטיפקט
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to read artifact: {e}"}), 500

    # יעד הווידאו
    src_video = artifact.get("video_path")
    if not src_video or not os.path.isfile(src_video):
        # אם video_path בארטיפקט אינו נתיב מלא, נסה לפתור מתוך MEDIA_DIR
        candidate = os.path.join(MEDIA_DIR, os.path.basename(src_video) if src_video else "")
        if os.path.isfile(candidate):
            src_video = candidate
        else:
            return jsonify({"error": "Source video not found for this artifact"}), 404

    base, _ = os.path.splitext(os.path.basename(src_video))
    out_path = os.path.join(MEDIA_DIR, base + "_analyzed.mp4")

    # רינדור וידאו מקור עם אוברליי זהה (ללא מדיה-פייפ מחדש)
    result = render_deadlift_video(
        artifact=artifact,
        out_path=out_path,
        preset="medium",  # לאיכות גבוהה; אם תרצה מהיר יותר: "ultrafast"
        crf=18            # 18≈איכות גבוהה; 20–23 מהיר/קטן יותר
    )
    if not result.get("ok"):
        return jsonify({"error": result.get("reason", "Failed to render video")}), 500

    video_url = request.host_url.rstrip('/') + '/media/' + os.path.basename(out_path)
    return jsonify({
        "analysis_id": artifact.get("analysis_id"),
        "video_url": video_url,
        "reps": result.get("reps")
    }), 200

