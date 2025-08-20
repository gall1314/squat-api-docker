from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os
import uuid
import shutil
import inspect

# ייבוא פונקציות ניתוח
from squat_analysis import run_analysis
from deadlift_analysis import run_deadlift_analysis
from bulgarian_split_squat_analysis import run_bulgarian_analysis
from pullup_analysis import run_pullup_analysis
from barbell_bicep_curl import run_barbell_bicep_curl_analysis
from bent_over_row_analysis import run_row_analysis  # NEW

app = Flask(__name__)
CORS(app)

MEDIA_DIR = "media"
os.makedirs(MEDIA_DIR, exist_ok=True)

EXERCISE_MAP = {
    "barbell squat": "squat",
    "barbell back squat": "squat",
    "squat": "squat",
    "deadlift": "deadlift",
    "bulgarian split squat": "bulgarian",
    "split squat": "bulgarian",
    "pull-up": "pullup",
    "pull up": "pullup",
    "barbell bicep curl": "bicep_curl",

    # NEW: Bent-Over Row aliases
    "bent-over row": "bent_row",
    "barbell bent-over row": "bent_row",
    "barbell bent over row": "bent_row",
    "bent over row": "bent_row",
    "barbell row": "bent_row",
    "row": "bent_row",
}

def _standardize_video_path(result_dict):
    """Normalize result to always contain result['video_path'].""" 
    if isinstance(result_dict, dict):
        if "video_path" not in result_dict and "analyzed_video_path" in result_dict:
            result_dict["video_path"] = result_dict["analyzed_video_path"]
    return result_dict

def _supports_arg(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False

def _run_with_tracks(func, src_path, out_video_path, fast: bool, *, frame_skip=3, scale=0.4, extra=None):
    """
    מריץ אנלייזר עם תמיכה בשני מסלולים:
    - fast=True  -> בלי וידאו (אם אפשר), לא מעבירים output_path / או None
    - fast=False -> עם וידאו ל-out_video_path (אם אפשר)
    מפעיל רק פרמטרים שהפונקציה תומכת בהם בפועל.
    """
    extra = extra or {}

    kwargs = {}
    # בסיס:
    if _supports_arg(func, "frame_skip"): kwargs["frame_skip"] = frame_skip
    if _supports_arg(func, "scale"): kwargs["scale"] = scale

    # מסלול וידאו/פאסט:
    if _supports_arg(func, "return_video"):
        kwargs["return_video"] = (not fast)
    if _supports_arg(func, "fast_mode"):
        kwargs["fast_mode"] = bool(fast)

    # יעד קובץ וידאו / תיקייה
    if not fast:
        # עם וידאו — נעדיף output_path אם קיים; אחרת output_dir אם קיים
        if _supports_arg(func, "output_path") and out_video_path:
            kwargs["output_path"] = out_video_path
        elif _supports_arg(func, "output_dir"):
            kwargs["output_dir"] = os.path.dirname(out_video_path) if out_video_path else MEDIA_DIR
    else:
        # פאסט — אם יש output_path, ננסה לא לכפות יצירה; יש אנליזרים שמקבלים None
        if _supports_arg(func, "output_path"):
            kwargs["output_path"] = None
        # ואם יש output_dir בלבד – לא נעביר אותו בכלל כדי לא לרנדר וידאו
        # (אם הפונקציה ממילא מייצרת וידאו – אין לנו איך למנוע בלי לשנותה)

    # פרמטרים ייעודיים נוספים אם ביקשת
    for k, v in (extra.items()):
        if _supports_arg(func, k):
            kwargs[k] = v

    return func(src_path, **kwargs)

@app.route('/analyze', methods=['POST'])
def analyze():
    print("==== POST RECEIVED ====")
    print("FORM KEYS:", list(request.form.keys()))
    print("FILES KEYS:", list(request.files.keys()))

    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400

    exercise_type = request.form.get('exercise_type')
    if not exercise_type:
        return jsonify({"error": "Missing exercise_type"}), 400
    exercise_type = exercise_type.lower().strip()
    resolved_type = EXERCISE_MAP.get(exercise_type)

    if not resolved_type:
        return jsonify({"error": f"Unsupported exercise type: {exercise_type}"}), 400

    # בוליאן fast
    fast_flag = request.form.get('fast', 'false').lower() == 'true'

    # שמירת קלט זמני
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp.name)

    unique_id = str(uuid.uuid4())[:8]
    base_filename = f"{resolved_type}_{unique_id}"
    raw_video_path = os.path.join(MEDIA_DIR, base_filename + ".mp4")

    shutil.copyfile(temp.name, raw_video_path)
    os.remove(temp.name)

    # יעד ברירת־מחדל לסרטון פלט
    analyzed_path = os.path.join(MEDIA_DIR, base_filename + "_analyzed.mp4")

    # מיפוי טיפוסים לפונקציות
    try:
        if resolved_type == 'squat':
            result = _run_with_tracks(
                run_analysis,
                raw_video_path,
                analyzed_path,
                fast_flag,
                frame_skip=3, scale=0.4
            )

        elif resolved_type == 'deadlift':
            result = _run_with_tracks(
                run_deadlift_analysis,
                raw_video_path,
                analyzed_path,
                fast_flag,
                frame_skip=3, scale=0.4
            )

        elif resolved_type == 'bulgarian':
            result = _run_with_tracks(
                run_bulgarian_analysis,
                raw_video_path,
                analyzed_path,
                fast_flag,
                frame_skip=3, scale=0.4
            )

        elif resolved_type == 'pullup':
            result = _run_with_tracks(
                run_pullup_analysis,
                raw_video_path,
                analyzed_path,
                fast_flag,
                frame_skip=3, scale=0.4
            )

        elif resolved_type == 'bicep_curl':
            result = _run_with_tracks(
                run_barbell_bicep_curl_analysis,
                raw_video_path,
                analyzed_path,
                fast_flag,
                frame_skip=3, scale=0.4
            )

        elif resolved_type == 'bent_row':
            # אנלייזר של ה־Row משתמש לעיתים ב-output_dir — העזר דואג לזה.
            result = _run_with_tracks(
                run_row_analysis,
                raw_video_path,
                analyzed_path,  # ישמש ל-output_dir או path לפי התמיכה
                fast_flag,
                frame_skip=3, scale=0.4,
                extra={"output_dir": MEDIA_DIR}  # אם הפונקציה תומכת — יעבור
            )
            result = _standardize_video_path(result)

        else:
            return jsonify({"error": f"Unhandled exercise type: {resolved_type}"}), 400

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    result = _standardize_video_path(result)

    # לייצר URL מלא אם יש וידאו בפלט
    output_path = result.get("video_path")
    full_url = None
    if output_path:
        full_url = request.host_url.rstrip('/') + '/media/' + os.path.basename(output_path)

    response = {
        "result": result,
        "video_url": full_url
    }
    return jsonify(response)

@app.route('/media/<filename>')
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

