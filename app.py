# -*- coding: utf-8 -*-
# app.py — unified API with fast/regular modes for all exercises

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid, shutil, inspect, importlib, sys, traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

MEDIA_DIR = "media"
os.makedirs(MEDIA_DIR, exist_ok=True)

EXERCISE_MAP = {
    "barbell squat": "squat", "barbell back squat": "squat", "squat": "squat",
    "deadlift": "deadlift",
    "bulgarian split squat": "bulgarian", "split squat": "bulgarian",
    "pull-up": "pullup", "pull up": "pullup", "pullups": "pullup",
    "barbell bicep curl": "bicep_curl", "bicep curl": "bicep_curl",
    "bent-over row": "bent_row", "barbell bent-over row": "bent_row",
    "barbell bent over row": "bent_row", "bent over row": "bent_row",
    "barbell row": "bent_row", "row": "bent_row",
}

@app.errorhandler(Exception)
def handle_exception(e):
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    print("\n=== UNCAUGHT EXCEPTION ===\n", tb, file=sys.stderr, flush=True)
    return jsonify({"error": "internal_error", "detail": str(e)}), 500

@app.before_request
def log_entry():
    print(f"--> {request.method} {request.path}", file=sys.stderr, flush=True)

def _standardize_video_path(result_dict):
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
    extra = extra or {}
    kwargs = {}
    if _supports_arg(func, "frame_skip"): kwargs["frame_skip"] = frame_skip
    if _supports_arg(func, "scale"): kwargs["scale"] = scale
    if _supports_arg(func, "return_video"): kwargs["return_video"] = (not fast)
    if _supports_arg(func, "fast_mode"): kwargs["fast_mode"] = bool(fast)

    if not fast:
        if _supports_arg(func, "output_path") and out_video_path:
            kwargs["output_path"] = out_video_path
        elif _supports_arg(func, "output_dir"):
            kwargs["output_dir"] = os.path.dirname(out_video_path) if out_video_path else MEDIA_DIR
    else:
        # fast => לא מייצרים וידאו חוזר
        if _supports_arg(func, "output_path"):
            kwargs["output_path"] = None

    for k, v in (extra.items()):
        if _supports_arg(func, k):
            kwargs[k] = v

    return func(src_path, **kwargs)

def _missing_stub(mod_name, fn_names):
    def _stub(*args, **kwargs):
        return {
            "error": "analyzer_unavailable",
            "detail": f"Module '{mod_name}' is missing or none of these functions exist: {', '.join(fn_names)}",
            "video_path": ""
        }
    return _stub

def load_func_soft(module_name, *func_names):
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[loader] FAILED import {module_name}: {e}", file=sys.stderr, flush=True)
        return _missing_stub(module_name, func_names)
    for fn in func_names:
        if hasattr(mod, fn):
            f = getattr(mod, fn)
            print(f"[loader] {module_name}.{fn} resolved", file=sys.stderr, flush=True)
            return f
    print(f"[loader] {module_name}: none of {func_names} found", file=sys.stderr, flush=True)
    return _missing_stub(module_name, func_names)

run_squat       = load_func_soft('squat_analysis', 'run_analysis')
run_deadlift    = load_func_soft('deadlift_analysis', 'run_deadlift_analysis', 'run_analysis')
run_bulgarian   = load_func_soft('bulgarian_split_squat_analysis', 'run_bulgarian_analysis', 'run_analysis')
run_pullup      = load_func_soft('pullup_analysis', 'run_pullup_analysis', 'run_analysis')
run_bicep_curl  = load_func_soft('barbell_bicep_curl', 'run_barbell_bicep_curl_analysis', 'run_analysis')
run_bent_row    = load_func_soft('bent_over_row_analysis', 'run_row_analysis', 'run_analysis')

def _save_upload_streaming(file_storage, dest_path):
    # שמירה ישירה ל-media בלי הקפצה ל-temp — חוסך כתיבה/קריאה כפולה
    filename = secure_filename(file_storage.filename or os.path.basename(dest_path))
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file_storage.stream, f, length=1024*1024)  # 1MB chunks

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("==== POST RECEIVED ====", file=sys.stderr, flush=True)
        print("Content-Type:", request.content_type, file=sys.stderr, flush=True)

        if not (request.content_type and "multipart/form-data" in request.content_type.lower()):
            return jsonify({"error": "Bad request", "detail": "Expected multipart/form-data with field 'video'"}), 400

        video_file = request.files.get('video')
        if not video_file:
            print("NO 'video' FILE FIELD", file=sys.stderr, flush=True)
            return jsonify({"error": "No video uploaded", "detail": "form-data field name must be 'video'"}), 400

        print("FORM KEYS:", list(request.form.keys()), file=sys.stderr, flush=True)
        print("FILES KEYS:", list(request.files.keys()), file=sys.stderr, flush=True)

        exercise_type = request.form.get('exercise_type')
        if not exercise_type:
            return jsonify({"error": "Missing exercise_type"}), 400

        exercise_type = exercise_type.lower().strip()
        resolved_type = EXERCISE_MAP.get(exercise_type)
        if not resolved_type:
            return jsonify({"error": f"Unsupported exercise type: {exercise_type}"}), 400

        fast_flag = request.form.get('fast', 'false').lower() == 'true'
        print(f"Resolved: {resolved_type} | fast={fast_flag}", file=sys.stderr, flush=True)

        unique_id = str(uuid.uuid4())[:8]
        base_filename = f"{resolved_type}_{unique_id}"
        raw_video_path = os.path.join(MEDIA_DIR, base_filename + ".mp4")
        analyzed_path = os.path.join(MEDIA_DIR, base_filename + "_analyzed.mp4")

        # שמירה ישירה בקבצים (streaming) — בלי temp ואח"כ copy
        _save_upload_streaming(video_file, raw_video_path)

        if resolved_type == 'squat':
            result = _run_with_tracks(run_squat, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
        elif resolved_type == 'deadlift':
            result = _run_with_tracks(run_deadlift, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
        elif resolved_type == 'bulgarian':
            result = _run_with_tracks(run_bulgarian, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
        elif resolved_type == 'pullup':
            result = _run_with_tracks(run_pullup, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
        elif resolved_type == 'bicep_curl':
            result = _run_with_tracks(run_bicep_curl, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
        elif resolved_type == 'bent_row':
            result = _run_with_tracks(run_bent_row, raw_video_path, analyzed_path, fast_flag,
                                      frame_skip=3, scale=0.4, extra={"output_dir": MEDIA_DIR})
            result = _standardize_video_path(result)
        else:
            return jsonify({"error": f"Unhandled exercise type: {resolved_type}"}), 400

        if not isinstance(result, dict):
            result = {"error": "invalid_result", "detail": "Analyzer did not return a dict", "video_path": ""}

        result = _standardize_video_path(result)
        output_path = result.get("video_path")
        full_url = request.host_url.rstrip('/') + '/media/' + os.path.basename(output_path) if output_path else None

        resp = {"result": result, "video_url": full_url}
        print("OK ->", full_url, file=sys.stderr, flush=True)
        return jsonify(resp)

    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print("\n*** analyze() EXCEPTION ***\n", tb, file=sys.stderr, flush=True)
        return jsonify({"error": "internal_error_in_analyze", "detail": str(e)}), 500

@app.route('/media/<filename>')
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)

@app.route('/healthz')
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

