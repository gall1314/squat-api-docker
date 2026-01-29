# -*- coding: utf-8 -*-
# app.py — unified API (multipart + raw), fast path, streaming saves

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid, shutil, inspect, importlib, sys, traceback, time
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

MEDIA_DIR = "media"
os.makedirs(MEDIA_DIR, exist_ok=True)

# -------- Aliases → internal type --------
EXERCISE_MAP = {
    "barbell squat": "squat", "barbell back squat": "squat", "squat": "squat",
    "deadlift": "deadlift",
    "romanian deadlift": "romanian_deadlift", "rdl": "romanian_deadlift",
    "bulgarian split squat": "bulgarian", "split squat": "bulgarian",
    "pull-up": "pullup", "pull up": "pullup", "pullups": "pullup",
    "barbell bicep curl": "bicep_curl", "bicep curl": "bicep_curl",
    "bent-over row": "bent_row", "barbell bent-over row": "bent_row",
    "barbell bent over row": "bent_row", "bent over row": "bent_row",
    "barbell row": "bent_row", "row": "bent_row",
    "good morning": "good_morning", "good mornings": "good_morning", "good morningg": "good_morning",
    "dips": "dips", "dip": "dips", "bench dips": "dips", "bench dip": "dips",
}

# -------- Error handling & logging --------
@app.errorhandler(Exception)
def handle_exception(e):
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    print("\n=== UNCAUGHT EXCEPTION ===\n", tb, file=sys.stderr, flush=True)
    return jsonify({"error": "internal_error", "detail": str(e)}), 500

@app.before_request
def log_entry():
    print(f"--> {request.method} {request.path}", file=sys.stderr, flush=True)

# -------- Utilities --------
def _standardize_video_path(result_dict):
    """Normalize result to always contain result['video_path'] if analyzer returned 'analyzed_video_path'."""
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
      fast=True  -> בלי וידאו (fast_mode=True)
      fast=False -> עם וידאו (output_path=out_video_path)
    """
    extra = extra or {}
    kwargs = {}
    
    # פרמטרים בסיסיים
    if _supports_arg(func, "frame_skip"): kwargs["frame_skip"] = frame_skip
    if _supports_arg(func, "scale"): kwargs["scale"] = scale
    
    # מסלול מהיר vs רגיל
    if _supports_arg(func, "fast_mode"): 
        kwargs["fast_mode"] = bool(fast)
    if _supports_arg(func, "return_video"): 
        kwargs["return_video"] = (not fast)

    # תמיד לשלוח output_path אם הפונקציה תומכת
    # (גם במסלול מהיר - הפונקציה תחליט אם להשתמש בו)
    if _supports_arg(func, "output_path"):
        if out_video_path:
            kwargs["output_path"] = out_video_path
            print(f"[_run_with_tracks] Passing output_path={out_video_path}", file=sys.stderr, flush=True)
    
    if _supports_arg(func, "output_dir"):
        kwargs["output_dir"] = os.path.dirname(out_video_path) if out_video_path else MEDIA_DIR

    for k, v in (extra.items()):
        if _supports_arg(func, k):
            kwargs[k] = v

    print(f"[_run_with_tracks] Calling func with kwargs keys: {list(kwargs.keys())}", file=sys.stderr, flush=True)
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
    """
    Import module and return the first existing attribute from func_names.
    If module/function missing — returns a stub that yields a clear JSON error instead of crashing.
    """
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

# ---- Resolve analyzers (לא מפיל אם חסר משהו) ----
run_squat       = load_func_soft('squat_analysis', 'run_analysis', 'run_squat_analysis')
run_deadlift    = load_func_soft('deadlift_analysis', 'run_deadlift_analysis', 'run_analysis')
run_rdl         = load_func_soft('romanian_deadlift_analysis', 'run_romanian_deadlift_analysis', 'run_analysis')
run_bulgarian   = load_func_soft('bulgarian_split_squat_analysis', 'run_bulgarian_analysis', 'run_analysis')
run_pullup      = load_func_soft('pullup_analysis', 'run_pullup_analysis', 'run_analysis')
run_bicep_curl  = load_func_soft('barbell_bicep_curl', 'run_barbell_bicep_curl_analysis', 'run_analysis')
run_bent_row    = load_func_soft('bent_over_row_analysis', 'run_row_analysis', 'run_analysis')
run_good_morning = load_func_soft('good_morning_analysis', 'run_good_morning_analysis', 'run_analysis')
run_dips        = load_func_soft('dips_analysis', 'run_dips_analysis', 'run_analysis')

# -------- Streaming saves --------
def _save_upload_streaming(file_storage, dest_path, chunk_size=8*1024*1024):
    # שמירה ישירה ל-media בלי temp/copy — חוסך I/O כפול
    filename = secure_filename(file_storage.filename or os.path.basename(dest_path))
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file_storage.stream, f, length=chunk_size)

def _save_raw_stream(raw_stream, dest_path, chunk_size=8*1024*1024):
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(raw_stream, f, length=chunk_size)

# -------- Core analyze logic (shared between multipart & raw) --------
def _do_analyze(resolved_type, raw_video_path, analyzed_path, fast_flag: bool):
    print(f"[_do_analyze] type={resolved_type}, raw={raw_video_path}, analyzed={analyzed_path}, fast={fast_flag}", file=sys.stderr, flush=True)
    
    if resolved_type == 'squat':
        return _run_with_tracks(run_squat, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
    elif resolved_type == 'deadlift':
        return _run_with_tracks(run_deadlift, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
    elif resolved_type == 'romanian_deadlift':
        return _run_with_tracks(run_rdl, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
    elif resolved_type == 'bulgarian':
        return _run_with_tracks(run_bulgarian, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
    elif resolved_type == 'pullup':
        return _run_with_tracks(run_pullup, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
    elif resolved_type == 'bicep_curl':
        return _run_with_tracks(run_bicep_curl, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
    elif resolved_type == 'bent_row':
        result = _run_with_tracks(run_bent_row, raw_video_path, analyzed_path, fast_flag,
                                  frame_skip=3, scale=0.4, extra={"output_dir": MEDIA_DIR})
        return _standardize_video_path(result)
    elif resolved_type == 'good_morning':
        return _run_with_tracks(run_good_morning, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
    elif resolved_type == 'dips':
        return _run_with_tracks(run_dips, raw_video_path, analyzed_path, fast_flag, frame_skip=3, scale=0.4)
    else:
        return {"error": f"Unhandled exercise type: {resolved_type}", "video_path": ""}

# -------- API --------
@app.route('/analyze', methods=['POST'])
def analyze():
    t0 = time.time()
    try:
        print("==== POST RECEIVED ====", file=sys.stderr, flush=True)
        ctype = (request.content_type or "").lower()
        print("Content-Type:", ctype, file=sys.stderr, flush=True)

        # ---------- RAW upload path (video/mp4 in body) ----------
        if ctype.startswith("video/"):
            exercise_type = (request.args.get('exercise_type') or "").strip().lower()
            if not exercise_type:
                return jsonify({"error": "Missing exercise_type (use query param)"}), 400
            resolved_type = EXERCISE_MAP.get(exercise_type)
            if not resolved_type:
                return jsonify({"error": f"Unsupported exercise type: {exercise_type}"}), 400

            fast_flag = request.args.get('fast', 'false').lower() == 'true'
            print(f"[RAW] Resolved: {resolved_type} | fast={fast_flag}", file=sys.stderr, flush=True)

            unique_id = str(uuid.uuid4())[:8]
            base_filename = f"{resolved_type}_{unique_id}"
            raw_video_path = os.path.join(MEDIA_DIR, base_filename + ".mp4")
            analyzed_path = os.path.join(MEDIA_DIR, base_filename + "_analyzed.mp4")

            t_save0 = time.time()
            _save_raw_stream(request.stream, raw_video_path, chunk_size=8*1024*1024)
            t_save1 = time.time()

            result = _do_analyze(resolved_type, raw_video_path, analyzed_path, fast_flag)
            if not isinstance(result, dict):
                result = {"error": "invalid_result", "detail": "Analyzer did not return a dict", "video_path": ""}

            result = _standardize_video_path(result)
            output_path = result.get("video_path") or ""
            print(f"[RAW] result video_path={output_path}", file=sys.stderr, flush=True)
            
            full_url = request.host_url.rstrip('/') + '/media/' + os.path.basename(output_path) if output_path else None

            print(f"OK -> {full_url}", file=sys.stderr, flush=True)
            print(f"[TIMING] save={(t_save1 - t_save0):.3f}s total={(time.time() - t0):.3f}s", file=sys.stderr, flush=True)
            return jsonify({"result": result, "video_url": full_url})

        # ---------- multipart/form-data path ----------
        if "multipart/form-data" not in ctype:
            return jsonify({"error": "Bad request", "detail": "Send video as multipart/form-data (field 'video') or raw video/mp4 with query params"}), 400

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
        print(f"[MP] Resolved: {resolved_type} | fast={fast_flag}", file=sys.stderr, flush=True)

        unique_id = str(uuid.uuid4())[:8]
        base_filename = f"{resolved_type}_{unique_id}"
        raw_video_path = os.path.join(MEDIA_DIR, base_filename + ".mp4")
        analyzed_path = os.path.join(MEDIA_DIR, base_filename + "_analyzed.mp4")
        
        print(f"[MP] raw_video_path={raw_video_path}, analyzed_path={analyzed_path}", file=sys.stderr, flush=True)

        t_save0 = time.time()
        _save_upload_streaming(video_file, raw_video_path, chunk_size=8*1024*1024)
        t_save1 = time.time()

        result = _do_analyze(resolved_type, raw_video_path, analyzed_path, fast_flag)
        if not isinstance(result, dict):
            result = {"error": "invalid_result", "detail": "Analyzer did not return a dict", "video_path": ""}

        result = _standardize_video_path(result)
        output_path = result.get("video_path") or ""
        print(f"[MP] result video_path={output_path}", file=sys.stderr, flush=True)
        
        full_url = request.host_url.rstrip('/') + '/media/' + os.path.basename(output_path) if output_path else None

        print("OK ->", full_url, file=sys.stderr, flush=True)
        print(f"[TIMING] save={(t_save1 - t_save0):.3f}s total={(time.time() - t0):.3f}s", file=sys.stderr, flush=True)
        return jsonify({"result": result, "video_url": full_url})

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
    # Fly.io expects 0.0.0.0:8080
    app.run(host="0.0.0.0", port=8080)
