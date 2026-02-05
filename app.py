# -*- coding: utf-8 -*-
# app.py — Fast/Slow dual-mode API

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid, shutil, inspect, importlib, importlib.util, sys, traceback, time
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, ClientDisconnected

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
    "dips": "dips", "dip": "dips", 
    "bench dips": "dips", "bench dip": "dips",
    "chest dips": "dips", "chest dip": "dips",
    "tricep dips": "dips", "triceps dips": "dips",
    "parallel bar dips": "dips", "bar dips": "dips",
    "overhead press": "overhead_press", "barbell overhead press": "overhead_press",
    "military press": "overhead_press", "standing press": "overhead_press",
    "shoulder press": "overhead_press", "strict press": "overhead_press",
    "push-up": "pushup", "push up": "pushup", "pushups": "pushup", "push ups": "pushup",
    "push-ups": "pushup", "regular push-up": "pushup", "standard push-up": "pushup",
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

def _run_analyzer(func, src_path, out_video_path, fast_mode: bool, *, frame_skip=3, scale=0.4, extra=None):
    """
    מריץ אנלייזר עם תמיכה בשני מסלולים:
      fast_mode=True  → רק JSON, ללא וידאו (10-15 שניות)
      fast_mode=False → JSON + וידאו מנותח (60 שניות)
    
    ** חשוב: התוצאות (reps, scores, feedback) זהות בשני המצבים! **
    """
    extra = extra or {}
    kwargs = {}
    
    # פרמטרים בסיסיים
    if _supports_arg(func, "frame_skip"): 
        kwargs["frame_skip"] = frame_skip
    if _supports_arg(func, "scale"): 
        kwargs["scale"] = scale
    
    # מסלול מהיר vs איטי
    if _supports_arg(func, "fast_mode"):
        kwargs["fast_mode"] = fast_mode
        print(f"[_run_analyzer] fast_mode={fast_mode}", file=sys.stderr, flush=True)
    
    if _supports_arg(func, "return_video"):
        kwargs["return_video"] = (not fast_mode)
        print(f"[_run_analyzer] return_video={not fast_mode}", file=sys.stderr, flush=True)
    
    # output_path - רק במצב איטי (עם וידאו)
    if not fast_mode and _supports_arg(func, "output_path") and out_video_path:
        kwargs["output_path"] = out_video_path
        print(f"[_run_analyzer] output_path={out_video_path}", file=sys.stderr, flush=True)
    
    if _supports_arg(func, "output_dir"):
        kwargs["output_dir"] = os.path.dirname(out_video_path) if out_video_path else MEDIA_DIR

    # פרמטרים נוספים
    for k, v in extra.items():
        if _supports_arg(func, k):
            kwargs[k] = v

    print(f"[_run_analyzer] Calling {func.__name__} with: {list(kwargs.keys())}", file=sys.stderr, flush=True)
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
    # Avoid noisy import traceback for analyzers that are not shipped in this build.
    if importlib.util.find_spec(module_name) is None:
        print(f"[loader] SKIP missing optional module {module_name}", file=sys.stderr, flush=True)
        return _missing_stub(module_name, func_names)

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

# ---- Resolve analyzers ----
run_squat       = load_func_soft('squat_analysis', 'run_analysis', 'run_squat_analysis')
run_deadlift    = load_func_soft('deadlift_analysis', 'run_deadlift_analysis', 'run_analysis')
run_rdl         = load_func_soft('romanian_deadlift_analysis', 'run_romanian_deadlift_analysis', 'run_analysis')
run_bulgarian   = load_func_soft('bulgarian_split_squat_analysis', 'run_bulgarian_analysis', 'run_analysis')
run_pullup      = load_func_soft('pullup_analysis', 'run_pullup_analysis', 'run_analysis')
run_bicep_curl  = load_func_soft('barbell_bicep_curl', 'run_barbell_bicep_curl_analysis', 'run_analysis')
run_bent_row    = load_func_soft('bent_over_row_analysis', 'run_row_analysis', 'run_analysis')
run_good_morning = load_func_soft('good_morning_analysis', 'run_good_morning_analysis', 'run_analysis')
run_dips        = load_func_soft('dips_analysis', 'run_dips_analysis', 'run_analysis')
run_overhead_press = load_func_soft('overhead_press_analysis', 'run_overhead_press_analysis', 'run_analysis')
run_pushup      = load_func_soft('pushup_analysis', 'run_pushup_analysis', 'run_analysis')

# -------- Streaming saves --------
def _save_upload_streaming(file_storage, dest_path, chunk_size=16*1024*1024):
    """שמירה ישירה עם chunks גדולים"""
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file_storage.stream, f, length=chunk_size)

def _save_raw_stream(raw_stream, dest_path, chunk_size=16*1024*1024):
    """שמירה ישירה עם chunks גדולים"""
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(raw_stream, f, length=chunk_size)

# -------- Core analyze logic --------
def _do_analyze(resolved_type, raw_video_path, analyzed_path, fast_mode: bool):
    """
    מריץ ניתוח על הוידאו
    
    fast_mode=True:  רק JSON (10-15 שניות)
    fast_mode=False: JSON + וידאו (60 שניות)
    
    ** התוצאות זהות בשני המצבים! **
    """
    print(f"[_do_analyze] type={resolved_type}, raw={raw_video_path}, analyzed={analyzed_path}, fast_mode={fast_mode}", 
          file=sys.stderr, flush=True)
    
    # פרמטרים אופטימליים
    frame_skip = 3
    scale = 0.4
    
    if resolved_type == 'squat':
        return _run_analyzer(run_squat, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    elif resolved_type == 'deadlift':
        return _run_analyzer(run_deadlift, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    elif resolved_type == 'romanian_deadlift':
        return _run_analyzer(run_rdl, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    elif resolved_type == 'bulgarian':
        return _run_analyzer(run_bulgarian, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    elif resolved_type == 'pullup':
        return _run_analyzer(run_pullup, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    elif resolved_type == 'bicep_curl':
        return _run_analyzer(run_bicep_curl, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    elif resolved_type == 'bent_row':
        result = _run_analyzer(run_bent_row, raw_video_path, analyzed_path, fast_mode,
                             frame_skip=frame_skip, scale=scale, 
                             extra={"output_dir": MEDIA_DIR})
        return _standardize_video_path(result)
    elif resolved_type == 'good_morning':
        return _run_analyzer(run_good_morning, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    elif resolved_type == 'dips':
        return _run_analyzer(run_dips, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    elif resolved_type == 'overhead_press':
        return _run_analyzer(run_overhead_press, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    elif resolved_type == 'pushup':
        return _run_analyzer(run_pushup, raw_video_path, analyzed_path, fast_mode,
                           frame_skip=frame_skip, scale=scale)
    else:
        return {"error": f"Unhandled exercise type: {resolved_type}", "video_path": ""}

# -------- API --------
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    POST /analyze
    
    Parameters:
    - exercise_type: string (required) - סוג התרגיל
    - fast: boolean (optional, default=true) - מצב מהיר או מלא
      * fast=true:  רק JSON, ללא וידאו (10-15 שניות)
      * fast=false: JSON + וידאו מנותח (60 שניות)
    
    Returns:
    - result: object - תוצאות הניתוח (reps, scores, feedback)
    - video_url: string|null - URL לוידאו מנותח (רק אם fast=false)
    """
    t0 = time.time()
    try:
        print("==== POST RECEIVED ====", file=sys.stderr, flush=True)
        ctype = (request.content_type or "").lower()
        print(f"Content-Type: {ctype}", file=sys.stderr, flush=True)

        # ---------- RAW upload path (video/mp4 in body) ----------
        if ctype.startswith("video/"):
            exercise_type = (request.args.get('exercise_type') or "").strip().lower()
            if not exercise_type:
                return jsonify({"error": "Missing exercise_type (use query param)"}), 400
            
            resolved_type = EXERCISE_MAP.get(exercise_type)
            if not resolved_type:
                return jsonify({"error": f"Unsupported exercise type: {exercise_type}"}), 400

            # fast mode - default true
            fast_str = request.args.get('fast', 'true').lower()
            fast_mode = (fast_str == 'true')
            print(f"[RAW] Resolved: {resolved_type} | fast_mode={fast_mode}", file=sys.stderr, flush=True)

            unique_id = str(uuid.uuid4())[:8]
            base_filename = f"{resolved_type}_{unique_id}"
            raw_video_path = os.path.join(MEDIA_DIR, base_filename + ".mp4")
            analyzed_path = os.path.join(MEDIA_DIR, base_filename + "_analyzed.mp4")

            t_save0 = time.time()
            _save_raw_stream(request.stream, raw_video_path)
            t_save1 = time.time()
            print(f"[RAW] Saved in {(t_save1 - t_save0):.3f}s", file=sys.stderr, flush=True)

            result = _do_analyze(resolved_type, raw_video_path, analyzed_path, fast_mode)
            if not isinstance(result, dict):
                result = {"error": "invalid_result", 
                         "detail": "Analyzer did not return a dict", 
                         "video_path": ""}

            result = _standardize_video_path(result)
            output_path = result.get("video_path") or ""
            print(f"[RAW] result video_path={output_path}", file=sys.stderr, flush=True)
            
            full_url = None
            if output_path and os.path.exists(output_path):
                full_url = request.host_url.rstrip('/') + '/media/' + os.path.basename(output_path)

            print(f"[RAW] OK -> {full_url}", file=sys.stderr, flush=True)
            print(f"[RAW] TOTAL TIME: {(time.time() - t0):.3f}s", file=sys.stderr, flush=True)
            return jsonify({"result": result, "video_url": full_url})

        # ---------- multipart/form-data path ----------
        if "multipart/form-data" not in ctype:
            return jsonify({
                "error": "Bad request", 
                "detail": "Send video as multipart/form-data (field 'video') or raw video/mp4 with query params"
            }), 400

        try:
            form = request.form
            files = request.files
        except (ClientDisconnected, BadRequest) as e:
            content_length = request.content_length
            print(f"[MP] request parsing failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            print(f"[MP] client likely disconnected while upload was in-flight (content_length={content_length})",
                  file=sys.stderr, flush=True)
            return jsonify({
                "error": "client_disconnected",
                "detail": "Upload interrupted before the multipart payload was fully received",
                "content_length": content_length
            }), 400

        video_file = files.get('video')
        if not video_file:
            print("NO 'video' FILE FIELD", file=sys.stderr, flush=True)
            return jsonify({
                "error": "No video uploaded", 
                "detail": "form-data field name must be 'video'"
            }), 400

        print(f"FORM KEYS: {list(form.keys())}", file=sys.stderr, flush=True)
        print(f"FILES KEYS: {list(files.keys())}", file=sys.stderr, flush=True)

        exercise_type = form.get('exercise_type')
        if not exercise_type:
            return jsonify({"error": "Missing exercise_type"}), 400

        exercise_type = exercise_type.lower().strip()
        resolved_type = EXERCISE_MAP.get(exercise_type)
        if not resolved_type:
            return jsonify({"error": f"Unsupported exercise type: {exercise_type}"}), 400

        # fast mode - default true
        fast_str = form.get('fast', 'true').lower()
        fast_mode = (fast_str == 'true')
        print(f"[MP] Resolved: {resolved_type} | fast_mode={fast_mode}", file=sys.stderr, flush=True)

        unique_id = str(uuid.uuid4())[:8]
        base_filename = f"{resolved_type}_{unique_id}"
        raw_video_path = os.path.join(MEDIA_DIR, base_filename + ".mp4")
        analyzed_path = os.path.join(MEDIA_DIR, base_filename + "_analyzed.mp4")
        
        print(f"[MP] raw_video_path={raw_video_path}, analyzed_path={analyzed_path}", 
              file=sys.stderr, flush=True)

        t_save0 = time.time()
        _save_upload_streaming(video_file, raw_video_path)
        t_save1 = time.time()
        print(f"[MP] Saved in {(t_save1 - t_save0):.3f}s", file=sys.stderr, flush=True)

        result = _do_analyze(resolved_type, raw_video_path, analyzed_path, fast_mode)
        if not isinstance(result, dict):
            result = {"error": "invalid_result", 
                     "detail": "Analyzer did not return a dict", 
                     "video_path": ""}

        result = _standardize_video_path(result)
        output_path = result.get("video_path") or ""
        print(f"[MP] result video_path={output_path}", file=sys.stderr, flush=True)
        
        full_url = None
        if output_path and os.path.exists(output_path):
            full_url = request.host_url.rstrip('/') + '/media/' + os.path.basename(output_path)

        print(f"[MP] OK -> {full_url}", file=sys.stderr, flush=True)
        print(f"[MP] TOTAL TIME: {(time.time() - t0):.3f}s", file=sys.stderr, flush=True)
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
    app.run(host="0.0.0.0", port=8080)
