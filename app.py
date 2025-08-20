# app.py
import os
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# אופציונלי: אפשר לאפשר CORS אם נדרש מהפרונט:
try:
    from flask_cors import CORS
except Exception:
    CORS = None

APP_NAME = "XLift-Analysis API"
APP_VERSION = "v1.0-fast-path"

app = Flask(__name__)
if CORS:
    CORS(app)

# ---------- Utilities ----------

def _as_bool(v, default=False):
    """Parse truthy strings/bools/ints safely."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")

def _json_error(message, status=400, **extra):
    payload = {"error": message}
    if extra:
        payload.update(extra)
    return jsonify(payload), status


# ---------- Health & Info ----------

@app.get("/")
def root():
    return jsonify(name=APP_NAME, version=APP_VERSION, ok=True), 200

@app.get("/healthz")
def healthz():
    return "ok", 200


# ---------- Main Analysis Endpoint ----------

@app.post("/analyze")
def analyze():
    """
    Body/Query params:
      - exercise: "pullup" | "deadlift"  (default: "pullup")
      - video_path: absolute/relative path on disk (optional if multipart 'file' sent)
      - file: multipart uploaded video (optional alternative to video_path)
      - return_video: bool (default: false)  -> fast path by default
      - preserve_quality: bool (default: false)
      - fast_mode: bool (default: None). If true => forces return_video=False.

    Response: JSON returned from the respective analysis function (+ error JSON on failure).
    """
    body = request.get_json(silent=True) or {}

    # Which exercise to run (default pullup so שלא ינסה לייבא deadlift סתם)
    exercise = (request.args.get("exercise") or body.get("exercise") or "pullup").strip().lower()

    # Where is the video coming from?
    video_path = body.get("video_path") or request.args.get("video_path")

    if "file" in request.files:
        f = request.files["file"]
        if not f.filename:
            return _json_error("uploaded file has no name")
        # save to /tmp
        fname = secure_filename(f.filename)
        save_path = os.path.join("/tmp", fname)
        try:
            f.save(save_path)
        except Exception as e:
            return _json_error("failed to save uploaded file", 500, detail=str(e))
        video_path = save_path

    if not video_path:
        return _json_error("provide 'video_path' or multipart 'file'")

    # Flags (fast path by default: no video rendering)
    return_video = _as_bool(request.args.get("return_video") or body.get("return_video"), default=False)
    preserve_quality = _as_bool(request.args.get("preserve_quality") or body.get("preserve_quality"), default=False)
    fast_mode = request.args.get("fast_mode", body.get("fast_mode", None))
    fast_mode = None if fast_mode is None else _as_bool(fast_mode, default=None)

    if fast_mode is True:
        return_video = False

    # Optional passthroughs (only if you want to override defaults in analyzers)
    # If not present, analyzers will use their own defaults.
    extra_kwargs = {}
    for key in ("frame_skip", "scale", "output_path", "feedback_path", "encode_crf"):
        if key in body:
            extra_kwargs[key] = body[key]
        elif request.args.get(key) is not None:
            # parse numerics when relevant
            val = request.args.get(key)
            if key in ("frame_skip", "encode_crf"):
                try:
                    val = int(val)
                except Exception:
                    pass
            elif key in ("scale",):
                try:
                    val = float(val)
                except Exception:
                    pass
            extra_kwargs[key] = val

    # Dispatch per exercise with lazy import so שהשרת לא יקרוס בזמן עלייה
    try:
        if exercise == "pullup":
            from pullup_analysis import run_pullup_analysis
            res = run_pullup_analysis(
                video_path,
                preserve_quality=preserve_quality,
                return_video=return_video,
                fast_mode=fast_mode,
                **extra_kwargs
            )
        elif exercise == "deadlift":
            from deadlift_analysis import run_deadlift_analysis
            res = run_deadlift_analysis(
                video_path,
                preserve_quality=preserve_quality,
                return_video=return_video,
                fast_mode=fast_mode,
                **extra_kwargs
            )
        else:
            return _json_error("unknown exercise (use 'pullup' or 'deadlift')")
    except ImportError as ie:
        # מודול חסר/שם פונקציה לא תואם – מחזירים JSON ולא מפילים שרת
        return _json_error("module_import_failed", 500, detail=str(ie))
    except Exception as e:
        # כל שגיאה בזמן ניתוח – להחזיר JSON ולא סטאק לוג ללקוח
        tb = traceback.format_exc()
        return _json_error("analysis_failed", 500, detail=str(e), traceback=tb)

    # Success
    return jsonify(res), 200


# ---------- Entrypoint ----------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"[{APP_NAME}] {APP_VERSION} listening on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

