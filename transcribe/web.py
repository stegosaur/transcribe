"""Local web UI: upload audio, watch progress, read/rename/download the transcript."""

import argparse
import json
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from .formats import FORMATS, render

STATIC = Path(__file__).parent / "static"
MIME = {"txt": "text/plain", "md": "text/markdown", "srt": "application/x-subrip",
        "vtt": "text/vtt", "json": "application/json"}


class Jobs:
    """Jobs live in one directory each (upload + job.json), so history
    survives restarts. A single worker thread feeds them to the GPU in order."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.queue: "queue.Queue[str]" = queue.Queue()
        for meta in sorted(self.root.glob("*/job.json")):
            try:
                job = json.loads(meta.read_text())
            except ValueError:
                continue
            if job["status"] in ("queued", "running"):   # interrupted by a restart
                job.update(status="queued", stage="queued", progress=None)
                self.queue.put(job["id"])
            self.jobs[job["id"]] = job

    def dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id or ""):
            abort(404)
        return self.root / job_id

    def save(self, job: dict):
        tmp = self.dir(job["id"]) / "job.json.tmp"
        tmp.write_text(json.dumps(job, ensure_ascii=False))
        tmp.replace(self.dir(job["id"]) / "job.json")

    def update(self, job_id: str, persist: bool = False, **fields):
        with self.lock:
            job = self.jobs[job_id]
            job.update(fields)
            if persist:
                self.save(job)

    def get(self, job_id: str) -> dict:
        self.dir(job_id)
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                abort(404)
            return dict(job)

    def summary(self, job: dict) -> dict:
        d = {k: v for k, v in job.items() if k != "result"}
        if job.get("result"):
            d["duration"] = job["result"]["duration"]
            d["speakers"] = len(job["result"]["speakers"])
        return d


def create_app(engine_box: dict, jobs: Jobs) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3   # 4 GB uploads

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    @app.get("/api/status")
    def status():
        engine = engine_box.get("engine")
        return jsonify({
            "ready": engine is not None,
            "error": engine_box.get("error"),
            "loading": engine_box.get("loading"),
            "device": engine_box.get("device"),
            "whisper": engine.whisper.model_id if engine else None,
            "diarization": engine.diarizer.model if engine and engine.diarizer else None,
            "queued": jobs.queue.qsize(),
        })

    @app.get("/api/jobs")
    def list_jobs():
        with jobs.lock:
            items = sorted(jobs.jobs.values(), key=lambda j: j["created"], reverse=True)
            return jsonify([jobs.summary(j) for j in items])

    @app.post("/api/jobs")
    def create_job():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"error": "no file uploaded"}), 400
        job_id = uuid.uuid4().hex
        d = jobs.dir(job_id)
        d.mkdir(parents=True)
        suffix = Path(upload.filename).suffix.lower()[:10] or ".bin"
        audio_path = d / f"audio{suffix}"
        upload.save(audio_path)

        def as_int(name):
            v = request.form.get(name, "").strip()
            return int(v) if v.isdigit() and int(v) > 0 else None

        options = {
            "num_speakers": as_int("num_speakers"),
            "min_speakers": as_int("min_speakers"),
            "max_speakers": as_int("max_speakers"),
            "language": (request.form.get("language") or "").strip().lower() or None,
            "task": "translate" if request.form.get("translate") in ("1", "true", "on") else "transcribe",
            "diarize": request.form.get("diarize", "1") not in ("0", "false", "off"),
        }
        job = {"id": job_id, "filename": Path(upload.filename).name, "audio": audio_path.name,
               "created": time.time(), "status": "queued", "stage": "queued", "progress": None,
               "options": options, "names": {}, "result": None, "error": None}
        with jobs.lock:
            jobs.jobs[job_id] = job
            jobs.save(job)
        jobs.queue.put(job_id)
        return jsonify(jobs.summary(job)), 201

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        return jsonify(jobs.get(job_id))

    @app.delete("/api/jobs/<job_id>")
    def delete_job(job_id):
        job = jobs.get(job_id)
        if job["status"] == "running":
            return jsonify({"error": "job is running"}), 409
        with jobs.lock:
            jobs.jobs.pop(job_id, None)
        shutil.rmtree(jobs.dir(job_id), ignore_errors=True)
        return "", 204

    @app.put("/api/jobs/<job_id>/names")
    def set_names(job_id):
        jobs.get(job_id)
        names = request.get_json(silent=True) or {}
        names = {str(k): str(v).strip()[:80] for k, v in names.items() if str(v).strip()}
        jobs.update(job_id, persist=True, names=names)
        return jsonify(names)

    @app.get("/api/jobs/<job_id>/audio")
    def audio(job_id):
        job = jobs.get(job_id)
        return send_file(jobs.dir(job_id) / job["audio"], conditional=True)

    @app.get("/api/jobs/<job_id>/download/<fmt>")
    def download(job_id, fmt):
        from .engine import Transcript

        job = jobs.get(job_id)
        if fmt not in FORMATS or not job.get("result"):
            abort(404)
        body = render(Transcript.from_dict(job["result"]), fmt, job.get("names"))
        name = f"{Path(job['filename']).stem}.{fmt}"
        return Response(body, mimetype=MIME[fmt] + "; charset=utf-8",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{_quote(name)}"})

    return app


def _quote(name: str) -> str:
    from urllib.parse import quote
    return quote(name)


def worker(engine_box: dict, jobs: Jobs):
    while True:
        job_id = jobs.queue.get()
        while engine_box.get("engine") is None:
            if engine_box.get("error"):
                jobs.update(job_id, persist=True, status="error", stage="failed",
                            error="models failed to load: " + engine_box["error"])
                break
            time.sleep(0.5)
        else:
            with jobs.lock:
                job = jobs.jobs.get(job_id)
            if job is None:   # deleted while queued
                continue
            jobs.update(job_id, persist=True, status="running", stage="starting", started=time.time())

            def progress(stage, frac, job_id=job_id):
                jobs.update(job_id, stage=stage, progress=frac)

            try:
                t = time.time()
                result = engine_box["engine"].run(str(jobs.dir(job_id) / job["audio"]), progress=progress,
                                                  display_name=job["filename"], **job["options"])
                jobs.update(job_id, persist=True, status="done", stage="done", progress=1.0,
                            result=result.to_dict(), elapsed=round(time.time() - t, 2))
            except Exception as e:
                traceback.print_exc()
                jobs.update(job_id, persist=True, status="error", stage="failed", error=str(e))


def main(argv=None):
    p = argparse.ArgumentParser(prog="transcribe-web", description="Web UI for transcribe")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other machines on your LAN")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--data-dir", default=str(Path.home() / ".local" / "share" / "transcribe" / "jobs"),
                   help="where uploads and finished transcripts are kept")
    p.add_argument("-m", "--model", default="turbo")
    p.add_argument("--diarization-model", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--hf-token")
    args = p.parse_args(argv)

    engine_box: dict = {"loading": "starting"}
    jobs = Jobs(Path(args.data_dir))

    def load():
        try:
            from .engine import Engine

            def log(msg):
                print(msg, file=sys.stderr)
                engine_box["loading"] = msg
                if msg.startswith("device: "):
                    engine_box["device"] = msg[len("device: "):]

            engine_box["engine"] = Engine(args.model, args.diarization_model, args.device,
                                          args.batch_size, args.hf_token, log=log)
            engine_box["loading"] = None
        except Exception as e:
            traceback.print_exc()
            engine_box["error"] = str(e)

    threading.Thread(target=load, daemon=True).start()
    threading.Thread(target=worker, args=(engine_box, jobs), daemon=True).start()

    print(f"transcribe web UI on http://{args.host}:{args.port}  (jobs in {args.data_dir})", file=sys.stderr)
    create_app(engine_box, jobs).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
