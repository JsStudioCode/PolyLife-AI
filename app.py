"""
PolyLife AI: Real-Time HTTP & Server-Sent Events (SSE) Server.
Serves the interactive UI, real-time ML/TOPSIS inference API, Firebase config persistence,
Kaggle dataset importer/retrainer, and live sensor telemetry stream.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import random
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from polylife_engine import (
    BASE_DIR,
    CRITERIA_NAMES,
    DEFAULT_WEIGHTS,
    FIREBASE_CFG_PATH,
    KAGGLE_CURATED_CATALOG,
    POLY_NAMES,
    POLY_PARAMS,
    ROUTES,
    PolyLifeEngine,
    generate_chemistry_dataset,
)

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)


class SSEBroker:
    """Broadcasts real-time JSON events to connected browser SSE clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[str]] = []

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=100)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, event_type: str, data: Any) -> None:
        payload = f"event: {event_type}\ndata: {json.dumps(data, default=float)}\n\n"
        with self._lock:
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._clients.remove(q)


ENGINE = PolyLifeEngine()
BROKER = SSEBroker()
STREAM_STATE = {"active": False, "interval_sec": 2.5, "count": 0}
STREAM_LOCK = threading.Lock()


def seed_initial_assessments() -> None:
    if ENGINE.list_assessments(limit=1):
        return
    seeds = [
        {"sample_name": "PP-AutoBumper-Lot04", "batch_id": "SORT-A101", "poly": "PP", "t": 1500, "I": 1.0, "T": 50, "RH": 60, "source": "Initial Seed"},
        {"sample_name": "PET-Bottle-Coastal", "batch_id": "SORT-B204", "poly": "PET", "t": 650, "I": 1.15, "T": 42, "RH": 75, "source": "Initial Seed"},
        {"sample_name": "HDPE-Crate-Outdoor", "batch_id": "SORT-A108", "poly": "HDPE", "t": 980, "I": 0.95, "T": 46, "RH": 55, "source": "Initial Seed"},
        {"sample_name": "LDPE-AgriFilm-UV", "batch_id": "SORT-C312", "poly": "LDPE", "t": 1720, "I": 1.25, "T": 58, "RH": 45, "source": "Initial Seed"},
        {"sample_name": "PS-Packaging-Clean", "batch_id": "SORT-D091", "poly": "PS", "t": 240, "I": 0.75, "T": 34, "RH": 40, "source": "Initial Seed"},
        {"sample_name": "PVC-Conduit-Aged", "batch_id": "SORT-E410", "poly": "PVC", "t": 1850, "I": 1.2, "T": 62, "RH": 70, "source": "Initial Seed"},
    ]
    now = time.time()
    for idx, s in enumerate(seeds):
        pred = ENGINE.predict(s)
        ENGINE.save_assessment(
            {
                "sample_name": s["sample_name"],
                "batch_id": s["batch_id"],
                "poly": s["poly"],
                "t": s["t"],
                "I": s["I"],
                "T": s["T"],
                "RH": s["RH"],
                "CI": pred["spectral"]["CI"],
                "HI": pred["spectral"]["HI"],
                "dXc": pred["spectral"]["dXc"],
                "ret": pred["module2"]["ret_ml"],
                "stage": pred["module1"]["stage"],
                "stage_probs": pred["module1"]["probabilities"],
                "route": pred["module3"]["best_route"],
                "topsis_scores": pred["module3"]["scores"],
                "source": s["source"],
                "created_at": now - (len(seeds) - idx) * 180,
            }
        )


def emit_sensor_sample(source: str = "Live Conveyor Sensor") -> dict[str, Any]:
    poly = random.choice(POLY_NAMES)
    t = round(random.uniform(80, 1950), 1)
    I = round(random.uniform(0.65, 1.38), 2)
    T = round(random.uniform(30, 68), 1)
    RH = round(random.uniform(25, 88), 1)
    pred = ENGINE.predict({"poly": poly, "t": t, "I": I, "T": T, "RH": RH})
    with STREAM_LOCK:
        STREAM_STATE["count"] += 1
        seq = STREAM_STATE["count"]
    rec = ENGINE.save_assessment(
        {
            "sample_name": f"{poly}-Line-{seq:03d}",
            "batch_id": f"CONV-{time.strftime('%H%M')}",
            "poly": poly,
            "t": t,
            "I": I,
            "T": T,
            "RH": RH,
            "CI": pred["spectral"]["CI"],
            "HI": pred["spectral"]["HI"],
            "dXc": pred["spectral"]["dXc"],
            "ret": pred["module2"]["ret_ml"],
            "stage": pred["module1"]["stage"],
            "stage_probs": pred["module1"]["probabilities"],
            "route": pred["module3"]["best_route"],
            "topsis_scores": pred["module3"]["scores"],
            "source": source,
        }
    )
    BROKER.publish("sensor_tick", rec)
    return rec


def sensor_worker() -> None:
    while True:
        time.sleep(0.5)
        with STREAM_LOCK:
            active = STREAM_STATE["active"]
            interval = STREAM_STATE["interval_sec"]
        if not active:
            continue
        try:
            emit_sensor_sample()
        except Exception:
            pass
        time.sleep(max(1.0, interval - 0.5))


class PolyLifeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return  # Keep console clean

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=float).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, rel_path: str) -> None:
        clean = rel_path.lstrip("/") or "index.html"
        target = (STATIC_DIR / clean).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        mime, _ = mimetypes.guess_type(str(target))
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_static("index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
            return

        if path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "polymers": POLY_PARAMS,
                    "routes": ROUTES,
                    "criteria": CRITERIA_NAMES,
                    "default_weights": DEFAULT_WEIGHTS,
                    "stream": STREAM_STATE,
                    "dataset_meta": ENGINE.dataset_meta,
                }
            )
            return

        if path == "/api/metrics":
            self._send_json(ENGINE.metrics)
            return

        if path == "/api/assessments":
            self._send_json({"items": ENGINE.list_assessments(limit=100)})
            return

        if path == "/api/kaggle/catalog":
            self._send_json(
                {
                    "catalog": KAGGLE_CURATED_CATALOG,
                    "logs": ENGINE.list_dataset_logs(),
                    "active": ENGINE.dataset_meta,
                }
            )
            return

        if path == "/api/firebase/config":
            cfg = None
            if FIREBASE_CFG_PATH.exists():
                try:
                    cfg = json.loads(FIREBASE_CFG_PATH.read_text(encoding="utf-8"))
                except Exception:
                    cfg = None
            self._send_json({"config": cfg})
            return

        if path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = BROKER.subscribe()
            try:
                hello = f"event: connected\ndata: {json.dumps({'stream': STREAM_STATE, 'dataset': ENGINE.dataset_meta})}\n\n"
                self.wfile.write(hello.encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                BROKER.unsubscribe(q)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_json()
        except Exception as e:
            self._send_json({"error": f"Invalid JSON body: {e}"}, status=400)
            return

        if path == "/api/predict":
            try:
                res = ENGINE.predict(body)
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path == "/api/assessments":
            try:
                pred = ENGINE.predict(body)
                rec = ENGINE.save_assessment(
                    {
                        "id": body.get("id"),
                        "sample_name": body.get("sample_name") or f"{pred['input']['poly']}-Specimen",
                        "batch_id": body.get("batch_id") or f"BATCH-{time.strftime('%H%M')}",
                        "poly": pred["input"]["poly"],
                        "t": pred["input"]["t"],
                        "I": pred["input"]["I"],
                        "T": pred["input"]["T"],
                        "RH": pred["input"]["RH"],
                        "CI": pred["spectral"]["CI"],
                        "HI": pred["spectral"]["HI"],
                        "dXc": pred["spectral"]["dXc"],
                        "ret": pred["module2"]["ret_ml"],
                        "stage": pred["module1"]["stage"],
                        "stage_probs": pred["module1"]["probabilities"],
                        "route": pred["module3"]["best_route"],
                        "topsis_scores": pred["module3"]["scores"],
                        "source": body.get("source") or "Lab Console",
                    }
                )
                BROKER.publish("assessment_created", rec)
                self._send_json({"item": rec}, status=201)
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path == "/api/stream/toggle":
            with STREAM_LOCK:
                if "active" in body:
                    STREAM_STATE["active"] = bool(body["active"])
                else:
                    STREAM_STATE["active"] = not STREAM_STATE["active"]
                if "interval_sec" in body:
                    STREAM_STATE["interval_sec"] = max(1.0, min(10.0, float(body["interval_sec"])))
                state_copy = dict(STREAM_STATE)
            BROKER.publish("stream_state", state_copy)
            self._send_json({"stream": state_copy})
            return

        if path == "/api/stream/tick":
            rec = emit_sensor_sample(source=str(body.get("source") or "Simulated Sensor Pulse"))
            self._send_json({"item": rec})
            return

        if path == "/api/kaggle/import":
            try:
                profile = ENGINE.fetch_kaggle_dataset(
                    slug=str(body.get("slug", "")),
                    username=body.get("username"),
                    key=body.get("key"),
                )
                BROKER.publish("dataset_imported", profile)
                self._send_json(profile)
            except PermissionError as e:
                self._send_json({"error": str(e), "needs_credentials": True}, status=401)
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path == "/api/kaggle/upload":
            try:
                csv_text = str(body.get("csv_text", ""))
                filename = str(body.get("filename", "kaggle_dataset.csv"))
                if not csv_text.strip():
                    raise ValueError("Uploaded CSV content is empty.")
                profile = ENGINE.register_csv_text(csv_text, filename=filename)
                BROKER.publish("dataset_imported", profile)
                self._send_json(profile)
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path == "/api/kaggle/retrain":
            try:
                ds_id = str(body.get("dataset_id", ""))
                mapping = body.get("mapping", {})
                title = body.get("title")
                if ds_id == "__builtin__":
                    df = generate_chemistry_dataset(1800, seed=42)
                    metrics = ENGINE.train_on_dataframe(df, source_label="Chemistry-Informed Photo-Oxidation Benchmark (Built-in)", save_cache=True)
                else:
                    metrics = ENGINE.retrain_from_imported_dataset(ds_id, mapping=mapping, title=title)
                BROKER.publish("model_retrained", metrics)
                self._send_json({"metrics": metrics})
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path == "/api/firebase/config":
            cfg = body.get("config")
            if cfg is None:
                if FIREBASE_CFG_PATH.exists():
                    FIREBASE_CFG_PATH.unlink()
                self._send_json({"config": None})
                return
            if not isinstance(cfg, dict) or not cfg.get("apiKey") or not cfg.get("projectId"):
                self._send_json({"error": "Firebase config must include at least 'apiKey' and 'projectId'."}, status=400)
                return
            FIREBASE_CFG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            self._send_json({"config": cfg})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/assessments/"):
            rec_id = parsed.path.split("/api/assessments/", 1)[1]
            ok = ENGINE.delete_assessment(rec_id)
            if ok:
                BROKER.publish("assessment_deleted", {"id": rec_id})
            self._send_json({"deleted": ok, "id": rec_id})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    seed_initial_assessments()
    t = threading.Thread(target=sensor_worker, daemon=True)
    t.start()
    server = ThreadingHTTPServer((host, port), PolyLifeHandler)
    print(f"PolyLife AI real-time server running at http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    host = os.environ.get("POLYLIFE_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    run_server(host=host, port=port)
