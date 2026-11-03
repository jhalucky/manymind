
"""ManyMind model API and optional production UI host.

The TypeScript client lives in ``web/``. During development, run Vite on
port 5173 and this API on 8787. After ``npm run build``, this server can
also serve ``web/dist``.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from manymind.minds.classifier.data import FEATURE_NAMES
from manymind.minds.classifier.model import ChurnClassifier
from manymind.minds.classifier.train import ARTIFACT_PATH as CLASSIFIER_PATH
from manymind.minds.sentinel.model import DEFAULT_THRESHOLD
from manymind.minds.sentinel.predict import MODEL_PATH as SENTINEL_PATH
from manymind.minds.classifier.predict import predict_frame as score_telco_frame
from manymind.minds.sentinel.evaluation import load_calibrated_threshold
from manymind.minds.kyc.predict import ARTIFACT_PATH as GUARD_PATH
from manymind.minds.kyc.predict import load_model as load_guard
from manymind.minds.sentinel.predict import load_model, predict_csv as score_card_csv, predict_frame as score_card_frame, predict_value


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "web" / "dist"
CARD_SAMPLE = PROJECT_ROOT / "tests" / "fixtures" / "creditcard_sample.csv"
TELCO_SAMPLE = PROJECT_ROOT / "tests" / "fixtures" / "telco_sample.csv"
HOST = "0.0.0.0"
PORT = 8787

_sentinel = None
_classifier = None
_classifier_source = "unavailable"
_guard = None


def get_sentinel():
    global _sentinel
    if _sentinel is None:
        _sentinel = load_model()
    return _sentinel


def get_classifier():
    global _classifier, _classifier_source
    if _classifier is not None:
        return _classifier

    if CLASSIFIER_PATH.is_file():
        try:
            loaded = ChurnClassifier().load(CLASSIFIER_PATH)
            fixture = PROJECT_ROOT / "tests" / "fixtures" / "telco_sample.csv"
            if fixture.is_file():
                loaded.predict(pd.read_csv(fixture).iloc[:1])
            _classifier = loaded
            _classifier_source = "artifact"
            return _classifier
        except Exception:
            pass

    from manymind.minds.classifier.train import generate_dataset

    X, y = generate_dataset()
    _classifier = ChurnClassifier().fit(X, y)
    _classifier_source = "in-memory fallback"
    return _classifier


def get_guard():
    global _guard
    if _guard is None:
        _guard = load_guard()
    return _guard


def model_status() -> dict:
    try:
        get_sentinel()
    except Exception:
        pass
    try:
        get_classifier()
    except Exception:
        pass
    try:
        get_guard()
    except Exception:
        pass
    return {
        "name": "ManyMind",
        "default_threshold": DEFAULT_THRESHOLD,
        "sentinel": {
            "available": SENTINEL_PATH.is_file(),
            "artifact": str(SENTINEL_PATH),
        },
        "classifier": {
            "available": CLASSIFIER_PATH.is_file() or _classifier is not None,
            "artifact": str(CLASSIFIER_PATH),
            "source": _classifier_source,
            "features": list(FEATURE_NAMES),
        },
        "guard": {
            "available": GUARD_PATH.is_file() or _guard is not None,
            "artifact": str(GUARD_PATH),
        },
        "disclaimer": (
            "Sentinel: public ULB cards. Classifier: IBM Telco. "
            "Guard: hand-written KYC seed, not a bank dump. "
            "Not a live bank, carrier, or police tool."
        ),
    }


MAX_UPLOAD_ROWS = 200


def _csv_frame(payload: dict) -> pd.DataFrame:
    text = payload.get("csv")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Upload a CSV with a header row.")
    frame = pd.read_csv(StringIO(text))
    if frame.empty:
        raise ValueError("CSV has no data rows.")
    if len(frame) > MAX_UPLOAD_ROWS:
        frame = frame.head(MAX_UPLOAD_ROWS)
    return frame


def _sentinel_threshold(payload: dict | None = None) -> float:
    if payload and payload.get("threshold") is not None:
        return float(payload["threshold"])
    calibrated = load_calibrated_threshold(required=False)
    return float(calibrated if calibrated is not None else DEFAULT_THRESHOLD)


def _classifier_rows(frame: pd.DataFrame) -> list[dict]:
    result = score_telco_frame(get_classifier(), frame)
    rows = []
    for index, row in result.iterrows():
        rows.append(
            {
                "index": int(index),
                "tenure": int(row["tenure"]) if pd.notna(row.get("tenure")) else 0,
                "prediction": str(row["prediction"]),
                "churn_probability": float(row["churn_probability"]),
            }
        )
    return rows


def run_sentinel(payload: dict) -> dict:
    threshold = _sentinel_threshold(payload)
    if payload.get("csv"):
        results = score_card_frame(get_sentinel(), _csv_frame(payload), threshold=threshold)
    else:
        raw_values = payload.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError("Upload a CSV or provide numeric values.")
        results = [predict_value(get_sentinel(), value, threshold=threshold) for value in raw_values]
    return {
        "mind": "sentinel",
        "threshold": float(threshold),
        "results": results,
        "disclaimer": model_status()["disclaimer"],
    }


def run_sentinel_sample() -> dict:
    threshold = load_calibrated_threshold(required=False)
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    results = score_card_csv(get_sentinel(), CARD_SAMPLE, threshold=threshold)
    return {
        "mind": "sentinel",
        "csv": str(CARD_SAMPLE),
        "threshold": float(threshold),
        "results": results,
        "disclaimer": model_status()["disclaimer"],
    }


def run_classifier_sample() -> dict:
    return {
        "mind": "classifier",
        "csv": str(TELCO_SAMPLE),
        "results": _classifier_rows(pd.read_csv(TELCO_SAMPLE)),
        "source": _classifier_source,
        "disclaimer": model_status()["disclaimer"],
    }


def run_classifier(payload: dict) -> dict:
    if payload.get("csv"):
        return {
            "mind": "classifier",
            "results": _classifier_rows(_csv_frame(payload)),
            "source": _classifier_source,
            "disclaimer": model_status()["disclaimer"],
        }

    row = {}
    for name in FEATURE_NAMES:
        if name not in payload:
            raise ValueError(f"Missing feature: {name}")
        row[name] = payload[name]

    model = get_classifier()
    frame = pd.DataFrame([row])
    label = int(model.predict(frame)[0])
    probability = float(model.score(frame)[0])
    return {
        "mind": "classifier",
        "prediction": "churn" if label == 1 else "retain",
        "label": label,
        "churn_probability": probability,
        "features": row,
        "results": [
            {
                "index": 0,
                "tenure": int(row.get("tenure") or 0),
                "prediction": "churn" if label == 1 else "retain",
                "churn_probability": probability,
            }
        ],
        "source": _classifier_source,
        "disclaimer": model_status()["disclaimer"],
    }


def run_guard(payload: dict) -> dict:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Paste an SMS or WhatsApp message.")
    if any(token in text.lower() for token in ("upi pin", "aadhaar number", "cvv")):
        raise ValueError("Do not paste OTP, UPI PIN, CVV, or Aadhaar numbers.")
    result = get_guard().explain(text.strip())
    result["mind"] = "guard"
    result["disclaimer"] = model_status()["disclaimer"]
    return result


class ManyMindHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[manymind] {self.address_string()} {format % args}")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._send_json(200, model_status())
            return
        try:
            if path == "/api/sentinel/sample":
                self._send_json(200, run_sentinel_sample())
                return
            if path == "/api/classifier/sample":
                self._send_json(200, run_classifier_sample())
                return
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_frontend(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/sentinel":
                self._send_json(200, run_sentinel(payload))
                return
            if path == "/api/classifier":
                self._send_json(200, run_classifier(payload))
                return
            if path == "/api/guard":
                self._send_json(200, run_guard(payload))
                return
        except FileNotFoundError as exc:
            self._send_json(404, {"error": str(exc)})
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(404, {"error": "Not found"})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _send_frontend(self, path: str) -> None:
        if not FRONTEND_DIST.is_dir():
            self._send_json(
                503,
                {
                    "error": (
                        "TypeScript UI is not built. "
                        "Run: cd web && npm install && npm run build"
                    ),
                    "dev": "cd web && npm run dev  (proxies /api to this server)",
                },
            )
            return

        relative = "index.html" if path in {"/", "/index.html"} else path.lstrip("/")
        candidate = (FRONTEND_DIST / relative).resolve()
        if FRONTEND_DIST not in candidate.parents and candidate != FRONTEND_DIST:
            self._send_json(404, {"error": "Not found"})
            return
        if candidate.is_file():
            types = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml",
                ".json": "application/json; charset=utf-8",
            }
            self._send(200, candidate.read_bytes(), types.get(candidate.suffix, "application/octet-stream"))
            return
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            self._send(200, index.read_bytes(), "text/html; charset=utf-8")
            return
        self._send_json(404, {"error": "Not found"})


class ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main(host: str = HOST, port: int = PORT) -> None:
    server = None
    last_error = None
    selected = port
    for candidate in range(port, port + 10):
        try:
            server = ReusableServer((host, candidate), ManyMindHandler)
            selected = candidate
            break
        except OSError as exc:
            last_error = exc
    if server is None:
        raise RuntimeError(f"Could not bind ManyMind API on {host}:{port}") from last_error

    print(f"ManyMind API  http://127.0.0.1:{selected}")
    print("UI dev server: cd web && npm run dev")
    print("Synthetic models only. Predictions are not real-world decisions.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
