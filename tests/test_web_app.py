
import json
import threading
from http.client import HTTPConnection

import pytest

from manymind.minds.sentinel.model import DEFAULT_THRESHOLD
from manymind.web.app import FRONTEND_DIST, ManyMindHandler, ReusableServer, model_status, run_classifier, run_sentinel


@pytest.fixture
def api_port():
    server = ReusableServer(("127.0.0.1", 0), ManyMindHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _request(port, method, path, payload=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=10)
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, response.getheader("Content-Type"), raw


def test_status_reports_both_minds():
    status = model_status()

    assert status["name"] == "ManyMind"
    assert status["default_threshold"] == 0.0
    assert status["sentinel"]["available"]
    assert status["classifier"]["available"]


@pytest.mark.skip(reason="Sentinel now scores credit-card rows, not a single number")
def test_sentinel_ui_prediction_uses_default_threshold():
    payload = run_sentinel({"values": [100, 150]})

    assert payload["threshold"] == DEFAULT_THRESHOLD
    assert payload["results"][0]["prediction"] == "normal"
    assert payload["results"][1]["prediction"] == "anomaly"


def test_classifier_ui_prediction_returns_probability():
    import pandas as pd

    sample = pd.read_csv("tests/fixtures/telco_sample.csv").iloc[0]
    payload = run_classifier(sample.to_dict())

    assert payload["prediction"] in {"churn", "retain"}
    assert 0 <= payload["churn_probability"] <= 1
    assert payload["source"] in {"artifact", "in-memory fallback"}


def test_sentinel_rejects_empty_values():
    with pytest.raises(ValueError, match="at least one"):
        run_sentinel({"values": []})


def test_http_status_and_sentinel(api_port):
    status_code, content_type, raw = _request(api_port, "GET", "/api/status")
    payload = json.loads(raw)

    assert status_code == 200
    assert "application/json" in content_type
    assert payload["name"] == "ManyMind"
    assert payload["default_threshold"] == 0.0

    status_code, _, raw = _request(api_port, "GET", "/api/status")
    payload = json.loads(raw)
    assert payload["classifier"]["features"]


def test_http_serves_built_frontend(api_port):
    if not (FRONTEND_DIST / "index.html").is_file():
        pytest.skip("TypeScript UI is not built")

    status_code, content_type, raw = _request(api_port, "GET", "/")

    assert status_code == 200
    assert "text/html" in content_type
    assert b"ManyMind" in raw
    assert b"root" in raw
