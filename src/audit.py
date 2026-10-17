"""Append-only decision audit log (JSON Lines).

One record per decision with everything needed to replay and review it: inputs the model saw, the
rule results, PD, decision, thresholds and model version. The PAN is already hashed upstream and only a
short prefix is logged. Disabled when no path is configured.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, request_id: str, request: dict, response: dict, latency_ms: float) -> None:
        if not self.path:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "pan_prefix": request["pan_hash"][:8],
            "request": {k: v for k, v in request.items() if k != "pan_hash"},
            "decision": response["decision"],
            "pd": response["pd"],
            "failed_rules": response["failed_rules"],
            "reasons": [r["feature"] for r in response["reasons"]],
            "thresholds": response["thresholds"],
            "model_version": response["model_version"],
            "features": response.get("features"),
            "latency_ms": round(latency_ms, 2),
        }
        line = json.dumps(record, separators=(",", ":"))
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")

    def read(self) -> list[dict]:
        if not self.path or not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
