"""Failure-safe structured run ledger for long statistical experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

# Keep this module usable both as ``scripts.run_ledger`` and when a maintainer
# executes the file directly while inspecting a failed run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.artifacts import atomic_write_json


class RunLedger:
    """Persist progress and failure state while a run is executing."""

    def __init__(
        self,
        path: Path,
        *,
        config_sha256: str,
        profile: str,
        seed: int,
        workers: int,
        execution_source_sha256: str | None = None,
    ):
        self.path = Path(path)
        self.started_at = datetime.now(timezone.utc).isoformat()
        if execution_source_sha256 is not None and (
            len(execution_source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in execution_source_sha256)
        ):
            raise ValueError("execution_source_sha256 must be a lowercase SHA-256 digest")
        self.data: dict[str, Any] = {
            "schema_version": "2.0",
            "run_id": hashlib.sha256(
                f"{config_sha256}|{execution_source_sha256 or ''}|{profile}|{int(seed)}".encode("utf-8")
            ).hexdigest()[:32],
            "status": "running",
            "started_at_utc": self.started_at,
            "config_sha256": config_sha256,
            "profile": profile,
            "seed": int(seed),
            "workers": int(workers),
            "events": [],
        }
        if execution_source_sha256 is not None:
            self.data["execution_source_sha256"] = execution_source_sha256
        self.flush()

    @property
    def run_id(self) -> str:
        return str(self.data["run_id"])

    def flush(self) -> None:
        atomic_write_json(self.path, self.data)

    def event(self, name: str, status: str, **fields: Any) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("ledger event name must be non-empty")
        if status not in {"started", "passed", "failed", "resumed", "checkpoint"}:
            raise ValueError(f"invalid ledger event status: {status!r}")
        record: dict[str, Any] = {
            "name": name,
            "status": status,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        record.setdefault("elapsed_seconds", 0.0)
        record.update(fields)
        self.data["events"].append(record)
        self.flush()

    def finish(self, status: str, **fields: Any) -> None:
        if status not in {"passed", "failed", "interrupted"}:
            raise ValueError(f"invalid terminal ledger status: {status!r}")
        self.data["status"] = status
        self.data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.data.update(fields)
        self.flush()

    def fail(self, exc: BaseException, **fields: Any) -> None:
        """Record a terminal failure while preserving declared gate outcomes.

        Callers may attach non-error terminal fields such as
        ``certificates_pass=False``.  The structured exception remains owned by
        the ledger so a caller cannot accidentally replace the failure record.
        """
        if "error" in fields:
            raise ValueError("ledger failure fields must not override error")
        # A second SIGINT while a long traceback or atomic replace is being
        # written must not strand the run in ``running``. Ignore SIGINT only
        # for this terminal ledger write; scientific work is already stopped.
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self.finish(
                "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": "".join(traceback.format_exception(exc))[-8000:],
                },
                **fields,
            )
        finally:
            signal.signal(signal.SIGINT, previous_handler)


__all__ = ["RunLedger"]
