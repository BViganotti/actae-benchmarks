#!/usr/bin/env python3
"""AI-agent durability benchmark: crash consistency of external side effects.

This measures the one thing that actually matters when a long-running agent
crashes, and it does so with real process death rather than a simulated
exception:

    A worker performs a side effect (a "tool") and the process dies in the
    window between the effect landing on an external system and the worker
    recording that it finished. On restart, how many times does the external
    effect fire again?

The external system is a durable, independent observer: either a local SQLite
database (``--service sqlite``) or a real local HTTP service
(``--service http``) with an idempotency key, a status-query endpoint, and a
``drop_after_commit`` mode that commits then drops the connection (a genuine
ambiguous timeout). The worker is a separate OS process that dies with
``os._exit(137)`` at an injected point, so in-memory progress is truly lost.

Recovery strategies:

    naive          no durable progress; a restart re-runs the whole pipeline.
    checkpoint     writes a progress marker *after* each step; a crash before
                   the marker re-runs that step.
    durable        claims each tool through Actae's idempotent execution ledger
                   with a stable per-step key, then completes it. A restart
                   replays completed steps and re-claims the crashed one.
    durable_query  like ``durable``, but on an ambiguous outcome (a reclaim or
                   a lost response) it QUERIES the downstream status endpoint
                   and reconciles instead of blindly re-applying.

Downstream behaviours:

    non_idempotent  the external system applies the effect every call.
    honors_key      the external system deduplicates on the caller-supplied key.
                    Only the ledger strategies have a stable key to supply;
                    naive/checkpoint have no durable identity, so each retry
                    presents a fresh key and the downstream cannot dedupe.

Crash points:

    after_effect    the ambiguous window: effect landed, completion unrecorded.
    after_complete  completion was recorded, then the process died.
    transport_drop  the service committed, then dropped the connection; the
                    caller sees a transport error with an unknown outcome
                    (HTTP service only).
    transport_fail  the service dropped the connection WITHOUT applying the
                    effect; again an unknown outcome, but the reconciling
                    "not applied" fork (HTTP service only).

The result: a durable ledger alone does **not** close the ambiguous window
against a non-idempotent downstream. It closes it when the downstream honours
the key, and ``durable_query`` closes it by *querying* — even against a
non-idempotent downstream. That is the honest boundary, and the numbers show it.

Usage (against a running Actae):

    ACTAE_URL=http://127.0.0.1:8002 ACTAE_API_KEY=sk-... \
        python3 examples/durability_benchmark.py --out examples/durability-benchmark-results.json

    # maximal evidence: both services, every strategy, crash at every step
    ACTAE_URL=... ACTAE_API_KEY=... \
        python3 examples/durability_benchmark.py --services sqlite http \
            --crash-at-all --out examples/durability-benchmark-results.json

The module is importable: ``run_matrix``, ``summarize_service``,
``summarize_application_rows`` and ``HttpEffectService`` have no top-level side
effects, so the arithmetic and the service are unit-testable without a server.
"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

STEPS = 6
CRASH_STEP = 3
LEASE_SECONDS = 2
RECOVERY_WAIT_SECONDS = 3.0
MAX_ATTEMPTS = 6

STRATEGIES = ("naive", "checkpoint", "durable", "durable_query")
CRASH_POINTS = ("after_effect", "after_complete", "transport_drop", "transport_fail")
DOWNSTREAMS = ("non_idempotent", "honors_key")
SERVICES = ("sqlite", "http")


class TransportError(Exception):
    """The service committed (or may have) but the response never arrived."""


# --------------------------------------------------------------------------- #
# Shared measurement
# --------------------------------------------------------------------------- #


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def summarize_application_rows(rows: List[Dict[str, str]], expected_keys: int) -> Dict[str, int]:
    """Summarize duplicate/lost effects by logical step key. Pure given rows."""
    counts: Dict[str, int] = {}
    for row in rows:
        key = row["key"]
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    distinct = len(counts)
    return {
        "total_applications": total,
        "distinct_keys": distinct,
        "expected_keys": expected_keys,
        "duplicate_applications": total - distinct,
        "max_applications_per_key": max(counts.values()) if counts else 0,
        "lost_keys": max(expected_keys - distinct, 0),
    }


# --------------------------------------------------------------------------- #
# External service A: SQLite (durable, independent of worker memory)
# --------------------------------------------------------------------------- #


def init_service(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS effects ("
        "key TEXT NOT NULL, downstream_key TEXT NOT NULL, fired_at TEXT NOT NULL)"
    )
    con.execute("CREATE INDEX IF NOT EXISTS effects_key ON effects (key)")
    con.execute("CREATE INDEX IF NOT EXISTS effects_downstream_key ON effects (downstream_key)")
    con.commit()
    con.close()


def service_apply(db_path: str, key: str, downstream_key: str, honors_key: bool) -> None:
    """Apply one external side effect to the SQLite service.

    ``key`` is the logical step identity used for measurement. ``downstream_key``
    is what the caller actually sends to the external system. ``honors_key``
    models a downstream that deduplicates on that key: a request whose
    ``downstream_key`` was already applied is a no-op.
    """
    con = sqlite3.connect(db_path)
    if honors_key:
        con.execute(
            "INSERT INTO effects (key, downstream_key, fired_at) "
            "SELECT ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM effects WHERE downstream_key = ?)",
            (key, downstream_key, now_iso(), downstream_key),
        )
    else:
        con.execute(
            "INSERT INTO effects (key, downstream_key, fired_at) VALUES (?, ?, ?)",
            (key, downstream_key, now_iso()),
        )
    con.commit()
    con.close()


def sqlite_rows(db_path: str) -> List[Dict[str, str]]:
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT key, downstream_key, fired_at FROM effects").fetchall()
    con.close()
    return [{"key": k, "downstream_key": d, "fired_at": f} for k, d, f in rows]


def summarize_service(db_path: str, expected_keys: int) -> Dict[str, int]:
    """Summarize the SQLite service. Pure given the db (kept for back-compat)."""
    return summarize_application_rows(sqlite_rows(db_path), expected_keys)


# --------------------------------------------------------------------------- #
# External service B: real HTTP service with idempotency + status query
# --------------------------------------------------------------------------- #


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that does not dump tracebacks on dropped sockets."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:  # noqa: ANN001
        return


class HttpEffectService:
    """A local HTTP side-effect service, durable and independent of the worker.

      * ``POST /effects`` ``{key, downstream_key, honors_key}`` — apply (dedupe
        on ``downstream_key`` when ``honors_key``). Responds
        ``{applied, applications}``; in ``drop_for_key`` mode the response for
        the first matching request is dropped *after* the commit, so the caller
        gets a transport error with an unknown outcome.
      * ``GET /effects/{key}`` — ``{key, applied, applications}`` (status query).
      * ``GET /effects`` — all rows (for summarization).
      * ``POST /reset`` — clear rows/one-shot state.

    ``drop_for_key`` makes the first matching request's response get dropped
    (connection closed). ``drop_before_commit`` selects whether the effect was
    applied before the drop: ``False`` = committed-then-dropped (the effect
    landed, outcome unknown), ``True`` = dropped-without-committing (the effect
    did NOT land, outcome unknown). The two are indistinguishable to the caller
    and are exactly the two forks a reconciling client must handle.
    """

    def __init__(
        self,
        drop_for_key: Optional[str] = None,
        delay_seconds: float = 0.0,
        drop_before_commit: bool = False,
        status_lag_seconds: float = 0.0,
        slow_for_key: Optional[str] = None,
        slow_seconds: float = 0.0,
    ) -> None:
        self._rows: List[Dict[str, object]] = []
        self._lock = threading.Lock()
        self._drop_for_key = drop_for_key
        self._drop_before_commit = drop_before_commit
        self._dropped = False
        self._delay = delay_seconds
        self._status_lag = status_lag_seconds
        self._slow_for_key = slow_for_key
        self._slow_seconds = slow_seconds
        self._server: Optional[ThreadingHTTPServer] = None

    def _visible_rows(self) -> List[Dict[str, object]]:
        """Rows whose effect is visible to a status query.

        ``status_lag_seconds`` models an eventually-consistent downstream: a
        just-applied effect is not yet reported for that window. Must be called
        with the lock held (it reads ``_rows``).
        """
        cutoff = time.time() - self._status_lag
        return [r for r in self._rows if float(r["_applied_at"]) <= cutoff]

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> str:
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # silence
                return

            def _json(self, code: int, obj: Dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read(self) -> Dict:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                return json.loads(raw or b"{}")

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/reset":
                    with service._lock:
                        service._rows = []
                        service._dropped = False
                    self._json(200, {"ok": True})
                    return
                if self.path != "/effects":
                    self.send_error(404)
                    return
                payload = self._read()
                key = str(payload["key"])
                downstream_key = str(payload["downstream_key"])
                honors = bool(payload.get("honors_key"))
                if service._delay:
                    time.sleep(service._delay)
                with service._lock:
                    already = any(r["downstream_key"] == downstream_key for r in service._rows)
                    drop = service._drop_for_key == key and not service._dropped
                    if drop:
                        service._dropped = True
                    # In drop_before_commit mode the request is dropped WITHOUT
                    # the effect landing (the "not applied" ambiguous fork).
                    skip_apply = drop and service._drop_before_commit
                    if honors and already:
                        applied = False
                    elif skip_apply:
                        applied = False
                    else:
                        service._rows.append(
                            {
                                "key": key,
                                "downstream_key": downstream_key,
                                "fired_at": now_iso(),
                                "_applied_at": time.time(),
                            }
                        )
                        applied = True
                    total = sum(1 for r in service._rows if r["downstream_key"] == downstream_key)
                if drop:
                    # Drop the response as a real timeout would. In
                    # drop_before_commit mode the effect never landed; in the
                    # default mode it landed but the caller cannot know.
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self.connection.close()
                    return
                if service._slow_for_key == key and service._slow_seconds:
                    # Slow-but-not-lost response: the effect is committed, the
                    # reply arrives after the client's timeout (a real
                    # in-flight timeout, not a dropped connection).
                    time.sleep(service._slow_seconds)
                self._json(200, {"applied": applied, "applications": total})

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/effects":
                    with service._lock:
                        rows = service._visible_rows()
                    self._json(200, {"rows": rows})
                    return
                if self.path.startswith("/effects/"):
                    key = urllib.parse.unquote(self.path[len("/effects/"):])
                    with service._lock:
                        applications = sum(
                            1 for r in service._visible_rows() if r["key"] == key
                        )
                    self._json(200, {"key": key, "applied": applications > 0, "applications": applications})
                    return
                self.send_error(404)

        self._server = _QuietThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        return self.url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def _http_request(url: str, method: str, path: str, payload: Optional[Dict] = None, timeout: float = 5.0) -> Dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        raise TransportError(f"{type(exc).__name__}: {exc}") from exc
    return json.loads(body) if body else {}


def http_apply(url: str, key: str, downstream_key: str, honors_key: bool, timeout: float = 5.0) -> Dict:
    return _http_request(
        url, "POST", "/effects",
        {"key": key, "downstream_key": downstream_key, "honors_key": honors_key},
        timeout=timeout,
    )


def http_status(url: str, key: str, timeout: float = 5.0) -> Dict:
    return _http_request(url, "GET", f"/effects/{urllib.parse.quote(key)}", timeout=timeout)


def summarize_http(url: str, expected_keys: int) -> Dict[str, int]:
    return summarize_application_rows(_http_request(url, "GET", "/effects")["rows"], expected_keys)


# --------------------------------------------------------------------------- #
# Service dispatch (shared by parent and worker subprocess)
# --------------------------------------------------------------------------- #


def apply_effect(service: str, target: str, key: str, downstream_key: str, honors_key: bool) -> Optional[Dict]:
    if service == "http":
        return http_apply(target, key, downstream_key, honors_key)
    service_apply(target, key, downstream_key, honors_key)
    return None


def status_effect(service: str, target: str, key: str) -> Dict:
    if service == "http":
        return http_status(target, key)
    rows = [r for r in sqlite_rows(target) if r["key"] == key]
    return {"key": key, "applied": bool(rows), "applications": len(rows)}


def summarize_target(service: str, target: str, expected_keys: int) -> Dict[str, int]:
    if service == "http":
        return summarize_http(target, expected_keys)
    return summarize_service(target, expected_keys)


# --------------------------------------------------------------------------- #
# Progress markers for the non-durable strategies
# --------------------------------------------------------------------------- #


def load_marker(marker_file: str) -> int:
    try:
        with open(marker_file, "r", encoding="utf-8") as handle:
            return int(json.load(handle).get("next", 0))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return 0


def save_marker(marker_file: str, next_step: int) -> None:
    tmp = f"{marker_file}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"next": next_step}, handle)
    os.replace(tmp, marker_file)


# --------------------------------------------------------------------------- #
# Worker child process
# --------------------------------------------------------------------------- #


async def _child_run(args: argparse.Namespace) -> int:
    from actae_client import ActaeClient

    service = args.service
    target = args.service_target
    start = load_marker(args.marker_file) if args.strategy == "checkpoint" else 0

    async with ActaeClient(endpoint=args.url, api_key=args.api_key) as client:
        for step in range(start, STEPS):
            key = f"step-{step}"
            params = {"step": step}

            if args.strategy in ("durable", "durable_query"):
                claim = await client.claim_execution(
                    args.channel, key, "effect", params, lease_seconds=LEASE_SECONDS
                )
                if claim.status == "replayed":
                    continue
                if claim.status == "in_progress":
                    # Lease still held by the crashed owner; the caller waits.
                    return 3
                if claim.status == "reclaimed" and args.strategy == "durable_query":
                    # Ambiguous prior attempt: query before re-applying.
                    try:
                        prior = status_effect(service, target, key)
                    except TransportError:
                        prior = {"applied": False}
                    if prior.get("applied"):
                        await client.complete_execution(
                            claim.execution.id, claim.claim_token,
                            result={"step": step, "reconciled": True},
                        )
                        continue
                try:
                    apply_effect(service, target, key, key, args.downstream == "honors_key")
                except TransportError:
                    # The effect may have landed. Reconcile if we can; otherwise
                    # treat the attempt as ambiguous and let recovery retry.
                    if args.strategy == "durable_query":
                        try:
                            prior = status_effect(service, target, key)
                        except TransportError:
                            prior = {"applied": False}
                        if prior.get("applied"):
                            await client.complete_execution(
                                claim.execution.id, claim.claim_token,
                                result={"step": step, "reconciled": True},
                            )
                            continue
                    return 3
                if args.crash_point == "after_effect" and args.crash_at == step:
                    os._exit(137)
                await client.complete_execution(
                    claim.execution.id, claim.claim_token, result={"step": step}
                )
            else:
                # Without a durable operation identity there is no stable key to
                # hand the downstream; each attempt looks like new work.
                downstream_key = f"{key}:{uuid.uuid4().hex}"
                try:
                    apply_effect(service, target, key, downstream_key, args.downstream == "honors_key")
                except TransportError:
                    return 3
                if args.crash_point == "after_effect" and args.crash_at == step:
                    os._exit(137)
                if args.strategy == "checkpoint":
                    save_marker(args.marker_file, step + 1)

            if args.crash_point == "after_complete" and args.crash_at == step:
                os._exit(137)
    return 0


def _child(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="durability_benchmark child")
    parser.add_argument("strategy", choices=STRATEGIES)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--service", choices=SERVICES, default="sqlite")
    parser.add_argument("--service-target", "--service-db", dest="service_target", required=True)
    parser.add_argument("--marker-file", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--downstream", choices=DOWNSTREAMS, required=True)
    parser.add_argument("--crash-at", type=int, default=-1)
    parser.add_argument("--crash-point", choices=CRASH_POINTS, default="after_effect")
    args = parser.parse_args(argv)
    return asyncio.run(_child_run(args))


def _summon_child(config: "Config", workdir: Path, crash: bool, target: str) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "child",
        config.strategy,
        "--channel",
        config.channel,
        "--service",
        config.service,
        "--service-target",
        target,
        "--marker-file",
        str(workdir / "progress.json"),
        "--url",
        config.url,
        "--api-key",
        config.api_key,
        "--downstream",
        config.downstream,
    ]
    if crash:
        cmd += ["--crash-at", str(config.crash_step), "--crash-point", config.crash_point]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode not in (0, 3, 137):
        raise RuntimeError(
            f"child failed rc={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.returncode


# --------------------------------------------------------------------------- #
# Matrix orchestration
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    strategy: str
    crash_point: str
    downstream: str
    channel: str
    url: str
    api_key: str
    crash_step: int = CRASH_STEP
    service: str = "sqlite"

    @property
    def name(self) -> str:
        return f"{self.strategy}|{self.crash_point}|{self.downstream}"


@dataclass
class Result:
    strategy: str
    crash_point: str
    downstream: str
    child_exit_codes: List[int]
    recovery_attempts: int
    metrics: Dict[str, int]
    crash_step: int = CRASH_STEP
    service: str = "sqlite"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_config(config: Config, workdir: Path, recovery_wait: float = RECOVERY_WAIT_SECONDS) -> Result:
    """Run one strategy x crash-point x downstream cell with a real service."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "progress.json").unlink(missing_ok=True)
    service_obj: Optional[HttpEffectService] = None
    if config.service == "http":
        drop_for = (
            f"step-{config.crash_step}"
            if config.crash_point in ("transport_drop", "transport_fail")
            else None
        )
        service_obj = HttpEffectService(
            drop_for_key=drop_for,
            drop_before_commit=(config.crash_point == "transport_fail"),
        )
        target = service_obj.start()
    else:
        target = str(workdir / "service.db")
        init_service(target)

    try:
        exit_codes = [_summon_child(config, workdir, crash=True, target=target)]
        attempts = 1
        while exit_codes[-1] != 0 and attempts < MAX_ATTEMPTS:
            time.sleep(recovery_wait)
            exit_codes.append(_summon_child(config, workdir, crash=False, target=target))
            attempts += 1
        metrics = summarize_target(config.service, target, STEPS)
    finally:
        if service_obj is not None:
            service_obj.stop()

    return Result(
        strategy=config.strategy,
        crash_point=config.crash_point,
        downstream=config.downstream,
        child_exit_codes=exit_codes,
        recovery_attempts=attempts,
        metrics=metrics,
        crash_step=config.crash_step,
        service=config.service,
    )


def run_matrix(
    url: str,
    api_key: str,
    strategies=STRATEGIES,
    crash_points=CRASH_POINTS,
    downstreams=DOWNSTREAMS,
    services=SERVICES,
    crash_steps=(CRASH_STEP,),
    root: Optional[Path] = None,
) -> List[Result]:
    run_id = uuid.uuid4().hex[:8]
    results: List[Result] = []
    base = Path(root) if root else Path(tempfile.mkdtemp(prefix="durability-bench-"))
    for service in services:
        for strategy in strategies:
            for crash_point in crash_points:
                # transport_drop / transport_fail are HTTP-only failure modes.
                if crash_point in ("transport_drop", "transport_fail") and service != "http":
                    continue
                for downstream in downstreams:
                    for crash_step in crash_steps:
                        channel = (
                            f"bench-{run_id}-{service}-{strategy}-{crash_point}-"
                            f"{downstream}-{crash_step}"
                        )
                        workdir = base / f"{service}-{strategy}-{crash_point}-{downstream}-{crash_step}"
                        config = Config(
                            strategy, crash_point, downstream, channel, url, api_key,
                            crash_step=crash_step, service=service,
                        )
                        result = run_config(config, workdir)
                        results.append(result)
                        summary = result.metrics
                        print(
                            f"  {service:<6} {strategy:<13} {crash_point:<14} {downstream:<15} "
                            f"step={crash_step} dup={summary['duplicate_applications']} "
                            f"lost={summary['lost_keys']} applications={summary['total_applications']} "
                            f"exits={result.child_exit_codes}",
                            flush=True,
                        )
    return results


def render_markdown(results: List[Result]) -> str:
    services = {r.service for r in results}
    crash_steps = {r.crash_step for r in results}
    extra = len(services) > 1 or len(crash_steps) > 1
    if not extra:
        lines = [
            "| Strategy | Crash point | Downstream | Duplicate effects | Lost effects | Total applications |",
            "|----------|-------------|------------|------------------:|-------------:|-------------------:|",
        ]
        for result in results:
            m = result.metrics
            lines.append(
                f"| {result.strategy} | {result.crash_point} | {result.downstream} | "
                f"{m['duplicate_applications']} | {m['lost_keys']} | {m['total_applications']} |"
            )
        return "\n".join(lines)

    lines = [
        "| Service | Strategy | Crash point | Crash step | Downstream | Duplicate effects | Lost effects | Total applications |",
        "|---------|----------|-------------|-----------:|------------|------------------:|-------------:|-------------------:|",
    ]
    for result in results:
        m = result.metrics
        lines.append(
            f"| {result.service} | {result.strategy} | {result.crash_point} | {result.crash_step} | "
            f"{result.downstream} | {m['duplicate_applications']} | {m['lost_keys']} | "
            f"{m['total_applications']} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "child":
        return _child(argv[1:])

    parser = argparse.ArgumentParser(description="AI-agent durability benchmark")
    parser.add_argument("--url", default=os.environ.get("ACTAE_URL", "http://127.0.0.1:8002"))
    parser.add_argument("--api-key", default=os.environ.get("ACTAE_API_KEY"))
    parser.add_argument("--out", default="examples/durability-benchmark-results.json")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES)
    parser.add_argument("--crash-points", nargs="+", choices=CRASH_POINTS)
    parser.add_argument("--downstreams", nargs="+", choices=DOWNSTREAMS)
    parser.add_argument("--services", nargs="+", choices=SERVICES)
    parser.add_argument(
        "--crash-at", type=int, action="append", default=None,
        help="crash step to inject (repeatable; default 3)",
    )
    parser.add_argument(
        "--crash-at-all", action="store_true",
        help="inject a crash at every step (overrides --crash-at)",
    )
    parser.add_argument("--json", action="store_true", help="print the JSON report instead of markdown")
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error("--api-key or ACTAE_API_KEY is required")

    crash_steps = tuple(range(STEPS)) if args.crash_at_all else tuple(args.crash_at or [CRASH_STEP])

    services = tuple(args.services or SERVICES)
    results = run_matrix(
        args.url,
        args.api_key,
        strategies=tuple(args.strategies or STRATEGIES),
        crash_points=tuple(args.crash_points or CRASH_POINTS),
        downstreams=tuple(args.downstreams or DOWNSTREAMS),
        services=services,
        crash_steps=crash_steps,
    )
    report = {
        "generated_at": now_iso(),
        "steps": STEPS,
        "crash_step": CRASH_STEP,
        "crash_steps": list(crash_steps),
        "services": list(services),
        "lease_seconds": LEASE_SECONDS,
        "results": [result.to_dict() for result in results],
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print()
    print(render_markdown(results))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
