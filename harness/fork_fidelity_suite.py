#!/usr/bin/env python3
"""
Fidelity suite — LLM-judged evidence that fork output == no-fork re-run.

Runs five unrelated pipelines against a real Actae server: for each, a
baseline, a full no-fork re-run, and a fork at step N. A DeepSeek judge
(temperature 0) scores semantic equivalence of the tail outputs, and the
suite asserts:

  1. fork fidelity >= threshold             (fork output equivalent to baseline)
  2. fork fidelity >= rerun fidelity - tolerance   (forking not worse than re-running)

The inherited prefix (steps 1..N) is verified byte-identical.

Prints a summary, writes a JSON evidence report, and exits 0 (all scenarios
passed) or 1 (any failed / hard error).

Requirements:
    pip install -e ./sdks/python
    # DEEPSEEK_API_KEY in actae/.env (not needed with --dry-run)
    cd actae && cargo run -- --dev          # Actae on :8002

Usage:
    python examples/fork_fidelity_suite.py                # real evidence run
    python examples/fork_fidelity_suite.py --dry-run      # plumbing, no API key
    python examples/fork_fidelity_suite.py --scenarios s3-localization,s5-product-brief
    python examples/fork_fidelity_suite.py --judge-reps 3 --threshold 7.0 --tolerance 1.5

Methodology + how to read the report: docs/FORK_FIDELITY.md
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from actae_client import ActaeClient

from fork_fidelity_lib import evaluate_suite
from fork_fidelity_scenarios import ALL_SCENARIOS, scenarios_by_name

ACTAE_URL = os.environ.get("ACTAE_URL", "http://localhost:8002")
ACTAE_API_KEY = os.environ.get("ACTAE_API_KEY", "sk-dev-0000000000000000000000")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "fidelity-reports"

DEEPSEEK_API_KEY: Optional[str] = None


def _load_deepseek_key() -> None:
    global DEEPSEEK_API_KEY
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        DEEPSEEK_API_KEY = key
        return
    env_path = Path(__file__).resolve().parent.parent / "actae" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                DEEPSEEK_API_KEY = line.split("=", 1)[1].strip().strip("\"'")
                return
    raise SystemExit(
        "DEEPSEEK_API_KEY not found. Set it in actae/.env or the environment "
        "(or run with --dry-run, which needs no API key)."
    )


def _summary(report) -> str:
    lines = [
        "Fidelity suite — LLM-judged fork vs no-fork re-run",
        f"generated_at: {report.generated_at}   pipeline_model: {report.pipeline_model}   "
        f"judge_model: {report.judge_model} (temp {report.judge_temperature}, reps {report.judge_reps})",
        f"threshold: {report.threshold}   tolerance: {report.tolerance}   dry_run: {report.dry_run}   "
        f"repeats: {len(report.scenarios[0].repeats) if report.scenarios else 0}",
        "",
    ]
    for r in report.scenarios:
        status = "PASS" if r.passed else "FAIL"
        repeats_txt = ", ".join(
            f"fork={row['fork_fidelity']:.2f}/rerun={row['rerun_fidelity']:.2f}" for row in r.repeats
        )
        lines.append(
            f"  [{status}] {r.name:<22} mean fork={r.fork.score:>4.1f} mean rerun={r.rerun.score:>4.1f} "
            f"prefix_identical={r.prefix_identical}"
        )
        lines.append(f"          per-repeat: {repeats_txt}")
        if r.fail_reason:
            lines.append(f"          reason: {r.fail_reason}")
        lines.append(f"          judge: {r.fork.verdict} — {r.fork.rationale[:140]}")
    lines.append("")
    o = report.to_dict()["overall"]
    lines.append(
        f"  OVERALL: {'PASS' if report.passed else 'FAIL'}  {report.passed_count}/{report.scenario_count} "
        f"scenarios  mean fork={o['mean_fork_fidelity']}  mean rerun={o['mean_rerun_fidelity']}  "
        f"worst fork={o['worst_fork_fidelity']}  calls={o['total_llm_calls']} tokens={o['total_tokens']}"
    )
    return "\n".join(lines)


def _write_report(report, report_dir: Path, run_id: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"fork-fidelity-{run_id}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    (report_dir / "latest.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return path


async def main(args: argparse.Namespace) -> int:
    scenarios = (
        scenarios_by_name([s.strip() for s in args.scenarios.split(",") if s.strip()])
        if args.scenarios
        else ALL_SCENARIOS
    )

    chat = None
    if not args.dry_run:
        _load_deepseek_key()
        chat = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com", timeout=180)

    actae = ActaeClient(api_key=ACTAE_API_KEY, endpoint=ACTAE_URL)
    await actae.connect()
    try:
        report = await evaluate_suite(
            actae,
            scenarios,
            chat=chat,
            pipeline_model=args.model,
            judge_model=args.judge_model,
            dry_run=args.dry_run,
            judge_reps=args.judge_reps,
            threshold=args.threshold,
            tolerance=args.tolerance,
            repeats=args.repeats,
            run_id=args.run_id,
        )
    finally:
        await actae.disconnect()

    print(_summary(report))
    path = _write_report(report, Path(args.report_dir), args.run_id or report.generated_at.replace(":", "-").replace("T", "-")[:19])
    print(f"\n  report: {path}")
    return 0 if report.passed else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-judged fork-resume fidelity suite (marketing-grade evidence).",
    )
    p.add_argument("--dry-run", action="store_true", help="deterministic fake pipeline + judge; no API key (plumbing only)")
    p.add_argument("--scenarios", type=str, default=None, help="comma-separated scenario names (default: all)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="pipeline model (default: deepseek-chat)")
    p.add_argument("--judge-model", default=DEFAULT_MODEL, help="judge model (default: deepseek-chat)")
    p.add_argument("--judge-reps", type=int, default=2, help="judge samples averaged per comparison (default 2)")
    p.add_argument("--repeats", type=int, default=1, help="independent scenario runs averaged per scenario (default 1; 3+ for marketing evidence)")
    p.add_argument("--threshold", type=float, default=7.0, help="min fork fidelity to pass (default 7.0)")
    p.add_argument("--tolerance", type=float, default=1.5, help="max gap fork below re-run to pass (default 1.5)")
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="directory for JSON evidence reports")
    p.add_argument("--run-id", type=str, default=None, help="stable run id (default: timestamp)")
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    if _args.judge_reps < 1 or _args.repeats < 1:
        raise SystemExit("--judge-reps and --repeats must be >= 1")
    sys.exit(asyncio.run(main(_args)))
