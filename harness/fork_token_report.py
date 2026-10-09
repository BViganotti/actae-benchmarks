"""Token & wall-clock accounting for the fork-resume demo.

This module is the *numbers backend* behind `examples/fork_resume_demo.py`.
It is deliberately standalone: no Actae client, no LLM SDK, no API keys,
stdlib only — so the arithmetic the demo prints to customers is unit-testable
in isolation (`sdks/example-tests/python/test_fork_token_savings.py`).

Honesty contract
----------------
Every figure in a report is the **sum of values the model provider returned
for real calls** (`usage.prompt_tokens` / `usage.completion_tokens` and a
wall-clock duration). Nothing is modelled, extrapolated, averaged or rounded
down. A "saved" figure is literally `baseline_total - fork_total`; the
percentage is that difference divided by the baseline total. If the fork tail
happened to cost *more* than the baseline's same steps (larger prompts from
inherited context), that shows up in the per-step rows and in a note — the
report never hides it.

A caller feeds the registry one `record(...)` per completed LLM call; the
report renderer then computes everything else from those sums.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

# Per-step accumulator shape: step name -> {prompt, completion}
_Usage = Dict[str, Dict[str, int]]


class StepRegistry:
    """Per-step accumulator for calls, prompt/completion tokens and elapsed
    wall-clock time.

    All values are additive sums; `record()` is the single write path and is
    called exactly once per completed LLM call. Failed calls are not recorded
    (nothing was billed/returned), so `total_calls()` equals successful calls.
    """

    def __init__(self) -> None:
        self._calls: Dict[str, int] = {}
        self._tokens: _Usage = {}
        self._elapsed_ms: Dict[str, int] = {}

    def record(self, step: str, prompt_tokens: int, completion_tokens: int, elapsed_ms: int) -> None:
        """Add one call's usage for `step`. Negative values are rejected —
        a real provider never reports negative usage, so accepting them would
        let a bug silently inflate the "savings" number."""
        if prompt_tokens < 0 or completion_tokens < 0 or elapsed_ms < 0:
            raise ValueError(
                f"negative usage for {step!r}: prompt={prompt_tokens} "
                f"completion={completion_tokens} elapsed_ms={elapsed_ms}"
            )
        acc = self._tokens.setdefault(step, {"prompt": 0, "completion": 0})
        acc["prompt"] += prompt_tokens
        acc["completion"] += completion_tokens
        self._elapsed_ms[step] = self._elapsed_ms.get(step, 0) + elapsed_ms
        self._calls[step] = self._calls.get(step, 0) + 1

    def mark(self, step: str) -> None:
        """Count an attempt that did not (yet) produce usage. Use this only
        when you want to count a call that yielded no `usage` object; for
        normal calls `record()` already increments the count."""
        self._calls[step] = self._calls.get(step, 0) + 1

    def count(self, step: str) -> int:
        return self._calls.get(step, 0)

    def calls_by_step(self) -> Dict[str, int]:
        return dict(self._calls)

    def total_calls(self) -> int:
        return sum(self._calls.values())

    def prompt_tokens(self, step: str) -> int:
        return self._tokens.get(step, {}).get("prompt", 0)

    def completion_tokens(self, step: str) -> int:
        return self._tokens.get(step, {}).get("completion", 0)

    def step_tokens(self, step: str) -> int:
        acc = self._tokens.get(step)
        return (acc["prompt"] + acc["completion"]) if acc else 0

    def total_tokens(self) -> int:
        return sum(self.step_tokens(s) for s in self._tokens)

    def elapsed_ms(self, step: str) -> int:
        return self._elapsed_ms.get(step, 0)

    def total_elapsed_ms(self) -> int:
        return sum(self._elapsed_ms.values())

    def steps(self) -> Tuple[str, ...]:
        """Steps in first-recorded order."""
        return tuple(self._tokens)


def _fmt(n: int) -> str:
    return f"{n:,}"


def _ms(ms: int) -> str:
    return f"{ms / 1000.0:.1f}s"


def format_token_report(
    baseline: StepRegistry,
    fork: StepRegistry,
    steps: Sequence[str],
    *,
    mode: str = "measured",
) -> str:
    """Render the token & wall-clock savings report.

    `steps` is the ordered step list (e.g. the pipeline names). A step the
    fork never invoked renders as "—" (it contributed nothing). Everything
    else is computed from the two registries' recorded sums.

    Returns the report as a multi-line string; it is what the demo prints and
    what the tests assert against.
    """
    base_total = baseline.total_tokens()
    fork_total = fork.total_tokens()
    saved = base_total - fork_total
    pct = (saved / base_total * 100.0) if base_total else 0.0
    base_ms = baseline.total_elapsed_ms()
    fork_ms = fork.total_elapsed_ms()

    name_w = max(4, *(len(str(s)) for s in steps)) + 2
    line = "-" * (name_w + 11 + 11)

    out = [
        "  ── Token & wall-clock report ──",
        f"  source: {mode}",
        "",
        "  " + "step".ljust(name_w) + "baseline".rjust(11) + "fork".rjust(11),
        "  " + line,
    ]
    for step in steps:
        b = baseline.step_tokens(step) if baseline.count(step) else 0
        f = fork.step_tokens(step) if fork.count(step) else None
        fcell = _fmt(f) if f is not None else "—"
        out.append(
            "  " + str(step).ljust(name_w) + _fmt(b).rjust(11) + fcell.rjust(11)
        )
    out.append("  " + line)
    out.append(
        "  " + "TOTAL".ljust(name_w) + _fmt(base_total).rjust(11) + _fmt(fork_total).rjust(11)
    )
    out.append("")

    if base_total:
        out.append(f"  tokens saved on the fork run: {_fmt(saved)} ({pct:.1f}% of the baseline)")
    else:
        out.append("  tokens saved on the fork run: 0 (no baseline usage recorded)")
    out.append(
        f"  wall-clock (LLM calls): baseline {_ms(base_ms)} · fork {_ms(fork_ms)} · "
        f"{_ms(base_ms - fork_ms)} saved"
    )
    out.append("")
    out.append("  Per-iteration economics (from the measured numbers):")
    out.append(f"    • naive re-run of all steps:          {_fmt(base_total)} tokens / iteration")
    out.append(f"    • fork run (inherited prefix + tail): {_fmt(fork_total)} tokens / iteration")
    out.append(f"    • each refinement iteration avoids ~{_fmt(saved)} tokens — the prefix the fork never re-runs")
    out.append("")

    tail_diff = sum(
        fork.step_tokens(s) - baseline.step_tokens(s)
        for s in steps
        if fork.count(s) and baseline.count(s)
    )
    if tail_diff > 0:
        out.append(
            "  Note: the fork's tail steps ran with the inherited prefix output in their"
        )
        out.append(
            f"  prompts, so they cost {_fmt(tail_diff)} tokens more than baseline's same steps "
            "(visible in the rows above)."
        )
        out.append("  The saving is the skipped prefix — it is not inflated by the tail.")
    out.append("")

    return "\n".join(out)
