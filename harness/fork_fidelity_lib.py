"""LLM-judged fidelity infrastructure for fork/resume — production-grade.

Purpose
-------
This is the evidence engine behind the marketing claim:

  > For the same prompt, a fork produces output equivalent to a full
  > no-fork re-run — judged by an LLM, not by string similarity.

For each *scenario* (an unrelated multi-step pipeline) we run three things on
a real Actae server:

  1. **baseline** — the full pipeline,
  2. **re-run**   — the full pipeline again ("the run without the fork"),
  3. **fork**     — fork at step N, re-run only the tail (steps N+1..M),
                     inheriting steps 1..N byte-identically.

A judge LLM (DeepSeek by default, temperature 0) then scores semantic
equivalence of the TAIL outputs:

  * `fork_fidelity`  = judge(baseline_tail, fork_tail)
  * `rerun_fidelity` = judge(baseline_tail, rerun_tail)

Per-scenario pass requires BOTH:

  * `fork_fidelity  >= threshold`                 — the fork output is
                                                    equivalent to baseline,
  * `fork_fidelity  >= rerun_fidelity - tolerance` — forking is not worse
                                                    than re-running.

The inherited prefix (steps 1..N) is verified **byte-identical** structurally
(the fork copies the stored snapshot; there is nothing to judge).

Why the tail?
-------------
The fork's prefix IS the baseline's output (copied). Comparing whole outputs
would trivially score the prefix as identical and mask tail drift. Comparing
only the tail isolates exactly what forking recomputes vs what a re-run
recomputes. Because the fork's tail inputs are byte-identical to baseline's
while a re-run's inputs drift (its steps 1..N differ), the fork has no
disadvantage here — the judge confirms or refutes that empirically.

Honesty contract
----------------
- The judge is the ONLY fidelity metric (no difflib/char-similarity gates).
- Failure is loud: a scenario below threshold or worse than the re-run
  control sets `pass = false` and the suite exits non-zero.
- The report records every score, verdict and judge rationale verbatim, plus
  config and per-run call/token totals, so the evidence is auditable.
- `dry_run` mode uses a deterministic judge and deterministic pipeline output
  — it exercises the plumbing, never the judgment. Only the real mode (real
  LLM judge + real pipeline) produces evidence to present.

Run via `examples/fork_fidelity_suite.py`; tested by
`sdks/example-tests/python/test_fork_fidelity_lib.py` (unit) and
`test_fork_fidelity_suite_live.py` (opt-in live).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from actae_client import ActaeClient
from actae_client.errors import APIError, ActaeConnectionError, RateLimitError
from actae_client.session import AgentSession, SessionError

from fork_token_report import StepRegistry

# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


class FidelityError(Exception):
    """Raised for configuration or judge failures that make a result unusable."""


@dataclass
class Step:
    """One pipeline step: an LLM call producing text stored under `key`."""

    key: str
    system: str
    user: Callable[[Dict[str, Any]], str]
    temperature: float = 0.6
    max_tokens: int = 500


@dataclass
class Scenario:
    """One self-contained, unrelated pipeline to test.

    `fork_at_step` is the number of inherited steps (1-based): the fork keeps
    steps `1..fork_at_step` and recomputes `fork_at_step+1..len(steps)`.
    `input_text` (optional) is embedded into the live state as `_input` for
    steps that consume a fixed input (code snippets, source documents, ...).
    """

    name: str
    description: str
    topic: str
    steps: List[Step]
    fork_at_step: int
    input_text: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


def validate_scenario(sc: Scenario) -> None:
    """Reject a scenario whose configuration would produce meaningless data."""
    if not sc.name or not re.match(r"^[a-z0-9][a-z0-9_-]*$", sc.name):
        raise FidelityError(f"scenario name {sc.name!r} must be a lowercase slug")
    if not sc.steps:
        raise FidelityError(f"scenario {sc.name}: no steps")
    keys = [s.key for s in sc.steps]
    if len(set(keys)) != len(keys):
        raise FidelityError(f"scenario {sc.name}: duplicate step keys {keys}")
    if not (1 <= sc.fork_at_step < len(sc.steps)):
        raise FidelityError(
            f"scenario {sc.name}: fork_at_step={sc.fork_at_step} must be in "
            f"[1, {len(sc.steps) - 1}] (needs ≥1 inherited step and ≥1 tail step)"
        )


def inherited_keys(sc: Scenario) -> List[str]:
    return [s.key for s in sc.steps[: sc.fork_at_step]]


def tail_keys(sc: Scenario) -> List[str]:
    return [s.key for s in sc.steps[sc.fork_at_step :]]


# ---------------------------------------------------------------------------
# Judge — the ONLY fidelity metric
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an impartial fidelity judge for agent-pipeline experiments.

You compare TWO OUTPUTS of the same multi-step pipeline and score how
semantically equivalent they are. A CANDIDATE output is judged against a
REFERENCE output.

Rules:
- Judge FACTS, CONCLUSIONS and COVERAGE — never wording, style, length or
  formatting. Different phrasing, ordering or emphasis is NORMAL (the model
  samples) and must not lower the score.
- When a pipeline step is asked to PROPOSE or INVENT specific values (prices,
  targets, estimates, timelines), different proposed values for the SAME item
  are NOT factual contradictions — both are valid samples. What matters is
  whether the candidate covered the same set of items and its reasoning
  matches. Do NOT penalize differing invented numbers; DO penalize a missing
  item, a genuinely wrong one, or reversed reasoning.
- Equivalent = same facts and conclusions, same coverage of key points.
- Missing, added, or contradicted KEY facts lower the score.
- Return ONLY a JSON object with exactly these fields:
  {"score": <integer 1-10>, "verdict": "equivalent"|"minor_differences"|"substantive_differences", "rationale": "<1-2 sentences>"}
- score 9-10: equivalent; 7-8: minor differences, no substantive change;
  4-6: some key points missing or wrong; 1-3: substantially different."""


def _build_judge_user(reference: str, candidate: str) -> str:
    return (
        "REFERENCE output:\n"
        "--------------------\n"
        f"{reference}\n\n"
        "CANDIDATE output:\n"
        "--------------------\n"
        f"{candidate}\n"
    )


def parse_judge_json(text: str) -> Dict[str, Any]:
    """Parse the judge's JSON reply, tolerating code fences and loose output."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw).rstrip("`").strip()
    try:
        obj = json.loads(raw)
    except ValueError:
        m_score = re.search(r'"score"\s*:\s*(\d{1,2})', raw)
        if not m_score:
            raise FidelityError(f"judge reply unparseable: {text[:200]!r}")
        m_verdict = re.search(r'"verdict"\s*:\s*"([^"]+)"', raw)
        obj = {
            "score": int(m_score.group(1)),
            "verdict": m_verdict.group(1) if m_verdict else "unknown",
            "rationale": raw[:300],
        }
    score = obj.get("score")
    if not isinstance(score, int) or not (1 <= score <= 10):
        raise FidelityError(f"judge score out of range: {score!r} ({text[:200]!r})")
    verdict = obj.get("verdict", "unknown")
    rationale = obj.get("rationale", "")
    return {"score": score, "verdict": verdict, "rationale": rationale}


@dataclass
class JudgeSample:
    score: int
    verdict: str
    rationale: str


@dataclass
class FidelityComparison:
    """Aggregated judge result for one (reference, candidate) pair."""

    score: float
    verdict: str
    rationale: str
    samples: List[JudgeSample]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "verdict": self.verdict,
            "rationale": self.rationale,
            "samples": [
                {"score": s.score, "verdict": s.verdict, "rationale": s.rationale}
                for s in self.samples
            ],
        }


def _verdict_for(score: float) -> str:
    if score >= 9.0:
        return "equivalent"
    if score >= 7.0:
        return "minor_differences"
    if score >= 4.0:
        return "some_substantive_differences"
    return "substantively_different"


class Judge:
    """Protocol: score(reference, candidate) -> FidelityComparison."""

    async def score(self, reference: str, candidate: str) -> FidelityComparison:
        raise NotImplementedError


class LLMJudge(Judge):
    """Real judge: an LLM call per sample (DeepSeek by default, temp 0).

    `reps` samples are averaged; all sample scores, verdicts and rationales
    are kept for the auditable report. Retries on rate limits and on
    unparseable replies (bounded)."""

    def __init__(
        self,
        chat: Any,
        *,
        model: str,
        reps: int = 2,
        max_tokens: int = 256,
        retries: int = 3,
    ) -> None:
        self._chat = chat
        self._model = model
        self._reps = reps
        self._max_tokens = max_tokens
        self._retries = retries

    async def score(self, reference: str, candidate: str) -> FidelityComparison:
        samples: List[JudgeSample] = []
        for _ in range(self._reps):
            samples.append(await self._sample(reference, candidate))
        score = sum(s.score for s in samples) / len(samples)
        rationale = " | ".join(s.rationale for s in samples)
        return FidelityComparison(
            score=score,
            verdict=_verdict_for(score),
            rationale=rationale,
            samples=samples,
        )

    async def _sample(self, reference: str, candidate: str) -> JudgeSample:
        last_err: Optional[Exception] = None
        for attempt in range(self._retries):
            try:
                resp = await self._chat.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": _build_judge_user(reference, candidate)},
                    ],
                    temperature=0.0,
                    max_tokens=self._max_tokens,
                )
                content = resp.choices[0].message.content or ""
                parsed = parse_judge_json(content)
                return JudgeSample(
                    score=parsed["score"],
                    verdict=parsed["verdict"],
                    rationale=parsed["rationale"],
                )
            except RateLimitError as e:  # retry with backoff
                last_err = e
                await asyncio.sleep(min(5 * (2 ** attempt), 30))
            except FidelityError as e:  # unparseable → retry once, then give up
                last_err = e
                if attempt == self._retries - 1:
                    raise FidelityError(f"judge failed after {self._retries} attempts: {e}") from e
                await asyncio.sleep(1)
        raise FidelityError(f"judge failed after {self._retries} attempts: {last_err}")


class DeterministicJudge(Judge):
    """dry-run judge: char-similarity mapped onto 1-10. Plumbing only.

    Never used for evidence — it exists so the whole suite (runs, extraction,
    thresholds, report) can be exercised without an API key."""

    async def score(self, reference: str, candidate: str) -> FidelityComparison:
        import difflib

        ratio = difflib.SequenceMatcher(None, reference, candidate).ratio()
        score = int(round(1 + ratio * 9))  # 1..10
        verdict = _verdict_for(float(score))
        sample = JudgeSample(score, verdict, f"dry-run deterministic (char-ratio {ratio:.2f})")
        return FidelityComparison(score=float(score), verdict=verdict, rationale=sample.rationale, samples=[sample])


# ---------------------------------------------------------------------------
# Pipeline execution (baseline / re-run / fork)
# ---------------------------------------------------------------------------

DEFAULT_DRY_SYSTEM = "dry-run deterministic step"


async def _llm_text(
    chat: Any,
    model: str,
    step_key: str,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    retries: int = 3,
) -> Tuple[str, int, int]:
    """One LLM call. Real mode calls the model; dry-run returns a
    deterministic token budget. Returns (content, prompt_tokens,
    completion_tokens)."""
    if dry_run:
        # Deterministic: depends only on the step, not on sampling.
        content = f"[dry-run:{step_key}] {system[:80]}"
        return content, 60 + len(user) // 4, 20 + len(content) // 4
    for attempt in range(retries):
        try:
            resp = await chat.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = getattr(resp, "usage", None)
            return (
                (resp.choices[0].message.content or "").strip(),
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
        except RateLimitError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(min(5 * (2 ** attempt), 30))


def _initial_live(sc: Scenario) -> Dict[str, Any]:
    live: Dict[str, Any] = {"topic": sc.topic}
    if sc.input_text:
        live["_input"] = sc.input_text
    return live


async def _run_steps(
    session: AgentSession,
    sc: Scenario,
    registry: StepRegistry,
    live: Dict[str, Any],
    *,
    chat: Any,
    model: str,
    dry_run: bool,
    from_idx: int,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Run scenario steps `from_idx..end` on an already-started session."""
    for i in range(from_idx, len(sc.steps)):
        step = sc.steps[i]
        user = step.user(live)
        content, pt, ct = await _llm_text(
            chat, model, step.key, step.system, user,
            temperature=step.temperature, max_tokens=step.max_tokens, dry_run=dry_run,
        )
        registry.record(step.key, pt, ct, 0)
        delta = {step.key: content}
        live.update(delta)
        if trace is not None:
            trace.append({
                "step_number": i + 1,
                "key": step.key,
                "system": step.system,
                "user": user,
                "temperature": step.temperature,
                "max_tokens": step.max_tokens,
                "output": content,
            })
        await session.step(
            f"step.{i + 1}.{step.key}",
            input=user,
            output={"tokens": registry.step_tokens(step.key)},
            context=delta,
        )


async def run_pipeline(
    actae: ActaeClient,
    sc: Scenario,
    channel: str,
    *,
    chat: Any,
    model: str,
    dry_run: bool,
) -> Tuple[Dict[str, Any], StepRegistry, List[Dict[str, Any]]]:
    """Run the full pipeline (baseline or re-run) on a fresh channel."""
    validate_scenario(sc)
    registry = StepRegistry()
    trace: List[Dict[str, Any]] = []
    live = _initial_live(sc)
    async with AgentSession(
        actae, channel,
        display_name=f"Fidelity {sc.name}",
        state_fn=lambda: dict(live),
        params=dict(sc.params or {}, run="fidelity-pipeline"),
    ) as session:
        await _run_steps(session, sc, registry, live, chat=chat, model=model, dry_run=dry_run, from_idx=0, trace=trace)
    return dict(live), registry, trace


async def run_fork(
    actae: ActaeClient,
    sc: Scenario,
    base_channel: str,
    fork_channel: str,
    *,
    chat: Any,
    model: str,
    dry_run: bool,
) -> Tuple[Dict[str, Any], StepRegistry, List[Dict[str, Any]], Dict[str, Any]]:
    """Fork `base_channel` at `sc.fork_at_step`, run only the tail."""
    validate_scenario(sc)
    registry = StepRegistry()
    trace: List[Dict[str, Any]] = []
    live = _initial_live(sc)

    session = await AgentSession.resume(
        actae, base_channel,
        fork_at_step=sc.fork_at_step,
        name=fork_channel,
        state_fn=lambda: dict(live),
        params=dict(sc.params or {}, run="fidelity-fork"),
    )
    inherited = session.inherited_state or {}
    provenance = {
        "boundary_restorable": session.boundary_restorable,
        "requested_boundary_cursor": session.requested_boundary_cursor,
        "resolved_boundary_cursor": session.resolved_boundary_cursor,
        "source_state_version": session.source_state_version,
        "source_state_sha256": session.source_state_sha256,
        "reproducibility": session.reproducibility,
        "inherited_state": inherited,
    }
    if session.boundary_restorable is not True:
        raise FidelityError(f"{sc.name}: fork boundary is not restorable")
    if session.requested_boundary_cursor != session.resolved_boundary_cursor:
        raise FidelityError(
            f"{sc.name}: requested cursor {session.requested_boundary_cursor} resolved "
            f"to {session.resolved_boundary_cursor}"
        )
    if not session.source_state_sha256:
        raise FidelityError(f"{sc.name}: fork receipt has no source-state fingerprint")
    live.update({k: v for k, v in inherited.items()})
    if sc.input_text and "_input" not in live:
        live["_input"] = sc.input_text

    async with session:
        await _run_steps(session, sc, registry, live, chat=chat, model=model, dry_run=dry_run, from_idx=sc.fork_at_step, trace=trace)
    return dict(live), registry, trace, provenance


# ---------------------------------------------------------------------------
# Comparison helpers (pure)
# ---------------------------------------------------------------------------


def tail_text(state: Dict[str, Any], sc: Scenario) -> str:
    """Join the tail steps' outputs (steps fork_at_step+1..M) into one text."""
    parts = []
    for key in tail_keys(sc):
        val = state.get(key)
        if val:
            parts.append(f"[{key}]\n{val}")
    return "\n\n".join(parts)


def prefix_identical(base_state: Dict[str, Any], fork_state: Dict[str, Any], sc: Scenario) -> Tuple[bool, List[str]]:
    """Byte-compare the inherited steps between baseline and fork."""
    keys = inherited_keys(sc)
    mismatches = [k for k in keys if fork_state.get(k) != base_state.get(k)]
    return (not mismatches), mismatches


# ---------------------------------------------------------------------------
# Scenario evaluation + aggregation
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    description: str
    topic: str
    step_count: int
    fork_at_step: int
    inherited_steps: List[str]
    tail_steps: List[str]
    prefix_identical: bool
    prefix_mismatches: List[str]
    fork: FidelityComparison
    rerun: FidelityComparison
    repeats: List[Dict[str, Any]]
    passed: bool
    fail_reason: Optional[str]
    calls: int
    tokens: int
    channels: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "topic": self.topic,
            "step_count": self.step_count,
            "fork_at_step": self.fork_at_step,
            "inherited_steps": self.inherited_steps,
            "tail_steps": self.tail_steps,
            "prefix_identical": self.prefix_identical,
            "prefix_mismatches": self.prefix_mismatches,
            "fork_fidelity": self.fork.to_dict(),
            "rerun_fidelity": self.rerun.to_dict(),
            "repeats": self.repeats,
            "pass": self.passed,
            "fail_reason": self.fail_reason,
            "llm_calls": self.calls,
            "tokens": self.tokens,
            "channels": self.channels,
        }

    @staticmethod
    def from_parts(
        sc: Scenario,
        prefix_ok: bool,
        prefix_mismatches: List[str],
        fork: FidelityComparison,
        rerun: FidelityComparison,
        repeats: List[Dict[str, Any]],
        calls: int,
        tokens: int,
        channels: List[Dict[str, str]],
    ) -> "ScenarioResult":
        return ScenarioResult(
            name=sc.name,
            description=sc.description,
            topic=sc.topic,
            step_count=len(sc.steps),
            fork_at_step=sc.fork_at_step,
            inherited_steps=inherited_keys(sc),
            tail_steps=tail_keys(sc),
            prefix_identical=prefix_ok,
            prefix_mismatches=prefix_mismatches,
            fork=fork,
            rerun=rerun,
            repeats=repeats,
            passed=False,
            fail_reason=None,
            calls=calls,
            tokens=tokens,
            channels=channels,
        )


def aggregate_comparison(comparisons: List[FidelityComparison]) -> FidelityComparison:
    """Combine judge comparisons from repeated runs into one (mean) comparison.

    Scores are averaged; verdicts map from the mean; all sample rationales are
    kept for the auditable report."""
    score = sum(c.score for c in comparisons) / len(comparisons)
    rationale = " | ".join(c.rationale for c in comparisons if c.rationale)
    samples = [s for c in comparisons for s in c.samples]
    return FidelityComparison(score=score, verdict=_verdict_for(score), rationale=rationale, samples=samples)


def _is_transient(exc: Exception) -> bool:
    """True for errors worth retrying the whole scenario repeat.

    Covers connection drops, rate limits, 5xx, and the dev server's embedded
    postgres restarting mid-run (which makes an earlier channel 404 on the
    subsequent fork — the channel it referenced no longer exists). Each repeat
    is an independent measurement, so re-running it on fresh channels does not
    bias the evidence; the report records that a retry happened."""
    if isinstance(exc, (ActaeConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIError):
        return exc.status_code >= 500 or exc.status_code == 429
    if isinstance(exc, SessionError):
        return True  # e.g. 'Channel ... not found' after a server restart
    return False


async def _run_one_repeat(
    actae: ActaeClient,
    sc: Scenario,
    *,
    chat: Any,
    pipeline_model: str,
    dry_run: bool,
    judge: Judge,
) -> Dict[str, Any]:
    """Run baseline + re-run + fork once, plus both judge comparisons.

    Returns the raw measurements for one independent repeat:
    {prefix_ok, prefix_mismatches, fork, rerun, calls, tokens, channels}."""
    suffix = uuid.uuid4().hex[:8]
    base_ch = f"fid-{sc.name}-{suffix}"
    rerun_ch = f"{base_ch}-rerun"
    fork_ch = f"{base_ch}-fork"

    base_state, base_reg, base_trace = await run_pipeline(actae, sc, base_ch, chat=chat, model=pipeline_model, dry_run=dry_run)
    rerun_state, rerun_reg, rerun_trace = await run_pipeline(actae, sc, rerun_ch, chat=chat, model=pipeline_model, dry_run=dry_run)
    fork_state, fork_reg, fork_trace, fork_provenance = await run_fork(actae, sc, base_ch, fork_ch, chat=chat, model=pipeline_model, dry_run=dry_run)

    prefix_ok, prefix_mismatches = prefix_identical(base_state, fork_state, sc)
    fork_cmp = await judge.score(tail_text(base_state, sc), tail_text(fork_state, sc))
    rerun_cmp = await judge.score(tail_text(base_state, sc), tail_text(rerun_state, sc))

    return {
        "prefix_ok": prefix_ok,
        "prefix_mismatches": prefix_mismatches,
        "fork": fork_cmp,
        "rerun": rerun_cmp,
        "calls": base_reg.total_calls() + rerun_reg.total_calls() + fork_reg.total_calls(),
        "tokens": base_reg.total_tokens() + rerun_reg.total_tokens() + fork_reg.total_tokens(),
        "channels": {"baseline": base_ch, "rerun": rerun_ch, "fork": fork_ch},
        "observations": {
            "baseline": {"state": base_state, "trace": base_trace, "tail": tail_text(base_state, sc)},
            "rerun": {"state": rerun_state, "trace": rerun_trace, "tail": tail_text(rerun_state, sc)},
            "fork": {"state": fork_state, "trace": fork_trace, "tail": tail_text(fork_state, sc)},
        },
        "fork_provenance": fork_provenance,
    }


def scenario_passed(
    prefix_ok: bool,
    prefix_mismatches: List[str],
    fork: FidelityComparison,
    rerun: FidelityComparison,
    *,
    threshold: float,
    tolerance: float,
) -> Tuple[bool, Optional[str]]:
    if not prefix_ok:
        return False, f"inherited prefix mismatch: {', '.join(prefix_mismatches)}"
    if fork.score < threshold:
        return False, f"fork fidelity {fork.score:.1f} < threshold {threshold:.1f}"
    if fork.score < rerun.score - tolerance:
        return False, (
            f"fork fidelity {fork.score:.1f} is > {tolerance:.1f} below the no-fork "
            f"re-run control ({rerun.score:.1f})"
        )
    return True, None


@dataclass
class SuiteReport:
    generated_at: str
    pipeline_model: str
    judge_model: str
    judge_temperature: float
    judge_reps: int
    threshold: float
    tolerance: float
    dry_run: bool
    scenarios: List[ScenarioResult]
    passed: bool
    passed_count: int
    scenario_count: int
    mean_fork_fidelity: float
    mean_rerun_fidelity: float
    worst_fork_fidelity: float
    total_llm_calls: int
    total_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "pipeline_model": self.pipeline_model,
            "judge_model": self.judge_model,
            "judge_temperature": self.judge_temperature,
            "judge_reps": self.judge_reps,
            "threshold": self.threshold,
            "tolerance": self.tolerance,
            "dry_run": self.dry_run,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "overall": {
                "pass": self.passed,
                "passed": self.passed_count,
                "scenarios": self.scenario_count,
                "mean_fork_fidelity": round(self.mean_fork_fidelity, 2),
                "mean_rerun_fidelity": round(self.mean_rerun_fidelity, 2),
                "worst_fork_fidelity": round(self.worst_fork_fidelity, 2),
                "total_llm_calls": self.total_llm_calls,
                "total_tokens": self.total_tokens,
            },
        }


def aggregate(results: List[ScenarioResult], *, threshold: float, tolerance: float, dry_run: bool, pipeline_model: str, judge_model: str, judge_reps: int) -> SuiteReport:
    passed = [r for r in results if r.passed]
    fork_scores = [r.fork.score for r in results] or [0.0]
    rerun_scores = [r.rerun.score for r in results] or [0.0]
    return SuiteReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        pipeline_model=pipeline_model,
        judge_model=judge_model,
        judge_temperature=0.0,
        judge_reps=judge_reps,
        threshold=threshold,
        tolerance=tolerance,
        dry_run=dry_run,
        scenarios=results,
        passed=len(passed) == len(results),
        passed_count=len(passed),
        scenario_count=len(results),
        mean_fork_fidelity=sum(fork_scores) / len(fork_scores),
        mean_rerun_fidelity=sum(rerun_scores) / len(rerun_scores),
        worst_fork_fidelity=min(fork_scores),
        total_llm_calls=sum(r.calls for r in results),
        total_tokens=sum(r.tokens for r in results),
    )


# ---------------------------------------------------------------------------
# Full suite driver
# ---------------------------------------------------------------------------


async def _repeat_with_retry(
    actae: ActaeClient,
    sc: Scenario,
    *,
    chat: Any,
    pipeline_model: str,
    dry_run: bool,
    judge: Judge,
    retries: int = 2,
) -> Tuple[Dict[str, Any], bool]:
    """Run one scenario repeat, retrying on transient failures (server
    restarts, dropped connections, rate limits). Returns (row, was_retried)."""
    for attempt in range(retries + 1):
        try:
            return await _run_one_repeat(actae, sc, chat=chat, pipeline_model=pipeline_model, dry_run=dry_run, judge=judge), attempt > 0
        except Exception as exc:  # noqa: BLE001
            if not _is_transient(exc) or attempt == retries:
                raise
            await asyncio.sleep(min(10 * (2 ** attempt), 60))


async def evaluate_suite(
    actae: ActaeClient,
    scenarios: List[Scenario],
    *,
    chat: Any,
    pipeline_model: str,
    judge_model: str,
    dry_run: bool = False,
    judge_reps: int = 2,
    threshold: float = 7.0,
    tolerance: float = 1.5,
    repeats: int = 1,
    repeat_retries: int = 2,
    run_id: Optional[str] = None,
) -> SuiteReport:
    """Run every scenario (baseline + re-run + fork) and judge the tails.

    `repeats` controls how many independent times each scenario is run; the
    pass/fail claim is evaluated on the MEAN scores across repeats (a single
    LLM sample is noisy — repeats give the honest statistical answer), with
    per-repeat scores kept in the report. `repeat_retries` makes the suite
    survive transient server failures (e.g. the dev server restarting and
    losing in-flight channels) by re-running the affected repeat on fresh
    channels; each repeat is an independent measurement, so a retry does not
    bias the evidence.

    Requires a connected `actae`. `chat` is an AsyncOpenAI client used for
    both the pipeline (real mode) and the judge; it is unused in dry-run."""
    judge: Judge = (
        DeterministicJudge()
        if dry_run
        else LLMJudge(chat, model=judge_model, reps=judge_reps)
    )

    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    results: List[ScenarioResult] = []
    for sc in scenarios:
        fork_cmps: List[FidelityComparison] = []
        rerun_cmps: List[FidelityComparison] = []
        repeat_rows: List[Dict[str, Any]] = []
        prefix_ok_all = True
        prefix_mismatches_all: List[str] = []
        channels_all: List[Dict[str, str]] = []
        total_calls = 0
        total_tokens = 0

        for _ in range(repeats):
            row, retried = await _repeat_with_retry(actae, sc, chat=chat, pipeline_model=pipeline_model, dry_run=dry_run, judge=judge)
            prefix_ok_all = prefix_ok_all and row["prefix_ok"]
            prefix_mismatches_all.extend(row["prefix_mismatches"])
            channels_all.append(row["channels"])
            fork_cmps.append(row["fork"])
            rerun_cmps.append(row["rerun"])
            repeat_rows.append({
                "fork_fidelity": round(row["fork"].score, 2),
                "rerun_fidelity": round(row["rerun"].score, 2),
                "prefix_identical": row["prefix_ok"],
                "retried": retried,
                "channels": row["channels"],
                "observations": row["observations"],
                "fork_provenance": row["fork_provenance"],
            })
            total_calls += row["calls"]
            total_tokens += row["tokens"]

        fork_agg = aggregate_comparison(fork_cmps)
        rerun_agg = aggregate_comparison(rerun_cmps)
        passed, fail_reason = scenario_passed(
            prefix_ok_all, prefix_mismatches_all, fork_agg, rerun_agg,
            threshold=threshold, tolerance=tolerance,
        )
        result = ScenarioResult.from_parts(
            sc=sc,
            prefix_ok=prefix_ok_all,
            prefix_mismatches=prefix_mismatches_all,
            fork=fork_agg,
            rerun=rerun_agg,
            repeats=repeat_rows,
            calls=total_calls,
            tokens=total_tokens,
            channels=channels_all,
        )
        result.passed = passed
        result.fail_reason = fail_reason
        results.append(result)

    return aggregate(
        results,
        threshold=threshold,
        tolerance=tolerance,
        dry_run=dry_run,
        pipeline_model=pipeline_model,
        judge_model=judge_model,
        judge_reps=judge_reps,
    )
