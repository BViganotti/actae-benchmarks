"""Fidelity test scenarios — five unrelated pipelines.

Each scenario is a different domain with a different structure (step count,
fork point, temperature profile, input kind) so the fidelity claim is not
pinned to one shape. See `docs/FORK_FIDELITY.md` for the methodology.

Scenarios:
  s1-market-entry       market analysis brief         (5 steps, fork@2, topic)
  s2-code-review        Python code review report     (5 steps, fork@3, input_text)
  s3-localization       FR translation + adaptation  (3 steps, fork@1, input_text)
  s4-incident-analysis  outage postmortem            (4 steps, fork@1, input_text)
  s5-product-brief      feature spec + GTM           (4 steps, fork@3, topic)
"""

from typing import Any, Dict

from fork_fidelity_lib import Scenario, Step, validate_scenario


def _s(state: Dict[str, Any], *keys: str) -> str:
    """Join the given live-state keys into a context block for a step prompt."""
    return "\n\n".join(state[k] for k in keys if state.get(k))


# ---------------------------------------------------------------------------
# s1 — market analysis brief
# ---------------------------------------------------------------------------

S1_MARKET = Scenario(
    name="s1-market-entry",
    description="Market-entry analysis for a plant-based milk startup in Germany",
    topic="Launching a plant-based milk startup in Germany in 2026",
    fork_at_step=2,
    params={"domain": "market-analysis"},
    steps=[
        Step(
            key="research",
            system=(
                "You are a market research analyst. Produce exactly 3 concrete, "
                "quantified findings about this market (size, growth, consumers)."
            ),
            user=lambda state: state["topic"],
            temperature=0.5,
            max_tokens=350,
        ),
        Step(
            key="competitors",
            system=(
                "You are a competitive strategist. Name the 3 most relevant "
                "competitors and one positioning fact about each."
            ),
            user=lambda state: _s(state, "research"),
            temperature=0.5,
            max_tokens=300,
        ),
        Step(
            key="pricing",
            system=(
                "You are a pricing strategist. Recommend a concrete pricing model "
                "with specific price points and margins."
            ),
            user=lambda state: _s(state, "research", "competitors"),
            temperature=0.6,
            max_tokens=350,
        ),
        Step(
            key="risks",
            system=(
                "You are a risk analyst. List the top 4 risks with a one-line mitigation each."
            ),
            user=lambda state: _s(state, "research", "competitors", "pricing"),
            temperature=0.6,
            max_tokens=350,
        ),
        Step(
            key="recommendation",
            system=(
                "You are the lead consultant. Write the final go/no-go recommendation "
                "with the 3 strongest reasons."
            ),
            user=lambda state: _s(state, "research", "competitors", "pricing", "risks"),
            temperature=0.4,
            max_tokens=350,
        ),
    ],
)

# ---------------------------------------------------------------------------
# s2 — code review report
# ---------------------------------------------------------------------------

S2_CODE_SNIPPET = """def process_payment(order, user):
    total = sum(item["price"] * item["qty"] for item in order["items"])
    if total > user["balance"]:
        return "insufficient funds"
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO payments (user_id, total, ref) VALUES ('%s', '%s', '%s')" %
        (user["id"], total, order["ref"])
    )
    cursor.execute(
        "UPDATE users SET balance = balance - %s WHERE id = '%s'" %
        (total, user["id"])
    )
    conn.commit()
    return "ok"
"""

S2_CODE_REVIEW = Scenario(
    name="s2-code-review",
    description="Security/correctness review of a payment-handling Python snippet",
    topic="Code review of a payment handler",
    input_text=S2_CODE_SNIPPET,
    fork_at_step=3,
    params={"domain": "code-review"},
    steps=[
        Step(
            key="understand",
            system="You are a senior engineer. Summarize in 3-4 sentences what this code does.",
            user=lambda state: _s(state, "_input"),
            temperature=0.3,
            max_tokens=250,
        ),
        Step(
            key="bugs",
            system=(
                "You are a code reviewer. List concrete bugs with severity and the "
                "line/section where each occurs."
            ),
            user=lambda state: _s(state, "_input", "understand"),
            temperature=0.4,
            max_tokens=400,
        ),
        Step(
            key="security",
            system=(
                "You are a security reviewer. List security issues: injection, "
                "authorization, data exposure. Be specific."
            ),
            user=lambda state: _s(state, "_input"),
            temperature=0.5,
            max_tokens=350,
        ),
        Step(
            key="improvements",
            system="You are a performance/quality expert. Suggest 3 concrete improvements.",
            user=lambda state: _s(state, "understand", "bugs"),
            temperature=0.5,
            max_tokens=300,
        ),
        Step(
            key="verdict",
            system=(
                "You are the review lead. Give the final verdict: is this mergeable, "
                "and what are the top blocking issues."
            ),
            user=lambda state: _s(state, "bugs", "security", "improvements"),
            temperature=0.4,
            max_tokens=350,
        ),
    ],
)

# ---------------------------------------------------------------------------
# s3 — translation + adaptation
# ---------------------------------------------------------------------------

S3_SOURCE_TEXT = (
    "Autoregressive large language models generate text token by token: at each "
    "step the model predicts a probability distribution over the vocabulary and "
    "samples the next token. Decoding strategies such as temperature scaling "
    "reshape that distribution to trade off diversity against determinism. KV "
    "caching avoids recomputing attention over earlier tokens, which is why "
    "longer contexts increase latency roughly linearly rather than quadratically."
)

S3_LOCALIZATION = Scenario(
    name="s3-localization",
    description="Translate an English technical paragraph to French and adapt it for a general audience",
    topic="Localization of an LLM technical explainer",
    input_text=S3_SOURCE_TEXT,
    fork_at_step=1,
    params={"domain": "localization"},
    steps=[
        Step(
            key="translate",
            system=(
                "You are a professional technical translator. Translate the text "
                "to French, keeping the technical terms accurate."
            ),
            user=lambda state: _s(state, "_input"),
            temperature=0.3,
            max_tokens=350,
        ),
        Step(
            key="adapt",
            system=(
                "You are an editor. Rewrite the French translation so a general "
                "(non-technical) audience can understand it, without changing the facts."
            ),
            user=lambda state: _s(state, "translate"),
            temperature=0.5,
            max_tokens=350,
        ),
        Step(
            key="glossary",
            system=(
                "You are a terminology manager. List the 5 most important technical "
                "terms: English term, French translation, one-line explanation."
            ),
            user=lambda state: _s(state, "_input", "translate"),
            temperature=0.4,
            max_tokens=300,
        ),
    ],
)

# ---------------------------------------------------------------------------
# s4 — incident analysis (postmortem)
# ---------------------------------------------------------------------------

S4_INCIDENT = (
    "At 09:14 UTC the payments API started returning 5xx for ~12% of traffic. "
    "P95 latency rose from 120ms to 8s. At 09:22 a second deployment rolled out "
    "new auth middleware. At 09:31 the error rate hit 38% and the API was put in "
    "read-only mode. At 09:47 the auth middleware was rolled back; latency did "
    "NOT improve. At 10:02 the nightly batch job was restarted (it had crashed at "
    "09:08 and left 40,000 jobs queued, each holding a DB connection); latency "
    "recovered within minutes and the queue drained by 10:40. The DB connection "
    "pool maxed out at 09:26. Post-incident investigation confirmed the batch "
    "job's crash left queued jobs holding pooled connections until the pool "
    "exhausted; the auth middleware rollout was coincidental and verified blameless."
)

S4_INCIDENT_ANALYSIS = Scenario(
    name="s4-incident-analysis",
    description="Postmortem of a fictional payments-API outage",
    topic="Outage postmortem analysis",
    input_text=S4_INCIDENT,
    fork_at_step=1,
    params={"domain": "incident-analysis"},
    steps=[
        Step(
            key="timeline",
            system=(
                "You are an SRE. Reconstruct the incident timeline from the report, "
                "in order with timestamps and one-line evidence."
            ),
            user=lambda state: _s(state, "_input"),
            temperature=0.3,
            max_tokens=300,
        ),
        Step(
            key="rootcause",
            system=(
                "You are the incident commander. Identify the most likely root cause "
                "and why it was not caught by monitoring."
            ),
            user=lambda state: _s(state, "_input", "timeline"),
            temperature=0.5,
            max_tokens=350,
        ),
        Step(
            key="blastradius",
            system=(
                "You are a systems analyst. Determine the blast radius (users, "
                "services, data integrity) and the customer impact."
            ),
            user=lambda state: _s(state, "timeline", "rootcause"),
            temperature=0.5,
            max_tokens=300,
        ),
        Step(
            key="actions",
            system=(
                "You are a reliability engineer. List concrete action items with "
                "owner and priority to prevent recurrence."
            ),
            user=lambda state: _s(state, "timeline", "rootcause", "blastradius"),
            temperature=0.4,
            max_tokens=350,
        ),
    ],
)

# ---------------------------------------------------------------------------
# s5 — product brief (offline mode)
# ---------------------------------------------------------------------------

S5_PRODUCT = Scenario(
    name="s5-product-brief",
    description="Product brief for adding offline mode to a note-taking app",
    topic="Adding an offline mode to a cross-platform note-taking app",
    fork_at_step=3,
    params={"domain": "product-brief"},
    steps=[
        Step(
            key="userresearch",
            system=(
                "You are a product researcher. Synthesize 3 concrete user needs for "
                "offline mode, each with a supporting scenario."
            ),
            user=lambda state: state["topic"],
            temperature=0.5,
            max_tokens=350,
        ),
        Step(
            key="spec",
            system=(
                "You are a product manager. Write a concise functional spec: scope, "
                "MVP, non-goals, and 2 edge cases."
            ),
            user=lambda state: _s(state, "userresearch"),
            temperature=0.4,
            max_tokens=400,
        ),
        Step(
            key="gtm",
            system=(
                "You are a growth lead. Outline the launch plan: target segments, "
                "messaging angle, and rollout."
            ),
            user=lambda state: _s(state, "userresearch", "spec"),
            temperature=0.5,
            max_tokens=300,
        ),
        Step(
            key="metrics",
            system=(
                "You are an analytics lead. Define 5 success metrics with baseline "
                "and target values."
            ),
            user=lambda state: _s(state, "userresearch", "spec", "gtm"),
            temperature=0.4,
            max_tokens=300,
        ),
    ],
)


ALL_SCENARIOS: list = [
    S1_MARKET,
    S2_CODE_REVIEW,
    S3_LOCALIZATION,
    S4_INCIDENT_ANALYSIS,
    S5_PRODUCT,
]

for _sc in ALL_SCENARIOS:
    validate_scenario(_sc)


def scenarios_by_name(names: list) -> list:
    by_name = {sc.name: sc for sc in ALL_SCENARIOS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown scenarios: {', '.join(unknown)}; available: {', '.join(by_name)}")
    return [by_name[n] for n in names]
