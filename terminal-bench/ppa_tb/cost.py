"""Cost estimation and budget guard.

The point of this module is to make the expensive decision visible *before* the
spend, because the grid multiplies faster than intuition suggests. The full
factorial in the design document is

    tasks x rejection-cells x arms x affordance x models x reps
    24 x 9 x 2 x 3 x 2 x 3 = 23,328 episodes

which is not a pilot, it is a research program. Everything below exists to find a
configuration that produces a publishable table at two orders of magnitude less.

Two cost centres, and they behave differently:

  * **Agent tokens** — scales with episode count AND trajectory length. Dominant.
  * **Container time** — scales with episode count plus gate adjudication. Small
    per unit, but the gate re-runs multiply it (see ``gate.PredicateRunner``,
    which is lazy for exactly this reason).

Defaults are deliberately rough. Replace them with measured values from your own
pilot before trusting any total: ``ppa_tb estimate --from-job`` does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostModel:
    """Per-episode cost assumptions. Override from a measured pilot."""

    # agent side
    mean_input_tokens: int = 180_000        # long trajectories re-send context
    mean_output_tokens: int = 25_000
    input_price_per_mtok: float = 3.00
    output_price_per_mtok: float = 15.00

    # container side
    episode_container_minutes: float = 12.0
    gate_container_minutes: float = 1.5     # per predicate evaluation
    compute_price_per_hour: float = 0.10    # local docker ~ 0; cloud sandboxes > 0

    # adjudication behaviour
    frac_episodes_touching_gate: float = 0.15
    mean_predicate_runs_when_touched: float = 3.3   # measured, lazy (eager = 15)

    def agent_cost_per_episode(self) -> float:
        return (self.mean_input_tokens / 1e6 * self.input_price_per_mtok
                + self.mean_output_tokens / 1e6 * self.output_price_per_mtok)

    def compute_cost_per_episode(self) -> float:
        gate = (self.frac_episodes_touching_gate
                * self.mean_predicate_runs_when_touched
                * self.gate_container_minutes)
        return (self.episode_container_minutes + gate) / 60 * self.compute_price_per_hour

    def per_episode(self) -> float:
        return self.agent_cost_per_episode() + self.compute_cost_per_episode()


@dataclass
class Estimate:
    n_episodes: int
    agent_cost: float
    compute_cost: float
    wall_clock_hours: float
    breakdown: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.agent_cost + self.compute_cost

    def render(self) -> str:
        lines = [
            f"episodes                : {self.n_episodes}",
            f"agent cost              : ${self.agent_cost:,.0f}",
            f"compute cost            : ${self.compute_cost:,.0f}",
            f"TOTAL                   : ${self.total:,.0f}",
            f"wall clock (concurrent) : {self.wall_clock_hours:.1f} h",
        ]
        if self.breakdown:
            lines.append("")
            lines.append("grid factors:")
            for k, v in self.breakdown.items():
                lines.append(f"  {k:<22} x{v}")
        return "\n".join(lines)


def estimate(cfg, model: CostModel | None = None) -> Estimate:
    m = model or CostModel()
    n = cfg.size()
    per_ep_minutes = m.episode_container_minutes + (
        m.frac_episodes_touching_gate * m.mean_predicate_runs_when_touched
        * m.gate_container_minutes)
    return Estimate(
        n_episodes=n,
        agent_cost=n * m.agent_cost_per_episode(),
        compute_cost=n * m.compute_cost_per_episode(),
        wall_clock_hours=n * per_ep_minutes / 60 / max(1, cfg.n_concurrent),
        breakdown={
            "tasks": len(cfg.tasks),
            "models": len(cfg.models),
            "arms": len(cfg.arms),
            "finality levels": len(cfg.finalities),
            "personalization levels": len(cfg.personalizations),
            "affordance levels": len(cfg.affordance_levels),
            "flaw types": len(cfg.flaw_types),
            "repetitions": cfg.repetitions,
        },
    )


class BudgetExceeded(RuntimeError):
    pass


def guard(cfg, budget_usd: float, model: CostModel | None = None) -> Estimate:
    """Refuse to launch a grid whose projected cost exceeds the cap.

    Raises rather than warns. A warning printed above 120 lines of harness output
    is a warning nobody reads, and the failure mode it protects against - noticing
    a 20x overspend after the run - is not recoverable.
    """
    est = estimate(cfg, model)
    if est.total > budget_usd:
        raise BudgetExceeded(
            f"projected ${est.total:,.0f} exceeds budget ${budget_usd:,.0f} "
            f"for {est.n_episodes} episodes.\n"
            f"Cheapest levers, in order of effect:\n"
            f"  - pin the rejection matrix to one cell   (up to 9x)\n"
            f"  - drop the affordance factor to one level (up to 3x)\n"
            f"  - reduce repetitions from {cfg.repetitions} to 2\n"
            f"  - cut tasks from {len(cfg.tasks)}\n"
            f"Note that cutting TASKS costs you the task-clustered CIs and the "
            f"domain stratification, so cut grid factors first."
        )
    return est


def calibrate_from_pilot(episodes_jsonl: str) -> CostModel:
    """Fit the cost model to a completed pilot job. Use before scaling."""
    import json
    from pathlib import Path

    n = sum(1 for _ in Path(episodes_jsonl).open())
    touched = 0
    for line in Path(episodes_jsonl).open():
        d = json.loads(line)
        if d.get("gate_verdict") != "no_gate_edit":
            touched += 1
    m = CostModel()
    if n:
        m.frac_episodes_touching_gate = touched / n
    return m


# --------------------------------------------------------------------------- #
# OpenRouter presets
# --------------------------------------------------------------------------- #
#
# Prices are USD per million tokens, verified August 2026. OpenRouter is
# passthrough: you pay the provider rate plus a credit-purchase fee (modelled as
# ``surcharge`` below), not a per-token markup. VERIFY AT openrouter.ai/models
# BEFORE RELYING ON A TOTAL - this table will go stale, and it has moved fast:
# OpenAI raised its top standard tier from $1.25/$10 to $5/$30 within a year
# while Anthropic cut Opus and added a higher tier above it.

MODEL_PRICES: dict[str, tuple[float, float]] = {
    # frontier
    "openrouter/anthropic/claude-fable-5":      (10.00, 50.00),
    "openrouter/openai/gpt-5.5-pro":            (30.00, 180.00),
    "openrouter/openai/gpt-5.6-sol":             (5.00, 30.00),
    "openrouter/openai/gpt-5.5":                 (5.00, 30.00),
    "openrouter/anthropic/claude-opus-4.8":      (5.00, 25.00),
    # mid
    "openrouter/anthropic/claude-sonnet-5":      (3.00, 15.00),
    "openrouter/openai/gpt-5.6-terra":           (2.00, 12.00),
    "openrouter/google/gemini-3.1-pro":          (2.00, 12.00),
    "openrouter/x-ai/grok-4.5":                  (2.00,  6.00),
    # budget
    "openrouter/openai/gpt-5.1":                 (1.25, 10.00),
    "openrouter/deepseek/deepseek-v4-flash":     (0.14,  0.28),
}


@dataclass
class MixedCostModel(CostModel):
    """Cost model that prices each model in the grid separately.

    The flat single-price CostModel is fine for a rough total but wrong for this
    design, because the grid deliberately spans a capability range: the whole
    point of the frontier plot is to contrast a strong model against a weak one,
    and on current pricing that spread is also a 35x price spread. Averaging them
    hides which half of the bill you are actually paying.
    """

    surcharge: float = 0.055          # OpenRouter credit-purchase fee
    cache_hit_rate: float = 0.0       # see note in estimate_mixed()

    def cost_for(self, model_id: str) -> float:
        price = MODEL_PRICES.get(model_id)
        if price is None:
            price = (self.input_price_per_mtok, self.output_price_per_mtok)
        pin, pout = price
        billed_in = self.mean_input_tokens * (1 - self.cache_hit_rate * 0.9)
        return (billed_in / 1e6 * pin
                + self.mean_output_tokens / 1e6 * pout) * (1 + self.surcharge)


def estimate_mixed(cfg, model: MixedCostModel | None = None) -> Estimate:
    """Per-model estimate. Episodes split evenly across the models in the grid."""
    m = model or MixedCostModel()
    per_model = cfg.size() // max(1, len(cfg.models))
    agent = 0.0
    breakdown = {}
    for mid in cfg.models:
        c = m.cost_for(mid) * per_model
        agent += c
        known = "" if mid in MODEL_PRICES else "  (PRICE UNKNOWN - using defaults)"
        breakdown[mid] = f"{per_model} ep x ${m.cost_for(mid):.2f} = ${c:,.0f}{known}"
    per_ep_minutes = m.episode_container_minutes + (
        m.frac_episodes_touching_gate * m.mean_predicate_runs_when_touched
        * m.gate_container_minutes)
    return Estimate(
        n_episodes=cfg.size(),
        agent_cost=agent,
        compute_cost=cfg.size() * m.compute_cost_per_episode(),
        wall_clock_hours=cfg.size() * per_ep_minutes / 60 / max(1, cfg.n_concurrent),
        breakdown=breakdown,
    )
