"""Per-PR cost telemetry — price table + cost math (SPEC 7.3).

Prompt caching is the #1 cost lever (cache the ~25K shared prefix once per PR,
read it at ~0.1x per file). To prove that lever works we need to *measure* it:
the orchestrator sums the neutral :class:`~openrabbit.domain.Usage` across every
model call in a review and turns that total into a :class:`CostSummary`
(input/output/cache-read/cache-write token totals + an optional USD estimate).

This module is pure arithmetic over token counts — no cloud SDKs, no network. It
deliberately holds only a *small* per-model table (the default Bedrock roles);
unknown models still report token totals, just without a dollar figure.

Prices are USD per **1,000,000 tokens** (per-MTok), the unit every vendor
publishes. They are best-effort public-list estimates and are config-overridable
later; the goal is an order-of-magnitude cost signal in CI, not billing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional

from openrabbit.domain import Usage

#: One million — prices are quoted per this many tokens.
_MTOK = 1_000_000

#: Decimals kept on the USD estimate when serialized for display.
_USD_DISPLAY_DECIMALS = 4


@dataclass(frozen=True)
class ModelPrice:
    """USD price per 1M tokens for one model, split by token kind.

    ``cache_read`` is the discounted price for tokens served from the prompt
    cache; ``cache_write`` is the (usually small) surcharge to *populate* the
    cache. Both default to the input price's neighborhood for models where a
    finer split is unknown.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float


#: Best-effort public-list prices (USD / 1M tokens) for the default roles plus a
#: few likely models. Cache reads are the ~0.1x lever; cache writes ~1.25x input
#: (the common Bedrock/Anthropic caching economics). Override-friendly: callers
#: can pass their own :class:`ModelPrice` to :func:`estimate_cost`.
PRICE_TABLE: dict[str, ModelPrice] = {
    # DEPRECATED legacy Gen-1 finder: 5K output cap, no reasoning, no global
    # profile, EOL signal — prefer amazon.nova-2-lite-v1:0; priced here only for
    # backward-compat so existing configs still get a cost estimate.
    "amazon.nova-pro-v1:0": ModelPrice(
        input_per_mtok=0.80,
        output_per_mtok=3.20,
        cache_read_per_mtok=0.20,
        cache_write_per_mtok=1.00,
    ),
    "amazon.nova-lite-v1:0": ModelPrice(
        input_per_mtok=0.06,
        output_per_mtok=0.24,
        cache_read_per_mtok=0.015,
        cache_write_per_mtok=0.075,
    ),
    "amazon.nova-micro-v1:0": ModelPrice(
        input_per_mtok=0.035,
        output_per_mtok=0.14,
        cache_read_per_mtok=0.00875,
        cache_write_per_mtok=0.04375,
    ),
    # Verifier / judge — strongest available, touches few files.
    # gpt-5.5 is the real Bedrock list rate (was 1.25/10.00 here — a 4x
    # understatement; corrected 2026-06-23 against the Bedrock pricing page).
    "openai.gpt-5.5": ModelPrice(
        input_per_mtok=5.50,
        output_per_mtok=33.00,
        cache_read_per_mtok=0.55,
        cache_write_per_mtok=5.50,
    ),
    # gpt-5.4 — current default verifier (lower fabrication rate, ~half the
    # gpt-5.5 cost; see openrabbit-model-choice). Best-effort Bedrock list rate.
    "openai.gpt-5.4": ModelPrice(
        input_per_mtok=2.75,
        output_per_mtok=16.50,
        cache_read_per_mtok=0.275,
        cache_write_per_mtok=2.75,
    ),
    # nova-2-lite — current default triage/finder (Converse, global. profile).
    # cache-read rate: AWS's standard Nova prompt-cache discount is ~10% of
    # input ($0.03/MTok), so $0.03 is the AWS-documented-standard rate. The
    # $0.075 used here is a deliberately conservative figure — FLAG for
    # empirical verification before locking budgets.
    "amazon.nova-2-lite-v1:0": ModelPrice(
        input_per_mtok=0.30,
        output_per_mtok=2.50,
        cache_read_per_mtok=0.075,
        cache_write_per_mtok=0.30,
    ),
    # nova-premier — EOL 2026-09-14 — do not assign to new roles.
    "amazon.nova-premier-v1:0": ModelPrice(
        input_per_mtok=2.50,
        output_per_mtok=12.50,
        cache_read_per_mtok=0.625,
        cache_write_per_mtok=2.50,
    ),
    # Premium (optional, cost-gated) — Claude on Bedrock.
    "anthropic.claude-opus-4-6-v1": ModelPrice(
        input_per_mtok=15.00,
        output_per_mtok=75.00,
        cache_read_per_mtok=1.50,
        cache_write_per_mtok=18.75,
    ),
    "anthropic.claude-sonnet-4-5-v1": ModelPrice(
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cache_read_per_mtok=0.30,
        cache_write_per_mtok=3.75,
    ),
}

#: Cross-region inference-profile prefixes that wrap a base model id. A profile
#: id like ``us.openai.gpt-5.5`` or ``global.anthropic.claude-...`` prices the
#: same as its bare base model, so we strip these before the table lookup.
_PROFILE_PREFIXES: tuple[str, ...] = (
    "us.",
    "use1.",
    "use2.",
    "apac.",
    "eu.",
    "global.",
)


def lookup_price(model: str) -> Optional[ModelPrice]:
    """Return the :class:`ModelPrice` for ``model``, or ``None`` if unknown.

    Tries an exact match first, then retries after stripping a known
    cross-region inference-profile prefix (``us.``/``global.``/...), so a profile
    id resolves to its base model's price.
    """
    price = PRICE_TABLE.get(model)
    if price is not None:
        return price
    for prefix in _PROFILE_PREFIXES:
        if model.startswith(prefix):
            base = model[len(prefix) :]
            price = PRICE_TABLE.get(base)
            if price is not None:
                return price
    return None


def estimate_cost(usage: Usage, price: ModelPrice) -> float:
    """USD cost for ``usage`` priced with ``price`` (per-MTok arithmetic)."""
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
        + usage.cache_read * price.cache_read_per_mtok
        + usage.cache_write * price.cache_write_per_mtok
    ) / _MTOK


def estimate_cost_for_model(usage: Usage, model: str) -> Optional[float]:
    """USD cost for ``usage`` under ``model``'s price, or ``None`` if unpriced."""
    price = lookup_price(model)
    if price is None:
        return None
    return estimate_cost(usage, price)


@dataclass(frozen=True)
class CostSummary:
    """A per-PR cost roll-up: token totals + an optional USD estimate.

    ``model`` and ``usd_estimate`` are optional: when the review mixes models or
    targets an unpriced model, token totals are still reported and the dollar
    figure is simply omitted (``None``).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0
    model: Optional[str] = None
    usd_estimate: Optional[float] = None

    @classmethod
    def from_usage(
        cls,
        usage: Usage,
        *,
        model: Optional[str] = None,
        calls: int = 0,
    ) -> CostSummary:
        """Build a summary from a (typically aggregated) :class:`Usage`.

        When ``model`` is priced, attach a USD estimate; otherwise leave it
        ``None`` (token totals still carry the cost signal).
        """
        usd = estimate_cost_for_model(usage, model) if model else None
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read=usage.cache_read,
            cache_write=usage.cache_write,
            calls=calls,
            model=model,
            usd_estimate=usd,
        )

    @classmethod
    def from_role_usages(
        cls,
        role_usages: Iterable[tuple[Optional[str], Usage]],
        *,
        calls: int = 0,
    ) -> CostSummary:
        """Build a summary pricing each ``(model, usage)`` at *its own* rate.

        A review touches multiple model roles (Nova finder vs. GPT-5.5 verifier)
        whose per-token prices differ sharply. Summing all tokens and pricing
        them at a single model's rate understates (or overstates) the bill — so
        each role's :class:`Usage` is priced at its own model's rate and the
        per-role dollar figures are summed (SPEC 7.3, item 1).

        Token totals are the aggregate across every role. The USD estimate is the
        sum of the *priced* roles' costs; unpriced (or model-less) roles still
        contribute their tokens but no dollars. When no role is priced the
        estimate is ``None`` (token totals still carry the signal). ``model`` is
        left ``None`` because the summary spans multiple models.
        """
        total_usage = Usage()
        total_usd: Optional[float] = None
        for model, usage in role_usages:
            total_usage = total_usage + usage
            cost = estimate_cost_for_model(usage, model) if model else None
            if cost is not None:
                total_usd = cost if total_usd is None else total_usd + cost
        return cls(
            input_tokens=total_usage.input_tokens,
            output_tokens=total_usage.output_tokens,
            cache_read=total_usage.cache_read,
            cache_write=total_usage.cache_write,
            calls=calls,
            model=None,
            usd_estimate=total_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly camelCase dict (CLI output / logs)."""
        return {
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "cacheRead": self.cache_read,
            "cacheWrite": self.cache_write,
            "calls": self.calls,
            "model": self.model,
            "usdEstimate": (
                round(self.usd_estimate, _USD_DISPLAY_DECIMALS)
                if self.usd_estimate is not None
                else None
            ),
        }
