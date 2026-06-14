"""Tests for ``openrabbit.pricing`` — per-PR cost telemetry (SPEC 7.3).

Covers the price table, the per-model cost calculation, and aggregation of a
neutral :class:`~openrabbit.domain.Usage` total into a :class:`CostSummary`.
No network: this module is pure arithmetic over token counts.
"""

from __future__ import annotations

import pytest

from openrabbit import pricing
from openrabbit.domain import Usage


# --------------------------------------------------------------------------- #
# price table                                                                  #
# --------------------------------------------------------------------------- #
class TestPriceTable:
    def test_known_models_present(self):
        table = pricing.PRICE_TABLE
        # The default Bedrock roles (Nova finder, GPT-5.5 verifier) must have a
        # price so the default config produces a $ estimate.
        assert "amazon.nova-pro-v1:0" in table
        assert "openai.gpt-5.5" in table

    def test_price_fields_are_per_million_tokens(self):
        price = pricing.PRICE_TABLE["openai.gpt-5.5"]
        # All four token kinds are priced (USD per 1M tokens).
        assert price.input_per_mtok > 0
        assert price.output_per_mtok > 0
        assert price.cache_read_per_mtok >= 0
        assert price.cache_write_per_mtok >= 0

    def test_cache_read_cheaper_than_input(self):
        # The whole point of prompt caching (SPEC 7.3): a cache read is far
        # cheaper than a fresh input token for every priced model.
        for price in pricing.PRICE_TABLE.values():
            assert price.cache_read_per_mtok <= price.input_per_mtok


# --------------------------------------------------------------------------- #
# price lookup                                                                 #
# --------------------------------------------------------------------------- #
class TestPriceLookup:
    def test_exact_match(self):
        price = pricing.lookup_price("openai.gpt-5.5")
        assert price is pricing.PRICE_TABLE["openai.gpt-5.5"]

    def test_unknown_model_returns_none(self):
        assert pricing.lookup_price("mystery.model-v9:0") is None

    def test_inference_profile_prefix_strip(self):
        # Cross-region inference profiles prefix the model id (e.g.
        # "us.openai.gpt-5.5" or "global.anthropic..."); the lookup should still
        # resolve to the base model price.
        price = pricing.lookup_price("us.amazon.nova-pro-v1:0")
        assert price is pricing.PRICE_TABLE["amazon.nova-pro-v1:0"]


# --------------------------------------------------------------------------- #
# cost math                                                                    #
# --------------------------------------------------------------------------- #
class TestEstimateCost:
    def test_cost_math_on_known_tokens(self):
        # 1,000,000 input @ $X, 500,000 output @ $Y, etc — exact arithmetic.
        price = pricing.ModelPrice(
            input_per_mtok=3.0,
            output_per_mtok=15.0,
            cache_read_per_mtok=0.3,
            cache_write_per_mtok=3.75,
        )
        usage = Usage(
            input_tokens=1_000_000,
            output_tokens=500_000,
            cache_read=2_000_000,
            cache_write=100_000,
        )
        cost = pricing.estimate_cost(usage, price)
        # 3.0 + 7.5 + 0.6 + 0.375 = 11.475
        assert cost == pytest.approx(11.475)

    def test_zero_usage_zero_cost(self):
        price = pricing.PRICE_TABLE["openai.gpt-5.5"]
        assert pricing.estimate_cost(Usage(), price) == pytest.approx(0.0)

    def test_estimate_cost_for_model_unknown_returns_none(self):
        assert pricing.estimate_cost_for_model(Usage(input_tokens=1), "nope") is None

    def test_estimate_cost_for_model_known(self):
        cost = pricing.estimate_cost_for_model(
            Usage(input_tokens=1_000_000), "openai.gpt-5.5"
        )
        assert cost is not None
        assert cost > 0


# --------------------------------------------------------------------------- #
# cost summary                                                                 #
# --------------------------------------------------------------------------- #
class TestCostSummary:
    def test_from_usage_carries_token_totals(self):
        usage = Usage(
            input_tokens=120,
            output_tokens=30,
            cache_read=900,
            cache_write=80,
        )
        summary = pricing.CostSummary.from_usage(usage, model="openai.gpt-5.5")
        assert summary.input_tokens == 120
        assert summary.output_tokens == 30
        assert summary.cache_read == 900
        assert summary.cache_write == 80
        assert summary.calls == 0  # set explicitly by the caller

    def test_from_usage_with_known_model_has_dollar_estimate(self):
        usage = Usage(input_tokens=1_000_000)
        summary = pricing.CostSummary.from_usage(usage, model="openai.gpt-5.5", calls=4)
        assert summary.usd_estimate is not None
        assert summary.usd_estimate > 0
        assert summary.calls == 4

    def test_from_usage_with_unknown_model_has_no_dollar_estimate(self):
        usage = Usage(input_tokens=1_000_000)
        summary = pricing.CostSummary.from_usage(usage, model="mystery.x")
        assert summary.usd_estimate is None

    def test_from_usage_no_model_has_no_dollar_estimate(self):
        usage = Usage(input_tokens=1_000_000)
        summary = pricing.CostSummary.from_usage(usage)
        assert summary.usd_estimate is None

    def test_to_dict_is_json_serializable(self):
        import json

        usage = Usage(input_tokens=10, output_tokens=5, cache_read=3, cache_write=1)
        summary = pricing.CostSummary.from_usage(usage, model="openai.gpt-5.5", calls=2)
        d = summary.to_dict()
        # Round-trips through JSON (CLI emits this in the offline payload).
        round = json.loads(json.dumps(d))
        assert round["inputTokens"] == 10
        assert round["outputTokens"] == 5
        assert round["cacheRead"] == 3
        assert round["cacheWrite"] == 1
        assert round["calls"] == 2
        assert "usdEstimate" in round

    def test_to_dict_rounds_dollar_estimate(self):
        # The $ estimate is rounded to a sane number of decimals for display.
        usage = Usage(input_tokens=1_234_567)
        summary = pricing.CostSummary.from_usage(usage, model="openai.gpt-5.5")
        d = summary.to_dict()
        assert isinstance(d["usdEstimate"], float)


# --------------------------------------------------------------------------- #
# per-role cost aggregation (item 1)                                           #
# --------------------------------------------------------------------------- #
class TestPerRoleCostSummary:
    def test_each_role_priced_at_its_own_rate(self):
        # Two roles with DIFFERENT models: each role's usage must be priced at
        # its own model rate, then summed (not all at the first model's rate).
        finder_usage = Usage(input_tokens=1_000_000, output_tokens=500_000)
        verifier_usage = Usage(input_tokens=200_000, output_tokens=100_000)
        summary = pricing.CostSummary.from_role_usages(
            [
                ("amazon.nova-pro-v1:0", finder_usage),
                ("openai.gpt-5.5", verifier_usage),
            ],
            calls=3,
        )
        expected = pricing.estimate_cost_for_model(
            finder_usage, "amazon.nova-pro-v1:0"
        ) + pricing.estimate_cost_for_model(verifier_usage, "openai.gpt-5.5")
        assert summary.usd_estimate == pytest.approx(expected)
        # Token totals are the aggregate across roles.
        assert summary.input_tokens == 1_200_000
        assert summary.output_tokens == 600_000
        assert summary.calls == 3

    def test_understating_verifier_at_finder_rate_is_avoided(self):
        # The whole point of item 1: pricing the (pricier) verifier tokens at the
        # (cheaper) finder rate understates the bill. Per-role pricing is higher.
        finder_usage = Usage(input_tokens=1_000_000)
        verifier_usage = Usage(input_tokens=1_000_000)
        per_role = pricing.CostSummary.from_role_usages(
            [
                ("amazon.nova-pro-v1:0", finder_usage),
                ("openai.gpt-5.5", verifier_usage),
            ]
        )
        naive = pricing.estimate_cost_for_model(
            finder_usage + verifier_usage, "amazon.nova-pro-v1:0"
        )
        assert per_role.usd_estimate > naive

    def test_unpriced_role_contributes_tokens_but_no_dollars(self):
        # A role whose model is unpriced still adds its tokens but yields no $ for
        # that role; priced roles still produce an estimate (best-effort signal).
        priced = Usage(input_tokens=1_000_000)
        unpriced = Usage(input_tokens=500_000)
        summary = pricing.CostSummary.from_role_usages(
            [
                ("openai.gpt-5.5", priced),
                ("mystery.model-v9:0", unpriced),
            ]
        )
        assert summary.usd_estimate == pytest.approx(
            pricing.estimate_cost_for_model(priced, "openai.gpt-5.5")
        )
        assert summary.input_tokens == 1_500_000

    def test_all_roles_unpriced_has_no_estimate(self):
        summary = pricing.CostSummary.from_role_usages(
            [("mystery.a", Usage(input_tokens=10)), (None, Usage(input_tokens=5))]
        )
        assert summary.usd_estimate is None
        assert summary.input_tokens == 15

    def test_empty_roles_is_zeroed(self):
        summary = pricing.CostSummary.from_role_usages([])
        assert summary.input_tokens == 0
        assert summary.usd_estimate is None
        assert summary.calls == 0
