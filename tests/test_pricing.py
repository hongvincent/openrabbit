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
        summary = pricing.CostSummary.from_usage(
            usage, model="openai.gpt-5.5", calls=4
        )
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
        summary = pricing.CostSummary.from_usage(
            usage, model="openai.gpt-5.5", calls=2
        )
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
