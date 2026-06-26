"""Cost hook — accumulates token usage and dollar cost across a run.

Reads usage off each Response. Since `after_tool` doesn't see the Response, the
loop doesn't push usage here directly in the MVP; instead the provider's Usage
is summed via `record()` which a caller (or a future after_model hook) invokes.
Kept simple: a tally object with per-model pricing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.core.types import Usage

# (input, output) USD per 1M tokens. Cache reads bill at ~0.1x input.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usd: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, model: str, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        self.by_model[model] = self.by_model.get(model, 0) + 1

        in_price, out_price = PRICING.get(model, (0.0, 0.0))
        # cache reads bill at 0.1x input; cache writes at 1.25x input
        billable_in = usage.input_tokens + 0.1 * usage.cache_read_tokens \
            + 1.25 * usage.cache_write_tokens
        self.usd += (billable_in * in_price + usage.output_tokens * out_price) / 1_000_000

    def summary(self) -> str:
        return (
            f"${self.usd:.4f} | in={self.input_tokens} out={self.output_tokens} "
            f"cache_r={self.cache_read_tokens} cache_w={self.cache_write_tokens}"
        )
