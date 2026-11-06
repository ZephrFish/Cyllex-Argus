"""Provider-neutral LLM interface. Every model call in Argus goes through LLMProvider."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMRequest(BaseModel):
    system: str
    content: str
    prompt_version: str
    tools: list[str] = []
    # Which pipeline agent is making this request. The LLMRouter looks up `cfg.agents.<name>`
    # to decide whether to send it to the base provider or an agent-specific override.
    # Known values: "triage", "analyst", "summary", "ttps", "stix_import", "web_search", "".
    # Unknown or empty = base provider is used verbatim.
    agent: str = ""


class LLMUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    model: str
    duration_ms: int
    # Server-side web searches billed on this call (Anthropic web_search tool). 0 for every
    # provider/call that did not search. Priced separately from tokens ($10 / 1000 searches).
    web_search_requests: int = 0
    # False when the trace must NOT price this call from the static table. Set by the
    # subscription provider under `[llm] subscription_cost_basis = "none"`, where per-token
    # pricing describes no real charge. Tokens and model are still recorded either way.
    price_from_table: bool = True


class LLMResponse[T: BaseModel](BaseModel):
    output: T
    usage: LLMUsage | None = None  # None when the provider does not report it (e.g. the fake)


class LLMError(Exception):
    """Base class for provider errors."""


class RateLimited(LLMError):
    """Usage or rate limit reached: save progress and pause the run."""


class LLMUnavailable(LLMError):
    """The provider cannot be used at all (not installed, not logged in)."""


class InvalidOutput(LLMError):
    """The provider answered, but not with valid structured output."""


class LLMTimeout(InvalidOutput):
    """The provider did not answer in time. Re-sending the same request would only waste quota."""


class LLMRefused(LLMError):
    """The provider declined the request on safety-policy grounds (e.g. Anthropic's cyber
    classifier blocking offensive content). A sibling of InvalidOutput, NOT a subclass: a
    refusal will not go away on retry, so it must bypass the retry helper and propagate so the
    pipeline can report it as a distinct, actionable condition rather than a generic failure."""

    def __init__(self, message: str | None = None, *, category: str | None = None,
                 explanation: str | None = None) -> None:
        self.category = category
        self.explanation = explanation
        super().__init__(message or explanation or f"request refused by the model's safety policy ({category or 'unspecified'})")


class LLMProvider(Protocol):
    def complete_json(self, request: LLMRequest, schema: type[T]) -> LLMResponse[T]: ...
