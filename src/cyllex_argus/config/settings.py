"""Configuration file (argus.toml) and data directory layout."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, Field, ValidationError, field_validator

Reliability = Literal["A", "B", "C", "D", "E", "F"]


class ConfigError(Exception):
    """Raised when argus.toml cannot be read or is invalid."""


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:32b-instruct"
    keep_alive: str = "10m"  # how long Ollama keeps the model loaded between calls


class AnthropicApiConfig(BaseModel):
    """Bring-your-own-key Anthropic Messages API provider.

    The `api_key` field is read only when the `ANTHROPIC_API_KEY` env var is empty. Env is the
    preferred route: the key stays out of argus.toml, which is a plaintext file. Both are hashed
    on each request via bcrypt-compare before `x-api-key` is set; nothing persists in memory
    longer than the request.
    """
    api_key: str = ""
    model: str = "claude-sonnet-5-5"
    base_url: str = "https://api.anthropic.com"
    anthropic_version: str = "2023-06-01"


class OpenAIApiConfig(BaseModel):
    """Bring-your-own-key OpenAI Chat Completions provider.

    `api_key` is read only when `OPENAI_API_KEY` is unset. Env preferred for the same reason as
    Anthropic. `organization` is optional (OpenAI's `OpenAI-Organization` header) and defaults
    off; set it when a key is scoped to multiple orgs.
    """
    api_key: str = ""
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    organization: str = ""


class LLMSearchConfig(BaseModel):
    # "claude-code-cli", "anthropic-api", or "disabled". Claude Code CLI uses its agentic WebSearch
    # tool; the Anthropic API uses its server-side web_search tool. Ollama / OpenAI-compatible have
    # no search wired in and mean "disabled" here.
    provider: str = "claude-code-cli"
    # How many primary sources a single web search asks for. None (default) ties the fan-out to
    # the run's max_docs, so the document budget the operator picks also governs discovery. Set a
    # number to pin it regardless of max_docs (e.g. to cap cost: each web search request is billed).
    max_results: int | None = Field(default=None, ge=1)


class AgentModelConfig(BaseModel):
    """Per-agent override. Both fields optional: empty = inherit the base `[llm]` provider /
    model. `provider` must be one of the known providers (claude-code-cli, ollama, anthropic-api,
    openai-api) or ''. The router validates at startup.

    This lets an operator run Triage on a cheap/fast model (e.g. openai-api + gpt-4o-mini) while
    keeping Analyst on the expensive one (anthropic-api + claude-sonnet-5-5) — the usual sweet
    spot since Triage is a yes/no decision and Analyst does the heavy extraction.
    """
    provider: str = ""
    model: str = ""


class PerAgentModelsConfig(BaseModel):
    """One override per agent that calls the LLM. Omitting an agent (or leaving its fields '')
    makes it inherit the base `[llm]` settings. Validator is deterministic (quote verification +
    rule-based scoring, no LLM) and STIX import is a parser-only path, so neither appears here."""
    triage: AgentModelConfig = Field(default_factory=AgentModelConfig)
    analyst: AgentModelConfig = Field(default_factory=AgentModelConfig)
    summary: AgentModelConfig = Field(default_factory=AgentModelConfig)


class LLMConfig(BaseModel):
    # "claude-code-cli", "ollama", "anthropic-api", or "openai-api".
    provider: str = "claude-code-cli"
    model: str | None = None
    timeout_seconds: int = 600
    claude_binary: str = "claude"
    # How the trace prices a `claude-code-cli` call. That provider runs on a Claude Pro/Max
    # subscription, which is flat-rate: no tokens are billed, so any figure is notional.
    #   "list" (default) - price it from PRICES_USD_PER_MTOK like every other provider, which
    #                      answers "what would this run have cost on a metered API key?" and keeps
    #                      the dashboard cost column populated and comparable across providers.
    #   "none"           - record no cost for subscription calls. Tokens, model and duration are
    #                      still recorded; only the money column stays empty, because on a
    #                      subscription it is not money that was spent.
    # Metered providers (anthropic-api, openai-api) ignore this; Ollama is already zero-cost.
    subscription_cost_basis: Literal["list", "none"] = "list"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)  # read only when provider = "ollama"
    anthropic: AnthropicApiConfig = Field(default_factory=AnthropicApiConfig)  # provider = "anthropic-api"
    openai: OpenAIApiConfig = Field(default_factory=OpenAIApiConfig)  # provider = "openai-api"
    agents: PerAgentModelsConfig = Field(default_factory=PerAgentModelsConfig)
    search: LLMSearchConfig = Field(default_factory=LLMSearchConfig)


class LimitsConfig(BaseModel):
    max_docs: int = 25
    chunk_chars: int = 60_000
    max_chunks_per_document: int = 12  # caps LLM calls per document; later parts are not analyzed
    download_timeout_seconds: float = 30.0
    max_download_bytes: int = 25_000_000


class PipelineConfig(BaseModel):
    """Behaviour of a run that is not a hard limit.

    `summary_on_finish` costs one extra LLM call per investigation and is what makes the
    dashboard answer "what did we learn?" the moment a run ends, instead of showing the
    operator an empty card and a button. Turn it off on a metered provider, or when the
    operator prefers to pay for a summary only when they ask for one.
    """

    summary_on_finish: bool = True


class ScoringConfig(BaseModel):
    base_weight: float = 0.6
    reliability_weights: dict[str, float] = Field(
        default_factory=lambda: {"A": 1.0, "B": 0.85, "C": 0.7, "D": 0.5, "E": 0.3, "F": 0.5}
    )
    recency_years: int = 3
    stale_factor: float = 0.75
    # Separate (harsher) discount for claims whose evidence has NO publication date at all. An
    # undated page is weaker than a known-ancient page: with "published 2019" we at least know
    # when the claim was true; with "undated" we do not. Both used to share stale_factor (the
    # audit's "dated-and-ancient scores same as undated" footgun). Default 0.5 keeps undated
    # claims scoring below ancient-but-dated ones without zeroing them out entirely.
    undated_factor: float = 0.5
    fuzzy_threshold: float = 95
    # A refuter contests an assessment only when its (effective) reliability weight is at least
    # this threshold. Keeps a single F- or E-grade source from flipping a well-corroborated fact
    # to CONTESTED. Default 0.6 matches `base_weight` (so A/B/C refuters count; D/E/F do not).
    contested_min_weight: float = 0.6

    @field_validator("reliability_weights", mode="before")
    @classmethod
    def validate_reliability_weights(cls, v):
        if not isinstance(v, dict):
            return v
        defaults = {"A": 1.0, "B": 0.85, "C": 0.7, "D": 0.5, "E": 0.3, "F": 0.5}
        invalid_keys = set(v.keys()) - set(defaults.keys())
        if invalid_keys:
            raise ValueError(f"Invalid reliability weights keys: {invalid_keys}. Must be one of {set(defaults.keys())}")
        result = defaults.copy()
        result.update(v)
        return result


class LoggingConfig(BaseModel):
    # Three modes for how much LLM I/O survives in the pipeline trace:
    # - "off"    (default): zero storage, best privacy, hard to debug after the fact.
    # - "errors": store raw ONLY on `decision in {rejected, failed, parse-error, error}`.
    #             90% of the debugging value at ~10% of the storage cost on a healthy run.
    # - "full":  store every triage/analyst prompt+response. Pays its weight when you want
    #            to replay a scoring pass without re-calling the LLM.
    # Accepts the legacy `true`/`false` shape (true → full, false → off) for one-release
    # backward compatibility.
    retain_raw_llm: Literal["off", "errors", "full"] | bool = "off"

    @field_validator("retain_raw_llm", mode="after")
    @classmethod
    def _coerce_bool(cls, v: object) -> str:  # type: ignore[override]
        if isinstance(v, bool):
            return "full" if v else "off"
        return str(v)


class EnricherConfig(BaseModel):
    """Per-enricher toggle plus the two optional credential fields used by key-gated
    enrichers (VirusTotal today; others later).

    `enabled = False` is the OPSEC-safe default — nothing crosses the machine's boundary
    unless the operator explicitly opts in to that specific enricher. `api_key` and
    `env_api_key` are empty by default; enrichers that need no key (URLhaus, GreyNoise
    Community, crt.sh, NVD, KEV) ignore both.
    """
    enabled: bool = False
    api_key: str = ""
    env_api_key: str = ""


class EnrichmentConfig(BaseModel):
    """[enrichment] section of argus.toml. Each key under `enrichers` matches an enricher's `name`."""
    enrichers: dict[str, EnricherConfig] = Field(default_factory=dict)


class FeedConfig(BaseModel):
    name: str
    url: str


class MispServerConfig(BaseModel):
    """One external MISP instance to pull Events from. Each `argus investigate` run queries
    each configured server for events whose `info` field mentions the target's aliases, then
    emits every link-shaped Attribute (`type` in {link, url, uri}) as a CandidateSource —
    same path as RSS and TAXII.

    Credential precedence per server: env var (if `env_api_key` names one, that wins) →
    `api_key` on the toml → no auth. Env is preferred because the key never touches disk.
    MISP uses `Authorization: <key>` (NOT `Bearer <key>`) — the connector keeps this header
    shape verbatim.

    `timestamp_days` limits the pull to the recent window; MISP accepts a duration string
    like `180d`. None lets the server choose. Default 180 keeps a long tail of context
    without turning each investigation into a multi-megabyte pull.
    """
    name: str
    url: str                    # MISP base URL (e.g. "https://misp.example.com")
    api_key: str = ""           # auth header stored in toml (fallback — env preferred)
    env_api_key: str = ""       # name of the env var that holds the key (preferred route)
    timestamp_days: int | None = 180
    enabled: bool = True


class WebhookConfig(BaseModel):
    """One outbound webhook subscribed to investigation terminal state transitions.

    Fires on FINISHED and/or PAUSED depending on `on_states`. The default (`[]`) means
    "fire on every terminal state"; set to `["finished"]` to only alert on success, or
    `["paused"]` to only alert on operator-stops / rate-limits.

    When `secret` is set, every POST carries an `X-Argus-Signature: sha256=<hex>` header
    (HMAC-SHA256 over `timestamp + "." + raw_body`) and an `X-Argus-Timestamp: <iso>`
    matching the body's `timestamp` field — the Stripe / GitHub webhook pattern. Env
    override via `env_secret` naming an env var; env wins over toml.
    """
    url: str
    secret: str = ""
    env_secret: str = ""
    on_states: list[str] = Field(default_factory=list)  # [] = both finished + paused
    enabled: bool = True


class TaxiiServerConfig(BaseModel):
    """One external TAXII 2.1 server to pull CTI from. Each `argus investigate` run queries each
    configured server for STIX `report` objects that mention the target's aliases, and feeds the
    matching links into the Collector pipeline as additional candidate documents — same path as
    RSS entries.

    Credential precedence per server: env var (if `env_api_key` names one, that wins) → `api_key`
    on the toml → `username` + `password` basic auth → no auth. Env is preferred because the key
    never touches disk.

    `added_after_days` limits the pull to the recent window; use None for "whatever the server
    has". Default 180 keeps a long-tail of context without turning each investigation into a
    multi-megabyte pull.
    """
    name: str
    url: str                    # TAXII 2.1 base URL (e.g. "https://taxii.example.com/taxii2/")
    collection: str             # collection id or alias; "*" means every readable collection
    api_key: str = ""           # Bearer token stored in toml (fallback — env preferred)
    env_api_key: str = ""       # name of the env var that holds the key (preferred route)
    username: str = ""          # optional HTTP Basic
    password: str = ""
    added_after_days: int | None = 180
    enabled: bool = True


def _default_feeds() -> list[FeedConfig]:
    return [
        FeedConfig(name="Microsoft Threat Intelligence", url="https://www.microsoft.com/en-us/security/blog/feed/"),
        FeedConfig(name="Unit 42", url="https://unit42.paloaltonetworks.com/feed/"),
        FeedConfig(name="Google Threat Intelligence", url="https://cloudblog.withgoogle.com/topics/threat-intelligence/rss/"),
        FeedConfig(name="CrowdStrike", url="https://www.crowdstrike.com/blog/feed/"),
        FeedConfig(name="Cisco Talos", url="https://blog.talosintelligence.com/rss/"),
        FeedConfig(name="CISA", url="https://www.cisa.gov/cybersecurity-advisories/all.xml"),
        FeedConfig(name="The DFIR Report", url="https://thedfirreport.com/feed/"),
        FeedConfig(name="Securelist", url="https://securelist.com/feed/"),
        FeedConfig(name="ESET WeLiveSecurity", url="https://www.welivesecurity.com/en/rss/feed/"),
    ]




class AuthConfig(BaseModel):
    """Admin credentials for the web login screen.

    - `ARGUS_ADMIN_PASSWORD` env (highest priority): plain password; hashed in-memory on boot
      and used verbatim for `/api/auth/login`. Preferred — keeps the hash off disk.
    - `admin_password_hash` (bcrypt): persisted hash; written by `argus set-admin-password`.
    - If BOTH are empty, the login endpoint returns 503 and the server stays in the URL-token
      model (dev / backwards-compat).

    `admin_username` can be overridden with `ARGUS_ADMIN_USERNAME` (default `admin`).
    """
    admin_username: str = "admin"
    admin_password_hash: str = ""  # bcrypt hash


class WatchersConfig(BaseModel):
    """Watchers (recurring monitors). `min_reliability` is the Admiralty source-reliability floor
    a change must clear to produce a feed event / alert — 'C' means "fairly reliable" and up."""
    min_reliability: Reliability = "C"


class ArgusConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    feeds: list[FeedConfig] = Field(default_factory=_default_feeds)
    taxii: list[TaxiiServerConfig] = Field(default_factory=list)
    misp: list[MispServerConfig] = Field(default_factory=list)
    webhooks: list[WebhookConfig] = Field(default_factory=list)
    publishers: dict[str, Reliability] = Field(default_factory=dict)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    watchers: WatchersConfig = Field(default_factory=WatchersConfig)


def load_config(path: Path) -> ArgusConfig:
    """Load argus.toml. A missing file yields the defaults."""
    if not path.exists():
        return ArgusConfig()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return ArgusConfig.model_validate(data)
    except (tomllib.TOMLDecodeError, ValidationError, OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Invalid configuration in {path}: {exc}") from exc


def save_config(cfg: ArgusConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(cfg.model_dump(exclude_none=True)), encoding="utf-8")


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def config_file(self) -> Path:
        return self.home / "argus.toml"

    @property
    def db_file(self) -> Path:
        return self.home / "argus.db"

    @property
    def raw_dir(self) -> Path:
        return self.home / "raw"

    @property
    def attack_file(self) -> Path:
        return self.home / "attack" / "enterprise-attack.json"

    def ensure(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.attack_file.parent.mkdir(parents=True, exist_ok=True)


def get_paths(home: Path | None = None) -> Paths:
    """Resolve the Argus home directory.

    Precedence: an explicit `home` arg (tests) → the `ARGUS_HOME` env var → `~/.cyllex-argus`.
    """
    if home is not None:
        return Paths(home)
    env = os.environ.get("ARGUS_HOME")
    if env:
        return Paths(Path(env))
    return Paths(Path.home() / ".cyllex-argus")


_TABLE_HEADER = re.compile(r"^\s*\[\[?\s*([^\[\]]+?)\s*\]\]?\s*(?:#.*)?$")
_LLM_TABLES = {"llm", "llm.ollama", "llm.search"}


def _llm_table(llm: LLMConfig) -> dict:
    return {"llm": llm.model_dump(exclude_none=True)}


def update_llm_config(path: Path, llm: LLMConfig) -> None:
    """Replace the [llm], [llm.ollama] and [llm.search] tables of argus.toml; every other byte stays as it was.

    The three tables are spliced out of the raw text, so comments and formatting elsewhere survive. If the file
    uses a layout the splice cannot handle (dotted keys, inline tables), it is rewritten from its parsed data with
    the new LLM tables; that fallback drops comments.
    """
    new_block = tomli_w.dumps(_llm_table(llm))
    if not path.exists():
        _write_atomic(path, new_block.encode("utf-8"))
        return
    try:
        original = path.read_bytes().decode("utf-8")
        before = tomllib.loads(original)
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Invalid configuration in {path}: {exc}") from exc
    eol = "\r\n" if "\r\n" in original else "\n"
    expected = {**before, **_llm_table(llm)}
    spliced = _splice_llm_tables(original, new_block.replace("\n", eol), eol)
    try:
        ok = tomllib.loads(spliced) == expected
    except tomllib.TOMLDecodeError:
        ok = False
    text = spliced if ok else tomli_w.dumps(expected)
    _write_atomic(path, text.encode("utf-8"))


def _splice_llm_tables(text: str, block: str, eol: str) -> str:
    kept: list[str] = []
    inserted = False
    skipping = False
    for line in text.splitlines(keepends=True):
        match = _TABLE_HEADER.match(line.rstrip("\r\n"))
        if match:
            name = re.sub(r"\s*\.\s*", ".", match.group(1))
            skipping = name in _LLM_TABLES and not line.lstrip().startswith("[[")
            if skipping and not inserted:
                kept.append(block + eol)
                inserted = True
        if not skipping:
            kept.append(line)
    if not inserted:
        if kept and not kept[-1].endswith("\n"):
            kept[-1] += eol
        if kept and kept[-1].strip():
            kept.append(eol)
        kept.append(block)
    return "".join(kept)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
