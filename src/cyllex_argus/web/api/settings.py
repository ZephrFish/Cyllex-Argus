import shutil
from datetime import UTC
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from cyllex_argus.config.settings import (
    AgentModelConfig,
    AnthropicApiConfig,
    ArgusConfig,
    ConfigError,
    LLMConfig,
    LLMSearchConfig,
    MispServerConfig,
    OllamaConfig,
    OpenAIApiConfig,
    PerAgentModelsConfig,
    TaxiiServerConfig,
    WebhookConfig,
    load_config,
    save_config,
    update_llm_config,
)
from cyllex_argus.knowledge.browse import BrowseQueries
from cyllex_argus.llm.prices import known_models_by_provider
from cyllex_argus.llm.router import KNOWN_PROVIDERS
from cyllex_argus.notify.webhook import WebhookDispatcher
from cyllex_argus.sources.misp import MispClient, MispError
from cyllex_argus.sources.taxii import TaxiiClient, TaxiiError
from cyllex_argus.web.deps import open_services

router = APIRouter()

PROVIDERS = ("claude-code-cli", "ollama", "anthropic-api", "openai-api")
SEARCH_PROVIDERS = ("claude-code-cli", "anthropic-api", "disabled")
MIN_TIMEOUT, MAX_TIMEOUT = 10, 3600
TEST_TIMEOUT_SECONDS = 5.0


class OllamaBody(BaseModel):
    base_url: str
    model: str
    keep_alive: str


class AnthropicBody(BaseModel):
    api_key: str = ""
    model: str = ""
    base_url: str = "https://api.anthropic.com"
    anthropic_version: str = "2023-06-01"


class OpenAIBody(BaseModel):
    api_key: str = ""
    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    organization: str = ""


class SearchBody(BaseModel):
    provider: str
    max_results: int | None = None


class AgentOverrideBody(BaseModel):
    provider: str = ""
    model: str = ""


class AgentsBody(BaseModel):
    """Per-agent overrides. Any missing agent inherits the base `[llm]` block."""
    triage: AgentOverrideBody = Field(default_factory=AgentOverrideBody)
    analyst: AgentOverrideBody = Field(default_factory=AgentOverrideBody)
    summary: AgentOverrideBody = Field(default_factory=AgentOverrideBody)


class LLMUpdate(BaseModel):
    provider: str
    model: str | None = None
    timeout_seconds: int
    claude_binary: str
    ollama: OllamaBody
    anthropic: AnthropicBody | None = None
    openai: OpenAIBody | None = None
    agents: AgentsBody | None = None
    search: SearchBody


class OllamaTestBody(BaseModel):
    base_url: str
    model: str | None = None


class LLMTest(BaseModel):
    provider: str
    claude_binary: str | None = None
    ollama: OllamaTestBody | None = None
    anthropic: AnthropicBody | None = None
    openai: OpenAIBody | None = None


def _bad(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _validate_base_url(value: str) -> str:
    if not value.startswith(("http://", "https://")) or " " in value or len(value) <= len("https://"):
        raise _bad("ollama.base_url must start with http:// or https://")
    return value


def _stored_llm(services) -> LLMConfig:
    """The LLM config as argus.toml holds it; a missing file means the live config (the defaults)."""
    path = services.paths.config_file
    return load_config(path).llm if path.exists() else services.config.llm


def llm_view(llm: LLMConfig, restart_required: bool) -> dict:
    # Masked view of the API keys: the UI shows whether a key is configured without exposing
    # the plaintext — the key is a secret, not something the Settings page needs to re-render.
    anthropic = llm.anthropic.model_dump()
    anthropic["api_key"] = _mask(anthropic.get("api_key", ""))
    openai = llm.openai.model_dump()
    openai["api_key"] = _mask(openai.get("api_key", ""))
    return {
        "provider": llm.provider,
        "model": llm.model,
        "timeout_seconds": llm.timeout_seconds,
        "claude_binary": llm.claude_binary,
        "ollama": llm.ollama.model_dump(),
        "anthropic": anthropic,
        "openai": openai,
        "agents": llm.agents.model_dump(),
        # claude-code-cli and anthropic-api have web search wired in; anything else means disabled.
        "search_provider": llm.search.provider if llm.search.provider in SEARCH_PROVIDERS else "disabled",
        "restart_required": restart_required,
    }


def _mask(value: str) -> str:
    """Return a UI-safe placeholder for a stored secret: '' when empty, '••••…' when set."""
    return "" if not value.strip() else "••••••••"


# Config keys that hold a secret anywhere in the tree. The full-config blob GET /api/settings
# returns is masked against this set so no secret (push token, source credentials, webhook secret,
# admin hash) ever leaves the server in plaintext — the reason masking only the dedicated views is
# not enough. Env-var *names* (`env_api_key`, `env_secret`) are not secrets and stay visible.
_SECRET_CONFIG_KEYS = frozenset({"api_key", "password", "secret", "token", "admin_password_hash"})


def _mask_config(value):
    """Recursively mask every secret-keyed string in a JSON-able config tree."""
    if isinstance(value, dict):
        return {
            key: (MASK_VALUE if value_ else "") if key in _SECRET_CONFIG_KEYS and isinstance(value_, str)
            else _mask_config(value_)
            for key, value_ in value.items()
        }
    if isinstance(value, list):
        return [_mask_config(item) for item in value]
    return value


MASK_VALUE = "••••••••"


def _merge_anthropic(body: "AnthropicBody | None", existing: AnthropicApiConfig) -> AnthropicApiConfig:
    """Build the AnthropicApiConfig to persist. The UI sends the masked placeholder (not the
    plaintext) when the operator did not touch the key field — in that case we keep the stored
    value. An empty string means "clear it". A real plaintext value replaces it.
    """
    if body is None:
        return existing
    api_key = existing.api_key if (body.api_key == MASK_VALUE or body.api_key is None) else body.api_key
    return AnthropicApiConfig(
        api_key=api_key,
        model=(body.model or existing.model).strip() or existing.model,
        base_url=(body.base_url or existing.base_url).strip() or existing.base_url,
        anthropic_version=(body.anthropic_version or existing.anthropic_version).strip() or existing.anthropic_version,
    )


def _merge_openai(body: "OpenAIBody | None", existing: OpenAIApiConfig) -> OpenAIApiConfig:
    if body is None:
        return existing
    api_key = existing.api_key if (body.api_key == MASK_VALUE or body.api_key is None) else body.api_key
    return OpenAIApiConfig(
        api_key=api_key,
        model=(body.model or existing.model).strip() or existing.model,
        base_url=(body.base_url or existing.base_url).strip() or existing.base_url,
        organization=(body.organization or existing.organization).strip(),
    )


def _merge_search(body: "SearchBody", existing: LLMSearchConfig) -> LLMSearchConfig:
    """Build the search config from the request, preserving a stored max_results when the body
    omits it. The Settings UI does not send max_results yet, so a save of an unrelated field
    must not silently wipe an operator's [llm.search] max_results."""
    return LLMSearchConfig(
        provider=body.provider,
        max_results=body.max_results if body.max_results is not None else existing.max_results,
    )


def _validate_and_build_agents(body: "AgentsBody | None") -> PerAgentModelsConfig:
    """Reject an unknown provider name per agent; otherwise coerce the AgentsBody into the
    stored config. A missing block defaults to empty overrides (every agent uses the base)."""
    if body is None:
        return PerAgentModelsConfig()
    out = {}
    for agent in ("triage", "analyst", "summary"):
        override = getattr(body, agent)
        provider = (override.provider or "").strip()
        if provider and provider not in KNOWN_PROVIDERS:
            raise _bad(
                f"unknown agents.{agent}.provider '{provider}' "
                f"(expected '' or one of: {', '.join(sorted(KNOWN_PROVIDERS))})"
            )
        out[agent] = AgentModelConfig(provider=provider, model=(override.model or "").strip())
    return PerAgentModelsConfig(**out)


def _restart_required(request: Request, stored: LLMConfig) -> bool:
    snapshot = getattr(request.app.state, "llm_config_snapshot", None)
    return snapshot is not None and stored.model_dump(mode="json") != snapshot


def _settings_payload(request: Request, services) -> dict:
    # Loaded, not merely downloaded: this flag drives the setup notice, and a notice that
    # does not appear is a prerequisite nobody is told about.
    present = services.attack_ready()
    with BrowseQueries(services.paths.db_file) as browse:
        counts = browse.attack_counts()
    stored = _stored_llm(services)
    return {
        "attack": {"present": present, **counts},
        "claude": {"found": bool(request.app.state.claude_available())},
        "config": _mask_config(services.config.model_dump(mode="json")),
        "home": str(services.paths.home),
        "llm": llm_view(stored, _restart_required(request, stored)),
        # Priced model ids per direct-API provider, so the Settings picker only suggests ids the
        # dashboard can cost. See known_models_by_provider() for why CLI aliases are excluded.
        "known_models": known_models_by_provider(),
    }


@router.get("/settings")
def settings(request: Request) -> dict:
    with open_services(request) as services:
        try:
            return _settings_payload(request, services)
        except ConfigError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/settings/llm")
def update_llm(body: LLMUpdate, request: Request) -> dict:
    if body.provider not in PROVIDERS:
        raise _bad(f"unknown provider '{body.provider}' (expected one of: {', '.join(PROVIDERS)})")
    if body.search.provider not in SEARCH_PROVIDERS:
        raise _bad(f"unknown search.provider '{body.search.provider}' (expected one of: {', '.join(SEARCH_PROVIDERS)})")
    if not MIN_TIMEOUT <= body.timeout_seconds <= MAX_TIMEOUT:
        raise _bad(f"timeout_seconds must be between {MIN_TIMEOUT} and {MAX_TIMEOUT}")
    if not body.claude_binary.strip():
        raise _bad("claude_binary must not be empty")
    _validate_base_url(body.ollama.base_url)
    if not body.ollama.model.strip():
        raise _bad("ollama.model must not be empty")
    if not body.ollama.keep_alive.strip():
        raise _bad("ollama.keep_alive must not be empty")
    agents_cfg = _validate_and_build_agents(body.agents)
    with open_services(request) as services:
        anthropic_cfg = _merge_anthropic(body.anthropic, services.config.llm.anthropic)
        openai_cfg = _merge_openai(body.openai, services.config.llm.openai)
        try:
            stored_llm = _stored_llm(services)
        except ConfigError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        stored_search = stored_llm.search
        llm = LLMConfig(
            provider=body.provider,
            model=(body.model or "").strip() or None,
            timeout_seconds=body.timeout_seconds,
            claude_binary=body.claude_binary.strip(),
            ollama=OllamaConfig(base_url=body.ollama.base_url, model=body.ollama.model.strip(),
                                keep_alive=body.ollama.keep_alive.strip()),
            anthropic=anthropic_cfg,
            openai=openai_cfg,
            agents=agents_cfg,
            search=_merge_search(body.search, stored_search),
            # Not exposed in the form, so carry the stored value forward: rebuilding LLMConfig
            # field by field would otherwise reset it to its default on every save.
            subscription_cost_basis=stored_llm.subscription_cost_basis,
        )
        try:
            update_llm_config(services.paths.config_file, llm)
        except ConfigError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not write argus.toml: {exc}") from exc
        # Every request and job rebuilds its Services from argus.toml (build_services), so the new
        # config is live at once. Advance the snapshot so the UI stops claiming a restart is needed.
        request.app.state.llm_config_snapshot = llm.model_dump(mode="json")
        return _settings_payload(request, services)


class LoggingUpdate(BaseModel):
    retain_raw_llm: Literal["off", "errors", "full"]


@router.put("/settings/logging")
def update_logging(body: LoggingUpdate, request: Request) -> dict:
    """Change the LLM-output retention mode without a server restart.

    Writes the new value to argus.toml so it survives, AND flips the live trace recorder
    so the next event emitted picks up the new policy immediately. This is the only
    settings surface that affects every subsequent pipeline event — so the UI should
    confirm with the operator before switching to `full` on a store that handles
    sensitive content.
    """
    with open_services(request) as services:
        try:
            _write_logging_mode(services.paths.config_file, body.retain_raw_llm)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not write argus.toml: {exc}") from exc
        services.config.logging.retain_raw_llm = body.retain_raw_llm
        # The live recorder may already be cached on the app for the current session; if the
        # services factory produces fresh ones per request this is a no-op, but when a
        # long-lived recorder exists we flip it in place so the mode takes effect without a
        # restart.
        recorder = services.trace_recorder()
        if hasattr(recorder, "set_retain_mode"):
            recorder.set_retain_mode(body.retain_raw_llm)
        return {"retain_raw_llm": body.retain_raw_llm}


def _write_logging_mode(path: Path, mode: str) -> None:
    """Rewrite argus.toml with the new [logging] section. Preserves every other section."""
    import tomllib

    import tomli_w

    from cyllex_argus.config.settings import _write_atomic as _atomic_write
    data = tomllib.loads(path.read_text("utf-8")) if path.exists() else {}
    data.setdefault("logging", {})["retain_raw_llm"] = mode
    _atomic_write(path, tomli_w.dumps(data).encode("utf-8"))


# -- External sources (MISP / TAXII) and outbound notifications (webhooks) -----------
#
# These three sections are list-shaped in argus.toml (`[[misp]]`, `[[taxii]]`, `[[webhook]]`).
# The pattern is the same: list, PUT bulk replace, test one entry. Masked secrets round-trip
# through the `MASK_VALUE` sentinel — the UI shows "••••••••" on reload and the server treats
# that string as "keep the stored value" so the operator does not have to re-type it to save
# an unrelated field.


class MispServerBody(BaseModel):
    name: str
    url: str
    api_key: str = ""
    env_api_key: str = ""
    timestamp_days: int | None = 180
    enabled: bool = True


class MispListBody(BaseModel):
    servers: list[MispServerBody] = Field(default_factory=list)


class TaxiiServerBody(BaseModel):
    name: str
    url: str
    collection: str
    api_key: str = ""
    env_api_key: str = ""
    username: str = ""
    password: str = ""
    added_after_days: int | None = 180
    enabled: bool = True


class TaxiiListBody(BaseModel):
    servers: list[TaxiiServerBody] = Field(default_factory=list)


class WebhookBody(BaseModel):
    url: str
    secret: str = ""
    env_secret: str = ""
    on_states: list[str] = Field(default_factory=list)
    enabled: bool = True


class WebhooksListBody(BaseModel):
    webhooks: list[WebhookBody] = Field(default_factory=list)


def _mask_secret(value: str) -> str:
    return MASK_VALUE if value.strip() else ""


def _unmask_or_fallback(client_value: str, stored_value: str) -> str:
    """UI echoes the mask when the operator did not touch the field; keep the stored value in
    that case. An empty string is "clear it"; any other value is a new plaintext."""
    return stored_value if client_value == MASK_VALUE else client_value


def _misp_view(cfg: MispServerConfig) -> dict:
    data = cfg.model_dump()
    data["api_key"] = _mask_secret(data.get("api_key", ""))
    return data


def _taxii_view(cfg: TaxiiServerConfig) -> dict:
    data = cfg.model_dump()
    data["api_key"] = _mask_secret(data.get("api_key", ""))
    data["password"] = _mask_secret(data.get("password", ""))
    return data


def _webhook_view(cfg: WebhookConfig) -> dict:
    data = cfg.model_dump()
    data["secret"] = _mask_secret(data.get("secret", ""))
    return data


def _stored_config(services) -> ArgusConfig:
    """The config as argus.toml holds it; a missing file means the live in-memory config.

    The list-shaped settings endpoints (misp, taxii, webhooks) use this to read: the test
    harness and some production paths share one in-memory `ArgusConfig` across requests, so
    the on-disk value — the one `save_config` just wrote — is the authoritative source for
    the next GET.
    """
    path = services.paths.config_file
    try:
        return load_config(path) if path.exists() else services.config
    except ConfigError:
        return services.config


def _save_sections(
    services,
    *,
    misp: list[MispServerConfig] | None = None,
    taxii: list[TaxiiServerConfig] | None = None,
    webhooks: list[WebhookConfig] | None = None,
) -> ArgusConfig:
    """Rewrite argus.toml, preserving every section except the ones named.

    Loads the on-disk config so comments the operator has outside the touched sections keep
    their place; mutates only the fields the caller passed; writes atomically through
    `save_config`. Also mutates `services.config` so the running process sees the change
    without a restart (for the Collector additions and the webhook dispatcher, which are
    rebuilt per `services.pipeline()` call anyway).
    """
    path = services.paths.config_file
    base = load_config(path) if path.exists() else services.config
    update: dict = {}
    if misp is not None:
        update["misp"] = misp
    if taxii is not None:
        update["taxii"] = taxii
    if webhooks is not None:
        update["webhooks"] = webhooks
    cfg = base.model_copy(update=update)
    save_config(cfg, path)
    services.config = cfg
    return cfg


# ---- MISP CRUD + test ----


@router.get("/settings/misp")
def list_misp(request: Request) -> dict:
    with open_services(request) as services:
        cfg = _stored_config(services)
        return {"servers": [_misp_view(server) for server in cfg.misp]}


@router.put("/settings/misp")
def save_misp(body: MispListBody, request: Request) -> dict:
    """Replace the entire `[[misp]]` list. Masked `api_key` values preserve the stored key by
    matching on `name` so an operator who only toggled `enabled` does not have to re-paste
    the key."""
    seen_names: set[str] = set()
    for server in body.servers:
        if not server.name.strip():
            raise _bad("every MISP server must have a non-empty name")
        if not server.url.strip():
            raise _bad(f"MISP server '{server.name}' must have a URL")
        if server.name in seen_names:
            raise _bad(f"duplicate MISP server name '{server.name}'")
        seen_names.add(server.name)
    with open_services(request) as services:
        stored = _stored_config(services)
        existing_by_name = {cfg.name: cfg for cfg in stored.misp}
        servers = []
        for server in body.servers:
            prior = existing_by_name.get(server.name)
            api_key = _unmask_or_fallback(server.api_key, prior.api_key if prior else "")
            servers.append(MispServerConfig(
                name=server.name, url=server.url,
                api_key=api_key, env_api_key=server.env_api_key,
                timestamp_days=server.timestamp_days,
                enabled=server.enabled,
            ))
        cfg = _save_sections(services, misp=servers)
        return {"servers": [_misp_view(server) for server in cfg.misp]}


@router.post("/settings/misp/test")
def test_misp(body: MispServerBody, request: Request) -> dict:
    """Probe one MISP server. Issues the same `/events/restSearch` call the Collector would,
    with a harmless `value=` to count events without pulling them all. Masked `api_key` falls
    back to the stored key for the same-named server."""
    with open_services(request) as services:
        stored = _stored_config(services)
        existing = next((cfg for cfg in stored.misp if cfg.name == body.name), None)
        api_key = _unmask_or_fallback(body.api_key, existing.api_key if existing else "")
        cfg = MispServerConfig(
            name=body.name, url=body.url, api_key=api_key, env_api_key=body.env_api_key,
            timestamp_days=body.timestamp_days, enabled=True,
        )
        transport = getattr(request.app.state, "http_transport", None)
        try:
            with httpx.Client(transport=transport, timeout=TEST_TIMEOUT_SECONDS) as client:
                events = MispClient(cfg, client).search_events("test", limit=1)
        except MispError as exc:
            return {"ok": False, "error": str(exc)}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"MISP unreachable: {type(exc).__name__}"}
        return {"ok": True, "events_sampled": len(events)}


# ---- TAXII CRUD + test ----


@router.get("/settings/taxii")
def list_taxii(request: Request) -> dict:
    with open_services(request) as services:
        cfg = _stored_config(services)
        return {"servers": [_taxii_view(server) for server in cfg.taxii]}


@router.put("/settings/taxii")
def save_taxii(body: TaxiiListBody, request: Request) -> dict:
    seen_names: set[str] = set()
    for server in body.servers:
        if not server.name.strip():
            raise _bad("every TAXII server must have a non-empty name")
        if not server.url.strip() or not server.collection.strip():
            raise _bad(f"TAXII server '{server.name}' needs a URL and a collection id")
        if server.name in seen_names:
            raise _bad(f"duplicate TAXII server name '{server.name}'")
        seen_names.add(server.name)
    with open_services(request) as services:
        stored = _stored_config(services)
        existing_by_name = {cfg.name: cfg for cfg in stored.taxii}
        servers = []
        for server in body.servers:
            prior = existing_by_name.get(server.name)
            api_key = _unmask_or_fallback(server.api_key, prior.api_key if prior else "")
            password = _unmask_or_fallback(server.password, prior.password if prior else "")
            servers.append(TaxiiServerConfig(
                name=server.name, url=server.url, collection=server.collection,
                api_key=api_key, env_api_key=server.env_api_key,
                username=server.username, password=password,
                added_after_days=server.added_after_days,
                enabled=server.enabled,
            ))
        cfg = _save_sections(services, taxii=servers)
        return {"servers": [_taxii_view(server) for server in cfg.taxii]}


@router.post("/settings/taxii/test")
def test_taxii(body: TaxiiServerBody, request: Request) -> dict:
    with open_services(request) as services:
        stored = _stored_config(services)
        existing = next((cfg for cfg in stored.taxii if cfg.name == body.name), None)
        api_key = _unmask_or_fallback(body.api_key, existing.api_key if existing else "")
        password = _unmask_or_fallback(body.password, existing.password if existing else "")
        cfg = TaxiiServerConfig(
            name=body.name, url=body.url, collection=body.collection,
            api_key=api_key, env_api_key=body.env_api_key,
            username=body.username, password=password,
            added_after_days=body.added_after_days, enabled=True,
        )
        transport = getattr(request.app.state, "http_transport", None)
        try:
            with httpx.Client(transport=transport, timeout=TEST_TIMEOUT_SECONDS) as client:
                objects = TaxiiClient(cfg, client).list_objects(match_type="report", limit=1)
        except TaxiiError as exc:
            return {"ok": False, "error": str(exc)}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"TAXII unreachable: {type(exc).__name__}"}
        return {"ok": True, "objects_sampled": len(objects)}


# ---- Webhooks CRUD + test ----


_VALID_STATES = ("finished", "paused")


@router.get("/settings/webhooks")
def list_webhooks(request: Request) -> dict:
    with open_services(request) as services:
        cfg = _stored_config(services)
        return {"webhooks": [_webhook_view(hook) for hook in cfg.webhooks]}


@router.put("/settings/webhooks")
def save_webhooks(body: WebhooksListBody, request: Request) -> dict:
    seen_urls: set[str] = set()
    for hook in body.webhooks:
        if not hook.url.strip():
            raise _bad("every webhook must have a non-empty url")
        if hook.url in seen_urls:
            raise _bad(f"duplicate webhook url '{hook.url}'")
        seen_urls.add(hook.url)
        for state in hook.on_states:
            if state not in _VALID_STATES:
                raise _bad(
                    f"webhook '{hook.url}' has an unknown state '{state}' in on_states "
                    f"(expected a subset of {list(_VALID_STATES)})"
                )
    with open_services(request) as services:
        stored = _stored_config(services)
        existing_by_url = {cfg.url: cfg for cfg in stored.webhooks}
        webhooks = []
        for hook in body.webhooks:
            prior = existing_by_url.get(hook.url)
            secret = _unmask_or_fallback(hook.secret, prior.secret if prior else "")
            webhooks.append(WebhookConfig(
                url=hook.url, secret=secret, env_secret=hook.env_secret,
                on_states=list(hook.on_states), enabled=hook.enabled,
            ))
        cfg = _save_sections(services, webhooks=webhooks)
        return {"webhooks": [_webhook_view(hook) for hook in cfg.webhooks]}


@router.post("/settings/webhooks/test")
def test_webhook(body: WebhookBody, request: Request) -> dict:
    """Fire a one-shot test POST to the webhook URL with a synthetic FINISHED payload. Masked
    `secret` falls back to the stored secret for the same-url webhook so the operator can test
    without re-typing the key."""
    from datetime import datetime

    from cyllex_argus.knowledge.models import InvestigationState
    from cyllex_argus.pipeline.investigation import InvestigationReport

    with open_services(request) as services:
        stored = _stored_config(services)
        existing = next((cfg for cfg in stored.webhooks if cfg.url == body.url), None)
        secret = _unmask_or_fallback(body.secret, existing.secret if existing else "")
        cfg = WebhookConfig(
            url=body.url, secret=secret, env_secret=body.env_secret,
            on_states=list(body.on_states), enabled=True,
        )
        report = InvestigationReport(
            investigation_id=0,
            target="Argus webhook test",
            state=InvestigationState.FINISHED,
            stage_counts={"test": 1},
            claims_accepted=0, claims_rejected=0,
            message="This is a webhook configuration test from Settings.",
        )
        transport = getattr(request.app.state, "http_transport", None)
        try:
            with httpx.Client(transport=transport, timeout=TEST_TIMEOUT_SECONDS) as client:
                dispatcher = WebhookDispatcher(
                    [cfg], client=client,
                    now=lambda: datetime.now(UTC),
                )
                results = dispatcher.dispatch_investigation(report)
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"webhook request failed: {type(exc).__name__}"}
        if not results:
            return {"ok": False, "error": "no result (unexpected)"}
        outcome = results[0]
        return {"ok": outcome.status == "sent", "status": outcome.status, "detail": outcome.detail}


@router.post("/settings/llm/test")
def test_llm(body: LLMTest, request: Request) -> dict:
    """Probe the configured provider. Reads nothing from, and writes nothing to, argus.toml."""
    if body.provider not in PROVIDERS:
        raise _bad(f"unknown provider '{body.provider}' (expected one of: {', '.join(PROVIDERS)})")
    if body.provider == "claude-code-cli":
        binary = (body.claude_binary or "").strip() or "claude"
        path = shutil.which(binary)
        if path is None:
            return {"ok": False, "error": f"{binary} not found on PATH"}
        return {"ok": True, "binary_path": path}
    if body.provider == "anthropic-api":
        return _test_anthropic(body.anthropic, request)
    if body.provider == "openai-api":
        return _test_openai(body.openai, request)
    if body.ollama is None:
        raise _bad("ollama settings are required to test Ollama")
    base_url = _validate_base_url(body.ollama.base_url).rstrip("/")
    transport = getattr(request.app.state, "http_transport", None)
    try:
        with httpx.Client(transport=transport, timeout=TEST_TIMEOUT_SECONDS) as client:
            response = client.get(base_url + "/api/tags")
    except httpx.ConnectError:
        return {"ok": False, "error": f"cannot connect to Ollama at {base_url}"}
    except httpx.TimeoutException:
        return {"ok": False, "error": f"Ollama at {base_url} did not answer within {TEST_TIMEOUT_SECONDS:g}s"}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"request failed: {type(exc).__name__}"}
    if response.status_code != 200:
        return {"ok": False, "error": f"Ollama answered HTTP {response.status_code}"}
    try:
        names = [str(m["name"]) for m in response.json()["models"]]
    except (ValueError, KeyError, TypeError):
        return {"ok": False, "error": "unexpected answer from /api/tags (is this an Ollama server?)"}
    result: dict = {"ok": True, "models": names}
    wanted = (body.ollama.model or "").strip()
    if wanted:
        # Ollama resolves a bare name to its ":latest" tag.
        result["model_present"] = wanted in names or (":" not in wanted and f"{wanted}:latest" in names)
    return result


def _test_anthropic(body: "AnthropicBody | None", request: Request) -> dict:
    """Probe the Anthropic API with a GET /v1/models call: cheapest round-trip that validates the
    key, the network path and that the model id (if any) exists on the account.

    The key resolution matches the real adapter: ANTHROPIC_API_KEY env wins, then the plaintext
    toml value sent by the UI (or the one already persisted if the UI echoed the mask).
    """
    import os as _os  # local import — avoid polluting the module-level namespace for a test helper

    body = body or AnthropicBody()
    env_key = _os.environ.get("ANTHROPIC_API_KEY", "").strip()
    # UI echoes the mask when the operator did not touch the field; fall back to argus.toml.
    plaintext = "" if body.api_key == MASK_VALUE else body.api_key.strip()
    stored = ""
    try:
        with open_services(request) as services:
            stored = services.config.llm.anthropic.api_key.strip()
    except Exception:  # noqa: BLE001 — a missing / malformed config is not a probe failure
        stored = ""
    key = env_key or plaintext or stored
    if not key:
        return {"ok": False, "error": "No API key: set ANTHROPIC_API_KEY or paste one in Settings."}
    base_url = (body.base_url or "https://api.anthropic.com").rstrip("/")
    headers = {
        "x-api-key": key,
        "anthropic-version": body.anthropic_version or "2023-06-01",
    }
    transport = getattr(request.app.state, "http_transport", None)
    try:
        with httpx.Client(transport=transport, timeout=TEST_TIMEOUT_SECONDS) as client:
            response = client.get(base_url + "/v1/models", headers=headers)
    except httpx.ConnectError:
        return {"ok": False, "error": f"cannot connect to Anthropic at {base_url}"}
    except httpx.TimeoutException:
        return {"ok": False, "error": f"Anthropic at {base_url} did not answer within {TEST_TIMEOUT_SECONDS:g}s"}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"request failed: {type(exc).__name__}"}
    if response.status_code == 401:
        return {"ok": False, "error": "Anthropic rejected the API key (401)."}
    if response.status_code != 200:
        return {"ok": False, "error": f"Anthropic answered HTTP {response.status_code}"}
    try:
        names = [str(m.get("id")) for m in (response.json().get("data") or [])]
    except (ValueError, KeyError, TypeError):
        return {"ok": False, "error": "unexpected answer from /v1/models (is this the Anthropic API?)"}
    result: dict = {"ok": True, "models": names}
    wanted = (body.model or "").strip()
    if wanted:
        result["model_present"] = wanted in names
    return result


def _test_openai(body: "OpenAIBody | None", request: Request) -> dict:
    """Probe the OpenAI API with GET /models. Same shape as the Anthropic probe."""
    import os as _os

    body = body or OpenAIBody()
    env_key = _os.environ.get("OPENAI_API_KEY", "").strip()
    plaintext = "" if body.api_key == MASK_VALUE else body.api_key.strip()
    stored = ""
    try:
        with open_services(request) as services:
            stored = services.config.llm.openai.api_key.strip()
    except Exception:  # noqa: BLE001
        stored = ""
    key = env_key or plaintext or stored
    if not key:
        return {"ok": False, "error": "No API key: set OPENAI_API_KEY or paste one in Settings."}
    base_url = (body.base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {key}"}
    org = (body.organization or "").strip()
    if org:
        headers["OpenAI-Organization"] = org
    transport = getattr(request.app.state, "http_transport", None)
    try:
        with httpx.Client(transport=transport, timeout=TEST_TIMEOUT_SECONDS) as client:
            response = client.get(base_url + "/models", headers=headers)
    except httpx.ConnectError:
        return {"ok": False, "error": f"cannot connect to OpenAI at {base_url}"}
    except httpx.TimeoutException:
        return {"ok": False, "error": f"OpenAI at {base_url} did not answer within {TEST_TIMEOUT_SECONDS:g}s"}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"request failed: {type(exc).__name__}"}
    if response.status_code == 401:
        return {"ok": False, "error": "OpenAI rejected the API key (401)."}
    if response.status_code != 200:
        return {"ok": False, "error": f"OpenAI answered HTTP {response.status_code}"}
    try:
        names = [str(m.get("id")) for m in (response.json().get("data") or [])]
    except (ValueError, KeyError, TypeError):
        return {"ok": False, "error": "unexpected answer from /models (is this the OpenAI API?)"}
    result: dict = {"ok": True, "models": names}
    wanted = (body.model or "").strip()
    if wanted:
        result["model_present"] = wanted in names
    return result
