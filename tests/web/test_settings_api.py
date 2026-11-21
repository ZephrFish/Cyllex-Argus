import tomllib

import httpx
import pytest

from tests.support import World
from tests.web.conftest import make_client as _make_client

SEEDED_TOML = """\
# Hand-written notes survive an LLM settings save.
[llm]
provider = "claude-code-cli"
timeout_seconds = 600

[llm.search]
provider = "claude-code-cli"

[limits]
max_docs = 7  # keep it small
chunk_chars = 50000

[scoring]
base_weight = 0.5
recency_years = 2

[logging]
retain_raw_llm = true

[publishers]
"Mandiant" = "A"
"Some Blog" = "D"

[[feeds]]
name = "CISA"
url = "https://www.cisa.gov/cybersecurity-advisories/all.xml"

[[feeds]]
name = "Unit 42"
url = "https://unit42.paloaltonetworks.com/feed/"
"""

PAYLOAD = {
    "provider": "ollama",
    "model": "opus",
    "timeout_seconds": 900,
    "claude_binary": "claude-custom",
    "ollama": {"base_url": "http://10.0.0.5:11434", "model": "llama3.1:8b", "keep_alive": "30m"},
    "search": {"provider": "disabled"},
}


def make_client(world, **kwargs):
    client = _make_client(world, **kwargs)
    client.headers["X-Argus-CSRF"] = "1"  # non-GET requests must carry it
    return client


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture
def client(world):
    return make_client(world)


def _payload(**changes):
    return {**PAYLOAD, **changes}


def _without_restart(llm: dict) -> dict:
    return {k: v for k, v in llm.items() if k != "restart_required"}


def _default_anthropic_view() -> dict:
    """The llm_view shape for a never-touched Anthropic config: defaults from AnthropicApiConfig
    with the api_key masked (empty when nothing is set)."""
    return {
        "api_key": "", "model": "claude-sonnet-5-5",
        "base_url": "https://api.anthropic.com", "anthropic_version": "2023-06-01",
    }


def _default_openai_view() -> dict:
    return {
        "api_key": "", "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1", "organization": "",
    }


def _default_agents_view() -> dict:
    """llm_view shape for a never-touched agents block: four empty overrides."""
    empty = {"provider": "", "model": ""}
    return {"triage": empty, "analyst": empty, "summary": empty}


def test_put_round_trips_every_field(client, world):
    response = client.put("/api/settings/llm", json=PAYLOAD)
    assert response.status_code == 200
    expected = {
        "provider": "ollama", "model": "opus", "timeout_seconds": 900, "claude_binary": "claude-custom",
        "ollama": PAYLOAD["ollama"],
        "anthropic": _default_anthropic_view(),
        "openai": _default_openai_view(),
        "agents": _default_agents_view(),
        "search_provider": "disabled",
    }
    assert _without_restart(response.json()["llm"]) == expected
    assert _without_restart(client.get("/api/settings").json()["llm"]) == expected
    stored = tomllib.loads(world.paths.config_file.read_text(encoding="utf-8"))["llm"]
    assert stored["provider"] == "ollama" and stored["ollama"]["keep_alive"] == "30m"
    assert stored["search"] == {"provider": "disabled"}
    # Same shape as GET /api/settings.
    assert set(response.json()) == set(client.get("/api/settings").json())


def test_settings_lists_priced_models_without_cli_aliases(client):
    known = client.get("/api/settings").json()["known_models"]
    # Grouped by direct-API provider, every id drawn from the price table.
    assert "claude-opus-5-5" in known["anthropic-api"]
    assert "gpt-4o" in known["openai-api"]
    # CLI short aliases are not API ids, and the [1m] context variant is not a selectable id.
    assert "opus" not in known["anthropic-api"]
    assert not any("[" in model for models in known.values() for model in models)


def test_put_with_null_model_omits_the_key(client, world):
    assert client.put("/api/settings/llm", json=_payload(model=None)).status_code == 200
    assert "model" not in tomllib.loads(world.paths.config_file.read_text(encoding="utf-8"))["llm"]
    assert client.get("/api/settings").json()["llm"]["model"] is None


@pytest.mark.parametrize("changes", [
    {"provider": "openai"},
    {"search": {"provider": "ollama"}},
    {"timeout_seconds": 9},
    {"timeout_seconds": 3601},
    {"timeout_seconds": "soon"},
    {"claude_binary": "  "},
    {"ollama": {**PAYLOAD["ollama"], "base_url": "ftp://host"}},
    {"ollama": {**PAYLOAD["ollama"], "base_url": "127.0.0.1:11434"}},
    {"ollama": {**PAYLOAD["ollama"], "model": ""}},
    {"ollama": {**PAYLOAD["ollama"], "keep_alive": " "}},
])
def test_put_rejects_invalid_values_without_writing(client, world, changes):
    response = client.put("/api/settings/llm", json=_payload(**changes))
    assert response.status_code == 400 and response.json()["error"]
    assert not world.paths.config_file.exists()


def test_put_keeps_the_base_url_verbatim(client):
    ollama = {**PAYLOAD["ollama"], "base_url": "http://host:11434/"}
    client.put("/api/settings/llm", json=_payload(ollama=ollama))
    assert client.get("/api/settings").json()["llm"]["ollama"]["base_url"] == "http://host:11434/"


def test_put_preserves_every_other_block_byte_for_byte(client, world):
    world.paths.config_file.write_bytes(SEEDED_TOML.encode("utf-8"))
    assert client.put("/api/settings/llm", json=PAYLOAD).status_code == 200
    after = world.paths.config_file.read_text(encoding="utf-8")
    # limits, scoring, logging, publishers and feeds, with their comments, are untouched.
    assert after.endswith(SEEDED_TOML[SEEDED_TOML.index("[limits]"):])
    assert after.startswith("# Hand-written notes survive an LLM settings save.\n")
    data = tomllib.loads(after)
    assert data["llm"]["provider"] == "ollama" and data["publishers"]["Some Blog"] == "D" and len(data["feeds"]) == 2


def test_put_preserves_crlf_files(client, world):
    world.paths.config_file.write_bytes(SEEDED_TOML.replace("\n", "\r\n").encode("utf-8"))
    client.put("/api/settings/llm", json=PAYLOAD)
    raw = world.paths.config_file.read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n")
    assert raw.endswith(SEEDED_TOML[SEEDED_TOML.index("[limits]"):].replace("\n", "\r\n").encode("utf-8"))


def test_put_appends_the_llm_tables_when_the_file_has_none(client, world):
    world.paths.config_file.write_text("[limits]\nmax_docs = 3\n", encoding="utf-8")
    client.put("/api/settings/llm", json=PAYLOAD)
    after = world.paths.config_file.read_text(encoding="utf-8")
    assert after.startswith("[limits]\nmax_docs = 3\n")
    assert tomllib.loads(after)["llm"]["provider"] == "ollama"


def test_put_falls_back_to_a_full_rewrite_for_dotted_keys(client, world):
    world.paths.config_file.write_text('llm.provider = "claude-code-cli"\n[limits]\nmax_docs = 3\n', encoding="utf-8")
    assert client.put("/api/settings/llm", json=PAYLOAD).status_code == 200
    data = tomllib.loads(world.paths.config_file.read_text(encoding="utf-8"))
    assert data["llm"]["provider"] == "ollama" and data["limits"] == {"max_docs": 3}


def test_put_on_an_invalid_toml_is_a_409_and_leaves_the_file(client, world):
    world.paths.config_file.write_text("[llm\n", encoding="utf-8")
    assert client.put("/api/settings/llm", json=PAYLOAD).status_code == 409
    assert world.paths.config_file.read_text(encoding="utf-8") == "[llm\n"


def test_settings_endpoints_require_a_session(world):
    anonymous = make_client(world, authenticated=False)
    assert anonymous.put("/api/settings/llm", json=PAYLOAD).status_code == 401
    assert anonymous.post("/api/settings/llm/test", json={"provider": "ollama"}).status_code == 401


# --- restart_required -------------------------------------------------------------------------------------------

def test_settings_take_effect_live_without_a_restart(client):
    """Every request and job rebuilds Services from argus.toml, so a saved change is live at
    once: the UI must not claim a restart is required."""
    assert client.get("/api/settings").json()["llm"]["restart_required"] is False
    response = client.put("/api/settings/llm", json=PAYLOAD)
    assert response.json()["llm"]["restart_required"] is False
    assert client.get("/api/settings").json()["llm"]["restart_required"] is False


def test_restart_required_clears_when_the_values_match_the_active_ones_again(client):
    client.put("/api/settings/llm", json=PAYLOAD)
    default = {
        "provider": "claude-code-cli", "model": None, "timeout_seconds": 600, "claude_binary": "claude",
        "ollama": {"base_url": "http://127.0.0.1:11434", "model": "qwen2.5:32b-instruct", "keep_alive": "10m"},
        "anthropic": {"api_key": "", "model": "claude-sonnet-5-5",
                      "base_url": "https://api.anthropic.com", "anthropic_version": "2023-06-01"},
        "openai": {"api_key": "", "model": "gpt-4o",
                   "base_url": "https://api.openai.com/v1", "organization": ""},
        "agents": {
            "triage": {"provider": "", "model": ""},
            "analyst": {"provider": "", "model": ""},
            "summary": {"provider": "", "model": ""},
        },
        "search": {"provider": "claude-code-cli"},
    }
    assert client.put("/api/settings/llm", json=default).json()["llm"]["restart_required"] is False


def test_restart_required_is_false_when_the_put_repeats_the_active_values(world):
    world.config.llm.provider = "ollama"  # the process started with Ollama
    client = make_client(world)
    same = client.get("/api/settings").json()["llm"]
    body = {**same, "search": {"provider": same["search_provider"]}}
    del body["search_provider"], body["restart_required"]
    assert client.put("/api/settings/llm", json=body).json()["llm"]["restart_required"] is False


# --- test connection --------------------------------------------------------------------------------------------

def _tags(*names):
    return lambda request: httpx.Response(200, json={"models": [{"name": n} for n in names]})


def _ollama_test(client, model="qwen2.5:32b-instruct", base_url="http://127.0.0.1:11434"):
    return client.post("/api/settings/llm/test",
                       json={"provider": "ollama", "ollama": {"base_url": base_url, "model": model}})


def _client_with(world, handler):
    return make_client(world, http_transport=httpx.MockTransport(handler))


def test_connection_success_lists_models_and_finds_the_model(world):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:32b-instruct"}, {"name": "llama3.1:8b"}]})

    result = _ollama_test(_client_with(world, handler), base_url="http://host:11434/").json()
    assert result == {"ok": True, "models": ["qwen2.5:32b-instruct", "llama3.1:8b"], "model_present": True}
    assert seen == ["http://host:11434/api/tags"]


def test_connection_reports_a_missing_model(world):
    result = _ollama_test(_client_with(world, _tags("llama3.1:8b"))).json()
    assert result["ok"] is True and result["model_present"] is False and result["models"] == ["llama3.1:8b"]


def test_connection_resolves_a_bare_model_name_to_latest(world):
    assert _ollama_test(_client_with(world, _tags("mistral:latest")), model="mistral").json()["model_present"] is True


def test_connection_failure_modes(world):
    def refuse(request):
        raise httpx.ConnectError("refused")

    def slow(request):
        raise httpx.ReadTimeout("slow")

    result = _ollama_test(_client_with(world, refuse)).json()
    assert result["ok"] is False and "cannot connect" in result["error"]
    assert "did not answer" in _ollama_test(_client_with(world, slow)).json()["error"]
    assert _ollama_test(_client_with(world, lambda r: httpx.Response(500))).json() == {
        "ok": False, "error": "Ollama answered HTTP 500"}
    assert _ollama_test(_client_with(world, lambda r: httpx.Response(200, text="<html>"))).json()["ok"] is False


def test_connection_does_not_write_to_disk(world):
    _ollama_test(_client_with(world, _tags("a")))
    assert not world.paths.config_file.exists()


def test_connection_rejects_a_bad_url_and_an_unknown_provider(client):
    assert _ollama_test(client, base_url="file:///etc").status_code == 400
    assert client.post("/api/settings/llm/test", json={"provider": "x"}).status_code == 400


def test_claude_connection_checks_the_binary_on_path(client, monkeypatch):
    monkeypatch.setattr("cyllex_argus.web.api.settings.shutil.which",
                        lambda name: "C:/bin/claude.exe" if name == "claude" else None)
    found = client.post("/api/settings/llm/test", json={"provider": "claude-code-cli", "claude_binary": "claude"})
    assert found.json() == {"ok": True, "binary_path": "C:/bin/claude.exe"}
    missing = client.post("/api/settings/llm/test", json={"provider": "claude-code-cli", "claude_binary": "nope"})
    assert missing.json() == {"ok": False, "error": "nope not found on PATH"}


def test_put_persists_and_preserves_search_max_results(client, world):
    # A PUT carrying max_results writes it under [llm.search].
    client.put("/api/settings/llm", json=_payload(search={"provider": "disabled", "max_results": 60}))
    stored = tomllib.loads(world.paths.config_file.read_text(encoding="utf-8"))["llm"]
    assert stored["search"] == {"provider": "disabled", "max_results": 60}
    # A later PUT that omits max_results (the Settings UI does not send it) keeps the stored value
    # instead of silently wiping it.
    client.put("/api/settings/llm", json=_payload(search={"provider": "disabled"}))
    stored = tomllib.loads(world.paths.config_file.read_text(encoding="utf-8"))["llm"]
    assert stored["search"] == {"provider": "disabled", "max_results": 60}


def test_put_preserves_subscription_cost_basis(client, world):
    """`subscription_cost_basis` is not on the Settings form. The handler rebuilds LLMConfig
    field by field, so without carrying the stored value forward an unrelated save would
    silently reset an operator's "none" back to the "list" default."""
    seeded = SEEDED_TOML.replace(
        'provider = "claude-code-cli"\ntimeout_seconds = 600',
        'provider = "claude-code-cli"\ntimeout_seconds = 600\nsubscription_cost_basis = "none"',
        1,
    )
    world.paths.config_file.write_text(seeded, encoding="utf-8")
    assert tomllib.loads(seeded)["llm"]["subscription_cost_basis"] == "none"  # guard the fixture

    assert client.put("/api/settings/llm", json=PAYLOAD).status_code == 200

    stored = tomllib.loads(world.paths.config_file.read_text(encoding="utf-8"))["llm"]
    assert stored["subscription_cost_basis"] == "none"
