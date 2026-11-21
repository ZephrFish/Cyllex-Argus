# Configuration

_[Back to the docs index](README.md)_

## Configuration

`argus.toml` lives in `~/.cyllex-argus/` (or `$ARGUS_HOME`). `argus init` writes it with every
default; `argus.toml` from an older install is auto-migrated.

```toml
[llm]
provider = "claude-code-cli"    # claude-code-cli | anthropic-api | openai-api | ollama — runs Analyst, Triage, Validator, summary calls
model = "opus"                  # claude-code-cli only; optional, uses the Claude Code default when absent
timeout_seconds = 600           # per LLM call
claude_binary = "claude"        # claude-code-cli only; or an absolute path to the CLI
subscription_cost_basis = "list"  # claude-code-cli only; "list" or "none", see below

[llm.ollama]                    # read only when provider = "ollama"
base_url = "http://127.0.0.1:11434"
model = "qwen2.5:32b-instruct"  # swap per hardware, e.g. "qwen2.5:14b-instruct"
keep_alive = "10m"

[llm.anthropic]                 # read only when provider = "anthropic-api" (bring your own key)
api_key = ""                    # prefer the ANTHROPIC_API_KEY env var — keeps the key off disk
model = "claude-sonnet-5-5"
base_url = "https://api.anthropic.com"

# `argus init` also writes [llm.openai], per-agent overrides under [llm.agents.*], and the
# [[taxii]] / [[misp]] / [[webhook]] / [auth] blocks documented in their own sections. This sample
# shows only the common ones; generate the full file with `argus init`.

[llm.search]
provider = "claude-code-cli"    # Collector's web search; or "disabled"

[limits]
max_docs = 25
chunk_chars = 60000
max_chunks_per_document = 12
download_timeout_seconds = 30
max_download_bytes = 25000000

[scoring]
base_weight = 0.6
recency_years = 3
stale_factor = 0.75
fuzzy_threshold = 95            # quote match threshold (rapidfuzz, 0 – 100)

[scoring.reliability_weights]   # Admiralty A..F multipliers
A = 1
B = 0.85
C = 0.7
D = 0.5
E = 0.3
F = 0.5

[logging]
retain_raw_llm = "off"          # "off" / "errors" / "full" — see pipeline trace

[publishers]
"Mandiant" = "B"

[[feeds]]                       # RSS feeds; listing any replaces the defaults
name = "Unit 42"
url = "https://unit42.paloaltonetworks.com/feed/"

# Enrichment is disabled by default. Opt in per connector; every enricher documents its own
# outbound statement (see the Enrichment section).
[enrichment.enrichers."crt.sh"]
enabled = false
```

### Cost on a subscription

`claude-code-cli` runs on a Claude Pro/Max subscription, which is flat-rate: a run bills no
tokens, so any per-token figure is an estimate of what the same work would have cost on a
metered key, not money that left your account. Claude Code labels its own figure the same way
(`costBasis: "list"`). `[llm] subscription_cost_basis` chooses what the trace records:

| Value | Cost column | Use when |
|---|---|---|
| `"list"` (default) | priced from the static table, like every other provider | you want runs comparable across providers, or you are deciding whether to move to a metered key |
| `"none"` | empty | you would rather see nothing than a number you were not charged |

Tokens, model and duration are recorded either way; only the money column changes. Metered
providers (`anthropic-api`, `openai-api`) ignore this setting, and Ollama is already zero-cost.

### Local models

Argus can run every call that sees document content on a local [Ollama](https://ollama.com) model.

- **Hybrid default.** `[llm] provider = "ollama"` + `[llm.search] provider = "claude-code-cli"`.
  Analyst, Triage, Validator and summaries run locally. Claude Code only runs the
  Collector's web search (public queries, no document text).
- **Strict mode.** Set `[llm.search] provider = "disabled"`: the Collector uses RSS feeds and
  ingested documents only; nothing reaches Anthropic.
- **Model.** `qwen2.5:32b-instruct` for 24 GB VRAM; `qwen2.5:14b-instruct` for 16 GB.
- **Install.** `ollama pull qwen2.5:32b-instruct`, start the Ollama server (`ollama serve`), then
  `argus serve`.
- **Trade-off.** Local models paraphrase quotes more often than Claude, so the Validator rejects
  more claims. In exchange, no document content leaves the machine.


## Exports

![An exported dossier](images/argus-dossier.png)

- **Markdown** (`argus export "<entity>" --format markdown`): human-readable profile with tables
  of facts and citations. `--no-summary` skips the AI executive summary. `--tlp-cap <tlp>` strips
  citations above the cap.
  - Opens with an **Evidence base** section: how many of the documents collected for this threat
    actually produced facts, and how many could not be retrieved, parsed or were ruled off-topic.
    A profile built from 25 of 40 documents looks exactly like one built from 25 of 25, so the
    file has to say which it is — and has to say that a finding missing below may be missing from
    the reporting, or missing only from what this collection could read. The printable web dossier
    carries the same statement on its cover.
- **STIX 2.1** (`--format stix`): a bundle you can import into any STIX-aware tool. ATT&CK
  objects are referenced by their official ids and NOT re-emitted; load the official ATT&CK
  bundle alongside.


## Where things are stored

```
~/.cyllex-argus/
├── argus.db                    SQLite database (every table, every event)
├── argus.toml                  Configuration
├── attack/
│   └── enterprise-attack.json  Latest downloaded MITRE ATT&CK Enterprise bundle
├── raw/
│   └── <content-hash>          Original bytes of every downloaded document (dedupe by SHA-256)
└── uploads/                    Files you ingested with `argus ingest`
```

Legacy `argus.db` / `argus.toml` from older installs are auto-recognised and renamed on first
run.


---
_[Back to the docs index](README.md)_
