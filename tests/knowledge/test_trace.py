from datetime import UTC, date, datetime

import pytest

from cyllex_argus.knowledge import sqlite_store
from cyllex_argus.knowledge.schema import MIGRATIONS
from cyllex_argus.knowledge.sqlite_store import SQLiteStore
from cyllex_argus.knowledge.trace import NullRecorder, PipelineEvent, SQLiteTraceRecorder
from cyllex_argus.llm.base import LLMUsage
from cyllex_argus.llm.prices import cost_usd


@pytest.fixture
def store(tmp_path):
    with SQLiteStore(tmp_path / "argus.db") as s:
        yield s


def _event(investigation_id: int, **kwargs) -> PipelineEvent:
    defaults = dict(document_id=None, stage="collector", decision="collected")
    return PipelineEvent(investigation_id=investigation_id, **{**defaults, **kwargs})


def test_write_and_read_one_event(store):
    inv = store.create_investigation("APT29")
    recorder = SQLiteTraceRecorder(store._conn, retain_raw=False)
    usage = LLMUsage(input_tokens=1000, output_tokens=200, model="claude-opus-5-5", duration_ms=850)
    recorder.record(_event(inv.id, stage="triage", decision="triaged-in", reason="mentions twice", usage=usage))
    [event] = recorder.list_events(inv.id)
    assert (event.stage, event.decision, event.reason) == ("triage", "triaged-in", "mentions twice")
    assert event.usage == usage
    assert event.duration_ms == 850
    assert event.cost_usd == cost_usd("claude-opus-5-5", 1000, 200)
    assert event.id is not None and event.created_at is not None


def test_event_without_usage_has_no_llm_columns(store):
    inv = store.create_investigation("APT29")
    recorder = SQLiteTraceRecorder(store._conn, retain_raw=False)
    recorder.record(_event(inv.id, duration_ms=12))
    [event] = recorder.list_events(inv.id)
    assert event.usage is None and event.cost_usd is None and event.duration_ms == 12


def test_raw_is_dropped_unless_retained(store):
    inv = store.create_investigation("APT29")
    off = SQLiteTraceRecorder(store._conn, retain_raw=False)
    off.record(_event(inv.id, raw_prompt="p", raw_response="r"))
    [dropped] = off.list_events(inv.id)
    assert off.raw(dropped.id) is None

    on = SQLiteTraceRecorder(store._conn, retain_raw=True)
    on.record(_event(inv.id, raw_prompt="p", raw_response="r"))
    kept = on.list_events(inv.id)[-1]
    assert on.raw(kept.id) == ("p", "r")


def test_migration_on_an_older_schema_adds_table_and_indexes(tmp_path, monkeypatch):
    db = tmp_path / "argus.db"
    monkeypatch.setattr(sqlite_store, "MIGRATIONS", MIGRATIONS[:2])
    SQLiteStore(db).close()
    monkeypatch.setattr(sqlite_store, "MIGRATIONS", MIGRATIONS)
    with SQLiteStore(db) as store:
        assert store.schema_version() == len(MIGRATIONS)
        names = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master")}
    assert {"pipeline_events", "pipeline_events_investigation", "pipeline_events_document"} <= names


def test_delete_for_removes_only_that_investigation(store):
    a, b = store.create_investigation("A"), store.create_investigation("B")
    recorder = SQLiteTraceRecorder(store._conn, retain_raw=False)
    recorder.record(_event(a.id))
    recorder.record(_event(a.id, stage="triage", decision="triaged-out"))
    recorder.record(_event(b.id))
    assert recorder.delete_for(a.id) == 2
    assert recorder.list_events(a.id) == []
    assert len(recorder.list_events(b.id)) == 1


def test_deleting_the_investigation_cascades(store):
    inv = store.create_investigation("A")
    recorder = SQLiteTraceRecorder(store._conn, retain_raw=False)
    recorder.record(_event(inv.id))
    store._conn.execute("DELETE FROM investigations WHERE id = ?", (inv.id,))
    assert recorder.size()[0] == 0


def test_delete_before_and_size(store):
    inv = store.create_investigation("A")
    recorder = SQLiteTraceRecorder(store._conn, retain_raw=True)
    recorder.record(_event(inv.id, raw_prompt="x" * 100, raw_response="y" * 50))
    rows, size = recorder.size()
    assert rows == 1 and size >= 150
    assert recorder.delete_before(date(2000, 1, 1)) == 0
    assert recorder.delete_before(datetime.now(UTC).date().replace(year=2100)) == 1
    assert recorder.size() == (0, 0)


def test_null_recorder_records_nothing():
    NullRecorder().record(_event(1))


def test_price_from_table_false_stores_tokens_but_no_cost(store):
    """`[llm] subscription_cost_basis = "none"` arrives here as price_from_table=False. A
    flat-rate subscription bills no tokens, so the money column stays empty while tokens,
    model and duration are still recorded."""
    inv = store.create_investigation("APT29")
    recorder = SQLiteTraceRecorder(store._conn, retain_raw=False)
    usage = LLMUsage(input_tokens=2166, output_tokens=4, model="claude-opus-5-5",
                     duration_ms=10, price_from_table=False)
    recorder.record(_event(inv.id, stage="analyst", decision="analyzed", usage=usage))
    [event] = recorder.list_events(inv.id)
    assert event.cost_usd is None
    assert event.usage.input_tokens == 2166 and event.usage.model == "claude-opus-5-5"
    assert event.duration_ms == 10


def test_price_from_table_default_still_prices(store):
    """The default must not change for any metered provider: a priceable model still gets a cost."""
    inv = store.create_investigation("APT29")
    recorder = SQLiteTraceRecorder(store._conn, retain_raw=False)
    usage = LLMUsage(input_tokens=2166, output_tokens=4, model="claude-opus-5-5", duration_ms=10)
    recorder.record(_event(inv.id, stage="analyst", decision="analyzed", usage=usage))
    [event] = recorder.list_events(inv.id)
    assert event.cost_usd == cost_usd("claude-opus-5-5", 2166, 4)
    assert event.cost_usd is not None
