"""Pipeline trace: every Collector/triage/Analyst/Validator decision, with LLM usage and cost."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from typing import Literal, Protocol

from pydantic import BaseModel

from cyllex_argus.llm.base import LLMUsage
from cyllex_argus.llm.prices import cost_usd

RetainRaw = Literal["off", "errors", "full"]

# Decisions that count as a failure for the "errors-only" retention mode. Rejects and
# parse errors are what an analyst wants to debug after the fact; everything else
# (accepted, triaged-in, collected, …) is noise 90% of the time.
_ERROR_DECISIONS = frozenset({"rejected", "failed", "parse-error", "parse_error", "error"})


def _coerce_retain_mode(value: object) -> RetainRaw:
    """Accept the historical bool shape (True → full, False → off) alongside the new string
    enum. Lets a 0.8-era `argus.toml` with `retain_raw_llm = true` upgrade cleanly."""
    if isinstance(value, bool):
        return "full" if value else "off"
    if value in {"off", "errors", "full"}:
        return value  # type: ignore[return-value]
    return "off"


class PipelineEvent(BaseModel):
    investigation_id: int
    document_id: int | None = None
    stage: str
    decision: str
    reason: str | None = None
    duration_ms: int | None = None
    usage: LLMUsage | None = None
    raw_prompt: str | None = None
    raw_response: str | None = None
    # Filled in when an event is read back; ignored on write.
    id: int | None = None
    created_at: datetime | None = None
    cost_usd: float | None = None


class TraceRecorder(Protocol):
    def record(self, event: PipelineEvent) -> None: ...


class NullRecorder:
    """Records nothing."""

    def record(self, event: PipelineEvent) -> None:
        return None


class SQLiteTraceRecorder:
    def __init__(self, conn: sqlite3.Connection, retain_raw: RetainRaw | bool = "off") -> None:
        self._conn = conn
        self._retain_raw: RetainRaw = _coerce_retain_mode(retain_raw)

    def _should_store_raw(self, event: PipelineEvent) -> bool:
        """Decide per event whether the raw prompt/response survives the write.

        - `off`:   never store. Zero storage impact at the cost of debugging ability.
        - `errors`: store only when the decision is a rejection or a failure. 90% of
                    debugging value at ~10% of the storage cost on a healthy pipeline.
        - `full`:  always store (when the stage bothered to populate them).
        """
        if self._retain_raw == "off":
            return False
        if self._retain_raw == "full":
            return True
        return event.decision in _ERROR_DECISIONS

    def record(self, event: PipelineEvent) -> None:
        usage = event.usage
        cost = (
            cost_usd(usage.model, usage.input_tokens, usage.output_tokens, usage.web_search_requests)
            if usage and usage.price_from_table
            else None
        )
        duration = event.duration_ms if event.duration_ms is not None else (usage.duration_ms if usage else None)
        store_raw = self._should_store_raw(event)
        with self._conn:
            self._conn.execute(
                "INSERT INTO pipeline_events (investigation_id, document_id, stage, decision, reason, created_at,"
                " duration_ms, input_tokens, output_tokens, cost_usd, model, raw_prompt, raw_response)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.investigation_id, event.document_id, event.stage, event.decision, event.reason,
                    datetime.now(UTC).isoformat(), duration,
                    usage.input_tokens if usage else None, usage.output_tokens if usage else None,
                    cost, usage.model if usage else None,
                    event.raw_prompt if store_raw else None,
                    event.raw_response if store_raw else None,
                ),
            )

    def set_retain_mode(self, mode: RetainRaw | bool) -> None:
        """Live-update the retention mode without reconstructing the recorder. Used by the
        /api/settings PUT path so the next event emitted picks up the new policy."""
        self._retain_raw = _coerce_retain_mode(mode)

    @property
    def retain_mode(self) -> RetainRaw:
        return self._retain_raw

    def purge_raw_only(self, before: date | None = None) -> int:
        """Null out `raw_prompt` and `raw_response` on existing rows (preserving the events).

        Lets the operator reclaim space without losing the audit trail that an event
        happened — the counts, timings and decisions stay. `before` scopes the purge to
        events older than that day; omitting it clears raw bytes from every row."""
        if before is not None:
            cutoff = datetime(before.year, before.month, before.day, tzinfo=UTC).isoformat()
            params: tuple = (cutoff,)
            where = " WHERE created_at < ? AND (raw_prompt IS NOT NULL OR raw_response IS NOT NULL)"
        else:
            params = ()
            where = " WHERE raw_prompt IS NOT NULL OR raw_response IS NOT NULL"
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE pipeline_events SET raw_prompt = NULL, raw_response = NULL" + where, params,
            )
        return cursor.rowcount

    def list_events(self, investigation_id: int) -> list[PipelineEvent]:
        rows = self._conn.execute(
            "SELECT id, investigation_id, document_id, stage, decision, reason, created_at, duration_ms,"
            " input_tokens, output_tokens, cost_usd, model FROM pipeline_events"
            " WHERE investigation_id = ? ORDER BY created_at, id",
            (investigation_id,),
        ).fetchall()
        events = []
        for r in rows:
            usage = None
            if r[8] is not None and r[9] is not None and r[11]:
                usage = LLMUsage(input_tokens=r[8], output_tokens=r[9], model=r[11], duration_ms=r[7] or 0)
            events.append(PipelineEvent(
                id=r[0], investigation_id=r[1], document_id=r[2], stage=r[3], decision=r[4], reason=r[5],
                created_at=datetime.fromisoformat(r[6]), duration_ms=r[7], usage=usage, cost_usd=r[10],
            ))
        return events

    def raw(self, event_id: int) -> tuple[str, str] | None:
        """Fetch the stored (prompt, response) pair for one event, or None when nothing was
        kept. The None covers two legitimate cases a caller must handle: this event never
        had raw I/O (collector/validator stages don't call an LLM), or retention was off/
        errors-only when the event was recorded."""
        row = self._conn.execute(
            "SELECT raw_prompt, raw_response FROM pipeline_events WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        return row[0], row[1]

    def delete_for(self, investigation_id: int) -> int:
        with self._conn:
            return self._conn.execute(
                "DELETE FROM pipeline_events WHERE investigation_id = ?", (investigation_id,)
            ).rowcount

    def delete_before(self, day: date) -> int:
        """Delete every event created before the start (UTC) of `day`."""
        cutoff = datetime(day.year, day.month, day.day, tzinfo=UTC).isoformat()
        with self._conn:
            return self._conn.execute("DELETE FROM pipeline_events WHERE created_at < ?", (cutoff,)).rowcount

    def delete_matching(self, before: date | None, investigation_id: int | None) -> int:
        """Delete the events that satisfy every given filter (both narrow the selection)."""
        clauses, params = [], []
        if before is not None:
            clauses.append("created_at < ?")
            params.append(datetime(before.year, before.month, before.day, tzinfo=UTC).isoformat())
        if investigation_id is not None:
            clauses.append("investigation_id = ?")
            params.append(investigation_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn:
            return self._conn.execute(f"DELETE FROM pipeline_events{where}", params).rowcount

    def size(self) -> tuple[int, int]:
        """Number of rows and the approximate bytes of stored text."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(COALESCE(reason, '')) + LENGTH(COALESCE(raw_prompt, ''))"
            " + LENGTH(COALESCE(raw_response, ''))), 0) FROM pipeline_events"
        ).fetchone()
        return row[0], row[1]
