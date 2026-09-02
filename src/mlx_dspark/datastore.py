"""Global, persistent, cross-session datastore of previously generated/seen token
sequences (llm-caching project, NOW.md Phase D1 — the datastore this project builds
on top of the per-request :class:`~mlx_dspark.lookup.NGramIndex` and DFlash).

Unlike ``lookup.NGramIndex`` (per-request, dies with the request), a
:class:`GlobalDatastore` survives across requests and processes: sessions feed it their
completed outputs, and later sessions draft from it. This is the one gap the project's
prior-art search found nobody on any platform ships (see ``docs/research/
PRIOR_ART_DATASTORE_SPECULATION.md`` in the llm-caching repo) — vLLM's SuffixDecoding
global tree lives only for the process, llama.cpp's persisted n-gram cache is unused.

Design per NOW.md Phase D1 step 1:

  - **In-RAM only**, under a byte quota (default 2 GiB) — a single NVMe page fault would
    cost a whole speculation round, so no block/page storage; see NOW.md's "Explicitly
    parked" list.
  - **Persistence = an append-only CRC32'd token log**, one JSON line per ingested
    sequence, replayed on start. A truncated or corrupt final line (a kill mid-write) is
    tolerated: replay stops there and treats everything before it as durable, exactly
    like ``run_d0b.py``'s ``--resume`` checkpoint (same failure mode, same fix).
  - **FIFO eviction by sequence** once the byte quota is exceeded (oldest ingested
    sequence first), followed by an index rebuild over the surviving tail — this is the
    "idle compaction" NOW.md allows in place of a fancier live-eviction scheme, since it
    only runs when the budget is actually exceeded.
  - **Match-scored, not latest-occurrence-only.** An earlier D0a draft used
    latest-occurrence + "longer draft always wins" as the tie-break, which NOW.md records
    as a real methodological trap: a longer proposed draft is not necessarily a more
    ACCURATE one, and picking on length alone lets a source's sheer data volume win ties
    against a less-relevant match. Every candidate occurrence of the query trigram is
    scored by how far its PRECEDING context agrees with the query's preceding context
    (the same fix D0a applied, ``simulate_d0a.NGramIndex._backward_score``), and the
    best-scoring occurrence wins, recency breaking ties.

Bytes/token is workload-dependent (NOW.md: ~10 B/token on repetitive output, hundreds on
high-entropy text; D0a measures it per project) — ``BYTES_PER_TOKEN_ESTIMATE`` is a
conservative placeholder for the eviction budget, not a measurement.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

MIN_MATCH_NGRAM = 3          # trigram key, matches lookup.NGramIndex's default minimum
MAX_OCCURRENCES_PER_KEY = 8  # bound memory/time; oldest occurrences evicted first
BACKWARD_SCORE_CAP = 24      # how far back to score match quality, beyond the trigram itself


class GlobalDatastore:
    """In-RAM trigram-keyed, match-scored index over token sequences ingested across
    requests/sessions, with FIFO-by-sequence eviction under a byte quota and an
    append-only CRC32'd log for persistence across process restarts.
    """

    BYTES_PER_TOKEN_ESTIMATE = 10

    def __init__(self, log_path: str | Path | None = None, *,
                 max_bytes: int = 2 * 1024 ** 3, replay: bool = True):
        self.tokens: list[int] = []
        self.index: dict[tuple[int, ...], list[int]] = {}
        self._seq_bounds: list[int] = [0]  # cumulative end-positions, oldest sequence first
        self.max_bytes = max_bytes
        self.n_sequences = 0
        self.n_evicted_sequences = 0
        self.log_path = Path(log_path) if log_path else None
        self._log_f = None
        self.n_replayed = 0
        self.replay_truncated = False
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_f = self.log_path.open("a+")
            if replay:
                self._replay()

    @property
    def nbytes_estimate(self) -> int:
        return len(self.tokens) * self.BYTES_PER_TOKEN_ESTIMATE

    def close(self) -> None:
        if self._log_f is not None:
            self._log_f.close()
            self._log_f = None

    # -- ingestion --------------------------------------------------------------------

    def ingest(self, seq: list[int], *, persist: bool = True) -> None:
        """Add a completed sequence (e.g. one turn's generated tokens) to the datastore."""
        if not seq:
            return
        if persist and self._log_f is not None:
            self._append_log(seq)
        self._extend_index(seq)
        self._seq_bounds.append(len(self.tokens))
        self.n_sequences += 1
        self._evict_if_over_budget()

    def _extend_index(self, new_tokens: list[int]) -> None:
        for tok in new_tokens:
            if len(self.tokens) >= MIN_MATCH_NGRAM - 1:
                key = tuple(self.tokens[-(MIN_MATCH_NGRAM - 1):] + [tok])
                occs = self.index.setdefault(key, [])
                occs.append(len(self.tokens))  # position of `tok`
                if len(occs) > MAX_OCCURRENCES_PER_KEY:
                    occs.pop(0)
            self.tokens.append(tok)

    def _evict_if_over_budget(self) -> None:
        if self.nbytes_estimate <= self.max_bytes or len(self._seq_bounds) <= 2:
            return
        total = len(self.tokens)
        # pop the oldest sequence boundary while what would SURVIVE it is still over
        # budget -- not while the pre-eviction total is over budget, or this always
        # evicts down to a single sequence regardless of how much headroom one eviction
        # actually buys back.
        while len(self._seq_bounds) > 2:
            cut = self._seq_bounds[0]
            if (total - cut) * self.BYTES_PER_TOKEN_ESTIMATE <= self.max_bytes:
                break
            self._seq_bounds.pop(0)
            self.n_evicted_sequences += 1
        cut = self._seq_bounds[0]
        surviving = self.tokens[cut:]
        self._seq_bounds = [b - cut for b in self._seq_bounds]
        self.tokens = []
        self.index = {}
        self._extend_index(surviving)

    # -- lookup -----------------------------------------------------------------------

    def _backward_score(self, trigram_start: int, query_context: list[int]) -> int:
        """How many tokens immediately BEFORE the stored trigram occurrence (starting at
        ``trigram_start``) agree with the tokens immediately before ``query_context``'s own
        trailing trigram (``query_context``'s last ``MIN_MATCH_NGRAM`` tokens ARE that
        trigram, per every call site below) — i.e. both walks start one token to the left
        of their respective trigrams, so they compare like-for-like context, not the
        query's own trigram tail against the stored occurrence's pre-trigram context."""
        score = 0
        i = trigram_start - 1
        j = len(query_context) - MIN_MATCH_NGRAM - 1
        while score < BACKWARD_SCORE_CAP and i >= 0 and j >= 0 and self.tokens[i] == query_context[j]:
            score += 1
            i -= 1
            j -= 1
        return score

    def match(
        self, query_suffix: tuple[int, ...], max_width: int,
        query_context: list[int] | None = None,
    ) -> tuple[list[int], int]:
        """Return ``(drafted_continuation, backward_score)``. ``backward_score`` is how many
        tokens preceding the matched occurrence agree with ``query_context``'s tail — the
        confidence signal used both for occurrence tie-breaking and by
        :func:`expected_accept_len` for the drafting gate. ``([], 0)`` on no match."""
        occs = self.index.get(query_suffix)
        if not occs:
            return [], 0
        if query_context:
            best_pos, best_score = occs[-1], -1
            for pos in occs:
                score = self._backward_score(pos - (MIN_MATCH_NGRAM - 1), query_context)
                if score >= best_score:  # >= prefers the more recent occurrence on ties
                    best_score, best_pos = score, pos
            pos = best_pos
        else:
            pos, best_score = occs[-1], 0
        start = pos + 1
        return self.tokens[start:start + max_width], best_score

    # -- persistence --------------------------------------------------------------------

    def _append_log(self, seq: list[int]) -> None:
        crc = zlib.crc32(json.dumps(seq).encode())
        self._log_f.write(json.dumps({"tokens": seq, "crc32": crc}) + "\n")
        self._log_f.flush()

    def _replay(self) -> None:
        self._log_f.seek(0)
        for line in self._log_f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seq, crc = rec["tokens"], rec["crc32"]
            except (json.JSONDecodeError, KeyError, TypeError):
                self.replay_truncated = True
                break
            if zlib.crc32(json.dumps(seq).encode()) != crc:
                self.replay_truncated = True
                break
            self.ingest(seq, persist=False)
            self.n_replayed += 1
        self._log_f.seek(0, 2)  # back to end; 'a+' appends there regardless, but be explicit


def cost_at_width(curve: dict[int, float], width: int) -> float:
    """Interpolate a measured verify-cost curve (``{width: relative_ms}``, ``c(0) == 1.0``
    == one plain decode step) at an arbitrary width. Same interpolation as the D0a/D0b
    offline studies (``artifacts/mlx/*/d0a/simulate_d0a.py``) — duplicated here rather than
    imported so the production package has no dependency on the research scripts."""
    if width in curve:
        return curve[width]
    keys = sorted(curve)
    if width <= keys[0]:
        return curve[keys[0]]
    if width >= keys[-1]:
        w0, w1 = keys[-2], keys[-1]
        slope = (curve[w1] - curve[w0]) / (w1 - w0)
        return curve[w1] + slope * (width - w1)
    lo = max(k for k in keys if k <= width)
    hi = min(k for k in keys if k >= width)
    if lo == hi:
        return curve[lo]
    frac = (width - lo) / (hi - lo)
    return curve[lo] + frac * (curve[hi] - curve[lo])


def expected_accept_len(match_len: int, backward_score: int) -> float:
    """First-cut, conservative estimate of E[accepted tokens] for a candidate draft:
    greedy acceptance needs an EXACT token match, and ``backward_score`` (how much
    preceding context already agreed) is the only confidence signal available before
    verifying — so predict the draft is good only as far as the evidence for it goes.
    This is deliberately simple; NOW.md scopes calibrating it against this host's real
    acceptance rates (from the round log below) to Phase D2, not D1.
    """
    return float(min(match_len, backward_score))


def should_draft(drafted_len: int, score: float, width: int, verify_curve: dict | None) -> bool:
    """NOW.md's drafting gate: draft only when ``E[accepted] + 1 > c(width)`` under the
    host's measured verify-cost curve — i.e. only when the predicted round is expected to
    beat a plain decode step. No curve (``verify_curve is None``) = gate disabled (always
    draft on a match), matching D0a/D0b's placeholder-curve fallback."""
    if drafted_len <= 0:
        return False
    if verify_curve is None:
        return True
    predicted = expected_accept_len(drafted_len, score)
    return (predicted + 1.0) > cost_at_width(verify_curve, drafted_len)


class RoundLog:
    """One row per speculation round (NOW.md Phase D1 step 3): source, match length,
    predicted score, drafted/accepted counts, verify width, gate state. Kept as plain
    dicts (no I/O) so callers choose whether/how to persist — ``totals()`` gives the
    per-process aggregate NOW.md asks for without re-scanning ``rows`` by hand."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, *, source: str, match_len: int, score: float, predicted_accept: float,
               drafted: int, accepted: int, width: int, gate_state: str,
               draft_ms: float = 0.0, verify_ms: float = 0.0) -> None:
        self.rows.append({
            "round": len(self.rows), "source": source, "match_len": match_len,
            "score": score, "predicted_accept": predicted_accept, "drafted": drafted,
            "accepted": accepted, "width": width, "gate_state": gate_state,
            "draft_ms": draft_ms, "verify_ms": verify_ms,
        })

    def totals(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for row in self.rows:
            src = row["source"]
            d = out.setdefault(src, {"rounds": 0, "drafted": 0, "accepted": 0, "gated_off": 0})
            d["rounds"] += 1
            d["drafted"] += row["drafted"]
            d["accepted"] += row["accepted"]
            if row["gate_state"] == "gated_off":
                d["gated_off"] += 1
        return out
