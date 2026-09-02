"""Model-free tests for the global cross-session datastore (llm-caching NOW.md Phase D1):
match scoring, byte-quota eviction, and append-only CRC log persistence/replay."""

from __future__ import annotations

import json

from mlx_dspark.datastore import (
    GlobalDatastore,
    RoundLog,
    cost_at_width,
    expected_accept_len,
    should_draft,
)


def test_no_match_returns_empty():
    ds = GlobalDatastore()
    ds.ingest([1, 2, 3, 4])
    draft, score = ds.match((9, 9, 9), 8)
    assert draft == []
    assert score == 0


def test_match_finds_earlier_occurrence():
    ds = GlobalDatastore()
    ds.ingest([4, 5, 6, 7, 8])
    draft, _score = ds.match((4, 5, 6), 8)
    assert draft == [7, 8]


def test_cross_sequence_match_is_the_whole_point():
    # session 1 ingests a completed output; session 2 (a DIFFERENT ingest call) drafts
    # from it -- this is what a per-request-only index (lookup.NGramIndex) cannot do.
    ds = GlobalDatastore()
    ds.ingest([10, 11, 12, 13, 14])          # "session 1"
    draft, _score = ds.match((10, 11, 12), 8)  # "session 2" queries cold
    assert draft == [13, 14]


def test_match_scoring_prefers_better_backward_agreement_over_recency():
    # two earlier occurrences of the trigram (5,6,7); the OLDER one has a preceding
    # context that agrees with the query far more than the more recent one does --
    # match-scoring must prefer it over "latest occurrence wins" (the trap NOW.md flags).
    ds = GlobalDatastore()
    old_ctx = [100, 101, 102, 103, 104, 105, 106, 107]
    ds.ingest(old_ctx + [5, 6, 7] + [200, 201])       # older, well-matching occurrence
    ds.ingest([9, 9, 9, 9, 9, 9, 9, 9] + [5, 6, 7] + [300, 301])  # newer, poor match
    query_context = old_ctx + [5, 6, 7]
    draft, score = ds.match((5, 6, 7), 8, query_context)
    assert draft[:2] == [200, 201]
    assert score >= 6  # most of `old_ctx`'s tail agreed


def test_persistence_round_trip(tmp_path):
    log_path = tmp_path / "ds.log"
    ds = GlobalDatastore(log_path)
    ds.ingest([1, 2, 3, 4, 5])
    ds.ingest([6, 7, 8])
    ds.close()

    ds2 = GlobalDatastore(log_path)
    assert ds2.tokens == [1, 2, 3, 4, 5, 6, 7, 8]
    assert ds2.n_replayed == 2
    assert not ds2.replay_truncated
    draft, _ = ds2.match((1, 2, 3), 2)
    assert draft == [4, 5]


def test_replay_tolerates_truncated_last_line(tmp_path):
    log_path = tmp_path / "ds.log"
    ds = GlobalDatastore(log_path)
    ds.ingest([1, 2, 3, 4, 5])
    ds.close()
    with log_path.open("a") as f:
        f.write('{"tokens": [6, 7], "crc32": 12345')  # truncated mid-write (no closing brace)

    ds2 = GlobalDatastore(log_path)
    assert ds2.tokens == [1, 2, 3, 4, 5]  # only the complete, valid record survives
    assert ds2.n_replayed == 1
    assert ds2.replay_truncated


def test_replay_rejects_corrupted_record(tmp_path):
    log_path = tmp_path / "ds.log"
    ds = GlobalDatastore(log_path)
    ds.ingest([1, 2, 3])
    ds.close()
    # a well-formed but bit-flipped record (wrong crc) must be rejected, not trusted
    with log_path.open("a") as f:
        f.write(json.dumps({"tokens": [9, 9, 9], "crc32": 0}) + "\n")

    ds2 = GlobalDatastore(log_path)
    assert ds2.tokens == [1, 2, 3]
    assert ds2.replay_truncated


def test_eviction_bounds_memory_and_drops_oldest_first():
    ds = GlobalDatastore(max_bytes=GlobalDatastore.BYTES_PER_TOKEN_ESTIMATE * 12)
    ds.ingest(list(range(0, 5)))     # oldest -- should be evicted first
    ds.ingest(list(range(100, 105)))
    ds.ingest(list(range(200, 205)))
    assert ds.nbytes_estimate <= ds.max_bytes + GlobalDatastore.BYTES_PER_TOKEN_ESTIMATE * 5
    assert 0 not in ds.tokens          # oldest sequence's tokens are gone
    assert 204 in ds.tokens            # newest sequence survives
    assert ds.n_evicted_sequences >= 1


def test_eviction_keeps_index_consistent_with_surviving_tokens():
    ds = GlobalDatastore(max_bytes=GlobalDatastore.BYTES_PER_TOKEN_ESTIMATE * 10)
    for base in (0, 100, 200, 300):
        ds.ingest([base + 1, base + 2, base + 3, base + 4, base + 5])
    # whatever survived must still be queryable through the rebuilt index
    last_base = 300
    draft, _ = ds.match((last_base + 1, last_base + 2, last_base + 3), 8)
    assert draft == [last_base + 4, last_base + 5]


def test_ingest_empty_sequence_is_a_noop():
    ds = GlobalDatastore()
    ds.ingest([])
    assert ds.tokens == []
    assert ds.n_sequences == 0


def test_cost_at_width_matches_known_points_and_interpolates():
    curve = {0: 1.0, 8: 1.4, 32: 7.6}
    assert cost_at_width(curve, 8) == 1.4
    assert cost_at_width(curve, 0) == 1.0
    mid = cost_at_width(curve, 20)
    assert 1.4 < mid < 7.6
    # extrapolates past the last point using the last segment's slope
    assert cost_at_width(curve, 40) > cost_at_width(curve, 32)


def test_expected_accept_len_is_bounded_by_evidence():
    assert expected_accept_len(match_len=10, backward_score=3) == 3.0
    assert expected_accept_len(match_len=2, backward_score=20) == 2.0


def test_should_draft_disabled_gate_always_drafts_on_a_match():
    assert should_draft(drafted_len=5, score=0, width=5, verify_curve=None) is True
    assert should_draft(drafted_len=0, score=0, width=5, verify_curve=None) is False


def test_should_draft_gate_rejects_low_confidence_wide_draft():
    # a long draft (width 32, expensive per the curve) with almost no backward evidence
    # should NOT clear the gate: predicted accept (~1) + 1 <= c(32).
    curve = {0: 1.0, 1: 1.05, 8: 1.4, 16: 2.9, 32: 7.6}
    assert should_draft(drafted_len=32, score=1, width=32, verify_curve=curve) is False
    # the same width with strong backward evidence should clear it
    assert should_draft(drafted_len=32, score=30, width=32, verify_curve=curve) is True


def test_round_log_totals_aggregate_by_source():
    log = RoundLog()
    log.record(source="datastore", match_len=8, score=6, predicted_accept=6, drafted=8,
               accepted=8, width=8, gate_state="drafted")
    log.record(source="lookup", match_len=3, score=3, predicted_accept=3, drafted=3,
               accepted=1, width=3, gate_state="drafted")
    log.record(source="datastore", match_len=32, score=1, predicted_accept=1, drafted=0,
               accepted=0, width=32, gate_state="gated_off")
    totals = log.totals()
    assert totals["datastore"]["rounds"] == 2
    assert totals["datastore"]["accepted"] == 8
    assert totals["datastore"]["gated_off"] == 1
    assert totals["lookup"]["rounds"] == 1
    assert totals["lookup"]["accepted"] == 1
