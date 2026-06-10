"""
Tests for the multi-dataset extractors added in Phase 1 (MOSS, QQP).

These mirror the schema produced by the existing LMSYS loader
(`extract_first_user_queries`) so that the shared preprocessor / sampler /
embedding / eviction pipeline works unchanged across all three datasets.

Extractors are tested against small synthetic fixtures (no large downloads),
so the parsing logic is verified locally; full-scale runs happen on GLC.
"""

import pandas as pd
import pytest

from src.data.loader import extract_moss_queries, extract_qqp_queries


# ── Shared schema expectation ────────────────────────────────────────
REQUIRED_COLUMNS = {
    "query_id", "query_text", "original_index",
    "model", "language", "num_turns", "conversation_id",
}


# ═════════════════════════════════════════════════════════════════════
# MOSS extractor
# ═════════════════════════════════════════════════════════════════════

def _moss_record(rec_id, num_turns, plain_text):
    return {
        "id": rec_id,
        "prefix": "MOSS is an AI assistant.",
        "num_turns": num_turns,
        "plain_text": plain_text,
    }


class TestExtractMossQueries:
    def test_extracts_first_human_turn(self):
        """The query is the text between the first [Human]: and the first <eoh>."""
        recs = [_moss_record(
            0, 1,
            "[Human]: how do i bake bread<eoh> [MOSS]: Use flour and water.<eoa>",
        )]
        df = extract_moss_queries(recs)
        assert len(df) == 1
        assert df.iloc[0]["query_text"] == "how do i bake bread"

    def test_only_first_human_turn_in_multiturn(self):
        """In a multi-turn conversation, only the FIRST human utterance is taken."""
        recs = [_moss_record(
            7, 2,
            "[Human]: first question<eoh> [MOSS]: ans one.<eoa> "
            "[Human]: second question<eoh> [MOSS]: ans two.<eoa>",
        )]
        df = extract_moss_queries(recs, single_turn_only=False)
        assert len(df) == 1
        assert df.iloc[0]["query_text"] == "first question"
        assert "second question" not in df.iloc[0]["query_text"]

    def test_single_turn_only_filters_multiturn(self):
        """single_turn_only=True keeps only num_turns == 1 conversations."""
        recs = [
            _moss_record(1, 1, "[Human]: keep me<eoh> [MOSS]: ok.<eoa>"),
            _moss_record(2, 3, "[Human]: drop me<eoh> [MOSS]: ok.<eoa>"),
        ]
        df = extract_moss_queries(recs, single_turn_only=True)
        assert len(df) == 1
        assert df.iloc[0]["query_text"] == "keep me"

    def test_single_turn_only_false_keeps_all(self):
        recs = [
            _moss_record(1, 1, "[Human]: a<eoh> [MOSS]: ok.<eoa>"),
            _moss_record(2, 3, "[Human]: b<eoh> [MOSS]: ok.<eoa>"),
        ]
        df = extract_moss_queries(recs, single_turn_only=False)
        assert len(df) == 2

    def test_schema_and_metadata(self):
        recs = [_moss_record(5, 1, "[Human]: hello<eoh> [MOSS]: hi.<eoa>")]
        df = extract_moss_queries(recs)
        assert REQUIRED_COLUMNS.issubset(set(df.columns))
        row = df.iloc[0]
        assert row["model"] == "moss"
        assert row["language"] == "en"
        assert row["num_turns"] == 1

    def test_skips_records_without_human_marker(self):
        recs = [
            _moss_record(1, 1, "no markers here at all"),
            _moss_record(2, 1, "[Human]: valid<eoh> [MOSS]: ok.<eoa>"),
        ]
        df = extract_moss_queries(recs)
        assert len(df) == 1
        assert df.iloc[0]["query_text"] == "valid"

    def test_strips_whitespace(self):
        recs = [_moss_record(1, 1, "[Human]:   padded query   <eoh> [MOSS]: ok.<eoa>")]
        df = extract_moss_queries(recs)
        assert df.iloc[0]["query_text"] == "padded query"

    def test_max_rows_limit(self):
        recs = [_moss_record(i, 1, f"[Human]: q{i}<eoh> [MOSS]: a.<eoa>") for i in range(10)]
        df = extract_moss_queries(recs, max_rows=3)
        assert len(df) == 3


# ═════════════════════════════════════════════════════════════════════
# QQP extractor
# ═════════════════════════════════════════════════════════════════════

def _qqp_row(qid1, q1, qid2, q2, dup="0"):
    return {
        "qid1": str(qid1), "question1": q1,
        "qid2": str(qid2), "question2": q2,
        "is_duplicate": dup,
    }


class TestExtractQqpQueries:
    def test_flattens_pair_into_two_questions(self):
        rows = [_qqp_row(1, "What is Python?", 2, "How to learn Python?")]
        df = extract_qqp_queries(rows)
        texts = set(df["query_text"])
        assert texts == {"What is Python?", "How to learn Python?"}

    def test_dedups_repeated_qid(self):
        """The same question (same qid) appearing in multiple pairs yields one row."""
        rows = [
            _qqp_row(1, "What is X?", 2, "What is Y?"),
            _qqp_row(1, "What is X?", 3, "What is Z?"),  # qid 1 repeats
        ]
        df = extract_qqp_queries(rows)
        # unique qids: 1, 2, 3 → 3 rows, not 4
        assert len(df) == 3
        assert df["query_id"].is_unique

    def test_schema_and_metadata(self):
        rows = [_qqp_row(1, "q one", 2, "q two")]
        df = extract_qqp_queries(rows)
        assert REQUIRED_COLUMNS.issubset(set(df.columns))
        assert (df["model"] == "qqp").all()
        assert (df["language"] == "en").all()
        assert (df["num_turns"] == 1).all()

    def test_skips_empty_questions(self):
        rows = [_qqp_row(1, "", 2, "valid question")]
        df = extract_qqp_queries(rows)
        assert len(df) == 1
        assert df.iloc[0]["query_text"] == "valid question"

    def test_max_rows_limit_counts_pairs(self):
        rows = [_qqp_row(2 * i, f"q{2*i}", 2 * i + 1, f"q{2*i+1}") for i in range(10)]
        df = extract_qqp_queries(rows, max_rows=2)
        # 2 pairs → up to 4 unique questions
        assert len(df) <= 4
        assert len(df) >= 1
