"""
Data loader for LMSYS-Chat-1M dataset.

Downloads the dataset from HuggingFace and extracts the first user message
from each conversation for use as cached queries.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_lmsys_dataset(
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> "Dataset":
    """
    Download / load LMSYS-Chat-1M from HuggingFace.

    Args:
        cache_dir: Directory to cache the raw HuggingFace download.
        streaming: If True, stream data instead of downloading all at once.

    Returns:
        HuggingFace Dataset object.
    """
    logger.info("Loading LMSYS-Chat-1M dataset from HuggingFace...")
    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        cache_dir=cache_dir,
        streaming=streaming,
    )
    logger.info(f"Dataset loaded: {len(ds) if not streaming else '(streaming)'} conversations")
    return ds


def extract_first_user_queries(dataset, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Extract the first user message from each conversation.

    Each row in LMSYS-Chat-1M has a 'conversation' field which is a list
    of dicts like [{"role": "user", "content": "..."}, {"role": "assistant", ...}, ...].
    We extract only the first user turn — this represents the initial query
    that would be cached in a semantic cache system.

    Args:
        dataset: HuggingFace Dataset with 'conversation' column.
        max_rows: Optional limit on number of rows to process (for dev/debugging).

    Returns:
        DataFrame with columns: [query_id, query_text, original_index, model, language, num_turns]
    """
    logger.info("Extracting first user queries from conversations...")

    records = []
    total = max_rows if max_rows else len(dataset)

    for idx, row in enumerate(tqdm(dataset, total=total, desc="Extracting queries")):
        if max_rows and idx >= max_rows:
            break

        conversation = row.get("conversation", [])
        if not conversation:
            continue

        # Find the first user message
        first_user_msg = None
        for turn in conversation:
            if turn.get("role") == "user":
                first_user_msg = turn.get("content", "").strip()
                break

        if first_user_msg:
            records.append({
                "query_id": idx,
                "query_text": first_user_msg,
                "original_index": idx,
                "model": row.get("model", "unknown"),
                "language": row.get("language", "unknown"),
                "num_turns": row.get("turn", 0),
                "conversation_id": row.get("conversation_id", ""),
            })

    df = pd.DataFrame(records)
    logger.info(f"Extracted {len(df)} user queries from {total} conversations")
    return df


def extract_moss_queries(
    records,
    max_rows: Optional[int] = None,
    single_turn_only: bool = True,
) -> pd.DataFrame:
    """
    Extract the first human query from each MOSS conversation.

    MOSS (`fnlp/moss-002-sft-data`) stores each English conversation as a dict
    ``{"id", "prefix", "num_turns", "plain_text"}`` where ``plain_text`` looks
    like ``"[Human]: <q1><eoh> [MOSS]: <a1><eoa> [Human]: <q2><eoh> ..."``.
    We take only the FIRST human utterance (the text between the first
    ``[Human]:`` and the first ``<eoh>``), mirroring the LMSYS loader's
    "first user turn" semantics so the downstream pipeline is identical.

    Args:
        records: Iterable of MOSS record dicts (parsed from an ``en_*.json``).
        max_rows: Optional cap on number of records processed (dev/smoke).
        single_turn_only: If True, keep only conversations with
            ``num_turns == 1`` (genuine single-turn exchanges, per the MOSS
            "medium-density" role). If False, take the first human turn from
            every conversation regardless of length.

    Returns:
        DataFrame with the shared schema:
        [query_id, query_text, original_index, model, language,
         num_turns, conversation_id].
    """
    logger.info("Extracting first human queries from MOSS conversations...")

    human_marker = "[Human]:"
    eoh_marker = "<eoh>"

    records_out = []
    for idx, rec in enumerate(records):
        if max_rows is not None and len(records_out) >= max_rows:
            break

        num_turns = rec.get("num_turns", 0)
        if single_turn_only and num_turns != 1:
            continue

        plain_text = rec.get("plain_text", "") or ""
        h_pos = plain_text.find(human_marker)
        if h_pos == -1:
            continue
        start = h_pos + len(human_marker)
        eoh_pos = plain_text.find(eoh_marker, start)
        query_text = (
            plain_text[start:eoh_pos] if eoh_pos != -1 else plain_text[start:]
        ).strip()

        if not query_text:
            continue

        rec_id = rec.get("id", idx)
        records_out.append({
            "query_id": rec_id,
            "query_text": query_text,
            "original_index": idx,
            "model": "moss",
            "language": "en",
            "num_turns": num_turns,
            "conversation_id": f"moss-{rec_id}",
        })

    df = pd.DataFrame(records_out)
    logger.info(f"Extracted {len(df)} MOSS queries (single_turn_only={single_turn_only})")
    return df


def extract_qqp_queries(rows, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Flatten Quora Question Pairs into a deduplicated list of unique questions.

    QQP (`quora-competitions/quora`) is distributed as a TSV with columns
    ``qid1, question1, qid2, question2, is_duplicate``. The paper uses the
    *question* text (not the duplicate label), so we flatten every pair into
    its two questions and deduplicate by question id — the same question
    appears in many pairs and must be cached once.

    Args:
        rows: Iterable of TSV row dicts (e.g. from ``csv.DictReader``).
        max_rows: Optional cap on number of PAIRS processed (dev/smoke).

    Returns:
        DataFrame with the shared schema:
        [query_id, query_text, original_index, model, language,
         num_turns, conversation_id].
    """
    logger.info("Extracting unique questions from Quora Question Pairs...")

    seen_qids: set[str] = set()
    records_out = []
    for pair_idx, row in enumerate(rows):
        if max_rows is not None and pair_idx >= max_rows:
            break

        for qid_key, q_key in (("qid1", "question1"), ("qid2", "question2")):
            qid = str(row.get(qid_key, "")).strip()
            text = (row.get(q_key, "") or "").strip()
            if not qid or not text or qid in seen_qids:
                continue
            seen_qids.add(qid)
            records_out.append({
                "query_id": int(qid) if qid.isdigit() else qid,
                "query_text": text,
                "original_index": len(records_out),
                "model": "qqp",
                "language": "en",
                "num_turns": 1,
                "conversation_id": f"qqp-{qid}",
            })

    df = pd.DataFrame(records_out)
    logger.info(f"Extracted {len(df)} unique Quora questions")
    return df


def save_raw_queries(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save extracted queries to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} raw queries to {output_path}")
    return output_path


def load_processed_queries(path: str | Path) -> pd.DataFrame:
    """Load previously saved processed queries from parquet."""
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} queries from {path}")
    return df
