"""Binary query-equivalence rubric for the independent cache-hit judge.

We judge whether a correct answer to the ORIGINAL cached question would also
correctly and completely answer the NEW question. This is deliberately stricter
than topical similarity and is independent of the embedding/cosine signal that
produced the hit (circularity-free)."""
from typing import Optional

SYSTEM = (
    "You are a strict evaluator of semantic cache hits. You decide whether a "
    "cached answer can be safely reused for a new question. Judge meaning, not "
    "wording. Be conservative: if reusing the original answer could mislead or "
    "omit something the new question specifically asks for, answer NO."
)


def build_messages(orig_query: str, new_query: str) -> list[dict]:
    user = (
        "A semantic cache returned the answer to a previously seen question in "
        "response to a new question.\n\n"
        f'ORIGINAL (cached) question:\n"""{orig_query}"""\n\n'
        f'NEW (incoming) question:\n"""{new_query}"""\n\n'
        "Would a correct, complete answer to the ORIGINAL question also be a "
        "correct and complete answer to the NEW question?\n"
        "Reply with exactly one word: YES or NO."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_verdict(text: str) -> Optional[bool]:
    t = (text or "").strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    return None
