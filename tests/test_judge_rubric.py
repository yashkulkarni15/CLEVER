from src.judge.rubric import build_messages, parse_verdict


def test_build_messages_contains_both_queries_and_asks_yes_no():
    msgs = build_messages("How tall is Everest?", "Height of Mount Everest?")
    blob = " ".join(m["content"] for m in msgs)
    assert "How tall is Everest?" in blob
    assert "Height of Mount Everest?" in blob
    assert "YES" in blob and "NO" in blob
    assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"


def test_parse_verdict():
    assert parse_verdict("YES") is True
    assert parse_verdict("yes.") is True
    assert parse_verdict("NO") is False
    assert parse_verdict("no, different topic") is False
    assert parse_verdict("MAYBE") is None
    assert parse_verdict("") is None
