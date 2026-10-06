import json

import path_report
import protocol


def test_clipped_observation_is_the_only_evidence_and_requires_later_delivery():
    record = protocol.new_record("episode", "What was the amount?")
    shown = protocol.observation(record, name="read_document", arguments={}, text="public value 100 " + "x" * 150 + " 99999",
                                 request=0, max_chars=100)
    assert len(shown) == 100 and "99999" not in shown
    protocol.delivered(record, request=0)
    assert not path_report.EpisodeLedger.from_record(record).get("obs_1").delivered
    protocol.delivered(record, request=1)
    obs = path_report.EpisodeLedger.from_record(record).get("obs_1")
    assert obs.delivered and obs.delivered_text == shown.split("\n", 1)[1]


def test_discarded_context_is_not_evidence_and_model_written_markers_are_not_parsed():
    record = protocol.new_record("episode", "q")
    protocol.observation(record, name="read_document", arguments={}, text="123", request=0, max_chars=100)
    protocol.submit(record, {"answer": "123", "path": [{"id": "a", "observation": "obs_1", "claim": "123"}]})
    protocol.finalize(record)
    ledger = path_report.EpisodeLedger.from_record(record)
    refs = path_report.check_references(path_report.parse_report(record["terminal_text"]), ledger)
    assert refs.codes == ["undelivered"]
    assert ledger.get("obs_999") is None


def test_nested_path_string_is_decoded_and_duplicate_keys_fail_closed():
    record = protocol.new_record("episode", "q")
    path = [{"id": "a", "observation": "obs_1", "claim": "value"}]
    protocol.submit(record, json.dumps({"answer": "1", "path": json.dumps(path)}))
    assert json.loads(record["terminal_text"])["path"] == path
    protocol.submit(record, '{"answer":"1","answer":"2","path":[]}')
    assert record["terminal_seen"] and not record["terminal_text"]
