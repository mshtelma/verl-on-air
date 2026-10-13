import hashlib
import json
import zipfile

import pytest

import build_corpus
import prep_data


def test_table_rendering_preserves_numeric_strings_empty_cells_and_element_order():
    document = {"document": {"elements": [
        {"content": "  First line\r\nSecond line  "},
        {"content": "<table><tr><th>Year</th><th>Receipte</th><th>Note</th></tr>"
                    "<tr><td>1940</td><td>001,234.00</td><td></td></tr></table>"},
        {"content": "Last line"},
    ]}}
    text = build_corpus.render_document(document).decode()
    assert text.startswith("First line\nSecond line\n\n")
    assert "| Year | Receipts | Note |\n| --- | --- | --- |\n| 1940 | 001,234.00 |  |" in text
    assert text.endswith("\n\nLast line\n\n")
    assert "NaN" not in text


def test_corpus_identity_ignores_zip_order_directories_and_timestamps(tmp_path):
    documents = {"treasury_bulletin_1939_01.txt": b"first text\n", "treasury_bulletin_1939_02.txt": b"second text\n"}
    archives = []
    for index, names in enumerate((list(documents), list(reversed(documents)))):
        path = tmp_path / f"{index}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                member = zipfile.ZipInfo(("nested/" if index else "") + name, (1980 + index, 1, 1, 0, 0, 0))
                archive.writestr(member, documents[name])
        archives.append(path)
    assert archives[0].read_bytes() != archives[1].read_bytes()
    assert build_corpus.archive_identity(archives[0]) == build_corpus.archive_identity(archives[1])


def test_duplicate_document_basenames_are_rejected(tmp_path):
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("one/doc.txt", "first")
        archive.writestr("two/doc.txt", "second")
    with pytest.raises(ValueError, match="duplicate"):
        build_corpus.archive_identity(path)


def test_rebuilding_identical_json_produces_identical_archive_bytes(tmp_path):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "treasury_bulletin_1939_01.json").write_text(json.dumps({
        "document": {"elements": [{"content": "Total: 1,234.00"}]}}))
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    assert build_corpus.build_archive(parsed, first) == build_corpus.build_archive(parsed, second)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()


def snapshot_fixture(tmp_path):
    csv = tmp_path / "officeqa_full.csv"
    csv.write_text("uid,question,answer\nsynthetic,Synthetic total?,1234\n")
    archive = tmp_path / build_corpus.ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("treasury_bulletin_1939_01.txt", "Total: 1,234.00\n")
    transformed = tmp_path / "transformed"
    transformed.mkdir()
    (transformed / "treasury_bulletin_1939_01.txt").write_text("Total: 1234.0\n")
    chunks = build_corpus.build_chunks(transformed, tmp_path / "chunks.jsonl")
    lock = {"source": {"repository": "synthetic", "revision": "test"},
            "question_csv": build_corpus.file_identity(csv),
            "corpus": build_corpus.archive_identity(archive), "chunks": chunks}
    return lock


def test_snapshot_keeps_reading_and_retrieval_renderings_separate(tmp_path):
    lock = snapshot_fixture(tmp_path)
    verified = prep_data.verify_snapshot(tmp_path, lock)
    assert verified["status"] == "MATCH" and verified["chunks"]["count"] == 1
    with zipfile.ZipFile(tmp_path / build_corpus.ARCHIVE_NAME) as archive:
        assert b"1,234.00" in archive.read("treasury_bulletin_1939_01.txt")
    chunk = json.loads((tmp_path / "chunks.jsonl").read_text())
    assert "1234.0" in chunk["content"] and "1,234.00" not in chunk["content"]


@pytest.mark.parametrize("changed", ["officeqa_full.csv", "chunks.jsonl", build_corpus.ARCHIVE_NAME])
def test_preparation_rejects_a_changed_question_corpus_or_index(tmp_path, changed):
    lock = snapshot_fixture(tmp_path)
    if changed.endswith(".zip"):
        with zipfile.ZipFile(tmp_path / changed, "w") as archive:
            archive.writestr("treasury_bulletin_1939_01.txt", "Total: 999\n")
    else:
        with (tmp_path / changed).open("a") as stream:
            stream.write("changed\n")
    with pytest.raises(ValueError, match="differs from the measured snapshot"):
        prep_data.verify_snapshot(tmp_path, lock)


def test_bad_parsed_document_cannot_be_silently_skipped(tmp_path):
    (tmp_path / "treasury_bulletin_1939_01.json").write_text('{"document": {}}')
    with pytest.raises(ValueError, match="elements"):
        build_corpus.build_archive(tmp_path, tmp_path / "corpus.zip")
    assert not (tmp_path / "corpus.zip").exists()
