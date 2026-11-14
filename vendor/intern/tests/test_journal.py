from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import intern.journal as journal_module
from intern.journal import Journal, MessageEntry, SessionEntry
from intern.messages import AssistantMessage, UserMessage


def test_a_journal_reloads_exactly_what_was_written(tmp_path: Path):
    path = tmp_path / "s.jsonl"
    journal = Journal(cwd="/work", path=path)
    journal.append(MessageEntry(message=UserMessage(content="hi")))
    journal.append(MessageEntry(message=AssistantMessage(text="hello", model="p/m")))

    reloaded = Journal(cwd="ignored", path=path)
    assert [e.model_dump() for e in reloaded.entries()] == [e.model_dump() for e in journal.entries()]
    assert reloaded.leaf_id == journal.leaf_id
    assert isinstance(reloaded.entries()[0], SessionEntry)


def test_branching_starts_a_new_path_and_keeps_the_old_one(tmp_path: Path):
    journal = Journal(cwd="/work", path=tmp_path / "s.jsonl")
    first = journal.append(MessageEntry(message=UserMessage(content="one")))
    journal.append(MessageEntry(message=UserMessage(content="two")))
    journal.branch(first.id)
    journal.append(MessageEntry(message=UserMessage(content="three")))

    contents = [
        e.message.content
        for e in journal.active_path()
        if isinstance(e, MessageEntry) and isinstance(e.message, UserMessage)
    ]
    assert contents == ["one", "three"]
    assert len(journal.entries()) == 4
    reloaded = Journal(cwd="/work", path=tmp_path / "s.jsonl")
    assert [e.id for e in reloaded.active_path()] == [e.id for e in journal.active_path()]


def test_branching_to_an_unknown_entry_fails():
    journal = Journal(cwd="/work")
    with pytest.raises(KeyError):
        journal.branch("missing")


@pytest.mark.parametrize("tail", [b'{"kind":"message",', b'{"content":"\xf0\x9f'])
def test_torn_tail_is_preserved_and_recovered_before_future_appends(tmp_path, tail):
    path = tmp_path / "s.jsonl"
    original = Journal(cwd="/work", path=path)
    original.append(MessageEntry(message=UserMessage(content="kept")))
    prefix = path.read_bytes()
    path.write_bytes(prefix + tail)
    with pytest.warns(RuntimeWarning, match="incomplete journal tail"):
        recovered = Journal(cwd="ignored", path=path)
    assert recovered.session_id == original.session_id
    assert path.read_bytes() == prefix
    assert next(tmp_path.glob("*.torn-*")).read_bytes() == tail
    recovered.append(MessageEntry(message=UserMessage(content="continued")))
    assert len(Journal(cwd="ignored", path=path).entries()) == 3


@pytest.mark.parametrize("bad", [b"invalid\n", b'{"kind":"unknown"}', b'{"kind":"message"}\n', b"{bad}", b"garbage"])
def test_complete_or_schema_invalid_records_are_not_silently_recovered(tmp_path, bad):
    path = tmp_path / "s.jsonl"
    Journal(cwd="/work", path=path)
    data = path.read_bytes() + bad
    path.write_bytes(data)
    with pytest.raises(ValidationError):
        Journal(cwd="ignored", path=path)
    assert path.read_bytes() == data and not list(tmp_path.glob("*.torn-*"))


def test_interior_corruption_is_not_recovered(tmp_path):
    path = tmp_path / "s.jsonl"
    Journal(cwd="/work", path=path)
    data = path.read_bytes() + b'broken\n{"kind":"session","cwd":"/work"}\n'
    path.write_bytes(data)
    with pytest.raises(ValidationError):
        Journal(cwd="ignored", path=path)
    assert path.read_bytes() == data


def test_complete_final_record_without_newline_stays_separate_from_the_next_append(tmp_path):
    path = tmp_path / "s.jsonl"
    original = Journal(cwd="/work", path=path)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    resumed = Journal(cwd="ignored", path=path)
    resumed.append(MessageEntry(message=UserMessage(content="next")))
    assert Journal(cwd="ignored", path=path).session_id == original.session_id
    assert len(path.read_bytes().splitlines()) == 2


def test_failed_sync_never_publishes_a_phantom_entry_and_requires_reopen(tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    journal = Journal(cwd="/work", path=path)
    before, leaf = journal.entries(), journal.leaf_id

    def failed_sync(fd):
        raise OSError("simulated persistence failure")

    with monkeypatch.context() as patch:
        patch.setattr(journal_module.os, "fsync", failed_sync)
        with pytest.raises(OSError, match="persistence failure"):
            journal.append(MessageEntry(message=UserMessage(content="not acknowledged")))
    assert journal.entries() == before and journal.leaf_id == leaf
    with pytest.raises(OSError, match="reopen"):
        journal.append(MessageEntry(message=UserMessage(content="must stop")))
    # A failed sync may still leave a complete line: reopening reconciles what is actually on disk.
    reopened = Journal(cwd="ignored", path=path)
    assert len(reopened.entries()) == 2


def test_stale_writer_is_rejected_without_corrupting_the_journal(tmp_path):
    path = tmp_path / "s.jsonl"
    first = Journal(cwd="/work", path=path)
    stale = Journal(cwd="ignored", path=path)
    first.append(MessageEntry(message=UserMessage(content="first writer")))
    with pytest.raises(OSError, match="another writer"):
        stale.append(MessageEntry(message=UserMessage(content="stale writer")))
    assert len(stale.entries()) == 1
    assert len(Journal(cwd="ignored", path=path).entries()) == 2
