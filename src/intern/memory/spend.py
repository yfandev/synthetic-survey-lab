from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from intern.messages import System1Usage, Usage

STEWARD_DIR = "steward"


class StewardSpend(BaseModel):
    # What the steward's runs on one session added up to, once it went idle or closed: it runs again on a resumed
    # session's follow-up and on a deferred session's retry.
    runs: int = 0
    system1: System1Usage = Field(default_factory=System1Usage)
    writer: Usage = Field(default_factory=Usage)

    @property
    def cost_usd(self) -> float:
        return self.system1.cost_usd + self.writer.cost_usd


def spend_file(home: Path, journal: Path) -> Path:
    return home / STEWARD_DIR / f"{journal.stem}.json"


def read_spend(home: Path, journal: Path) -> StewardSpend:
    try:
        return StewardSpend.model_validate_json(spend_file(home, journal).read_text())
    except (OSError, ValueError):
        return StewardSpend()


def add_spend(home: Path, journal: Path, system1: System1Usage, writer: Usage) -> None:
    if not (system1.asks or system1.failed or writer.input_tokens):
        return
    spent = read_spend(home, journal)
    total = System1Usage(**{k: getattr(spent.system1, k) + getattr(system1, k) for k in System1Usage.model_fields})
    path = spend_file(home, journal)
    path.parent.mkdir(parents=True, exist_ok=True)
    added = StewardSpend(runs=spent.runs + 1, system1=total, writer=spent.writer + writer)
    path.write_text(added.model_dump_json(indent=1))
