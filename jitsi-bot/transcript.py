"""Live meeting transcript ordered by when each sentence was spoken.

Sentences from different participants finish transcription out of order, so
the whole file is rewritten (atomically) in ``spoken_at`` order after each new
sentence.  An interrupted bot therefore leaves a complete, readable file.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _parse(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Transcript:
    def __init__(self, path: Path, *, room: str, session_id: str, started: datetime,
                 echo: bool = True):
        self.path, self.room, self.session_id = path, room, session_id
        self.started, self.echo = started, echo
        self.entries: list[dict] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write()

    def add(self, result: dict) -> None:
        text = (result.get("text") or "").strip()
        if not text:
            return
        entry = {
            "spoken_at": result.get("spoken_at") or result.get("created_at"),
            "speaker": result.get("speaker_label") or "Unknown",
            "participant_id": result.get("participant_id"),
            "text": text,
            "language": result.get("language"),
            "utterance_id": result.get("utterance_id"),
        }
        self.entries.append(entry)
        if self.echo:
            print(f"[{self._offset(entry)}] {entry['speaker']}: {text}", flush=True)
        self.write()

    def _offset(self, entry: dict) -> str:
        spoken = _parse(entry["spoken_at"])
        return _elapsed((spoken - self.started).total_seconds()) if spoken else "--:--:--"

    def ordered(self) -> list[dict]:
        epoch = datetime.min.replace(tzinfo=timezone.utc)
        return sorted(self.entries, key=lambda item: _parse(item["spoken_at"]) or epoch)

    def render(self) -> str:
        lines = [
            f"# Transcript: {self.room}",
            "",
            f"- Started: {self.started.isoformat(timespec='seconds')}",
            f"- Parascribe session: `{self.session_id}`",
            "- Speaker names are the display names participants chose in Jitsi.",
            "",
        ]
        previous = None
        for entry in self.ordered():
            if entry["speaker"] != previous:
                lines += ["", f"**{entry['speaker']}** · {self._offset(entry)}"]
                previous = entry["speaker"]
            lines.append(entry["text"])
        return "\n".join(lines).rstrip() + "\n"

    def write(self) -> None:
        self._replace(self.path, self.render())
        self._replace(self.path.with_suffix(".json"), json.dumps({
            "room": self.room, "session_id": self.session_id,
            "started": self.started.isoformat(), "utterances": self.ordered(),
        }, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _replace(path: Path, content: str) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
