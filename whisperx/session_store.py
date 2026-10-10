"""Persistent, session-scoped transcripts and speaker voiceprints.

Voiceprints are biometric data.  They deliberately live in their own table so
they can be deleted without deleting the transcript.  Speaker identifiers are
only meaningful inside one session; there is no global identity table and no
cross-session matching.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionNotFound(KeyError):
    """Raised when a session id is unknown."""


class SessionEnded(RuntimeError):
    """Raised when new speech is submitted to an ended session."""


class VoiceprintsDisabled(RuntimeError):
    """Raised when attempting to store a voiceprint after consent withdrawal."""


class SessionStore:
    """SQLite-backed session store safe to share between service replicas."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        # Overwrite deleted cells before releasing their pages.  This cannot
        # erase external backups or storage snapshots, which operators must
        # manage separately.
        connection.execute("PRAGMA secure_delete = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._schema_lock:
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL CHECK (status IN ('active', 'ended')),
                    capture_voiceprints INTEGER NOT NULL CHECK (capture_voiceprints IN (0, 1)),
                    signatures_deleted_at TEXT,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS voiceprints (
                    session_id TEXT NOT NULL,
                    speaker_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimensions INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL,
                    speech_seconds REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, speaker_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS utterances (
                    utterance_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT,
                    speaker_id TEXT,
                    speaker_similarity REAL,
                    audio_seconds REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    UNIQUE (session_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS utterances_session_idx
                    ON utterances(session_id, sequence);
                """
                )
                row = connection.execute(
                    "SELECT version FROM schema_info LIMIT 1").fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
                elif row["version"] != SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Unsupported session database schema {row['version']} "
                        f"(expected {SCHEMA_VERSION})"
                    )
                connection.commit()
            finally:
                connection.close()

    @staticmethod
    def _session(row: sqlite3.Row, *, speaker_count: int = 0, utterance_count: int = 0) -> dict:
        return {
            "session_id": row["session_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "ended_at": row["ended_at"],
            "capture_voiceprints": bool(row["capture_voiceprints"]),
            "signatures_deleted_at": row["signatures_deleted_at"],
            "metadata": json.loads(row["metadata_json"]),
            "speaker_count": speaker_count,
            "utterance_count": utterance_count,
        }

    def create_session(
        self, *, capture_voiceprints: bool = True, metadata: dict[str, Any] | None = None
    ) -> dict:
        session_id = uuid.uuid4().hex
        now = _now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO sessions(
                       session_id, created_at, updated_at, status,
                       capture_voiceprints, metadata_json
                   ) VALUES (?, ?, ?, 'active', ?, ?)""",
                (session_id, now, now, int(capture_voiceprints), json.dumps(metadata or {})),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str, *, include_utterances: bool = False) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise SessionNotFound(session_id)
            speaker_count = connection.execute(
                "SELECT COUNT(*) FROM voiceprints WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            utterance_count = connection.execute(
                "SELECT COUNT(*) FROM utterances WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            result = self._session(
                row, speaker_count=speaker_count, utterance_count=utterance_count
            )
            if include_utterances:
                result["utterances"] = [
                    dict(item)
                    for item in connection.execute(
                        """SELECT utterance_id, sequence, created_at, source, text,
                                  language, speaker_id, speaker_similarity, audio_seconds
                           FROM utterances WHERE session_id = ? ORDER BY sequence""",
                        (session_id,),
                    )
                ]
            return result

    @staticmethod
    def _normalise(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not vector.size or not np.all(np.isfinite(vector)):
            raise ValueError("voiceprint embedding must contain finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("voiceprint embedding cannot be zero")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    @staticmethod
    def _decode(row: sqlite3.Row) -> np.ndarray:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.size != row["dimensions"]:
            raise RuntimeError("stored voiceprint has invalid dimensions")
        return vector

    def match_voiceprint(
        self,
        session_id: str,
        embedding: np.ndarray,
        *,
        audio_seconds: float,
        similarity_threshold: float,
        update_threshold: float,
    ) -> tuple[str | None, float | None, bool]:
        """Match or enroll one speaker inside ``session_id``.

        Returns ``(speaker_id, similarity, enrolled)``.  Marginal matches are
        identified but do not update the centroid, which limits drift.
        """
        vector = self._normalise(embedding)
        now = _now()
        with self._transaction(immediate=True) as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise SessionNotFound(session_id)
            if session["status"] != "active":
                raise SessionEnded(session_id)
            if not session["capture_voiceprints"]:
                return None, None, False

            rows = list(
                connection.execute(
                    "SELECT * FROM voiceprints WHERE session_id = ? ORDER BY speaker_id",
                    (session_id,),
                )
            )
            compatible = [row for row in rows if row["dimensions"] == vector.size]
            if compatible:
                similarities = [float(np.dot(vector, self._decode(row))) for row in compatible]
                best_index = int(np.argmax(similarities))
                best = compatible[best_index]
                similarity = similarities[best_index]
                if similarity >= similarity_threshold:
                    if similarity >= update_threshold:
                        samples = int(best["sample_count"])
                        centroid = self._normalise(
                            (self._decode(best) * samples + vector) / (samples + 1)
                        )
                        connection.execute(
                            """UPDATE voiceprints
                               SET embedding = ?, sample_count = ?, speech_seconds = ?,
                                   updated_at = ?
                               WHERE session_id = ? AND speaker_id = ?""",
                            (
                                centroid.tobytes(), samples + 1,
                                float(best["speech_seconds"]) + audio_seconds,
                                now, session_id, best["speaker_id"],
                            ),
                        )
                    connection.execute(
                        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                        (now, session_id),
                    )
                    return best["speaker_id"], similarity, False

            next_number = 1 + max(
                (int(row["speaker_id"].split("_")[-1]) for row in rows), default=0
            )
            speaker_id = f"speaker_{next_number}"
            connection.execute(
                """INSERT INTO voiceprints(
                       session_id, speaker_id, embedding, dimensions, sample_count,
                       speech_seconds, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    session_id, speaker_id, vector.tobytes(), vector.size,
                    audio_seconds, now, now,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            return speaker_id, None, True

    def append_utterance(
        self,
        session_id: str,
        *,
        text: str,
        language: str | None,
        speaker_id: str | None,
        speaker_similarity: float | None,
        audio_seconds: float,
        source: str,
    ) -> dict:
        now = _now()
        utterance_id = uuid.uuid4().hex
        with self._transaction(immediate=True) as connection:
            session = connection.execute(
                "SELECT status FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise SessionNotFound(session_id)
            if session["status"] != "active":
                raise SessionEnded(session_id)
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM utterances WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO utterances(
                       utterance_id, session_id, sequence, created_at, source, text,
                       language, speaker_id, speaker_similarity, audio_seconds
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    utterance_id, session_id, sequence, now, source, text, language,
                    speaker_id, speaker_similarity, audio_seconds,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
        return {
            "utterance_id": utterance_id,
            "session_id": session_id,
            "sequence": sequence,
            "created_at": now,
            "source": source,
            "text": text,
            "language": language,
            "speaker_id": speaker_id,
            "speaker_similarity": speaker_similarity,
            "audio_seconds": audio_seconds,
        }

    def delete_voiceprints(self, session_id: str) -> int:
        now = _now()
        with self._transaction(immediate=True) as connection:
            if connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone() is None:
                raise SessionNotFound(session_id)
            deleted = connection.execute(
                "DELETE FROM voiceprints WHERE session_id = ?", (session_id,)
            ).rowcount
            connection.execute(
                """UPDATE sessions
                   SET capture_voiceprints = 0, signatures_deleted_at = ?, updated_at = ?
                   WHERE session_id = ?""",
                (now, now, session_id),
            )
        # Force deleted WAL content into the main database and truncate the WAL.
        with self._connection() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return deleted

    def end_session(self, session_id: str, *, delete_voiceprints: bool = False) -> dict:
        now = _now()
        with self._transaction(immediate=True) as connection:
            if connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone() is None:
                raise SessionNotFound(session_id)
            connection.execute(
                """UPDATE sessions SET status = 'ended', ended_at = COALESCE(ended_at, ?),
                       updated_at = ? WHERE session_id = ?""",
                (now, now, session_id),
            )
            if delete_voiceprints:
                connection.execute(
                    "DELETE FROM voiceprints WHERE session_id = ?", (session_id,))
                connection.execute(
                    """UPDATE sessions
                       SET capture_voiceprints = 0, signatures_deleted_at = ?, updated_at = ?
                       WHERE session_id = ?""",
                    (now, now, session_id),
                )
        if delete_voiceprints:
            with self._connection() as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return self.get_session(session_id, include_utterances=True)
