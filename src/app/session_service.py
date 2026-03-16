from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4


@dataclass
class SessionState:
    reference_id: str
    turns: list[str] = field(default_factory=list)


class SessionService:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()

    def create(self, reference_id: str) -> tuple[str, SessionState]:
        session_id = uuid4().hex
        state = SessionState(reference_id=reference_id, turns=[])
        with self._lock:
            self._sessions[session_id] = state
        return session_id, state

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            return self._sessions[session_id]

    def append_turn(self, session_id: str, text: str, max_turn_len: int) -> SessionState:
        cleaned = text.strip()
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            state = self._sessions[session_id]
            if cleaned:
                state.turns.append(cleaned)
            if len(state.turns) > max_turn_len:
                state.turns = state.turns[-max_turn_len:]
            return state

    def update_reference(self, session_id: str, reference_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            state = self._sessions[session_id]
            state.reference_id = reference_id
            return state

    def reset(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            state = self._sessions[session_id]
            state.turns = []
            return state
