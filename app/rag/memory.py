from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class SessionMemory:
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def history_for_prompt(self, limit: int = 8) -> list[dict[str, str]]:
        recent = self.messages[-limit:]
        return [{"role": msg.role, "content": msg.content} for msg in recent]


class MemoryStore:
    def __init__(self):
        self._sessions: dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: str | None) -> SessionMemory:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        new_id = session_id or str(uuid4())
        session = SessionMemory(session_id=new_id)
        self._sessions[new_id] = session
        return session

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


memory_store = MemoryStore()
