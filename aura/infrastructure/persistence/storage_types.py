"""Pure data types for session storage, shared across layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    created_at: datetime
    last_used_at: datetime
    message_count: int
    first_user_prompt: str
