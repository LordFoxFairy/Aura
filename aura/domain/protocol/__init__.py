"""Aura-owned canonical protocol payload aliases."""

from aura.domain.protocol.events import (
    CommonProtocolEvent,
    CoordinationEvent,
    SubagentProtocolEvent,
    SubagentTaskNotificationPayload,
    TeamMessagePayload,
    TeamProtocolEvent,
    WireEvent,
)
from aura.domain.protocol.requests import HeadlessRequest

__all__ = [
    "CommonProtocolEvent",
    "CoordinationEvent",
    "HeadlessRequest",
    "SubagentProtocolEvent",
    "SubagentTaskNotificationPayload",
    "TeamMessagePayload",
    "TeamProtocolEvent",
    "WireEvent",
]
