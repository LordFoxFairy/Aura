"""Aura-owned canonical protocol payload aliases."""

from aura.domain.protocol.events import (
    CommonProtocolEvent,
    CoordinationEvent,
    SubagentProtocolEvent,
    TeamProtocolEvent,
    WireEvent,
)
from aura.domain.protocol.requests import HeadlessRequest

__all__ = [
    "CommonProtocolEvent",
    "CoordinationEvent",
    "HeadlessRequest",
    "SubagentProtocolEvent",
    "TeamProtocolEvent",
    "WireEvent",
]
