"""对话生命周期的累计状态 — 随 AgentLoop 共享引用，跨 turn 存活。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopState:
    # turn_count 在每次 _invoke_model 入口前 +1，pre_model hook 看到的是"即将开始的第 N 轮"。
    turn_count: int = 0
    # total_tokens_used 由 make_usage_tracking_hook 在 post_model 阶段填入，loop 本身不写。
    total_tokens_used: int = 0
    # custom — per-session transient scratchpad for hooks / tools.
    #
    # Legitimate keys currently in use (contract lock — any new slot
    # here MUST land with a matching docstring update and a justified
    # owner, not silently):
    #
    # - denials sink: :data:`aura.core.permissions.denials.DENIALS_SINK_KEY`
    #   (G5). Shared list reference between the permission hook
    #   (writer) and :class:`aura.core.agent.Agent` (owner + reader
    #   via ``last_turn_denials()``).
    # - ``"_token_stats"`` — populated by ``make_usage_tracking_hook``;
    #   read by the status bar + ``/verbose``.
    # - ``"todos"`` — populated by ``todo_write`` tool; read by
    #   compact / system prompt assembly.
    #
    # Do NOT add new transient slots for one-shot hook→loop signalling:
    # G4 removed the last per-call decision side-channel in favor of
    # :class:`aura.core.hooks.PreToolOutcome` direct-return. New
    # lifecycle data should ride typed return values, not a dict slot
    # here.
    custom: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        # 必须原地 mutate：AgentLoop 持有同一个 LoopState 引用，新建对象不会被 loop 感知。
        self.turn_count = 0
        self.total_tokens_used = 0
        self.custom.clear()
