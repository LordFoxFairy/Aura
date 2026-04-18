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
    # custom 留给 hook 作者自扩展，避免 LoopState 字段随业务无限膨胀。
    custom: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        # 必须原地 mutate：AgentLoop 持有同一个 LoopState 引用，新建对象不会被 loop 感知。
        self.turn_count = 0
        self.total_tokens_used = 0
        self.custom.clear()
