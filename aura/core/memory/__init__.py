"""Memory 子系统 —— eager walk-up + progressive walk-down + rules + `@imports`。

`context.py` 是唯一的 message 组装点；其余模块（primary memory、rules、
system prompt）都是它的纯输入。
"""

from aura.core.memory.context import Context, NestedFragment
from aura.core.memory.system_prompt import build_system_prompt

__all__ = [
    "Context",
    "NestedFragment",
    "build_system_prompt",
]
