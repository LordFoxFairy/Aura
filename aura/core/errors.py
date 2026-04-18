"""异常根 —— 所有 Aura app-level 异常都继承 AuraError。"""

from __future__ import annotations

# AuraError 在 aura.config.schema 中定义，因为 schema 是 aura 异常层级的引导
# 起点（AuraConfigError 定义于此），且放置于此可避免 aura.core.__init__ 的循环
# 导入问题。aura.core.errors 作为对外公开的规范路径，直接重新导出。
from aura.config.schema import AuraError

__all__ = ["AuraError"]
