"""Provider-specific context-window lookups.

Bug pin: a user session showed ``40.2k/512k [█░░░░░░░░░] 8%`` on the status
bar while the actual prompt overflowed the model's real max — Aura was
returning the 512k UNKNOWN-model default for ``deepseek:glm-5``, lying by
4× to the operator. The default was lowered to 128k AND the table now
explicitly enumerates the common DeepSeek / GLM / Qwen / Kimi families.

Each parametrize entry below pins ONE family→window mapping the user
counts on; if a future refactor accidentally drops it, this suite catches
the regression. The list is exhaustive enough that adding a new family
forces a corresponding test row — that's the contract.
"""

from __future__ import annotations

import pytest

from aura.core.llm import get_context_window


class TestKnownFamilies:

    @pytest.mark.parametrize("spec,expected", [
        # Anthropic
        ("anthropic:claude-opus-4-7", 200_000),
        ("anthropic:claude-sonnet-4-6", 200_000),
        ("anthropic:claude-haiku-4-5", 200_000),
        ("claude-3-5-sonnet-20241022", 200_000),
        # OpenAI
        ("openai:gpt-4o-mini", 128_000),
        ("openai:gpt-4o-2024-11-20", 128_000),
        ("openai:gpt-5", 400_000),
        ("openai:gpt-3.5-turbo", 16_385),
        ("openai:gpt-4", 8_192),
        ("openai:o1", 200_000),
        ("openai:o3-mini", 200_000),
        # Google
        ("google:gemini-2-flash", 1_000_000),
        ("google:gemini-1.5-pro", 1_000_000),
        # DeepSeek (the user's actual case family)
        ("deepseek:deepseek-chat", 128_000),
        ("deepseek:deepseek-coder-v3", 128_000),
        ("deepseek:deepseek-reasoner", 64_000),
        ("deepseek:deepseek-v3", 128_000),
        # Zhipu GLM (the user's actual case family — glm-5 specifically)
        ("deepseek:glm-5", 128_000),
        ("zhipu:glm-4.5", 128_000),
        ("zhipu:glm-4-plus", 128_000),
        ("zhipu:glm-4-long", 1_000_000),
        ("zhipu:glm-4-air", 128_000),
        ("zhipu:glm-z1-flash", 128_000),
        # Aliyun DashScope / Qwen
        ("dashscope:qwen-max", 32_000),
        ("dashscope:qwen-plus", 128_000),
        ("dashscope:qwen-turbo", 128_000),
        ("dashscope:qwen-turbo-1m", 1_000_000),
        ("dashscope:qwen-long", 10_000_000),
        ("dashscope:qwen3-coder", 128_000),
        ("dashscope:qwen3-32b", 128_000),
        ("dashscope:qwen2.5-72b", 128_000),
        # Moonshot / Kimi
        ("moonshot:moonshot-v1-128k", 128_000),
        ("moonshot:moonshot-v1-32k", 32_000),
        ("moonshot:moonshot-v1-8k", 8_000),
        ("moonshot:kimi-k2", 128_000),
    ])
    def test_known_window(self, spec: str, expected: int) -> None:
        assert get_context_window(spec) == expected, (
            f"{spec} returned {get_context_window(spec)}, expected {expected}"
        )


class TestUnknownDefault:

    def test_unknown_model_falls_back_to_safe_default(self) -> None:
        """Default is now 128k (claude-code parity) — over-stated default
        was a real bug because the status bar lied to operators on common
        Chinese models.
        """
        assert get_context_window("totally-unknown-model-spec-2099") == 128_000

    def test_default_constant_is_128k(self) -> None:
        from aura.core.llm import _DEFAULT_CONTEXT_WINDOW
        assert _DEFAULT_CONTEXT_WINDOW == 128_000


class TestProviderPrefixIgnored:
    """Provider prefix doesn't change the lookup — only the model id matters."""

    @pytest.mark.parametrize("spec1,spec2", [
        ("anthropic:claude-opus-4", "claude-opus-4"),
        ("openai:gpt-4o", "gpt-4o"),
        ("deepseek:deepseek-chat", "deepseek-chat"),
        ("dashscope:qwen-plus", "qwen-plus"),
    ])
    def test_with_and_without_prefix_same(self, spec1: str, spec2: str) -> None:
        assert get_context_window(spec1) == get_context_window(spec2)


class TestLongestPrefixWins:
    """When multiple keys match, the longest one wins — so qwen-turbo-1m
    beats qwen-turbo, and glm-4-long beats glm-4. Regression: the lookup
    used to use insertion order which was fragile.
    """

    def test_glm_4_long_beats_glm_4(self) -> None:
        assert get_context_window("zhipu:glm-4-long") == 1_000_000

    def test_qwen_turbo_1m_beats_qwen_turbo(self) -> None:
        assert get_context_window("dashscope:qwen-turbo-1m") == 1_000_000

    def test_glm_5_does_not_match_glm_4(self) -> None:
        # No glm-5 key existed pre-fix → default. Now there's a real entry.
        assert get_context_window("deepseek:glm-5") == 128_000

    def test_gpt_4o_mini_beats_gpt_4(self) -> None:
        # Was already correct, but pin the regression.
        assert get_context_window("openai:gpt-4o-mini") == 128_000
