"""Integration test tier — exercise full LLM → tool → hook → asker → result loops.

Unit tests under ``tests/`` mock one component at a time; this tier wires the
real Agent / AgentLoop / HookChain / tool dispatch end-to-end, faking only
the LLM (via :class:`tests.conftest.FakeChatModel`) and, for one test,
the MCP transport.
"""
