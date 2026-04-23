"""End-to-end (pty / real-subprocess) scenarios for Aura.

These tests spawn a child Python interpreter that drives a real
:class:`aura.core.agent.Agent` (with a scripted FakeChatModel — no
network, no provider SDK). The parent communicates with the child via a
pty master fd OR via a signal file on disk, exactly like
``tests/test_g1_message_persistence.py``'s ``AC-G1-4`` dogfood case —
``StringIO`` / render-repr is not an acceptable substitute per
``feedback_dogfood_before_done.md``.

Select this tier with ``pytest -m e2e``.
"""
