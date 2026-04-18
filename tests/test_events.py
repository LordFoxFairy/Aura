"""Tests for aura.core.events — AgentEvent union and dataclasses."""

import dataclasses

import pytest

from aura.core.events import AgentEvent, AssistantDelta, Final, ToolCallCompleted, ToolCallStarted


class TestAssistantDelta:
    """Tests for AssistantDelta dataclass."""

    def test_construct_positional(self) -> None:
        """AssistantDelta constructs with positional arg."""
        ev = AssistantDelta("hello")
        assert ev.text == "hello"

    def test_construct_keyword(self) -> None:
        """AssistantDelta constructs with keyword arg."""
        ev = AssistantDelta(text="hello")
        assert ev.text == "hello"

    def test_equality(self) -> None:
        """Two AssistantDelta with same text are equal."""
        ev1 = AssistantDelta("hello")
        ev2 = AssistantDelta("hello")
        assert ev1 == ev2

    def test_inequality(self) -> None:
        """Two AssistantDelta with different text are not equal."""
        ev1 = AssistantDelta("hello")
        ev2 = AssistantDelta("world")
        assert ev1 != ev2

    def test_frozen_raises_on_mutation(self) -> None:
        """AssistantDelta is frozen; mutating field raises FrozenInstanceError."""
        ev = AssistantDelta("hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.text = "world"  # type: ignore[misc]

    def test_isinstance_agent_event(self) -> None:
        """AssistantDelta instance passes isinstance check for AgentEvent."""
        ev = AssistantDelta("hello")
        assert isinstance(ev, (AssistantDelta, ToolCallStarted, ToolCallCompleted, Final))


class TestToolCallStarted:
    """Tests for ToolCallStarted dataclass."""

    def test_construct_positional(self) -> None:
        """ToolCallStarted constructs with positional args."""
        ev = ToolCallStarted("bash", {"cmd": "ls"})
        assert ev.name == "bash"
        assert ev.input == {"cmd": "ls"}

    def test_construct_keyword(self) -> None:
        """ToolCallStarted constructs with keyword args."""
        ev = ToolCallStarted(name="bash", input={"cmd": "ls"})
        assert ev.name == "bash"
        assert ev.input == {"cmd": "ls"}

    def test_equality(self) -> None:
        """Two ToolCallStarted with same fields are equal."""
        ev1 = ToolCallStarted("bash", {"cmd": "ls"})
        ev2 = ToolCallStarted("bash", {"cmd": "ls"})
        assert ev1 == ev2

    def test_inequality(self) -> None:
        """Two ToolCallStarted with different fields are not equal."""
        ev1 = ToolCallStarted("bash", {"cmd": "ls"})
        ev2 = ToolCallStarted("bash", {"cmd": "pwd"})
        assert ev1 != ev2

    def test_frozen_raises_on_mutation(self) -> None:
        """ToolCallStarted is frozen; mutating field raises FrozenInstanceError."""
        ev = ToolCallStarted("bash", {"cmd": "ls"})
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.name = "python"  # type: ignore[misc]

    def test_isinstance_agent_event(self) -> None:
        """ToolCallStarted instance passes isinstance check for AgentEvent."""
        ev = ToolCallStarted("bash", {"cmd": "ls"})
        assert isinstance(ev, (AssistantDelta, ToolCallStarted, ToolCallCompleted, Final))


class TestToolCallCompleted:
    """Tests for ToolCallCompleted dataclass."""

    def test_construct_positional_with_output(self) -> None:
        """ToolCallCompleted constructs with positional args."""
        ev = ToolCallCompleted("bash", "exit 0")
        assert ev.name == "bash"
        assert ev.output == "exit 0"
        assert ev.error is None

    def test_construct_positional_with_all_args(self) -> None:
        """ToolCallCompleted constructs with all positional args including error."""
        ev = ToolCallCompleted("bash", "exit 1", "command failed")
        assert ev.name == "bash"
        assert ev.output == "exit 1"
        assert ev.error == "command failed"

    def test_construct_keyword(self) -> None:
        """ToolCallCompleted constructs with keyword args."""
        ev = ToolCallCompleted(name="bash", output="exit 0")
        assert ev.name == "bash"
        assert ev.output == "exit 0"
        assert ev.error is None

    def test_construct_keyword_with_error(self) -> None:
        """ToolCallCompleted constructs with keyword error arg."""
        ev = ToolCallCompleted(name="bash", output="exit 1", error="command failed")
        assert ev.name == "bash"
        assert ev.output == "exit 1"
        assert ev.error == "command failed"

    def test_error_defaults_to_none(self) -> None:
        """ToolCallCompleted.error defaults to None when not provided."""
        ev = ToolCallCompleted("bash", "exit 0")
        assert ev.error is None

    def test_error_can_be_set(self) -> None:
        """ToolCallCompleted.error can be explicitly set to a string."""
        ev = ToolCallCompleted("bash", "exit 1", error="boom")
        assert ev.error == "boom"

    def test_equality(self) -> None:
        """Two ToolCallCompleted with same fields are equal."""
        ev1 = ToolCallCompleted("bash", "exit 0", "error msg")
        ev2 = ToolCallCompleted("bash", "exit 0", "error msg")
        assert ev1 == ev2

    def test_equality_both_no_error(self) -> None:
        """Two ToolCallCompleted with both None error are equal."""
        ev1 = ToolCallCompleted("bash", "exit 0")
        ev2 = ToolCallCompleted("bash", "exit 0")
        assert ev1 == ev2

    def test_inequality_error_vs_no_error(self) -> None:
        """ToolCallCompleted with error differs from one without."""
        ev1 = ToolCallCompleted("bash", "exit 0")
        ev2 = ToolCallCompleted("bash", "exit 0", error="boom")
        assert ev1 != ev2

    def test_frozen_raises_on_mutation(self) -> None:
        """ToolCallCompleted is frozen; mutating field raises FrozenInstanceError."""
        ev = ToolCallCompleted("bash", "exit 0")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.output = "exit 1"  # type: ignore[misc]

    def test_isinstance_agent_event(self) -> None:
        """ToolCallCompleted instance passes isinstance check for AgentEvent."""
        ev = ToolCallCompleted("bash", "exit 0")
        assert isinstance(ev, (AssistantDelta, ToolCallStarted, ToolCallCompleted, Final))


class TestFinal:
    """Tests for Final dataclass."""

    def test_construct_positional(self) -> None:
        """Final constructs with positional arg."""
        ev = Final("Answer is 42")
        assert ev.message == "Answer is 42"

    def test_construct_keyword(self) -> None:
        """Final constructs with keyword arg."""
        ev = Final(message="Answer is 42")
        assert ev.message == "Answer is 42"

    def test_equality(self) -> None:
        """Two Final with same message are equal."""
        ev1 = Final("Answer is 42")
        ev2 = Final("Answer is 42")
        assert ev1 == ev2

    def test_inequality(self) -> None:
        """Two Final with different messages are not equal."""
        ev1 = Final("Answer is 42")
        ev2 = Final("Answer is 43")
        assert ev1 != ev2

    def test_frozen_raises_on_mutation(self) -> None:
        """Final is frozen; mutating field raises FrozenInstanceError."""
        ev = Final("Answer is 42")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.message = "Answer is 43"  # type: ignore[misc]

    def test_isinstance_agent_event(self) -> None:
        """Final instance passes isinstance check for AgentEvent."""
        ev = Final("Answer is 42")
        assert isinstance(ev, (AssistantDelta, ToolCallStarted, ToolCallCompleted, Final))


class TestAgentEventUnion:
    """Tests for AgentEvent type alias."""

    def test_agent_event_is_union_of_all_types(self) -> None:
        """AgentEvent type annotation includes all event classes."""
        # Verify at runtime that all types can be checked
        events: list[AgentEvent] = [
            AssistantDelta("hello"),
            ToolCallStarted("bash", {"cmd": "ls"}),
            ToolCallCompleted("bash", "exit 0"),
            Final("done"),
        ]
        assert len(events) == 4
        assert all(
            isinstance(e, (AssistantDelta, ToolCallStarted, ToolCallCompleted, Final))
            for e in events
        )
