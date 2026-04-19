"""Tests for aura.core.events — AgentEvent union and dataclasses."""

import dataclasses

import pytest

from aura.core.events import AgentEvent, AssistantDelta, Final, ToolCallCompleted, ToolCallStarted


class TestAssistantDelta:
    def test_construct_positional(self) -> None:
        ev = AssistantDelta("hello")
        assert ev.text == "hello"

    def test_construct_keyword(self) -> None:
        ev = AssistantDelta(text="hello")
        assert ev.text == "hello"

    def test_equality(self) -> None:
        ev1 = AssistantDelta("hello")
        ev2 = AssistantDelta("hello")
        assert ev1 == ev2

    def test_inequality(self) -> None:
        ev1 = AssistantDelta("hello")
        ev2 = AssistantDelta("world")
        assert ev1 != ev2

    def test_frozen_raises_on_mutation(self) -> None:
        ev = AssistantDelta("hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.text = "world"  # type: ignore[misc]

    def test_isinstance_agent_event(self) -> None:
        ev = AssistantDelta("hello")
        assert isinstance(ev, (AssistantDelta, ToolCallStarted, ToolCallCompleted, Final))


class TestToolCallStarted:
    def test_construct_positional(self) -> None:
        ev = ToolCallStarted("bash", {"cmd": "ls"})
        assert ev.name == "bash"
        assert ev.input == {"cmd": "ls"}

    def test_construct_keyword(self) -> None:
        ev = ToolCallStarted(name="bash", input={"cmd": "ls"})
        assert ev.name == "bash"
        assert ev.input == {"cmd": "ls"}

    def test_equality(self) -> None:
        ev1 = ToolCallStarted("bash", {"cmd": "ls"})
        ev2 = ToolCallStarted("bash", {"cmd": "ls"})
        assert ev1 == ev2

    def test_inequality(self) -> None:
        ev1 = ToolCallStarted("bash", {"cmd": "ls"})
        ev2 = ToolCallStarted("bash", {"cmd": "pwd"})
        assert ev1 != ev2

    def test_frozen_raises_on_mutation(self) -> None:
        ev = ToolCallStarted("bash", {"cmd": "ls"})
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.name = "python"  # type: ignore[misc]

    def test_isinstance_agent_event(self) -> None:
        ev = ToolCallStarted("bash", {"cmd": "ls"})
        assert isinstance(ev, (AssistantDelta, ToolCallStarted, ToolCallCompleted, Final))


class TestToolCallCompleted:
    def test_construct_positional_with_output(self) -> None:
        ev = ToolCallCompleted("bash", "exit 0")
        assert ev.name == "bash"
        assert ev.output == "exit 0"
        assert ev.error is None

    def test_construct_positional_with_all_args(self) -> None:
        ev = ToolCallCompleted("bash", "exit 1", "command failed")
        assert ev.name == "bash"
        assert ev.output == "exit 1"
        assert ev.error == "command failed"

    def test_construct_keyword(self) -> None:
        ev = ToolCallCompleted(name="bash", output="exit 0")
        assert ev.name == "bash"
        assert ev.output == "exit 0"
        assert ev.error is None

    def test_construct_keyword_with_error(self) -> None:
        ev = ToolCallCompleted(name="bash", output="exit 1", error="command failed")
        assert ev.name == "bash"
        assert ev.output == "exit 1"
        assert ev.error == "command failed"

    def test_error_defaults_to_none(self) -> None:
        ev = ToolCallCompleted("bash", "exit 0")
        assert ev.error is None

    def test_error_can_be_set(self) -> None:
        ev = ToolCallCompleted("bash", "exit 1", error="boom")
        assert ev.error == "boom"

    def test_equality(self) -> None:
        ev1 = ToolCallCompleted("bash", "exit 0", "error msg")
        ev2 = ToolCallCompleted("bash", "exit 0", "error msg")
        assert ev1 == ev2

    def test_equality_both_no_error(self) -> None:
        ev1 = ToolCallCompleted("bash", "exit 0")
        ev2 = ToolCallCompleted("bash", "exit 0")
        assert ev1 == ev2

    def test_inequality_error_vs_no_error(self) -> None:
        ev1 = ToolCallCompleted("bash", "exit 0")
        ev2 = ToolCallCompleted("bash", "exit 0", error="boom")
        assert ev1 != ev2

    def test_frozen_raises_on_mutation(self) -> None:
        ev = ToolCallCompleted("bash", "exit 0")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.output = "exit 1"  # type: ignore[misc]

    def test_isinstance_agent_event(self) -> None:
        ev = ToolCallCompleted("bash", "exit 0")
        assert isinstance(ev, (AssistantDelta, ToolCallStarted, ToolCallCompleted, Final))


class TestFinal:
    def test_construct_positional(self) -> None:
        ev = Final("Answer is 42")
        assert ev.message == "Answer is 42"

    def test_construct_keyword(self) -> None:
        ev = Final(message="Answer is 42")
        assert ev.message == "Answer is 42"

    def test_equality(self) -> None:
        ev1 = Final("Answer is 42")
        ev2 = Final("Answer is 42")
        assert ev1 == ev2

    def test_inequality(self) -> None:
        ev1 = Final("Answer is 42")
        ev2 = Final("Answer is 43")
        assert ev1 != ev2

    def test_frozen_raises_on_mutation(self) -> None:
        ev = Final("Answer is 42")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.message = "Answer is 43"  # type: ignore[misc]

    def test_isinstance_agent_event(self) -> None:
        ev = Final("Answer is 42")
        assert isinstance(ev, (AssistantDelta, ToolCallStarted, ToolCallCompleted, Final))


class TestAgentEventUnion:
    def test_agent_event_is_union_of_all_types(self) -> None:
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
