"""on_chunk streaming: text deltas flow provider -> loop -> callback."""

from __future__ import annotations

from types import SimpleNamespace

from conftest import echo_tool

from harness.core.loop import Agent, AgentConfig
from harness.providers.claude import ClaudeProvider, _text_delta


def _delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
    )


def _other_event() -> SimpleNamespace:
    return SimpleNamespace(type="message_start", delta=None)


class _FakeStream:
    def __init__(self, events, final):
        self._events = events
        self._final = final
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final

    def close(self):
        self.closed = True


class _FakeMessages:
    def __init__(self, events, final):
        self._events = events
        self._final = final
        self.last_kwargs = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._events, self._final)


class _FakeClient:
    def __init__(self, events, final):
        self.messages = _FakeMessages(events, final)


def _final_message(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=5, output_tokens=3,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )


def test_text_delta_extractor():
    assert _text_delta(_delta("hi")) == "hi"
    assert _text_delta(_other_event()) is None


def test_provider_forwards_deltas_to_on_chunk():
    events = [_other_event(), _delta("Hel"), _delta("lo"), _delta(" world")]
    client = _FakeClient(events, _final_message("Hello world"))
    prov = ClaudeProvider(model="claude-opus-4-8", api_key="x", client=client)

    chunks: list[str] = []
    resp = prov.stream("sys", [{"role": "user", "content": "hi"}], [], on_chunk=chunks.append)
    assert "".join(chunks) == "Hello world"
    assert resp.stop_reason == "end_turn"


def test_loop_threads_on_chunk_through_to_provider():
    events = [_delta("part1"), _delta("part2")]
    client = _FakeClient(events, _final_message("part1part2"))
    prov = ClaudeProvider(model="claude-opus-4-8", api_key="x", client=client)

    chunks: list[str] = []
    msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    Agent(AgentConfig(provider=prov, tools=[echo_tool()], on_chunk=chunks.append)).run(msgs)
    assert "".join(chunks) == "part1part2"


def test_fake_provider_without_on_chunk_still_runs():
    # conftest.FakeProvider.stream has no on_chunk param; loop must not pass it.
    from conftest import FakeProvider, assistant_text

    prov = FakeProvider([assistant_text("ok")])
    msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    # on_chunk set, but provider can't take it -> loop omits it, no crash
    final = Agent(AgentConfig(provider=prov, tools=[], on_chunk=lambda s: None)).run(msgs)
    assert final.stop_reason == "end_turn"
