from src.app.services.ai.llm import LLMResponse, ToolCall


def text(answer: str) -> LLMResponse:
    return LLMResponse(text=answer, tool_calls=[])


def call(name: str, **args) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[ToolCall(name=name, args=args)])


class FakeLLM:
    """Scripted LLM: returns queued responses in order and records every request."""

    model_name = "fake-model"

    def __init__(self, script=None):
        self.script = list(script or [])
        self.requests = []

    def queue(self, *responses):
        self.script.extend(responses)
        return self

    def generate(self, system, turns, tools=None):
        self.requests.append({"system": system, "turns": list(turns), "tools": tools})
        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
