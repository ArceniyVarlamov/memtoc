"""Provider-neutral single-agent runtime for agentic MemToC.

The runtime deliberately keeps model policy, tool execution, and tracing
separate.  MemToC tool calls execute frozen episode fixtures: no live network or
knowledge source is consulted, so the original experimental condition remains
the only source of the returned observation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .scoring import extract_final

TRACE_SCHEMA_VERSION = "agent-trace-v0"
POLICY_VERSION = "agentic-memtoc-loop-v0"
TOOL_ADAPTER_VERSION = "memtoc-fixture-v0"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_tool_schema(schema: dict) -> dict:
    """Convert an episode tool schema to the OpenAI function-tool envelope."""
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        return schema
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate_json_value(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Small deterministic validator for the JSON-Schema subset in ToolHop."""
    errors: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_matches_type(value, item) for item in expected):
            return [f"{path}: expected one of {expected}"]
    elif isinstance(expected, str) and not _matches_type(value, expected):
        return [f"{path}: expected {expected}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in enum")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: required property missing")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}.{name}: additional property not allowed")
        for name, item in value.items():
            if name in properties:
                errors.extend(validate_json_value(item, properties[name], f"{path}.{name}"))
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(validate_json_value(item, schema["items"], f"{path}[{index}]"))
    return errors


def oracle_value(schema: dict, question: str) -> Any:
    if schema.get("enum"):
        return schema["enum"][0]
    expected = schema.get("type")
    if expected == "string":
        return question
    if expected == "array":
        return []
    if expected == "boolean":
        return False
    if expected in ("integer", "number"):
        return 0
    if expected == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        return {
            name: oracle_value(properties.get(name, {}), question)
            for name in schema.get("required", [])
        }
    return None


@dataclass(frozen=True)
class ToolExecution:
    fixture_id: str
    condition: str
    status: str
    content: Any
    latency_ms: float
    runtime_retry_count: int = 0

    def as_trace(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "condition": self.condition,
            "status": self.status,
            "content": self.content,
            "content_sha256": sha256_json(self.content),
            "latency_ms": self.latency_ms,
            "runtime_retry_count": self.runtime_retry_count,
        }


class FixtureToolAdapter:
    """Execute exactly the observation frozen into one MemToC episode."""

    version = TOOL_ADAPTER_VERSION

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def validate(self, episode: dict, name: str, arguments: Any) -> list[str]:
        schema = episode["tool_schema"]
        if name != schema["name"]:
            return [f"unknown tool {name!r}; expected {schema['name']!r}"]
        if not isinstance(arguments, dict):
            return ["tool arguments must decode to a JSON object"]
        return validate_json_value(arguments, schema.get("parameters", {"type": "object"}))

    def oracle_arguments(self, episode: dict) -> dict:
        parameters = episode["tool_schema"].get("parameters", {"type": "object"})
        value = oracle_value(parameters, episode["question"])
        return value if isinstance(value, dict) else {}

    def execute(self, episode: dict, name: str, arguments: dict) -> ToolExecution:
        started = time.perf_counter()
        if not self.enabled:
            content = {"error": {"type": "tool_disabled", "message": "FC-style no-op backend"}}
            status = "recoverable_error"
        elif episode.get("condition") == "no_tool" or episode.get("tool_output") is None:
            content = {"error": {"type": "tool_unavailable", "message": "No tool in this arm"}}
            status = "recoverable_error"
        else:
            content = episode["tool_output"]
            status = "recoverable_error" if isinstance(content, dict) and "error" in content else "ok"
        return ToolExecution(
            fixture_id=episode["episode_id"],
            condition=episode["condition"],
            status=status,
            content=content,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


@dataclass
class ModelResponse:
    message: dict
    finish_reason: str | None = None
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class AgentModel(Protocol):
    name: str
    provider: str

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | dict | None,
    ) -> ModelResponse: ...


class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client with native function calling."""

    provider = "openai-compatible"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        max_tokens: int = 384,
        seed: int = 0,
    ):
        self.model = model
        self.name = model
        self.base_url = (base_url or os.environ.get("KCB_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("KCB_API_KEY", "EMPTY")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.seed = seed

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | dict | None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        started = time.perf_counter()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
            request_id = response.headers.get("x-request-id") or body.get("id")
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        message = dict(choice.get("message") or {})
        message.setdefault("role", "assistant")
        return ModelResponse(
            message=message,
            finish_reason=choice.get("finish_reason"),
            request_id=request_id,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


class ScriptedChatClient:
    """Deterministic model double used by unit/smoke tests."""

    provider = "scripted"

    def __init__(self, responses: list[dict], name: str = "scripted"):
        self.responses = list(responses)
        self.name = name
        self.requests: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict] | None, tool_choice: Any) -> ModelResponse:
        self.requests.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        if not self.responses:
            raise RuntimeError("scripted response queue exhausted")
        message = self.responses.pop(0)
        return ModelResponse(message=message, finish_reason="stop", input_tokens=1, output_tokens=1)


class FixtureSmokeClient:
    """Deterministic end-to-end smoke client; never use it for model metrics."""

    provider = "scripted-smoke"
    name = "scripted-smoke"

    def chat(self, messages: list[dict], tools: list[dict] | None, tool_choice: Any) -> ModelResponse:
        if messages and messages[-1].get("role") == "tool":
            observation = json.loads(messages[-1].get("content") or "{}")
            answer = observation.get("result", "UNKNOWN") if isinstance(observation, dict) else "UNKNOWN"
            return ModelResponse(
                message={"role": "assistant", "content": f"FINAL: {answer}"},
                finish_reason="stop",
                input_tokens=1,
                output_tokens=1,
            )
        if tools and tool_choice != "none":
            function = tools[0]["function"]
            question = next((str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "user"), "")
            arguments = oracle_value(function.get("parameters", {}), question)
            call = _tool_call(function["name"], arguments if isinstance(arguments, dict) else {}, "smoke-call")
            return ModelResponse(
                message={"role": "assistant", "content": None, "tool_calls": [call]},
                finish_reason="tool_calls",
                input_tokens=1,
                output_tokens=1,
            )
        return ModelResponse(
            message={"role": "assistant", "content": "FINAL: UNKNOWN"},
            finish_reason="stop",
            input_tokens=1,
            output_tokens=1,
        )


class TraceWriter:
    def __init__(self, path: str | Path, experiment_id: str, run_id: str, provenance: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.provenance = dict(provenance)

    def session(self, episode_id: str, trace_id: str | None = None) -> "TraceSession":
        return TraceSession(self, episode_id, trace_id or uuid.uuid4().hex)

    def _append(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


@dataclass
class TraceSession:
    writer: TraceWriter
    episode_id: str
    trace_id: str
    step_id: int = 0

    def emit(self, event_type: str, **fields: Any) -> dict:
        row = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "experiment_id": self.writer.experiment_id,
            "run_id": self.writer.run_id,
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "parent_step_id": None if self.step_id == 0 else self.step_id - 1,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "provenance": self.writer.provenance,
            "content_capture": "full",
            **fields,
        }
        self.writer._append(row)
        self.step_id += 1
        return row


@dataclass(frozen=True)
class AgentPolicy:
    arm: str
    max_tool_calls: int
    max_output_tokens_total: int
    recover_validation_errors: bool = False

    @classmethod
    def for_arm(cls, arm: str, max_calls: int = 3, max_tokens_total: int = 1536) -> "AgentPolicy":
        if arm not in {"s0", "s1", "f0", "a1", "a2", "a3", "o1"}:
            raise ValueError(f"unknown arm: {arm}")
        default_calls = {"s0": 0, "s1": 0, "f0": 1, "a1": 1, "a2": 1, "a3": max_calls, "o1": 1}
        return cls(
            arm=arm,
            max_tool_calls=default_calls[arm],
            max_output_tokens_total=max_tokens_total,
            recover_validation_errors=arm == "a3",
        )


@dataclass
class EpisodeResult:
    episode_id: str
    arm: str
    trace_id: str
    final_answer: str
    extracted_final: str
    termination: str
    model_calls: int
    agent_tool_calls: int
    tool_executions: int
    runtime_retries: int
    input_tokens: int
    output_tokens: int
    wall_time_ms: float
    trace: TraceSession = field(repr=False)

    def as_dict(self) -> dict:
        return {name: value for name, value in vars(self).items() if name != "trace"}


def _tool_call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": canonical_json(arguments)},
    }


def _parse_tool_call(call: dict) -> tuple[str, str, Any, list[str]]:
    errors: list[str] = []
    call_id = str(call.get("id") or uuid.uuid4().hex)
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments", "{}")
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            arguments = None
            errors.append(f"arguments are not valid JSON: {exc.msg}")
    else:
        arguments = raw_arguments
    if not name:
        errors.append("tool name missing")
    return call_id, name, arguments, errors


def _agent_messages(episode: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a tool-using question-answering agent. Use only tools that are provided, "
                "never invent a tool result, and decide when enough evidence is available. When you "
                "answer, end with one line beginning 'FINAL: ' followed by the answer."
            ),
        },
        {"role": "user", "content": episode["question"]},
    ]


def run_episode(
    episode: dict,
    model: AgentModel,
    policy: AgentPolicy,
    trace_writer: TraceWriter,
    adapter: FixtureToolAdapter | None = None,
) -> EpisodeResult:
    """Run one observable agent trajectory and append every state transition."""
    started = time.perf_counter()
    adapter = adapter or FixtureToolAdapter(enabled=policy.arm != "f0")
    trace = trace_writer.session(str(episode["episode_id"]))
    model_calls = agent_tool_calls = tool_executions = runtime_retries = 0
    input_tokens = output_tokens = 0
    final_answer = ""
    termination = ""

    tool_available = episode.get("condition") != "no_tool" and episode.get("tool_schema") is not None
    tool = normalize_tool_schema(episode["tool_schema"]) if tool_available else None
    tools = [tool] if tool else None

    if policy.arm == "s0":
        messages = [{"role": "user", "content": episode["prompts"]["closed_book"]}]
        tools = None
    elif policy.arm == "s1":
        prompt = episode["prompts"].get("with_tool") or episode["prompts"]["closed_book"]
        messages = [{"role": "user", "content": prompt}]
        tools = None
    else:
        messages = _agent_messages(episode)

    trace.emit(
        "agent_start",
        messages=messages,
        messages_sha256=sha256_json(messages),
        tool_definitions=tools or [],
        policy={
            "arm": policy.arm,
            "version": POLICY_VERSION,
            "max_tool_calls": policy.max_tool_calls,
            "max_output_tokens_total": policy.max_output_tokens_total,
        },
    )

    if policy.arm == "o1" and tool:
        arguments = adapter.oracle_arguments(episode)
        call = _tool_call(episode["tool_schema"]["name"], arguments, f"oracle-{trace.trace_id[:12]}")
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        agent_tool_calls += 1
        validation_errors = adapter.validate(episode, call["function"]["name"], arguments)
        trace.emit("tool_validation", tool_call={"id": call["id"], "name": call["function"]["name"], "arguments": arguments},
                   validation={"valid": not validation_errors, "errors": validation_errors, "source": "oracle"})
        if validation_errors:
            termination = "invalid_call"
        else:
            execution = adapter.execute(episode, call["function"]["name"], arguments)
            tool_executions += 1
            trace.emit("tool_execution", tool_call={"id": call["id"], "name": call["function"]["name"], "arguments": arguments},
                       tool_result=execution.as_trace())
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                             "content": canonical_json(execution.content)})

    while True:
        if policy.arm == "o1" and termination == "invalid_call":
            break
        if output_tokens >= policy.max_output_tokens_total:
            termination = "max_tokens"
            break

        if not tools or policy.arm in {"s0", "s1"}:
            tool_choice: str | dict | None = None
        elif policy.arm == "a1" and agent_tool_calls == 0:
            tool_choice = "required"
        elif policy.arm in {"a1", "a2", "o1"} and agent_tool_calls > 0:
            tool_choice = "none"
        else:
            tool_choice = "auto"

        accounting = {
            "model_calls": model_calls,
            "agent_tool_calls": agent_tool_calls,
            "runtime_retries": runtime_retries,
            "cumulative_input_tokens": input_tokens,
            "cumulative_output_tokens": output_tokens,
            "wall_time_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        trace.emit(
            "model_request",
            model={"provider": model.provider, "name": model.name},
            messages=messages,
            messages_sha256=sha256_json(messages),
            tool_definitions=tools or [],
            tool_choice=tool_choice,
            accounting=accounting,
        )
        response = model.chat(messages, tools, tool_choice)
        model_calls += 1
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        assistant = dict(response.message)
        assistant.setdefault("role", "assistant")
        trace.emit(
            "model_response",
            model={
                "provider": model.provider,
                "name": model.name,
                "request_id": response.request_id,
                "finish_reason": response.finish_reason,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
            },
            assistant_message=assistant,
        )
        messages.append(assistant)
        calls = assistant.get("tool_calls") or []

        if not calls:
            if policy.arm == "a1" and tool and agent_tool_calls == 0:
                termination = "invalid_call"
            else:
                final_answer = str(assistant.get("content") or "")
                termination = "final"
            break
        if len(calls) != 1 or not tools:
            termination = "invalid_call"
            break
        if agent_tool_calls >= policy.max_tool_calls:
            termination = "max_calls"
            break

        call_id, name, arguments, parse_errors = _parse_tool_call(calls[0])
        agent_tool_calls += 1
        validation_errors = parse_errors + adapter.validate(episode, name, arguments)
        trace_call = {"id": call_id, "name": name, "arguments": arguments}
        trace.emit("tool_validation", tool_call=trace_call,
                   validation={"valid": not validation_errors, "errors": validation_errors, "source": "agent"})
        if validation_errors:
            if policy.recover_validation_errors and agent_tool_calls < policy.max_tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name or episode["tool_schema"]["name"],
                    "content": canonical_json({"error": {"type": "validation_error", "details": validation_errors}}),
                })
                continue
            termination = "invalid_call"
            break

        execution = adapter.execute(episode, name, arguments)
        tool_executions += 1
        trace.emit("tool_execution", tool_call=trace_call, tool_result=execution.as_trace())
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": canonical_json(execution.content),
        })
        if execution.status == "fatal_error":
            termination = "tool_fatal"
            break

    extracted = extract_final(final_answer)
    wall_time_ms = round((time.perf_counter() - started) * 1000, 3)
    trace.emit(
        "termination",
        termination=termination,
        final_answer=final_answer,
        extracted_final=extracted,
        accounting={
            "model_calls": model_calls,
            "agent_tool_calls": agent_tool_calls,
            "tool_executions": tool_executions,
            "runtime_retries": runtime_retries,
            "cumulative_input_tokens": input_tokens,
            "cumulative_output_tokens": output_tokens,
            "wall_time_ms": wall_time_ms,
        },
    )
    return EpisodeResult(
        episode_id=str(episode["episode_id"]),
        arm=policy.arm,
        trace_id=trace.trace_id,
        final_answer=final_answer,
        extracted_final=extracted,
        termination=termination,
        model_calls=model_calls,
        agent_tool_calls=agent_tool_calls,
        tool_executions=tool_executions,
        runtime_retries=runtime_retries,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_time_ms=wall_time_ms,
        trace=trace,
    )
