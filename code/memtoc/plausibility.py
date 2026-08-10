"""Deterministic construction helpers for the MemToC plausibility-clean canon."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SEED = 20260714
TOKEN_RE = re.compile(r"[a-z0-9]+")
GENERIC = {
    "a", "an", "and", "advanced", "analysis", "analyze", "analyzing",
    "based", "comprehensive", "data", "designed", "detailed", "enhanced",
    "find", "finder", "for", "from", "get", "identify", "identifier",
    "information", "input", "lookup", "name", "of", "option", "options",
    "output", "provide", "provides", "query", "record", "records", "search",
    "specified", "string", "the", "to", "tool", "using", "value", "various",
    "with",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens(text: str, *, drop_generic: bool = True) -> frozenset[str]:
    values = set(TOKEN_RE.findall(text.lower().replace("_", " ")))
    return frozenset(values - GENERIC if drop_generic else values)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def required_tokens(schema: dict[str, Any]) -> frozenset[str]:
    required = schema.get("parameters", {}).get("required", []) or []
    return frozenset(
        token
        for name in required
        for token in tokens(str(name), drop_generic=False)
    )


def output_shape(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:%| percent)?", text, re.I):
        return "number"
    if re.fullmatch(r"(?:yes|no|true|false)", text, re.I):
        return "boolean"
    if re.fullmatch(r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text):
        return "date"
    return "text"


def normalize_tool_name(name: str) -> str:
    return "_".join(TOKEN_RE.findall(name.lower().replace("-", "_")))


def family_features(schema: dict[str, Any]) -> dict[str, frozenset[str]]:
    return {
        "name": tokens(schema.get("name", "")),
        "required": required_tokens(schema),
        "description": tokens(schema.get("description", "")),
    }


def family_score(target: dict[str, Any], source: dict[str, Any]) -> dict[str, float]:
    left, right = family_features(target), family_features(source)
    parts = {
        "name_jaccard": jaccard(left["name"], right["name"]),
        "required_jaccard": jaccard(left["required"], right["required"]),
        "description_jaccard": jaccard(left["description"], right["description"]),
    }
    parts["score"] = (
        0.50 * parts["name_jaccard"]
        + 0.25 * parts["required_jaccard"]
        + 0.25 * parts["description_jaccard"]
    )
    return parts


def equivalent_eligible(parts: dict[str, float]) -> bool:
    return (
        parts["name_jaccard"] > 0
        or parts["description_jaccard"] >= 0.05
        or parts["required_jaccard"] >= 0.50
    )


@dataclass(frozen=True)
class SourceHop:
    instance_id: int
    hop_idx: int
    question: str
    answer: str
    schema: dict[str, Any]
    function_code: str = ""

    @property
    def key(self) -> tuple[int, int]:
        return self.instance_id, self.hop_idx


def load_toolhop(path: Path) -> list[SourceHop]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[SourceHop] = []
    for instance in raw:
        subtasks = list(instance["sub_task"].items())
        schemas = list(instance["tools"].items())
        if len(subtasks) != len(schemas):
            raise ValueError(f"ToolHop instance {instance['id']} has misaligned hops")
        for hop_idx, ((question, answer), (_, schema)) in enumerate(zip(subtasks, schemas)):
            answer_text = str(answer).strip()
            if not answer_text:
                continue
            rows.append(
                SourceHop(
                    instance_id=int(instance["id"]),
                    hop_idx=hop_idx,
                    question=question,
                    answer=answer_text,
                    schema=schema,
                    function_code=str(instance.get("functions", [""])[hop_idx]),
                )
            )
    return rows


class _StaticUnknown:
    """Sentinel used by the conservative function-code audit below."""


STATIC_UNKNOWN = _StaticUnknown()


def _static_eval(node: ast.AST, env: dict[str, Any]) -> Any:
    """Evaluate the small, literal subset used by ToolHop mock functions.

    ToolHop functions are not executable production tools; they are generated
    Python snippets containing small dictionaries/lists and return statements.
    This evaluator deliberately refuses arbitrary calls and expressions.  It
    is therefore safe for an audit and, when it cannot prove a value, returns
    ``STATIC_UNKNOWN`` instead of guessing.
    """

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, STATIC_UNKNOWN)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_static_eval(item, env) for item in node.elts]
        cls = list if isinstance(node, ast.List) else tuple if isinstance(node, ast.Tuple) else set
        if any(value is STATIC_UNKNOWN for value in values):
            return STATIC_UNKNOWN
        return cls(values)
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:  # **mapping expansion is intentionally unknown.
                return STATIC_UNKNOWN
            key = _static_eval(key_node, env)
            value = _static_eval(value_node, env)
            if key is STATIC_UNKNOWN:
                continue
            result[key] = value
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _static_eval(node.operand, env)
        if isinstance(value, (int, float)):
            return -value if isinstance(node.op, ast.USub) else value
        return STATIC_UNKNOWN
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_eval(node.left, env), _static_eval(node.right, env)
        if isinstance(left, (str, int, float)) and isinstance(right, type(left)):
            return left + right
        return STATIC_UNKNOWN
    if isinstance(node, ast.Subscript):
        container = _static_eval(node.value, env)
        key = _static_eval(node.slice, env)
        if isinstance(container, dict):
            if key is STATIC_UNKNOWN:
                values = list(container.values())
                # A single-record mock database is common.  Preserve its
                # record shape so a later ``record[output_format]`` remains
                # statically evaluable; for genuinely ambiguous databases we
                # retain the conservative list-of-possibilities behavior.
                return values[0] if len(values) == 1 else values
            return container.get(key, STATIC_UNKNOWN)
        if isinstance(container, (list, tuple)):
            if isinstance(key, int) and -len(container) <= key < len(container):
                return container[key]
            return list(container) if key is STATIC_UNKNOWN else STATIC_UNKNOWN
        return STATIC_UNKNOWN
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        receiver = _static_eval(node.func.value, env)
        if node.func.attr == "get" and isinstance(receiver, dict):
            if not node.args:
                return STATIC_UNKNOWN
            key = _static_eval(node.args[0], env)
            if key is STATIC_UNKNOWN:
                return list(receiver.values())
            if key in receiver:
                return receiver[key]
            return _static_eval(node.args[1], env) if len(node.args) > 1 else None
        return STATIC_UNKNOWN
    if isinstance(node, ast.IfExp):
        condition = _static_eval(node.test, env)
        if condition is True:
            return _static_eval(node.body, env)
        if condition is False:
            return _static_eval(node.orelse, env)
        return STATIC_UNKNOWN
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = _static_eval(node.left, env), _static_eval(node.comparators[0], env)
        if left is STATIC_UNKNOWN or right is STATIC_UNKNOWN:
            return STATIC_UNKNOWN
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.In):
            try:
                return left in right
            except TypeError:
                return STATIC_UNKNOWN
        if isinstance(op, ast.NotIn):
            try:
                return left not in right
            except TypeError:
                return STATIC_UNKNOWN
        return STATIC_UNKNOWN
    if isinstance(node, ast.BoolOp):
        values = [_static_eval(value, env) for value in node.values]
        if isinstance(node.op, ast.And):
            if any(value is False for value in values):
                return False
            return True if all(value is True for value in values) else STATIC_UNKNOWN
        if isinstance(node.op, ast.Or):
            if any(value is True for value in values):
                return True
            return False if all(value is False for value in values) else STATIC_UNKNOWN
    if isinstance(node, ast.ListComp):
        # List comprehensions over literal lists are common in ToolHop mocks.
        if len(node.generators) != 1:
            return STATIC_UNKNOWN
        generator = node.generators[0]
        source = _static_eval(generator.iter, env)
        if not isinstance(source, (list, tuple)):
            return STATIC_UNKNOWN
        result: list[Any] = []
        for item in source:
            local = dict(env)
            if isinstance(generator.target, ast.Name):
                local[generator.target.id] = item
            else:
                return STATIC_UNKNOWN
            if generator.ifs:
                checks = [_static_eval(condition, local) for condition in generator.ifs]
                if any(check is False for check in checks):
                    continue
                if any(check is STATIC_UNKNOWN for check in checks):
                    continue
            result.append(_static_eval(node.elt, local))
        return result
    return STATIC_UNKNOWN


def _flatten_static(value: Any) -> set[str]:
    if value is STATIC_UNKNOWN or value is None:
        return set()
    if isinstance(value, (str, int, float, bool)):
        return {str(value).strip()}
    if isinstance(value, dict):
        result: set[str] = set()
        for item in value.values():
            result.update(_flatten_static(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for item in value:
            result.update(_flatten_static(item))
        return result
    return set()


def _static_run_block(statements: list[ast.stmt], env: dict[str, Any], returns: list[Any]) -> None:
    for statement in statements:
        if isinstance(statement, ast.Return):
            returns.append(_static_eval(statement.value, env) if statement.value else None)
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value_node = statement.value
            if value_node is None:
                continue
            value = _static_eval(value_node, env)
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    env[target.id] = value
            continue
        if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            current = env.get(statement.target.id, STATIC_UNKNOWN)
            value = _static_eval(statement.value, env)
            if isinstance(statement.op, ast.Add) and isinstance(current, list) and value is not STATIC_UNKNOWN:
                env[statement.target.id] = current + [value]
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "append" and isinstance(call.func.value, ast.Name):
                receiver = env.get(call.func.value.id)
                if isinstance(receiver, list) and call.args:
                    receiver.append(_static_eval(call.args[0], env))
            continue
        if isinstance(statement, ast.For):
            source = _static_eval(statement.iter, env)
            if isinstance(source, dict):
                source = list(source.keys())
            if not isinstance(source, (list, tuple, set)):
                continue
            if not isinstance(statement.target, ast.Name):
                continue
            for item in source:
                env[statement.target.id] = item
                _static_run_block(statement.body, env, returns)
            _static_run_block(statement.orelse, env, returns)
            continue
        if isinstance(statement, ast.If):
            condition = _static_eval(statement.test, env)
            if condition is True:
                _static_run_block(statement.body, env, returns)
            elif condition is False:
                _static_run_block(statement.orelse, env, returns)
            else:
                # Unknown input conditions are explored both ways.  Use a
                # copy for the alternate branch so the primary environment
                # remains usable for later statements.
                _static_run_block(statement.body, env, returns)
                alternate = dict(env)
                _static_run_block(statement.orelse, alternate, returns)
            continue
        if isinstance(statement, (ast.With, ast.Try)):
            body = statement.body
            _static_run_block(body, env, returns)
            continue
        # Nested helper functions are intentionally not executed.  If a
        # return depends on one, the audit will be UNKNOWN rather than guessed.


def function_return_values(function_code: str) -> set[str]:
    """Return statically reachable scalar values from a ToolHop function."""

    try:
        tree = ast.parse(function_code)
    except (SyntaxError, ValueError):
        return set()
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return set()
    env: dict[str, Any] = {}
    positional = list(function.args.posonlyargs) + list(function.args.args)
    defaults = [STATIC_UNKNOWN] * (len(positional) - len(function.args.defaults)) + [
        _static_eval(default, env) for default in function.args.defaults
    ]
    env.update({argument.arg: default for argument, default in zip(positional, defaults)})
    for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
        env[argument.arg] = _static_eval(default, env) if default is not None else STATIC_UNKNOWN
    returns: list[Any] = []
    _static_run_block(function.body, env, returns)
    result: set[str] = set()
    for value in returns:
        result.update(_flatten_static(value))
    return result


def audit_function_answer(function_code: str, answer: str) -> dict[str, Any]:
    """Classify a recorded ToolHop answer against the function's returns.

    ``pass`` means the answer is statically reachable; ``fail`` means the
    function has statically reachable values but this answer is not among them;
    ``unknown`` means the generated function is too dynamic for this audit.
    """

    values = function_return_values(function_code)
    answer_text = str(answer).strip()
    if not values:
        return {"status": "unknown", "possible_values": []}
    if answer_text in values:
        return {"status": "pass", "possible_values": sorted(values)}
    return {"status": "fail", "possible_values": sorted(values)}


def ranked_candidates(target: dict[str, Any], sources: list[SourceHop]) -> list[dict[str, Any]]:
    target_schema = target["tool_schema"]
    target_name = normalize_tool_name(target_schema["name"])
    target_key = (int(target["instance_id"]), int(target["hop_idx"]))
    target_gold = str(target["gold_answer"]).strip()
    shape = output_shape(target_gold)
    candidates: list[dict[str, Any]] = []

    for source in sources:
        if source.key == target_key or source.answer == target_gold:
            continue
        if output_shape(source.answer) != shape:
            continue
        source_name = normalize_tool_name(source.schema["name"])
        exact = source_name == target_name
        parts = family_score(target_schema, source.schema)
        if not exact and not equivalent_eligible(parts):
            continue
        tier = "same_tool" if exact else "equivalent_family"
        candidates.append(
            {
                "proposed_output": source.answer,
                "source_instance_id": source.instance_id,
                "source_hop_idx": source.hop_idx,
                "source_question": source.question,
                "source_tool_name": source.schema["name"],
                "source_tool_schema": source.schema,
                "selection_tier": tier,
                "output_shape": shape,
                "family_score": round(parts["score"], 8),
                "name_jaccard": round(parts["name_jaccard"], 8),
                "required_jaccard": round(parts["required_jaccard"], 8),
                "description_jaccard": round(parts["description_jaccard"], 8),
                "length_delta": abs(len(source.answer) - len(target_gold)),
            }
        )

    candidates.sort(
        key=lambda row: (
            0 if row["selection_tier"] == "same_tool" else 1,
            -row["family_score"],
            row["length_delta"],
            row["proposed_output"].casefold(),
            row["source_instance_id"],
            row["source_hop_idx"],
        )
    )
    return candidates


def blind_id(episode_id: str) -> str:
    value = hashlib.sha256(f"{SEED}:{episode_id}".encode()).hexdigest()[:12]
    return f"P{value.upper()}"
