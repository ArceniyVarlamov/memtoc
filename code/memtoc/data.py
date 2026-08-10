"""Loading ToolHop and selecting conflict-eligible hops.

The ToolHop instance schema is documented upstream. What matters here:
sub_task = dict {sub-question -> gold sub-answer}, one per hop;
functions[i] is the executable Python of hop i's tool; tools[i] its JSON schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


DATE_RE = re.compile(
    r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\s*$|"
    r"\b(january|february|march|april|may|june|july|august|september|october"
    r"|november|december)\b",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"^\s*-?[\d,.]+\s*$")


@dataclass
class Hop:
    """One hop of an instance: sub-question, gold answer and its tool."""

    instance_id: int
    hop_idx: int
    question: str
    gold_answer: str
    tool_schema: dict
    function_code: str
    qc_flags: list[str] = field(default_factory=list)

    @property
    def answer_kind(self) -> str:
        a = self.gold_answer.strip()
        if NUMERIC_RE.match(a):
            return "number"
        if DATE_RE.search(a):
            return "date"
        return "entity"


def load_toolhop(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and data, "unexpected ToolHop.json layout"
    return data


def iter_hops(instance: dict) -> list[Hop]:
    """Expand an instance into an ordered list of hops.
    
    In ToolHop `sub_task` and `tools` are dicts keyed by the sub-question
    TEXT (checked on all 995 instances — the keys agree); `functions` is a
    list in hop order. The tool is joined on the question key and the
    function by index; a missing key is marked with a QC flag.
    """
    subs = list(instance["sub_task"].items())
    tools = instance["tools"]
    funcs = instance["functions"]
    hops = []
    for i, (q, a) in enumerate(subs):
        h = Hop(
            instance_id=instance["id"],
            hop_idx=i,
            question=q,
            gold_answer=str(a),
            tool_schema=tools.get(q, {}),
            function_code=funcs[i] if i < len(funcs) else "",
        )
        if q not in tools:
            h.qc_flags.append("tool_schema_missing_for_question")
        if i >= len(funcs):
            h.qc_flags.append("function_missing_for_hop")
        hops.append(h)
    return hops


def conflict_eligible_hops(instance: dict) -> list[Hop]:
    """Hops where injecting a parametric conflict is meaningful.
    
    Heuristic: intermediate (non-final) hops with an entity answer —
    parametric knowledge lives on entities, not on the arithmetic of the
    tail.
    """
    hops = iter_hops(instance)
    return [h for h in hops[:-1] if h.answer_kind == "entity"]


def entity_distractor_pool(data: list[dict]) -> list[str]:
    """Pool of entity answers across the dataset, for tool_wrong substitutions."""
    pool = set()
    for inst in data:
        for h in iter_hops(inst):
            if h.answer_kind == "entity":
                pool.add(h.gold_answer)
    return sorted(pool)
