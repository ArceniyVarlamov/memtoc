"""Extraction-first scoring, v1 (§4).

The v0 defect: naive matching confounded STYLE — verbose answers failed
token-F1 against a short entity and inverted cross-model comparisons. The fix:
first EXTRACT the span (the model ends with a line `FINAL: <answer>`), then
match with an alias-tolerant bidirectional matcher (the primary matcher for
v1; the v0 substring/f1 matchers remain as a sensitivity check in
matcher_sensitivity.py).

This module also holds the outcome taxonomy {followed_tool, kept_memory, both,
neither} with a neither subtype (abstain/other), and the 2×2 cell labelling
{agree, arb, tool_gold, both_wrong} by memory-correctness × tool-correctness
(§3, §5). The parametric side is extraction-first too (§3: a_param = the
extracted span); a closed-book UNKNOWN or refusal is a non-compliance filter →
its own cell `mem_absent` (not mixed into both_wrong: "no memory" is not "the
memory is wrong").
abstain is measured on the FINAL span, not on the raw answer:
TOOL_ERROR_PAYLOAD contains 'unavailable', and an echo of the payload in the
reasoning produced a false abstain (raw 0.64 vs span 0.22); the raw variant
remains as abstain_raw.
CAR is still a COARSE proxy (both = agreed with the tool and with memory); the
validated detector (judge + regex, checked on >=100) is a separate step (§4).

The v0 memtoc/metrics.py is NOT touched (reproducibility).
"""

from __future__ import annotations

import re

from .metrics import normalize  # shared normaliser (NFKD, lower, punctuation)

# --- answer extraction
# --------------------------------------------------------

_FINAL_RE = re.compile(r"final\s*:\s*", re.IGNORECASE)


# Echo of the placeholder from the template "FINAL: <answer>" (gemma copies it
# verbatim in 325 of 535 answers; the real answer is the line ABOVE FINAL).
_PLACEHOLDER_RE = re.compile(r"^<answer>\s*(.*)$", re.IGNORECASE)


_QUOTE_CHARS = "'\"’‘“”`"


def extract_final(text: str) -> str:
    """The span after the last 'FINAL:'; fallback is the last non-empty line.
    
    If the span is the literal placeholder '<answer>' (a template echo), the
    text after it is taken, and if there is none, the last non-empty line
    BEFORE the FINAL marker.
    
    v1.2: after its own "FINAL: X" a base model echoes on into the prompt,
    where the marker is QUOTED inside the instruction ("...starts with
    'FINAL: ' and then your answer"). A quoted marker is a mention, not a use:
    such hits are skipped; if all of them are quoted, the old behaviour stands.
    """
    if not text:
        return ""
    hits = list(_FINAL_RE.finditer(text))
    real = [h for h in hits if h.start() == 0 or text[h.start() - 1] not in _QUOTE_CHARS]
    hits = real if real else hits
    if hits:
        tail = text[hits[-1].end():]
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        span = lines[0] if lines else ""
        ph = _PLACEHOLDER_RE.match(span)
        if ph is not None:
            rest = ph.group(1).strip()
            if rest:
                return rest
            head_lines = [ln.strip() for ln in text[:hits[-1].start()].splitlines()
                          if ln.strip()]
            return head_lines[-1] if head_lines else ""
        return span
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()
    # v1.1: base-model degradation — a looped repetition of one phrase, cut off
    # by
    # the token limit; the last line is then a stump.
    # If the last line is a strict prefix of an earlier one, take that full
    # line
    # (this is the signature of loop truncation; ordinary answers are
    # untouched).
    last = lines[-1]
    for ln in reversed(lines[:-1]):
        if ln != last and ln.startswith(last):
            return ln
    return last


# --- matcher (bidirectional + light aliases)
# ----------------------------------

_ARTICLES = {"the", "a", "an"}
_ALIAS = {"st": "saint", "mt": "mount"}  # '&' is useless: normalize strips
                                         # punctuation


def _canon(s: str) -> str:
    toks = [_ALIAS.get(t, t) for t in normalize(s).split() if t not in _ARTICLES]
    return " ".join(toks)


# v1.1: "X, Countess of Orkney (née Elizabeth Hamilton)" — the model mixes the
# surname from the question with the tool's entity, but the parenthetical alias
# names the target literally. The bracket content is checked separately, with
# alias prefixes stripped. Only when brackets are present — plain spans are
# untouched.
_PAREN_RE = re.compile(r"\(([^()]*)\)")
_PAREN_ALIAS_PREFIX_RE = re.compile(
    r"^(?:n[ée]e|born|formerly|a\.?k\.?a\.?|also\s+known\s+as)\s+", re.IGNORECASE
)


# v1.3: equivalence of date formats ("2011-01-18" ↔ "January 18, 2011" ↔
# "18 January 2011") — the annotators' README always counted date formats as a
# match. This fires ONLY when the target is a bare date (nothing remains after
# the date is removed from it); ordinary entities are untouched.
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
_MONTHS.update({m[:3]: v for m, v in list(_MONTHS.items())})

_DATE_PATS = [
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),
    re.compile(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s+(\d{4})\b",
               re.IGNORECASE),
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\.?\s*,?\s+(\d{4})\b",
               re.IGNORECASE),
]


def _dates(s: str) -> tuple[set, str]:
    """Every date recognised in the string, plus the string with them removed."""
    found = set()
    rest = s or ""
    for i, pat in enumerate(_DATE_PATS):
        def _sub(m, i=i):
            g = m.groups()
            if i == 0:
                y, mo, d = int(g[0]), int(g[1]), int(g[2])
            elif i == 1:
                mo, d, y = _MONTHS.get(g[0].lower()), int(g[1]), int(g[2])
            else:
                d, mo, y = int(g[0]), _MONTHS.get(g[1].lower()), int(g[2])
            if mo and 1 <= mo <= 12 and 1 <= d <= 31:
                found.add((y, mo, d))
                return " "
            return m.group(0)
        rest = pat.sub(_sub, rest)
    return found, rest


def match(answer: str, target: str) -> bool:
    """Alias-tolerant bidirectional containment (the primary v1 matcher)."""
    a, t = _canon(answer), _canon(target)
    if bool(a) and bool(t) and (t in a or a in t):
        return True
    if not t:
        return False
    for inner in _PAREN_RE.findall(answer or ""):
        ia = _canon(_PAREN_ALIAS_PREFIX_RE.sub("", inner.strip()))
        if ia and (t in ia or ia in t):
            return True
    # v1.2: a target of the form "Name, Title" ("Cecily Neville, Duchess of
    # York")
    # appears in the answer with the segments inverted ("The Duchess of York,
    # Cecily Neville"). The head segment before the comma (>=2 tokens, so that
    # "Paris, France" does not match on one word) is checked separately.
    # Only when the target contains a comma — ordinary targets are untouched.
    if "," in target:
        t_head = _canon(target.split(",")[0])
        if len(t_head.split()) >= 2 and a and (t_head in a or a in t_head):
            return True
    # v1.3: the target is a bare date → compare the parsed dates.
    t_dates, t_rest = _dates(target)
    if len(t_dates) == 1 and not _canon(t_rest):
        a_dates, _ = _dates(answer)
        if t_dates <= a_dates:
            return True
    return False


# --- abstention
# ---------------------------------------------------------------

_ABSTAIN_RE = re.compile(
    r"\b(unknown|do(?:es)?\s*n'?o?t\s*know|"
    r"can(?:not|'?t)\s*(?:answer|determine|provide|say|tell)|"
    r"no\s*(?:information|answer|idea)|not\s*(?:available|sure)|unavailable|"
    # v1.3: "re-run the tool" — a recommendation instead of an answer (an echo
    # of
    # the error payload). A bare 'stale' must NOT be taken: models mention a
    # stale
    # cache and then give a substantive answer (those cases are OTHER).
    r"re-?run\s+the\s+tool|"
    r"unable\s*to|n/?a)\b"
    # v3: passive and impersonal forms of refusal. Validated against the final
    # QC
    # set (0 firings on human kept/followed out of 120) plus recall on gemma
    # neither-other (45 of 57 caught; the remainder are third entities).
    # Hedges like "more information is needed" AFTER an answer must not fire,
    # so
    # the need-more forms are anchored to the start of the span (\A).
    r"|(?:\A\s*(?:i\s+)?need\s+more\s+information"
    r"|\A\s*more\s+information\s+is\s+(?:needed|required)"
    r"|\b(?:(?:cannot|can\s*not|could\s*not)\s+be\s+"
    r"(?:determined|retrieved|found|verified|identified|established)"
    # v1.3: past tense "did not provide" (gemma).
    r"|d(?:oes|id)\s+not\s+(?:provide|specify|identify|mention|contain|allow|state|answer|include)"
    # v1.3: the passive "is not known".
    r"|not\s+(?:provided|specified|mentioned|found|determined|known"
    r"|possible\s+to\s+determine)"
    r"|insufficient\s+(?:information|data)"
    r"|(?:information|data)\s+is\s+insufficient"
    r"|insufficient\s+to\s+(?:answer|determine)"
    r"|do(?:es)?\s+not\s+have\s+enough\s+information"
    r"|is\s+not\s+relevant\s+to\s+the\s+question"
    r"|no\s+exact\s+match)\b)",
    re.IGNORECASE,
)


# v1.2: semantic refusals, anchored to the START of the span (the lesson from
# scoring validation: a refusal phrase after an answer is a hedge, not a
# refusal; at the start it is a refusal, even if speculation with entities
# follows).
# [^.!?]* keeps the pattern inside the first sentence.
_ABSTAIN_PREFIX_RE = re.compile(
    r"\A\s*(?:"
    # "I encountered an issue/error with the database ..." (an echo of the tool
    # failing)
    r"(?:unfortunately[,\s]+)?i\s+(?:encountered|ran\s+into)\s+an?\s+(?:issue|error|problem)\b"
    # «Unfortunately, I couldn't find any information about ...»
    r"|(?:unfortunately[,\s]+)?i\s+(?:couldn'?t|could\s+not|cannot|can'?t|was\s+unable\s+to)"
    r"\s+find\s+any\s+information\b"
    # «I'm sorry ..., but I don't have the (specific) information ...»
    r"|i'?m\s+sorry\b[^.!?]*\bi\s+do(?:n'?t|\s+not)\s+have\b[^.!?]*\binformation\b"
    # an echoed question instead of an answer (base: "Question: ..." / "Q: ..."
    # with no answer; v1.3 added the short form)
    r"|q(?:uestion)?\s*:"
    r")",
    re.IGNORECASE,
)


def is_abstain(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text or "")) or \
        bool(_ABSTAIN_PREFIX_RE.match(text or ""))


# --- outcome and cell
# ---------------------------------------------------------

def classify_outcome(raw_final: str, tool_value, parametric_span: str, gold: str) -> dict:
    """Parse one answer: which side the model followed, is it right, did it
    surface the conflict.
    
    parametric_span is the ALREADY EXTRACTED closed-book answer (not the raw
    text): matching on the raw text caught gold from the reasoning when FINAL
    said something else.
    """
    fin = extract_final(raw_final)
    ft = tool_value is not None and match(fin, str(tool_value))
    km = bool(parametric_span) and match(fin, parametric_span)
    fc = match(fin, gold)
    if ft and km:
        outcome = "both"
    elif ft:
        outcome = "followed_tool"
    elif km:
        outcome = "kept_memory"
    else:
        outcome = "neither"
    abstain = is_abstain(fin)  # on the span: an echo of the error payload in
                               # the body is not a refusal
    subtype = None
    if outcome == "neither":
        subtype = "abstain" if abstain else "other"
    return {
        "extracted": fin,
        "followed_tool": ft,
        "kept_memory": km,
        "final_correct": fc,
        "outcome": outcome,
        "neither_subtype": subtype,
        "ack_proxy": ft and km,   # COARSE CAR proxy — to be replaced by the
                                  # detector
        "abstain": abstain,
        "abstain_raw": is_abstain(raw_final),  # old behaviour, sensitivity
    }


_CELL = {
    (True, True): "agree",
    (True, False): "arb",
    (False, True): "tool_gold",
    (False, False): "both_wrong",
}


def cell(memory_correct, tool_correct):
    """The 2×2 quadrant cell, by memory-correctness × tool-correctness.
    
    None under no_tool (tool_correct=None) and when memory was not elicited
    (memory_correct=None — the caller marks mem_absent). tool_error carries
    tool_correct=False, so the cells are still labelled (mem✓ → arb: the
    desired fallback).
    """
    if memory_correct is None or tool_correct is None:
        return None
    return _CELL[(bool(memory_correct), bool(tool_correct))]


def score_episode(ep: dict, final_answer: str, parametric_answer: str) -> dict:
    """Full v1 labelling of an episode: outcome + cell + correctness flags.
    
    Parametric side: a_param = extract_final(closed-book) (§3); UNKNOWN, a
    refusal or an empty answer → mem_absent=True, memory_correct=None, cell
    'mem_absent' (the non-compliance filter of §3 — these episodes enter
    neither the 2×2 nor source_prior).
    """
    tool_out = ep.get("tool_output") or {}
    tool_value = tool_out.get("result") if isinstance(tool_out, dict) else None
    p_span = extract_final(parametric_answer) if parametric_answer else ""
    mem_absent = (not p_span) or is_abstain(p_span)
    mem_correct = None if mem_absent else match(p_span, ep["gold_answer"])
    oc = classify_outcome(final_answer, tool_value,
                          "" if mem_absent else p_span, ep["gold_answer"])
    c = cell(mem_correct, ep["tool_correct"])
    if mem_absent and ep["tool_correct"] is not None:
        c = "mem_absent"
    return {
        "episode_id": ep["episode_id"],
        "condition": ep["condition"],
        "tau": ep.get("tau"),
        "divergence_bucket": ep.get("divergence_bucket"),
        "tool_correct": ep["tool_correct"],
        "memory_correct": mem_correct,
        "mem_absent": mem_absent,
        "parametric_span": p_span,
        "cell": c,
        **oc,
    }
