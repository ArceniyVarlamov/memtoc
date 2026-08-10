"""Channel A of gold verification: structural properties of Wikidata, no LLM.

Two thirds of pool v2 are not arbitrary facts but specific Wikidata properties
(father, mother, date of birth, director, timezone). There is no need to ask a
language model for those: the value is read by property identifier, the verdict
reproduces, and the provenance — "Q133622/P569, access date" — can be cited in
the paper. An LLM with web access is left for the free-form wordings (channel
B), where there is no structural property.

Cautions that surfaced in the pilot of 2026-07-20:
  * homonymy — "Thor Heyerdahl" is both the anthropologist and his father; when
    several candidates share the label no verdict is issued and the row goes to
    people rather than being resolved silently;
  * several statements of different precision — the same date of birth exists
    to the day and to the year; we take preferred rank and the highest
    precision, and a precision below day does not count as a verdict for "date
    of birth";
  * timezones do NOT come here: property P421 points at an item "EDT"/"Eastern
    Daylight Time" while the gold is written as the IANA identifier
    "America/New_York", so comparing labels yields a false mismatch on all 19
    rows;
  * several values of a property (two directors) — matching ANY of them counts
    as a match, but the row is flagged as non-unique: that bears directly on
    the criterion "the gold is unique".

Network: the public Wikidata API, read only. Responses are cached to disk and
there is a pause between requests — we do not hammer the service. A User-Agent
is required by Wikimedia's rules.

Run:
  python -m scripts.verify_gold_wikidata --limit 30      # trial
  python -m scripts.verify_gold_wikidata                 # the whole pool
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from memtoc.inject_nonentity import parse_date

ROOT = Path(__file__).resolve().parent.parent
API = "https://www.wikidata.org/w/api.php"
UA = ("MemToC-research/1.0 (knowledge-conflict benchmark; contact via "
      "github.com/annotator_a) python-urllib")

# Question template -> Wikidata property. Order matters: narrower templates
# must come before general ones.
PROPERTY_MAP: list[tuple[str, str, str]] = [
    (r"^who is the father of ", "P22", "father"),
    (r"^who is the mother of ", "P25", "mother"),
    (r"^who is the (spouse|husband|wife) of ", "P26", "spouse"),
    (r"^who is the (child|son|daughter) of ", "P40", "child"),
    (r"^who is the sibling of ", "P3373", "sibling"),
    (r"^what is the date of birth of ", "P569", "date of birth"),
    (r"^what is the date of death of ", "P570", "date of death"),
    (r"^what is the place of birth of ", "P19", "place of birth"),
    (r"^what is the place of death of ", "P20", "place of death"),
    (r"^who is the director of (the )?(film )?", "P57", "director"),
    (r"^who is the author of ", "P50", "author"),
    (r"^who is the publisher of ", "P123", "publisher"),
    (r"^who is the composer of ", "P86", "composer"),
    (r"^who is the producer of ", "P162", "producer"),
    (r"^who is the creator of ", "P170", "creator"),
    (r"^who is the cast member of ", "P161", "cast member"),
]
_PROP_RE = [(re.compile(p, re.I), prop, name) for p, prop, name in PROPERTY_MAP]


def norm(text: str) -> str:
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower()))
    return re.sub(r"\s+", " ", s).strip()


class Api:
    """Thin client with a disk cache: a repeat run does not touch the network."""

    def __init__(self, cache_path: Path, pause: float = 0.4):
        self.cache_path = cache_path
        self.pause = pause
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.calls = 0

    def _fetch(self, url: str) -> dict:
        """GET that respects the limits: on 429/503 back off and try again."""
        delay = self.pause
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                self.calls += 1
                time.sleep(self.pause)
                return data
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 503) or attempt == 5:
                    raise
                delay = min(delay * 2, 30) or 1.0
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def get(self, params: dict) -> dict:
        key = urllib.parse.urlencode(sorted(params.items()))
        if key in self.cache:
            return self.cache[key]
        data = self._fetch(API + "?" + key)
        self.cache[key] = data
        return data

    def entity_doc(self, qid: str) -> dict:
        """The full entity through Special:EntityData — served from a CDN.
        
        One document holds every property and label, so instead of three calls per
        row (claims + labels of the targets) there is one per entity, and almost
        everything comes from the cache: subjects and targets are reused.
        """
        key = "entitydata:" + qid
        if key in self.cache:
            return self.cache[key]
        doc = self._fetch(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
        ent = doc.get("entities", {}).get(qid, {})
        slim = {
            "labels": {"en": ent.get("labels", {}).get("en", {})},
            "aliases": {"en": ent.get("aliases", {}).get("en", [])},
            "claims": ent.get("claims", {}),
        }
        self.cache[key] = slim
        return slim

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False))

    def search(self, term: str) -> list[dict]:
        d = self.get({"action": "wbsearchentities", "search": term, "language": "en",
                      "format": "json", "limit": "5"})
        return d.get("search", [])

    def entities(self, ids: list[str]) -> dict:
        return {q: self.entity_doc(q) for q in ids}

    def claims(self, qid: str, prop: str) -> list[dict]:
        return self.entity_doc(qid).get("claims", {}).get(prop, [])


def property_for(question: str) -> tuple[str, str, str] | None:
    """(property, human-readable name, subject) or None."""
    q = question.strip()
    for rx, prop, name in _PROP_RE:
        m = rx.match(q)
        if m:
            subject = q[m.end():].strip().rstrip("?").strip()
            if subject:
                return prop, name, subject
    return None


def resolve_subject(api: Api, subject: str) -> tuple[str | None, list[str]]:
    """Q-id of the subject; homonymy is a flag for a human, not a verdict."""
    hits = api.search(subject)
    if not hits:
        return None, ["subject_not_found"]
    target = norm(subject)
    exact = [h for h in hits if norm(h.get("label", "")) == target]
    if not exact:
        return hits[0]["id"], ["subject_fuzzy_match"]
    if len(exact) > 1:
        return exact[0]["id"], ["subject_ambiguous"]
    return exact[0]["id"], []


def _best_time(claims: list[dict]) -> list[tuple[str, int]]:
    out = []
    for c in claims:
        if c.get("rank") == "deprecated":
            continue
        snak = c.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        v = snak.get("datavalue", {}).get("value", {})
        if "time" in v:
            out.append((v["time"], v.get("precision", 0)))
    return sorted(out, key=lambda x: -x[1])


def check_row(api: Api, question: str, gold: str) -> dict:
    got = property_for(question)
    if not got:
        return {"channel": "llm", "verdict": "no_property", "flags": []}
    prop, name, subject = got
    qid, flags = resolve_subject(api, subject)
    if qid is None:
        return {"channel": "wikidata", "property": prop, "property_name": name,
                "subject": subject, "verdict": "subject_not_found", "flags": flags}
    claims = api.claims(qid, prop)
    if not claims:
        return {"channel": "wikidata", "property": prop, "property_name": name,
                "subject": subject, "subject_qid": qid,
                "verdict": "no_claim", "flags": flags}

    base = {"channel": "wikidata", "property": prop, "property_name": name,
            "subject": subject, "subject_qid": qid, "n_values": len(claims)}
    if prop in ("P569", "P570"):
        parsed = parse_date(gold)
        times = _best_time(claims)
        base["wikidata_values"] = [t for t, _ in times]
        if parsed is None or parsed[0] is None:
            return {**base, "verdict": "gold_unparsed", "flags": flags}
        if not times:
            return {**base, "verdict": "no_claim", "flags": flags}
        g = parsed[0]
        for t, precision in times:
            m = re.match(r"^([+-])(\d{4})-(\d{2})-(\d{2})", t)
            if not m:
                continue
            y, mo, d = int(m.group(2)), int(m.group(3)), int(m.group(4))
            if precision >= 11 and (y, mo, d) == (g.year, g.month, g.day):
                return {**base, "verdict": "match",
                        "flags": flags + (["multiple_values"] if len(times) > 1 else [])}
            if precision == 10 and (y, mo) == (g.year, g.month):
                return {**base, "verdict": "match_month_precision", "flags": flags}
            if precision == 9 and y == g.year:
                return {**base, "verdict": "match_year_precision", "flags": flags}
        return {**base, "verdict": "mismatch", "flags": flags}

    # entity-valued properties: compare the label and the aliases of the target
    # Q
    targets = []
    for c in claims:
        snak = c.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        val = snak.get("datavalue", {}).get("value", {})
        if isinstance(val, dict) and "id" in val:
            targets.append(val["id"])
        elif isinstance(val, str):
            targets.append(val)
    qids = [t for t in targets if re.fullmatch(r"Q\d+", t)]
    ents = api.entities(qids) if qids else {}
    names = []
    for q in qids:
        e = ents.get(q, {})
        lbl = e.get("labels", {}).get("en", {}).get("value")
        if lbl:
            names.append(lbl)
        names += [a["value"] for a in e.get("aliases", {}).get("en", [])]
    names += [t for t in targets if not re.fullmatch(r"Q\d+", t)]
    base["wikidata_values"] = sorted(set(names))
    g = norm(gold)
    if any(norm(n) == g for n in names):
        return {**base, "verdict": "match",
                "flags": flags + (["multiple_values"] if len(qids) > 1 else [])}
    if any(g and (g in norm(n) or norm(n) in g) for n in names):
        return {**base, "verdict": "match_partial", "flags": flags}
    close = [n for n in names
             if difflib.SequenceMatcher(None, g, norm(n)).ratio() >= 0.85]
    if close:
        return {**base, "verdict": "match_fuzzy", "flags": flags + ["needs_human_check"]}
    return {**base, "verdict": "mismatch", "flags": flags}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "data" / "question_pool.json"))
    ap.add_argument("--cache", default=str(ROOT / "results" / "construction" / "wikidata_cache.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "construction"
                                         / "gold_verification_wikidata.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = json.loads(Path(args.pool).read_text())["questions"]
    if args.limit:
        rows = [r for r in rows if property_for(r["question"])][: args.limit]
    api = Api(Path(args.cache))
    results, errors = [], 0
    for i, r in enumerate(rows, start=1):
        try:
            res = check_row(api, r["question"], r["gold_answer"])
        except Exception as exc:  # noqa: BLE001 — the network must not bring
                                  # the run down
            res, errors = {"channel": "error", "verdict": "error",
                           "error": str(exc), "flags": []}, errors + 1
        results.append({"question": r["question"], "gold_answer": r["gold_answer"],
                        "gold_kind": r["gold_kind"], "family": r["family"], **res})
        if i % 25 == 0:
            api.save()
            print(f"  {i}/{len(rows)} (network calls {api.calls})", flush=True)
    api.save()

    from collections import Counter
    summary = {
        "checked": len(results),
        "channel": dict(Counter(r["channel"] for r in results)),
        "verdict": dict(Counter(r["verdict"] for r in results).most_common()),
        "flags": dict(Counter(f for r in results for f in r.get("flags", [])).most_common()),
        "errors": errors,
        "api_calls": api.calls,
        "accessed": time.strftime("%Y-%m-%d"),
        "source": "Wikidata API (wbsearchentities/wbgetclaims/wbgetentities)",
    }
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"summary": summary, "results": results},
                              ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
