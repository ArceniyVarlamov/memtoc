"""Distractor generator v2, through Wikidata "siblings".

Motivation. The old `pick_typed_distractor` substitutes a random entity of the
same TYPE drawn from the pool itself — the type matches but the field and era
do not. That is how "the Danish chemist Sorensen" gets Ferdinand de Lesseps
(who built Suez): the substitution is valid in form, but the model rejects it
because de Lesseps is plainly not a chemist, not because it KNOWS Sorensen. The
memory-versus-tool conflict is not tested on such a row.

Here the distractor is built from the gold ENTITY itself: we take its
occupation (P106) / sex (P21) / era (P569) / country and pull from ALL of
Wikidata another person of the same kind and era, certainly not equal to the
gold and not equal to the subject's genuine alternative answers. The type
matches by construction, so does the field, and there are no stubs, wrong sexes
or glued strings, because we take the live labels of real entities.

Division of labour (team decision, 2026-07-21):
  * date / year / timezone   -> delegated to memtoc.inject_nonentity (already done);
  * person / place / organization -> this module, Wikidata siblings;
  * work / other_string / anything where the gold entity is not recognised or
    has no distinguishing property -> the row is marked needs_human: those are
    authored by hand.

The old `pick_typed_distractor` is NOT touched — legacy must reproduce.

Network: read only. w/api.php (search/entities) reuses the caching client from
verify_gold_wikidata; siblings are pulled with a SPARQL query to WDQS and those
responses are cached to disk as well. Provenance (property, filter values,
access date) is written into every row.

Run:
  python -m scripts.build_distractors_wikidata --limit 20     # trial
  python -m scripts.build_distractors_wikidata --ids 53,148,208
  python -m scripts.build_distractors_wikidata                # the whole pool
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.parse
from collections import Counter
from pathlib import Path

from scripts.verify_gold_wikidata import Api, norm, property_for, resolve_subject
from memtoc.inject_nonentity import nonentity_distractor

ROOT = Path(__file__).resolve().parent.parent
WDQS = "https://query.wikidata.org/sparql"

SEX_MALE = "Q6581097"
SEX_FEMALE = "Q6581072"
HUMAN = "Q5"

# Era "nearby" for near and "obviously other" for far, in years from the gold.
NEAR_ERA = 45
FAR_ERA_MIN, FAR_ERA_MAX = 150, 600
SPARQL_LIMIT = 80
# The number of Wikipedia language editions as a proxy for "global fame". Above
# the threshold sit Schwarzenegger/Scorsese/Bruce Lee: a model certainly knows
# them and would reject the distractor "by fame", not by knowing the answer.
FAME_CAP = 40


# --------------------------------------------------------------------- WDQS
def sparql(api: Api, query: str) -> list[dict]:
    """Cached SPARQL to WDQS through the same client (backoff/pause/UA)."""
    key = "sparql:" + hashlib.sha1(query.encode()).hexdigest()
    if key in api.cache:
        return api.cache[key]
    url = WDQS + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    try:
        data = api._fetch(url)  # reuse the backoff / pause / User-Agent
    except Exception:  # a WDQS timeout on a heavy query — do not drop the row,
                       # do not poison the cache
        return []
    rows = data.get("results", {}).get("bindings", [])
    api.cache[key] = rows
    return rows


def _qid_of(binding: dict, var: str) -> str:
    return binding[var]["value"].rsplit("/", 1)[-1]


# ------------------------------------------------------- attributes of the
# gold
def claim_qids(doc: dict, prop: str) -> list[str]:
    out = []
    for c in doc.get("claims", {}).get(prop, []):
        snak = c.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        v = snak.get("datavalue", {}).get("value", {})
        if isinstance(v, dict) and "id" in v:
            out.append(v["id"])
    return out


def birth_year(doc: dict) -> int | None:
    best = None
    for prop in ("P569",):
        for c in doc.get("claims", {}).get(prop, []):
            if c.get("rank") == "deprecated":
                continue
            snak = c.get("mainsnak", {})
            if snak.get("snaktype") != "value":
                continue
            t = snak.get("datavalue", {}).get("value", {}).get("time", "")
            m = re.match(r"^([+-])(\d{4})", t)
            if m:
                y = int(m.group(2)) * (1 if m.group(1) == "+" else -1)
                best = y if best is None else best
    return best


def label_of(api: Api, qid: str) -> str | None:
    return api.entity_doc(qid).get("labels", {}).get("en", {}).get("value")


def resolve_gold(api: Api, gold: str) -> tuple[str | None, list[str]]:
    """Q-id of the gold answer's entity. Ambiguity is a flag, not a silent choice."""
    hits = api.search(gold)
    if not hits:
        return None, ["gold_not_found"]
    target = norm(gold)
    exact = [h for h in hits if norm(h.get("label", "")) == target]
    if not exact:
        return hits[0]["id"], ["gold_fuzzy"]
    if len(exact) > 1:
        return exact[0]["id"], ["gold_ambiguous"]
    return exact[0]["id"], []


# --------------------------------------------------------- choosing a
# candidate
def _pick(rows: list[dict], exclude: set[str], gold_norm: str,
          rng: random.Random) -> tuple[str, str] | None:
    """A suitable (qid, label). We prefer the NOT globally famous (by the number
    of Wikipedia language editions): otherwise the distractor is rejected "by
    fame".
    """
    valid = []
    for b in rows:
        qid = _qid_of(b, "p")
        label = b.get("pLabel", {}).get("value", "")
        if not label or qid in exclude:
            continue
        nl = norm(label)
        if not nl or nl == gold_norm:
            continue
        # do not substitute a piece of the gold, and do not take the gold as a
        # piece of the candidate
        if nl in gold_norm.split() or gold_norm in nl.split():
            continue
        sl = int(b["sl"]["value"]) if b.get("sl", {}).get("value") else 0
        valid.append((sl, qid, label))
    if not valid:
        return None
    modest = [v for v in valid if v[0] <= FAME_CAP]
    pick_from = modest if modest else sorted(valid)[:5]  # all are famous -> take the least
    rng.shuffle(pick_from)
    return pick_from[0][1], pick_from[0][2]


def people_query(occ: list[str], sex: str | None, lo: int | None, hi: int | None,
                 country: str | None = None) -> str:
    lines = []
    if occ:
        lines.append("VALUES ?occ { " + " ".join("wd:" + o for o in occ) + " }")
        lines.append("?p wdt:P106 ?occ .")
    else:
        lines.append("?p wdt:P31 wd:Q5 .")
    if sex:
        lines.append(f"?p wdt:P21 wd:{sex} .")
    if country:
        lines.append(f"?p wdt:P27 wd:{country} .")
    if lo is not None and hi is not None:
        lines.append("?p wdt:P569 ?dob .")
        lines.append(f"FILTER(YEAR(?dob) >= {lo} && YEAR(?dob) <= {hi})")
    body = "\n  ".join(lines)
    return (f"SELECT DISTINCT ?p ?pLabel ?sl WHERE {{\n  {body}\n"
            f'  ?p rdfs:label ?pLabel . FILTER(LANG(?pLabel)="en")\n'
            f"  OPTIONAL {{ ?p wikibase:sitelinks ?sl }}\n"
            f"}} LIMIT {SPARQL_LIMIT}")


def place_query(types: list[str], country: str | None, same_country: bool) -> str:
    lines = []
    if types:
        lines.append("VALUES ?t { " + " ".join("wd:" + t for t in types) + " }")
        lines.append("?p wdt:P31 ?t .")
    else:
        lines.append("?p wdt:P31 wd:Q515 .")  # city as the default
    if country:
        op = "=" if same_country else "!="
        lines.append(f"?p wdt:P17 ?c . FILTER(?c {op} wd:{country})")
    body = "\n  ".join(lines)
    return (f"SELECT DISTINCT ?p ?pLabel ?sl WHERE {{\n  {body}\n"
            f'  ?p rdfs:label ?pLabel . FILTER(LANG(?pLabel)="en")\n'
            f"  OPTIONAL {{ ?p wikibase:sitelinks ?sl }}\n"
            f"}} LIMIT {SPARQL_LIMIT}")


# -------------------------------------------------------------------- one row
def _cascade(api: Api, queries: list[tuple[str, list[str]]], exclude: set[str],
             gold_norm: str, rng: random.Random) -> tuple[tuple[str, str] | None, list[str]]:
    """Try the queries from narrowest to widest; take the first suitable one."""
    for q, flags in queries:
        hit = _pick(sparql(api, q), exclude, gold_norm, rng)
        if hit:
            return hit, flags
    return None, []


def build_person(api: Api, doc: dict, gold_norm: str, exclude: set[str],
                 rng: random.Random) -> dict:
    occ = claim_qids(doc, "P106")
    sex_vals = claim_qids(doc, "P21")
    sex = sex_vals[0] if sex_vals else None
    ctry = claim_qids(doc, "P27")
    country = ctry[0] if ctry else None
    by = birth_year(doc)
    prov = {"occupations": occ[:4], "sex": sex, "country": country, "birth_year": by}

    if not occ and by is None:
        return {"method": "human", "needs_human": True,
                "reason": "person_no_occupation_no_era", "provenance": prov}

    lo, hi = (by - NEAR_ERA, by + NEAR_ERA) if by is not None else (None, None)
    # near: narrow down in order — occupation+sex+country+era -> no country ->
    # no era
    near, near_flags = _cascade(api, [
        (people_query(occ, sex, lo, hi, country), []),
        (people_query(occ, sex, lo, hi, None), ["near_no_country"]),
        (people_query(occ, sex, None, None, country), ["near_no_era"]),
        (people_query(occ, sex, None, None, None), ["near_no_era", "near_no_country"]),
    ], exclude, gold_norm, rng)

    # Cross-cultural risk: the gold has a country, but no same-culture
    # neighbour
    # was found (a Japanese emperor offered for an Italian marquis).
    # We do not substitute just anyone — hand it to a human.
    if country and "near_no_country" in near_flags:
        return {"method": "human", "needs_human": True,
                "reason": "cross_culture_no_country_match",
                "provenance": prov, "flags": near_flags}

    # far: same occupation, era 150-600 years back -> same type, but plainly
    # not him
    far_queries = []
    if by is not None:
        far_queries.append((people_query(occ, sex, by - FAR_ERA_MAX, by - FAR_ERA_MIN, None), []))
    far_queries.append((people_query([], sex, None, None, None), ["far_generic"]))
    far, far_flags = _cascade(api, far_queries, exclude | ({near[0]} if near else set()),
                              gold_norm, rng)

    return _assemble(near, far, "person", prov, near_flags, far_flags)


def build_place(api: Api, doc: dict, gold_norm: str, exclude: set[str],
                kind: str, rng: random.Random) -> dict:
    types = claim_qids(doc, "P31")[:4]
    country_vals = claim_qids(doc, "P17")
    country = country_vals[0] if country_vals else None
    prov = {"instance_of": types, "country": country}
    near = _pick(sparql(api, place_query(types, country, same_country=True)),
                 exclude, gold_norm, rng)
    far = _pick(sparql(api, place_query(types, country, same_country=False)),
                exclude | ({near[0]} if near else set()), gold_norm, rng)
    return _assemble(near, far, kind, prov, [], [])


def _assemble(near, far, method, prov, near_flags, far_flags) -> dict:
    flags = list(near_flags) + list(far_flags)
    if near is None:
        flags.append("near_empty")
    if far is None:
        flags.append("far_empty")
    res = {"method": "wikidata_" + method, "provenance": prov, "flags": flags}
    if near:
        res["near"], res["near_qid"] = near[1], near[0]
    if far:
        res["far"], res["far_qid"] = far[1], far[0]
    if near is None:  # a bad near is pointless — hand it to a human
        res["needs_human"] = True
        res["reason"] = "no_sibling_found"
    return res


def resolve_via_property(api: Api, question: str, gold: str
                         ) -> tuple[str | None, set[str], list[str]]:
    """Exact QID of the gold from the subject's property, plus the exclusion set.
    
    For "who is the father of X" the subject X points with property P22 straight
    at the entity we need — more reliable than a search by name (it removes
    homonymy) and it also yields ALL the genuine values (a second father or
    director) that must be excluded from the distractors. Returns (gold_qid |
    None, exclusions, flags).
    """
    got = property_for(question)
    if not got:
        return None, set(), []
    prop, _name, subject = got
    subj_qid, sflags = resolve_subject(api, subject)
    if subj_qid is None:
        return None, set(), sflags
    vals = claim_qids(api.entity_doc(subj_qid), prop)
    exclude = {subj_qid} | set(vals)
    if not vals:
        return None, exclude, sflags + ["subject_no_value"]
    g = norm(gold)
    matched = [v for v in vals if norm(label_of(api, v) or "") == g]
    chosen = matched[0] if matched else (vals[0] if len(vals) == 1 else None)
    return chosen, exclude, sflags + (["gold_not_in_values"] if chosen is None else [])


def build_row(api: Api, rec: dict, seed: int) -> dict:
    kind = rec["gold_kind"]
    gold = rec["gold_answer"]
    rng = random.Random(f"{seed}|{rec['qkey']}")

    if kind in ("date", "year", "timezone"):
        near, nf = nonentity_distractor(gold, kind, rng, "near")
        far, ff = nonentity_distractor(gold, kind, rng, "far")
        out = {"method": "nonentity", "flags": nf + ff}
        if near:
            out["near"] = near
        if far:
            out["far"] = far
        if near is None:
            out["needs_human"], out["reason"] = True, "nonentity_failed"
        return out

    if kind not in ("person", "place", "organization"):
        return {"method": "human", "needs_human": True,
                "reason": f"kind_{kind}_by_hand"}

    # 1) exact QID from the subject's property (removes homonymy); 2) fall back
    # to search.
    gold_qid, exclude, gflags = resolve_via_property(api, rec["question"], gold)
    if gold_qid is None:
        gold_qid, sflags = resolve_gold(api, gold)
        gflags = gflags + sflags
        if gold_qid is None or "gold_ambiguous" in sflags:
            return {"method": "human", "needs_human": True,
                    "reason": "gold_unresolved", "flags": gflags}
    exclude.add(gold_qid)
    gold_norm = norm(gold)
    doc = api.entity_doc(gold_qid)

    # The type is taken from the gold ENTITY itself, not from the pool's label:
    # some
    # genealogical questions are mislabelled place/organization, and by label
    # they
    # went into the place branch -> and got random people (Stalin as a king's
    # father).
    p31 = claim_qids(doc, "P31")
    is_person = (HUMAN in p31) or (not p31 and kind == "person")
    if is_person:
        res = build_person(api, doc, gold_norm, exclude, rng)
    else:
        res = build_place(api, doc, gold_norm, exclude, kind, rng)
    if is_person != (kind == "person"):
        res.setdefault("flags", []).append(f"kind_relabeled_from:{kind}")
    res.setdefault("flags", []).extend(gflags)
    res["gold_qid"] = gold_qid
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "data" / "question_pool.json"))
    ap.add_argument("--cache", default=str(ROOT / "results" / "construction" / "wikidata_cache.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "construction" / "distractors_v2_wikidata.json"))
    ap.add_argument("--broken", default=str(ROOT / "results" / "construction" / "broken_gold_exclude.json"),
                    help="qkey->reason; these questions are dropped (broken gold is not used)")
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated: only these instance_id")
    args = ap.parse_args()

    recs = json.loads(Path(args.pool).read_text())["questions"]
    if args.ids:
        want = {int(x) for x in args.ids.split(",")}
        recs = [r for r in recs if r["instance_id"] in want]
    if args.limit:
        recs = recs[: args.limit]
    broken = json.loads(Path(args.broken).read_text()) if Path(args.broken).exists() else {}

    api = Api(Path(args.cache))
    results = {}
    for i, rec in enumerate(recs, 1):
        if rec["qkey"] in broken:  # broken gold — not used (team decision,
                                   # 07-21)
            res = {"method": "dropped_broken_gold", "dropped": True,
                   "reason": broken[rec["qkey"]].get("reason", "broken_gold")}
        else:
            try:
                res = build_row(api, rec, args.seed)
            except Exception as exc:  # noqa: BLE001 — the network does not
                                      # bring the run down
                res = {"method": "error", "needs_human": True, "reason": str(exc)}
        res.update({"qkey": rec["qkey"], "instance_id": rec["instance_id"],
                    "hop_idx": rec["hop_idx"], "question": rec["question"],
                    "gold_answer": rec["gold_answer"], "gold_kind": rec["gold_kind"],
                    "family": rec["family"]})
        results[rec["qkey"]] = res  # qkey is unique (605), instance_id is not
                                    # (310)
        if i % 20 == 0:
            api.save()
            print(f"  {i}/{len(recs)} (network calls {api.calls})", flush=True)
    api.save()

    dropped = [r for r in results.values() if r.get("dropped")]
    filled = [r for r in results.values() if not r.get("needs_human") and not r.get("dropped")]
    human = [r for r in results.values() if r.get("needs_human")]
    summary = {
        "total": len(results),
        "machine_filled": len(filled),
        "needs_human": len(human),
        "dropped_broken_gold": len(dropped),
        "by_method": dict(Counter(r.get("method", "unknown") for r in results.values())),
        "human_reasons": dict(Counter(r.get("reason", "") for r in human).most_common()),
        "flags": dict(Counter(f for r in results.values() for f in r.get("flags", [])).most_common()),
        "api_calls": api.calls,
        "accessed": time.strftime("%Y-%m-%d"),
        "seed": args.seed,
        "source": "Wikidata WDQS (P106/P21/P569/P31/P17) + w/api.php",
    }
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"summary": summary, "results": results},
                              ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
