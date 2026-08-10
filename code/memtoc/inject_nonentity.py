"""Distractors for non-entity gold types: date, year, timezone.

Why this is a separate module. `memtoc.inject.pick_typed_distractor` draws its
substitution from a typed pool of named entities
(person/place/organization/work). For "What is the date of birth of Thor
Heyerdahl?" no such pool exists: the distractor for a date is ANOTHER date,
not another entity. The question did not arise in the v1 benchmark because
selection admitted only entity-typed hops, and 82% of the items turned out to
be about people. The v2 pool adds 112 dates, 45 years and 19 timezones —
almost a third of the set, and without this branch they drop out.

The existing `pick_typed_distractor` is NOT touched: earlier builds must
reproduce bit-for-bit. The new episode builder calls this function explicitly,
by gold_kind.

Plausibility. A date is rendered in the ORIGINAL format of the gold: if the
gold is "19 May 1673" and the distractor "1673-05-19", the model tells the
substitution apart by shape rather than by knowledge — and we would measure
format recognition instead of conflict. The shift goes through `datetime`, so
31 February is never born.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
_MONTH_IDX = {m.lower(): i + 1 for i, m in enumerate(MONTH_NAMES)}

# Shifts in days. near = "plausibly nearby" (up to ~4 years), far = another era
# (50-300 years). The thresholds match the spirit of the entity branch's
# buckets: near must be indistinguishable in type and plausibility, far
# obviously not.
NEAR_DAYS = (15, 1500)
FAR_DAYS = (18_000, 110_000)
MAX_YEAR = 2026
NEAR_YEARS = (1, 15)
FAR_YEARS = (80, 400)

_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DMY_RE = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{3,4})$")
_MDY_RE = re.compile(r"^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{3,4})$")
_MY_RE = re.compile(r"^([A-Za-z]+)\s+(\d{3,4})$")


def parse_date(gold: str) -> tuple[date, str] | None:
    """(date, format name), or None if it could not be parsed."""
    g = gold.strip()
    m = _ISO_RE.match(g)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return _safe_date(y, mo, d), "iso"
    m = _DMY_RE.match(g)
    if m and m.group(2).lower() in _MONTH_IDX:
        return _safe_date(int(m.group(3)), _MONTH_IDX[m.group(2).lower()],
                          int(m.group(1))), "dmy"
    m = _MDY_RE.match(g)
    if m and m.group(1).lower() in _MONTH_IDX:
        return _safe_date(int(m.group(3)), _MONTH_IDX[m.group(1).lower()],
                          int(m.group(2))), "mdy"
    m = _MY_RE.match(g)
    if m and m.group(1).lower() in _MONTH_IDX:
        return _safe_date(int(m.group(2)), _MONTH_IDX[m.group(1).lower()], 1), "my"
    return None


def _safe_date(y: int, mo: int, d: int) -> date | None:
    try:
        return date(max(y, 1), mo, d)
    except ValueError:
        return None


def render_date(d: date, fmt: str) -> str:
    month = MONTH_NAMES[d.month - 1]
    if fmt == "iso":
        return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
    if fmt == "dmy":
        return f"{d.day} {month} {d.year}"
    if fmt == "mdy":
        return f"{month} {d.day}, {d.year}"
    return f"{month} {d.year}"


def _shift(rng, lo: int, hi: int) -> int:
    return rng.randint(lo, hi) * rng.choice((-1, 1))


def date_distractor(gold: str, rng, divergence: str = "near") -> tuple[str | None, list[str]]:
    parsed = parse_date(gold)
    if parsed is None or parsed[0] is None:
        return None, ["date_unparsed"]
    d, fmt = parsed
    lo, hi = NEAR_DAYS if divergence == "near" else FAR_DAYS
    for _ in range(12):
        try:
            cand = d + timedelta(days=_shift(rng, lo, hi))
        except OverflowError:
            continue
        # Ceiling of "today": a death date in 2161 is implausible by shape
        # alone,
        # and the model would reject it out of absurdity rather than knowledge.
        if cand != d and 1 <= cand.year <= MAX_YEAR:
            return render_date(cand, fmt), []
    # The forward shift hit the ceiling — move strictly into the past instead.
    for _ in range(12):
        try:
            cand = d - timedelta(days=rng.randint(lo, hi))
        except OverflowError:
            continue
        if cand != d and cand.year >= 1:
            return render_date(cand, fmt), ["date_forced_past"]
    return None, ["date_shift_failed"]


def year_distractor(gold: str, rng, divergence: str = "near") -> tuple[str | None, list[str]]:
    if not re.fullmatch(r"\d{3,4}", gold.strip()):
        return None, ["year_unparsed"]
    y = int(gold.strip())
    lo, hi = NEAR_YEARS if divergence == "near" else FAR_YEARS
    for _ in range(12):
        cand = y + _shift(rng, lo, hi)
        if cand != y and 1 <= cand <= MAX_YEAR:
            return str(cand), []
    return None, ["year_shift_failed"]


def timezone_distractor(gold: str, rng, divergence: str = "near",
                        pool: list[str] | None = None) -> tuple[str | None, list[str]]:
    """near = another zone of the same region, far = another region.
    
    `pool` is the set of zones observed in the data; without it a built-in
    minimum is used.
    """
    zones = list(pool or DEFAULT_TZ_POOL)
    g = gold.strip()
    region = g.split("/")[0]
    same = [z for z in zones if z.split("/")[0] == region and z != g]
    other = [z for z in zones if z.split("/")[0] != region]
    cands = (same or other) if divergence == "near" else (other or same)
    if not cands:
        return None, ["tz_pool_empty"]
    flags = [] if (same if divergence == "near" else other) else ["tz_fallback_region"]
    return rng.choice(sorted(cands)), flags


DEFAULT_TZ_POOL = [
    "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
    "America/Toronto", "America/Mexico_City", "America/Sao_Paulo",
    "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Madrid",
    "Europe/Rome", "Europe/Moscow", "Europe/Warsaw",
    "Asia/Tokyo", "Asia/Shanghai", "Asia/Kolkata", "Asia/Dubai",
    "Australia/Sydney", "Africa/Cairo", "Africa/Lagos", "Pacific/Auckland",
]

DISPATCH = {
    "date": date_distractor,
    "year": year_distractor,
    "timezone": timezone_distractor,
}


def nonentity_distractor(gold: str, gold_kind: str, rng,
                         divergence: str = "near", **kw) -> tuple[str | None, list[str]]:
    """Single entry point; None means "this type is not served here"."""
    fn = DISPATCH.get(gold_kind)
    if fn is None:
        return None, [f"unsupported_kind:{gold_kind}"]
    return fn(gold, rng, divergence, **kw)
