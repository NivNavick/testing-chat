from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import re
import math
import pandas as pd
from datetime import datetime, date, time

# =========================
# Config / constants
# =========================

# NOTE:
# - We intentionally do NOT use dateutil for *type detection* (it misclassifies numbers as dates).
# - Duration is a SEMANTIC type (stored as seconds), but we now gate it:
#   duration can ONLY win if there is explicit duration evidence (HH:MM / suffix like 'm'/'h'/'s'
#   or header hints), not just "small floats".

NULL_LIKE = {"", "na", "n/a", "null", "none", "-", "--", "nan"}
BOOL_TRUE = {"true", "t", "yes", "y", "1"}
BOOL_FALSE = {"false", "f", "no", "n", "0"}

RE_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")
RE_INT = re.compile(r"^\s*[-+]?\d+\s*$")
RE_FLOAT = re.compile(r"^\s*[-+]?\d+(\.\d+)?\s*$")
RE_IDLIKE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{3,}$")  # 4+ chars, stable-ish
RE_PURE_NUMBER = re.compile(r"^\s*\d+\s*$")
RE_DURATION_SUFFIX = re.compile(r"^\s*[-+]?\d+(\.\d+)?\s*[hmsHMS]\s*$")

# Header hints (lightweight, can extend later / per customer)
DURATION_HEADER_HINTS = {
    "duration", "dur", "elapsed", "turnover", "turn over", "tat", "t.a.t",
    "mins", "min", "minutes", "hours", "hrs", "sec", "seconds"
}

# Common datetime/date formats in exports
DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
]
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
]


# =========================
# Data model
# =========================

@dataclass
class ColumnProfile:
    column_name: str
    detected_type: str
    confidence: float

    non_null_count: int
    null_count: int
    parse_rate: float

    unique_count: int
    unique_ratio: float
    avg_strlen: float
    max_strlen: int

    datetime_min: Optional[str] = None
    datetime_max: Optional[str] = None
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None

    regex_hits: Dict[str, float] = None
    sample_values: List[str] = None
    candidate_scores: Dict[str, float] = None
    debug: Dict[str, Any] = None


# =========================
# Helpers
# =========================

def _normalize_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if s == "":
        return None
    if s.lower() in NULL_LIKE:
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def _sample_values(values: List[str], k: int = 15) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in values:
        if s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) >= k:
            break
    return out


def _clean_numeric(s: str) -> str:
    s2 = s.strip()
    s2 = s2.replace(",", "")
    s2 = re.sub(r"[\$₪€£]", "", s2)
    return s2


def _try_parse_with_formats(s: str, formats: List[str]) -> Optional[datetime]:
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


# STRICT parsing: no dateutil fallback here on purpose.
def _parse_datetime_strict(s: str) -> Optional[datetime]:
    if RE_PURE_NUMBER.match(s):
        return None
    return _try_parse_with_formats(s, DT_FORMATS)


def _parse_date_strict(s: str) -> Optional[date]:
    if RE_PURE_NUMBER.match(s):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def _parse_time_of_day(s: str) -> Optional[time]:
    m = RE_TIME.match(s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3)) if m.group(3) else 0
    if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
        return time(hour=hh, minute=mm, second=ss)
    return None


def _parse_duration_seconds(s: str) -> Optional[int]:
    """
    Returns duration in seconds if parseable.
    Supports:
      - HH:MM or HH:MM:SS (interpreted as duration)
      - "90m", "1.5h", "3600s"
      - plain number is NOT treated as duration here anymore
        (we only allow numeric-only duration if header hints / context says so)
    """
    s0 = s.strip().lower()

    # HH:MM(:SS)
    m = RE_TIME.match(s0)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3)) if m.group(3) else 0
        if 0 <= mm <= 59 and 0 <= ss <= 59:
            return hh * 3600 + mm * 60 + ss

    # unit suffix
    unit_m = re.match(r"^\s*([-+]?\d+(\.\d+)?)\s*([hms])\s*$", s0)
    if unit_m:
        val = float(unit_m.group(1))
        unit = unit_m.group(3)
        if unit == "h":
            return int(val * 3600)
        if unit == "m":
            return int(val * 60)
        if unit == "s":
            return int(val)
        return None

    return None


def _has_duration_header_hint(column_name: str) -> bool:
    c = (column_name or "").strip().lower()
    return any(h in c for h in DURATION_HEADER_HINTS)


# =========================
# Profiling
# =========================

def profile_column(
    series: pd.Series,
    column_name: str,
    sample_size: int = 2000,
) -> Dict[str, Any]:
    raw = series.tolist()
    norm_all = [_normalize_value(v) for v in raw]
    norm = [v for v in norm_all if v is not None]

    non_null = len(norm)
    null_count = len(raw) - non_null

    if non_null == 0:
        prof = ColumnProfile(
            column_name=str(column_name),
            detected_type="empty",
            confidence=1.0,
            non_null_count=0,
            null_count=int(null_count),
            parse_rate=0.0,
            unique_count=0,
            unique_ratio=0.0,
            avg_strlen=0.0,
            max_strlen=0,
            regex_hits={},
            sample_values=[],
            candidate_scores={"empty": 1.0},
            debug={"note": "column has no non-null values"},
        )
        return asdict(prof)

    sample = norm[: min(sample_size, non_null)]
    n = len(sample)

    unique_in_order = list(dict.fromkeys(sample))
    unique_count = len(set(sample))
    unique_ratio = unique_count / max(1, n)

    lengths = [len(s) for s in sample]
    avg_strlen = sum(lengths) / n
    max_strlen = max(lengths)

    def hit_rate(rx: re.Pattern) -> float:
        return sum(1 for s in sample if rx.match(s)) / n

    regex_hits = {
        "time_like": hit_rate(RE_TIME),
        "int_like": hit_rate(RE_INT),
        "float_like": hit_rate(RE_FLOAT),
        "id_like": hit_rate(RE_IDLIKE),
        "pure_number_like": hit_rate(RE_PURE_NUMBER),
        "duration_suffix_like": hit_rate(RE_DURATION_SUFFIX),
    }

    # Parse attempts
    dt_parsed: List[datetime] = []
    date_parsed: List[date] = []
    tod_parsed: List[time] = []

    dur_parsed: List[int] = []  # seconds
    explicit_duration_hits = 0   # counts values with explicit duration signal (HH:MM or unit suffix)

    ints_parsed: List[int] = []
    floats_parsed: List[float] = []
    bool_parsed: List[str] = []

    leading_zero_count = 0
    numeric_like_count = 0

    separator_rate = sum(
        1 for s in sample if any(ch in s for ch in (":", "-", "/", "T"))
    ) / n

    header_has_duration_hint = _has_duration_header_hint(str(column_name))

    for s in sample:
        # datetime/date strict detection
        dt = _parse_datetime_strict(s)
        if dt is not None:
            dt_parsed.append(dt)

        d = _parse_date_strict(s)
        if d is not None:
            date_parsed.append(d)

        tod = _parse_time_of_day(s)
        if tod is not None:
            tod_parsed.append(tod)

        # duration (explicit forms only)
        dur = _parse_duration_seconds(s)
        if dur is not None:
            dur_parsed.append(dur)
            explicit_duration_hits += 1

        # numeric
        s_num = _clean_numeric(s)
        if RE_FLOAT.match(s_num):
            numeric_like_count += 1
            if len(s_num) > 1 and s_num[0] == "0" and s_num[1].isdigit():
                leading_zero_count += 1
            try:
                f = float(s_num)
                floats_parsed.append(f)
                if f.is_integer():
                    ints_parsed.append(int(f))
            except Exception:
                pass

        # boolean
        sl = s.strip().lower()
        if sl in BOOL_TRUE or sl in BOOL_FALSE:
            bool_parsed.append(sl)

    # Parse success rates
    rates = {
        "datetime": len(dt_parsed) / n,
        "date": len(date_parsed) / n,
        "time_of_day": len(tod_parsed) / n,
        "duration": len(dur_parsed) / n,
        "integer": len(ints_parsed) / n,
        "float": len(floats_parsed) / n,
        "boolean": len(bool_parsed) / n,
    }

    # Candidate scoring
    scores: Dict[str, float] = {}

    # datetime sanity (year range) if parsed
    if dt_parsed:
        years = [x.year for x in dt_parsed]
        sane_year_rate = sum(1 for y in years if 2000 <= y <= 2050) / len(years)
    else:
        sane_year_rate = 0.0

    scores["datetime"] = rates["datetime"] * (0.7 + 0.3 * sane_year_rate)
    scores["date"] = rates["date"]
    scores["time_of_day"] = rates["time_of_day"]

    # Duration scoring is now GATED:
    explicit_duration_rate = explicit_duration_hits / n
    duration_allowed = (explicit_duration_rate >= 0.2) or header_has_duration_hint

    if duration_allowed and dur_parsed:
        within_24h = sum(1 for dsec in dur_parsed if 0 <= dsec <= 24 * 3600) / len(dur_parsed)
        scores["duration"] = rates["duration"] * (0.7 + 0.3 * within_24h)
    else:
        scores["duration"] = 0.0

    scores["integer"] = rates["integer"]
    scores["float"] = rates["float"]
    scores["boolean"] = rates["boolean"] if rates["boolean"] >= 0.9 else 0.0

    # Behavior-based
    cat_score = max(0.0, (0.2 - unique_ratio) / 0.2)  # 1 when unique_ratio=0; 0 at 0.2+
    free_text_score = 0.0
    if unique_ratio > 0.7 and avg_strlen >= 20:
        free_text_score = min(1.0, (unique_ratio - 0.7) / 0.3) * min(1.0, avg_strlen / 40)

    id_like_score = 0.0
    if unique_ratio > 0.7 and regex_hits["id_like"] > 0.6 and avg_strlen <= 24:
        id_like_score = min(1.0, (unique_ratio - 0.7) / 0.3) * regex_hits["id_like"]

    scores["categorical"] = cat_score
    scores["free_text"] = free_text_score
    scores["id_like"] = id_like_score

    # generic text baseline
    scores["text"] = 0.15 + 0.1 * (1.0 - cat_score)

    # Guard rails
    numeric_rate = numeric_like_count / max(1, n)
    leading_zero_rate = leading_zero_count / max(1, numeric_like_count)

    # If many leading zeros and numeric-like: prefer id_like over integer/float
    if leading_zero_rate >= 0.3 and unique_ratio > 0.5:
        scores["id_like"] = max(scores["id_like"], 0.75)

    # NUMERIC VETO: mostly numeric and few separators => cannot be date/datetime
    if numeric_rate >= 0.9 and separator_rate < 0.3:
        scores["datetime"] *= 0.05
        scores["date"] *= 0.05

    # SEPARATOR VETO: if values rarely contain time/date separators, downweight date/datetime
    if separator_rate < 0.2:
        scores["datetime"] *= 0.1
        scores["date"] *= 0.1

    # Ratio heuristic: if values look like ratios (0..~1), prefer float (not duration)
    if floats_parsed and max(floats_parsed) <= 1.2:
        scores["duration"] *= 0.1

    # If datetime is very strong, suppress date/time_of_day (prevents misclassifying full datetime)
    if scores["datetime"] >= 0.85:
        scores["date"] *= 0.3
        scores["time_of_day"] *= 0.3

    # Choose winner
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    confidence = max(0.0, min(1.0, 0.6 * best_score + 0.4 * (best_score - second_score)))

    # Summaries
    dt_min = dt_max = None
    if best_type == "datetime" and dt_parsed:
        dt_min = min(dt_parsed).isoformat(sep=" ")
        dt_max = max(dt_parsed).isoformat(sep=" ")

    num_min = num_max = None
    if best_type in ("integer", "float") and floats_parsed:
        num_min = float(min(floats_parsed))
        num_max = float(max(floats_parsed))
    elif best_type == "duration" and dur_parsed:
        num_min = float(min(dur_parsed))
        num_max = float(max(dur_parsed))

    prof = ColumnProfile(
        column_name=str(column_name),
        detected_type=best_type,
        confidence=float(confidence),
        non_null_count=int(non_null),
        null_count=int(null_count),
        parse_rate=float(rates.get(best_type, 0.0)),
        unique_count=int(unique_count),
        unique_ratio=float(unique_ratio),
        avg_strlen=float(avg_strlen),
        max_strlen=int(max_strlen),
        datetime_min=dt_min,
        datetime_max=dt_max,
        numeric_min=num_min,
        numeric_max=num_max,
        regex_hits=regex_hits,
        sample_values=_sample_values(unique_in_order, k=15),
        candidate_scores=scores,
        debug={
            "numeric_rate": float(numeric_rate),
            "leading_zero_rate": float(leading_zero_rate),
            "separator_rate": float(separator_rate),
            "explicit_duration_rate": float(explicit_duration_rate),
            "header_has_duration_hint": bool(header_has_duration_hint),
            "duration_allowed": bool(duration_allowed),
            "parsed_counts": {
                "datetime": len(dt_parsed),
                "date": len(date_parsed),
                "time_of_day": len(tod_parsed),
                "duration": len(dur_parsed),
                "integer": len(ints_parsed),
                "float": len(floats_parsed),
                "boolean": len(bool_parsed),
            },
            "ranked_top5": ranked[:5],
        },
    )
    return asdict(prof)
def profile_dataframe(df: pd.DataFrame, sample_size: int = 2000) -> List[Dict[str, Any]]:
    profiles = []
    for col in df.columns:
        profiles.append(profile_column(df[col], column_name=str(col), sample_size=sample_size))
    return profiles