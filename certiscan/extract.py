"""Turn OCR words + layout regions into a structured MTR record."""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

from certiscan.specs import parse_grade, resolve_spec_key
from certiscan.textlayout import (
    group_lines,
    line_text,
    median_height,
    normalize_label,
    parse_number,
    split_segments,
    words_in,
)
from certiscan.types import Region, Word

# ----------------------------------------------------------------------------- info block

FIELD_LABELS: Dict[str, List[str]] = {
    "cert_no": ["certificate no", "cert no", "certificate number", "mtc no", "test certificate no", "report no", "mtr no"],
    "date": ["date", "issue date", "date of issue"],
    "customer": ["customer", "purchaser", "sold to", "consignee"],
    "po_no": ["po no", "purchase order", "customer po", "order no", "po number"],
    "heat_list": ["heat no", "heat nos", "heat number", "melt no", "heat #"],
    "spec": ["specification", "spec", "material spec", "standard"],
    "grade": ["grade", "steel grade", "material grade"],
    "size": ["size", "dimensions", "product size"],
    "quantity": ["quantity", "qty"],
}
_LABEL_LOOKUP = {normalize_label(lbl): field for field, lbls in FIELD_LABELS.items() for lbl in lbls}

INFO_FIELDS = ["cert_no", "date", "customer", "po_no", "spec", "grade", "size", "quantity"]


def match_label(text: str, fuzzy: bool = True) -> Optional[str]:
    n = normalize_label(text)
    if not n:
        return None
    if n in _LABEL_LOOKUP:
        return _LABEL_LOOKUP[n]
    if not fuzzy:
        return None
    # tolerate OCR substitutions ("Cert1ficate"), not missing words ("Test Certificate" != "... No")
    close = [c for c in difflib.get_close_matches(n, list(_LABEL_LOOKUP), n=3, cutoff=0.85) if abs(len(c) - len(n)) <= 2]
    return _LABEL_LOOKUP[close[0]] if close else None


def _prefix_label(text: str) -> Optional[Tuple[str, str]]:
    # exact match only: fuzzy prefixes turn titles like "TEST CERTIFICATE EN 10204"
    # into a bogus "Test Certificate No." field
    tokens = text.split()
    for k in range(min(4, len(tokens) - 1), 0, -1):
        field = match_label(" ".join(tokens[:k]), fuzzy=False)
        if field:
            return field, " ".join(tokens[k:])
    return None


def _is_label_like(text: str) -> bool:
    return match_label(text.split(":")[0]) is not None


def parse_info(words: List[Word]) -> Dict[str, str]:
    """Key/value pairs from the header block. Handles 'Label: value', column-aligned
    'Label    value', and two pairs per line."""
    out: Dict[str, str] = {}

    def put(field: str, value: str) -> None:
        value = value.strip().lstrip(":").strip()
        if value and field not in out:
            out[field] = value

    for line in group_lines(words):
        # line-local height: a 40px title and 22px fields must not share one gap threshold
        segs = split_segments(line, gap_factor=1.2)
        i = 0
        while i < len(segs):
            text = segs[i].text
            def continuation(j: int, value: str):
                # OCR can split one value ('56 PCS', '/', '7566 KG') into several segments:
                # keep appending until the next label (e.g. the right-hand column) starts
                while j < len(segs) and not _is_label_like(segs[j].text):
                    value = f"{value} {segs[j].text}".strip()
                    j += 1
                return j, value

            if ":" in text:
                label, value = text.split(":", 1)
                field = match_label(label)
                if field:
                    i, value = continuation(i + 1, value.strip())
                    put(field, value)
                    continue
            field = match_label(text)
            if field and i + 1 < len(segs) and not _is_label_like(segs[i + 1].text):
                i, value = continuation(i + 1, "")
                put(field, value)
                continue
            pref = _prefix_label(text)
            if pref:
                put(*pref)
            i += 1
    return out


# ----------------------------------------------------------------------------- tables

ELEMENTS = ["C", "Mn", "P", "S", "Si", "Cu", "Ni", "Cr", "Mo", "V", "Nb", "Ti", "Al", "B", "N", "CE"]
_EL_LOOKUP = {e.upper(): e for e in ELEMENTS}
# common OCR confusions on element symbols
_EL_ALIASES = {"SL": "Si", "S1": "Si", "NL": "Ni", "N1": "Ni", "M0": "Mo", "AI": "Al", "A1": "Al", "T1": "Ti", "CU": "Cu"}
_SKIP_ROW = re.compile(r"\b(min|max|spec|limit|req|required|std|standard)\b", re.IGNORECASE)


def element_symbol(text: str) -> Optional[str]:
    t = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return _EL_LOOKUP.get(t) or _EL_ALIASES.get(t)


def _is_heat_header(text: str) -> bool:
    toks = re.findall(r"[a-z]+", text.lower())
    return any(t in ("heat", "melt", "cast", "lot") for t in toks)


MECH_ORDER = ["yield_mpa", "tensile_mpa", "elongation_pct"]


def mech_column(text: str) -> Tuple[Optional[str], str]:
    low = text.lower()
    toks = re.findall(r"[a-z0-9]+", low)
    unit = "ksi" if any(t in ("ksi", "ksl", "kst", "ks1", "kși") for t in toks) else "mpa"
    if _is_heat_header(text):
        return "heat", unit
    if any(t in ("yield", "ys", "proof", "rp", "rp0") for t in toks):
        return "yield_mpa", unit
    if any(t in ("tensile", "uts", "ts", "rm") for t in toks):
        return "tensile_mpa", unit
    if any(t.startswith("elong") for t in toks) or "el" in toks:
        return "elongation_pct", "pct"
    return None, unit


def _explode(word: Word) -> List[Word]:
    """Split a multi-token OCR phrase into per-token boxes (proportional to characters)."""
    tokens = word.text.split()
    if len(tokens) <= 1:
        return [word]
    total = sum(len(t) for t in tokens) + len(tokens) - 1
    x0, y0, x1, y1 = word.box
    per_char = (x1 - x0) / max(total, 1)
    out, cursor = [], x0
    for t in tokens:
        w = len(t) * per_char
        out.append(Word(t, (cursor, y0, cursor + w, y1), word.conf))
        cursor += w + per_char
    return out


def _columns(header: List[Word], kind: str, h: float) -> List[dict]:
    cols = []
    for seg in split_segments(header, gap_factor=1.0, ref_height=h):
        x0, _, x1, _ = seg.box
        if kind == "chem":
            key = "heat" if _is_heat_header(seg.text) else element_symbol(seg.text)
            unit = "pct"
        else:
            key, unit = mech_column(seg.text)
        cols.append({"key": key, "unit": unit, "x0": x0, "x1": x1, "cx": (x0 + x1) / 2, "text": seg.text})
    return cols


def _add_orphan_columns(cols: List[dict], data_lines: List[List[Word]]) -> List[dict]:
    """OCR sometimes misses a header (one-letter symbols like 'C' or 'P' are easy to drop).
    Numeric tokens far from every header column reveal such a column; add it, key unknown."""
    if len(cols) < 2:
        return cols
    centers = sorted(c["cx"] for c in cols)
    spacing = sorted(b - a for a, b in zip(centers, centers[1:]))[len(centers) // 2 - 1] if len(centers) > 2 else centers[1] - centers[0]
    orphans = []
    for line in data_lines:
        for w in line:
            for p in _explode(w):
                if not re.search(r"\d", p.text):
                    continue
                inside = any(c["x0"] - 4 <= p.cx <= c["x1"] + 4 for c in cols)
                if not inside and min(abs(c["cx"] - p.cx) for c in cols) > 0.4 * spacing:
                    orphans.append(p.cx)
    groups: List[List[float]] = []
    for x in sorted(orphans):
        if groups and x - groups[-1][-1] < 0.3 * spacing:
            groups[-1].append(x)
        else:
            groups.append([x])
    for g in groups:
        cx = sum(g) / len(g)
        cols.append({"key": None, "unit": cols[-1]["unit"] if cols else "mpa", "x0": cx, "x1": cx, "cx": cx,
                     "text": "", "inferred": True})
    return sorted(cols, key=lambda c: c["cx"])


def _infer_keys(cols: List[dict], order: List[str]) -> None:
    """Name inferred columns from the conventional column order (C, Mn, P, S, Si, ... or YS, UTS, El)
    when the gap between their labelled neighbours admits exactly as many candidates as columns."""
    used = {c["key"] for c in cols}
    i = 0
    while i < len(cols):
        if not cols[i].get("inferred") or cols[i]["key"] is not None:
            i += 1
            continue
        j = i
        while j < len(cols) and cols[j].get("inferred") and cols[j]["key"] is None:
            j += 1
        left = next((order.index(c["key"]) for c in reversed(cols[:i]) if c["key"] in order), -1)
        right = next((order.index(c["key"]) for c in cols[j:] if c["key"] in order), None)
        if right is not None:
            cands = [k for k in order[left + 1:right] if k not in used]
            if len(cands) == j - i:
                for c, k in zip(cols[i:j], cands):
                    c["key"] = k
                    used.add(k)
        i = j


def _header_score(line: List[Word], kind: str, h: float) -> int:
    segs = split_segments(line, gap_factor=1.0, ref_height=h)
    if kind == "chem":
        return sum(1 for s in segs if element_symbol(s.text))
    return sum(1 for s in segs if mech_column(s.text)[0] not in (None, "heat"))


def _assign(word: Word, cols: List[dict]) -> int:
    inside = [i for i, c in enumerate(cols) if c["x0"] - 4 <= word.cx <= c["x1"] + 4]
    if inside:
        return inside[0]
    return min(range(len(cols)), key=lambda i: abs(cols[i]["cx"] - word.cx))


def parse_table(words: List[Word], kind: str) -> List[dict]:
    """Parse a chemistry ('chem') or mechanical ('mech') table into rows keyed by heat."""
    if not words:
        return []
    h = median_height(words)
    lines = group_lines(words)
    scores = [_header_score(l, kind, h) for l in lines]
    if not scores or max(scores) < 2:
        return []
    hi = scores.index(max(scores))
    cols = _add_orphan_columns(_columns(lines[hi], kind, h), lines[hi + 1:])
    if cols and not any(c["key"] == "heat" for c in cols) and cols[0]["key"] is None:
        cols[0]["key"] = "heat"
    _infer_keys(cols, ELEMENTS if kind == "chem" else MECH_ORDER)
    heat_idx = next((i for i, c in enumerate(cols) if c["key"] == "heat"), None)
    if heat_idx is None:
        return []

    rows = []
    for line in lines[hi + 1:]:
        cells: Dict[int, List[str]] = {}
        for word in line:
            for piece in _explode(word):
                cells.setdefault(_assign(piece, cols), []).append(piece.text)
        heat_text = "".join(cells.get(heat_idx, []))
        if not heat_text or not re.search(r"\d", heat_text) or _SKIP_ROW.search(" ".join(cells.get(heat_idx, []))):
            continue
        row = {"heat_no": re.sub(r"[^A-Z0-9\-/]", "", heat_text.upper())}
        values = {}
        for i, col in enumerate(cols):
            if i == heat_idx or not col["key"] or i not in cells:
                continue
            raw = " ".join(cells[i])
            v = parse_number(raw)
            if v is None:
                continue
            if kind == "chem" and v >= 5:
                # no element in carbon steel is >= 5 wt%: OCR dropped the decimal point ('028' = 0.28)
                digits = re.sub(r"\D", "", raw)
                v = float("0." + digits.lstrip("0")) if digits.startswith("0") else None
                if v is None:
                    continue
            if col["unit"] == "ksi":
                v = round(v * 6.894757, 1)
            elif col["key"] in ("yield_mpa", "tensile_mpa") and v < 150:
                # no structural/pressure steel has strength < 150 MPa: the unit header was misread, it's ksi
                v = round(v * 6.894757, 1)
            values[col["key"]] = v
        row["values"] = values
        rows.append(row)
    return rows


def snap_heats(rows: List[dict], canonical: List[str], min_ratio: float = 0.6) -> None:
    """Replace OCR-garbled heat numbers ('BOL65', '13K9151016') with the closest heat listed
    elsewhere on the certificate (cross-field consistency)."""
    if not canonical:
        return
    for r in rows:
        if r["heat_no"] in canonical:
            continue
        best = max(canonical, key=lambda c: difflib.SequenceMatcher(None, r["heat_no"], c).ratio())
        if difflib.SequenceMatcher(None, r["heat_no"], best).ratio() >= min_ratio:
            r["heat_no"] = best


def _heat_list(text: Optional[str]) -> List[str]:
    if not text:
        return []
    toks = [re.sub(r"[^A-Z0-9\-/]", "", t.upper()) for t in re.split(r"[,;&]|\s+AND\s+|\s{2,}", text)]
    return [t for t in toks if re.search(r"\d", t) and len(t) >= 4]


def merge_heats(chem_rows: List[dict], mech_rows: List[dict]) -> List[dict]:
    heats: Dict[str, dict] = {}
    for r in chem_rows:
        heats.setdefault(r["heat_no"], {"heat_no": r["heat_no"], "chemistry": {}, "mechanical": {}})["chemistry"].update(r["values"])
    for r in mech_rows:
        key = r["heat_no"]
        if key not in heats and len(heats) == 1 and len(mech_rows) == 1:
            key = next(iter(heats))  # single heat, OCR disagreed on its number in one table
        heats.setdefault(key, {"heat_no": key, "chemistry": {}, "mechanical": {}})["mechanical"].update(r["values"])
    return list(heats.values())


# ----------------------------------------------------------------------------- layout fallback

_TITLE_PATTERNS = {
    "chem_table": re.compile(r"chemical", re.I),
    "mech_table": re.compile(r"mechanical|tensile test|tensile properties", re.I),
}
_END_PATTERN = re.compile(r"certify|remarks|authori[sz]ed|signatory", re.I)


def rule_regions(words: List[Word]) -> List[Region]:
    """Keyword-based layout when no detector is available (and as an ablation baseline)."""
    lines = group_lines(words)
    if not lines:
        return []
    marks = []
    for i, line in enumerate(lines):
        text = line_text(line)
        for cls, pat in _TITLE_PATTERNS.items():
            if pat.search(text) and not any(m[1] == cls for m in marks):
                marks.append((i, cls))
    end = next((i for i, l in enumerate(lines) if _END_PATTERN.search(line_text(l))), len(lines))
    marks.sort()
    regions = []

    def span(a: int, b: int, cls: str) -> None:
        ws = [w for l in lines[a:b] for w in l]
        if ws:
            regions.append(Region(cls, (min(w.box[0] for w in ws) - 5, min(w.box[1] for w in ws) - 5,
                                        max(w.box[2] for w in ws) + 5, max(w.box[3] for w in ws) + 5), 0.5))

    first_table = marks[0][0] if marks else end
    span(0, first_table, "info_block")
    for k, (i, cls) in enumerate(marks):
        stop = marks[k + 1][0] if k + 1 < len(marks) else max(end, i + 1)
        span(i, stop, cls)
    return regions


# ----------------------------------------------------------------------------- document

def _best(regions: List[Region], cls: str) -> Optional[Region]:
    cands = [r for r in regions if r.cls == cls]
    return max(cands, key=lambda r: r.conf) if cands else None


def extract_document(words: List[Word], regions: List[Region]) -> dict:
    info_r, chem_r, mech_r = (_best(regions, c) for c in ("info_block", "chem_table", "mech_table"))
    chem_words = words_in(words, chem_r) if chem_r else []
    mech_words = words_in(words, mech_r) if mech_r else []

    info = parse_info(words_in(words, info_r)) if info_r else {}
    # fill anything the info block missed from the rest of the page (outside the tables)
    table_ids = {id(w) for w in chem_words + mech_words}
    rest = [w for w in words if id(w) not in table_ids]
    for k, v in parse_info(rest).items():
        info.setdefault(k, v)

    chem_rows, mech_rows = parse_table(chem_words, "chem"), parse_table(mech_words, "mech")
    canonical = _heat_list(info.get("heat_list")) or [r["heat_no"] for r in chem_rows]
    snap_heats(chem_rows, canonical)
    snap_heats(mech_rows, canonical)
    heats = merge_heats(chem_rows, mech_rows)
    doc = {f: info.get(f) for f in INFO_FIELDS}
    doc["grade"] = parse_grade(info.get("spec"), info.get("grade"))
    doc["spec_key"] = resolve_spec_key(info.get("spec"), info.get("grade"))
    doc["heats"] = heats
    return doc
