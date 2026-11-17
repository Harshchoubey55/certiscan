"""Loading spec limits and resolving a free-text specification to a spec key."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

SPECS_PATH = Path(__file__).parent / "data" / "specs.json"


@lru_cache(maxsize=4)
def load_specs(path: Optional[str] = None) -> dict:
    data = json.loads(Path(path or SPECS_PATH).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


_SPEC_RE = re.compile(r"\b(?:ASTM|ASME)?\s*S?A\s*-?\s*(\d{2,4})", re.IGNORECASE)
# OCR-tolerant fallback: 'ASTM ASI6' (S->5, I->1) is A516; needs at least one real digit
_SPEC_LOOSE_RE = re.compile(r"A\s*-?\s*([0-9SIlOBZ]{2,4})")
_DIGIT_FIX = str.maketrans({"S": "5", "I": "1", "l": "1", "O": "0", "B": "8", "Z": "2"})
# no leading \b: OCR often drops the space ('ASTMA516M17GR70')
_GRADE_RE = re.compile(r"GR(?:ADE)?\.?\s*([A-Z0-9]{1,3})\b", re.IGNORECASE)


def parse_spec_number(text: str) -> Optional[str]:
    """'ASTM A106/A106M-19a Gr. B' -> 'A106'.  'ASME SA516' -> 'A516'.  OCR'd 'ASTM ASI6' -> 'A516'."""
    if not text:
        return None
    m = _SPEC_RE.search(text)
    if m:
        return f"A{m.group(1)}"
    for m in _SPEC_LOOSE_RE.finditer(text.upper().replace("ASTM", " ").replace("ASME", " ")):
        if re.search(r"\d", m.group(1)):
            return "A" + m.group(1).translate(_DIGIT_FIX)
    return None


def parse_grade(spec_text: Optional[str], grade_text: Optional[str]) -> Optional[str]:
    if grade_text:
        g = grade_text.strip().upper()
        m = _GRADE_RE.search(g)
        if m:
            return m.group(1)
        g = re.sub(r"[^A-Z0-9]", "", g)
        return g or None
    if spec_text:
        m = _GRADE_RE.search(spec_text)
        if m:
            return m.group(1).upper()
    return None


def resolve_spec_key(spec_text: Optional[str], grade_text: Optional[str], specs: Optional[dict] = None) -> Optional[str]:
    """Map extracted spec + grade strings onto a key of specs.json (e.g. 'A106-B', 'A36')."""
    specs = specs if specs is not None else load_specs()
    number = parse_spec_number(spec_text or "")
    if not number:
        return None
    grade = parse_grade(spec_text, grade_text)
    if grade and f"{number}-{grade}" in specs:
        return f"{number}-{grade}"
    if number in specs:
        return number
    return f"{number}-{grade}" if grade else number
