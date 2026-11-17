"""Geometry helpers: group OCR words into lines and lines into cell-like segments.

These are deliberately independent of the OCR engine's granularity: EasyOCR returns
short phrases, Tesseract returns single words, and the synthetic ground truth returns
single words. All three end up as the same lines/segments.
"""
from __future__ import annotations

import re
import statistics
from typing import List, Optional

from certiscan.types import Region, Segment, Word


def median_height(words: List[Word]) -> float:
    return statistics.median([w.h for w in words]) if words else 20.0


def group_lines(words: List[Word], overlap: float = 0.5) -> List[List[Word]]:
    """Cluster words into text lines by vertical overlap. Assumes the page is deskewed."""
    lines: List[dict] = []
    for w in sorted(words, key=lambda w: (w.cy, w.box[0])):
        best, best_ov = None, 0.0
        for line in lines:
            top, bottom = line["y0"], line["y1"]
            inter = min(bottom, w.box[3]) - max(top, w.box[1])
            ov = inter / min(w.h, bottom - top)
            if ov > best_ov:
                best, best_ov = line, ov
        if best is not None and best_ov >= overlap:
            best["words"].append(w)
            # track a running median-ish band rather than the full union, so tall
            # words do not swallow neighbouring lines
            n = len(best["words"])
            best["y0"] = (best["y0"] * (n - 1) + w.box[1]) / n
            best["y1"] = (best["y1"] * (n - 1) + w.box[3]) / n
        else:
            lines.append({"y0": w.box[1], "y1": w.box[3], "words": [w]})
    lines.sort(key=lambda l: (l["y0"] + l["y1"]) / 2)
    return [sorted(l["words"], key=lambda w: w.box[0]) for l in lines]


def split_segments(line: List[Word], gap_factor: float = 1.0, ref_height: Optional[float] = None) -> List[Segment]:
    """Split a line into segments wherever the horizontal gap exceeds gap_factor * text height."""
    if not line:
        return []
    h = ref_height or median_height(line)
    segments = [Segment([line[0]])]
    for prev, w in zip(line, line[1:]):
        gap = w.box[0] - prev.box[2]
        if gap > gap_factor * h:
            segments.append(Segment([w]))
        else:
            segments[-1].words.append(w)
    return segments


def words_in(words: List[Word], region: Region) -> List[Word]:
    return [w for w in words if region.contains(w)]


def line_text(line: List[Word]) -> str:
    return " ".join(w.text for w in line)


def normalize_label(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[.:()\[\]]", "", t)
    t = re.sub(r"[^a-z0-9#/ ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_NUM_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", ",": "."})


def parse_number(text: str) -> Optional[float]:
    """Parse OCR'd numbers such as '0.18', '.035', '0,18', '62.5%', '0.O35'."""
    if text is None:
        return None
    t = text.strip().replace(" ", "").rstrip("%")
    if not t:
        return None
    if re.fullmatch(r"[0-9OolI,.\-]+", t):
        t = t.translate(_NUM_FIX)
    m = re.search(r"-?\d*\.?\d+", t)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None
