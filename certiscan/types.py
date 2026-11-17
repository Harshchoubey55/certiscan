"""Core data types shared by OCR, layout detection and extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

Box = Tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels

REGION_CLASSES = [
    "title_block",
    "info_block",
    "chem_table",
    "mech_table",
    "stamp",
    "signature",
    "remarks",
]


@dataclass
class Word:
    """A piece of OCR'd text with its bounding box. May be one word or a short phrase."""

    text: str
    box: Box
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def h(self) -> float:
        return max(1.0, self.box[3] - self.box[1])

    @property
    def w(self) -> float:
        return max(1.0, self.box[2] - self.box[0])

    def to_dict(self) -> dict:
        return {"text": self.text, "box": [round(v, 1) for v in self.box], "conf": round(self.conf, 3)}

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(d["text"], tuple(d["box"]), d.get("conf", 1.0))


@dataclass
class Region:
    """A detected layout block (table, info block, stamp, ...)."""

    cls: str
    box: Box
    conf: float = 1.0

    def contains(self, word: Word) -> bool:
        x0, y0, x1, y1 = self.box
        return x0 <= word.cx <= x1 and y0 <= word.cy <= y1

    def to_dict(self) -> dict:
        return {"cls": self.cls, "box": [round(v, 1) for v in self.box], "conf": round(self.conf, 3)}

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        return cls(d["cls"], tuple(d["box"]), d.get("conf", 1.0))


@dataclass
class Segment:
    """Horizontally adjacent words merged into one cell-like unit."""

    words: List[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def box(self) -> Box:
        return (
            min(w.box[0] for w in self.words),
            min(w.box[1] for w in self.words),
            max(w.box[2] for w in self.words),
            max(w.box[3] for w in self.words),
        )

    @property
    def cx(self) -> float:
        b = self.box
        return (b[0] + b[2]) / 2
