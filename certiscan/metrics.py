"""Evaluation metrics: field-level P/R/F1, per-field accuracy, compliance accuracy, CER."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from certiscan.textlayout import group_lines, line_text
from certiscan.types import Region, Word

SCALAR_FIELDS = ["cert_no", "date", "customer", "po_no", "spec", "grade", "spec_key", "size", "quantity"]


def norm_text(value) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper()) if value not in (None, "") else ""


def flatten(doc: dict) -> List[Tuple[str, str]]:
    """Leaf (key, value) pairs; numbers rounded to the precision they are printed with."""
    pairs = []
    for f in SCALAR_FIELDS:
        v = norm_text(doc.get(f))
        if v:
            pairs.append((f, v))
    for heat in doc.get("heats") or []:
        h = norm_text(heat.get("heat_no"))
        pairs.append(("heat_no", h))
        for el, v in (heat.get("chemistry") or {}).items():
            if v is not None:
                pairs.append((f"chem.{el}@{h}", f"{v:.4f}"))
        for k, v in (heat.get("mechanical") or {}).items():
            if v is not None:
                pairs.append((f"mech.{k}@{h}", f"{round(v):d}" if k != "elongation_pct" else f"{v:.1f}"))
    return pairs


def group_of(key: str) -> str:
    if key.startswith("chem."):
        return "chemistry"
    if key.startswith("mech."):
        return "mechanical"
    return key


class Scorer:
    def __init__(self):
        self.tp = self.n_pred = self.n_gt = 0
        self.group_tp: Dict[str, int] = defaultdict(int)
        self.group_gt: Dict[str, int] = defaultdict(int)
        self.docs = self.docs_exact = 0
        self.compliance_correct = 0
        self.confusion: Counter = Counter()
        self.cer_edits = self.cer_chars = 0

    def add(self, pred: dict, gt: dict) -> dict:
        p, g = Counter(flatten(pred)), Counter(flatten(gt))
        inter = p & g
        tp = sum(inter.values())
        self.tp += tp
        self.n_pred += sum(p.values())
        self.n_gt += sum(g.values())
        for (k, _), c in g.items():
            self.group_gt[group_of(k)] += c
        for (k, _), c in inter.items():
            self.group_tp[group_of(k)] += c
        self.docs += 1
        exact = p == g
        self.docs_exact += exact
        ps = (pred.get("compliance") or {}).get("status")
        gs = (gt.get("compliance") or {}).get("status")
        self.compliance_correct += ps == gs
        self.confusion[(gs, ps)] += 1
        missed = sorted(set(g) - set(p))
        wrong = sorted(set(p) - set(g))
        return {"exact": exact, "compliance_ok": ps == gs, "missed": missed[:10], "spurious": wrong[:10]}

    def add_cer(self, edits: int, chars: int) -> None:
        self.cer_edits += edits
        self.cer_chars += chars

    def summary(self) -> dict:
        prec = self.tp / self.n_pred if self.n_pred else 0.0
        rec = self.tp / self.n_gt if self.n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out = {
            "documents": self.docs,
            "field_precision": round(prec, 4),
            "field_recall": round(rec, 4),
            "field_f1": round(f1, 4),
            "document_exact_match": round(self.docs_exact / self.docs, 4) if self.docs else 0.0,
            "compliance_accuracy": round(self.compliance_correct / self.docs, 4) if self.docs else 0.0,
            "per_field_recall": {k: round(self.group_tp[k] / v, 4) for k, v in sorted(self.group_gt.items())},
            "compliance_confusion": {f"{g}->{p}": c for (g, p), c in sorted(self.confusion.items(), key=str)},
        }
        if self.cer_chars:
            out["cer_data_regions"] = round(self.cer_edits / self.cer_chars, 4)
        return out


def levenshtein(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return Levenshtein.distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def region_text(words: Iterable[Word], region: Region) -> str:
    inside = [w for w in words if region.contains(w)]
    return "\n".join(line_text(l) for l in group_lines(inside))


def cer(ocr_words: List[Word], gt_words: List[Word], regions: List[Region]) -> Tuple[int, int]:
    """Character edits between OCR and ground-truth text inside the given regions."""
    edits = chars = 0
    for r in regions:
        ref = region_text(gt_words, r).replace(" ", "")
        hyp = region_text(ocr_words, r).replace(" ", "")
        edits += levenshtein(hyp, ref)
        chars += len(ref)
    return edits, chars
