"""Synthetic Material Test Report (MTR) generator.

Real MTRs are customer documents and no public dataset exists, so we render our own.
Every page comes with free, exact labels:
  * YOLO bounding boxes for 7 layout classes (title, info block, chemistry table,
    mechanical table, stamp, signature, remarks)
  * word-level boxes + text (for OCR error measurement and oracle evaluation)
  * the structured ground-truth record and its expected compliance verdict

Variation: 4 specs, label synonyms, 1-2 heats, 3 table grid styles, MPa / N/mm2 / ksi
units, optional spec-limit rows and hardness columns, swapped table order, 3 font
families, ~25% deliberately out-of-spec documents, and scan-like augmentation
(skew, blur, noise, JPEG, low-dpi resampling, paper tint).

Usage:
    python -m certiscan.synth.generate --out data/synth --n 600 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
import string
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from certiscan.fonts import available_families, get_font
from certiscan.specs import load_specs
from certiscan.types import REGION_CLASSES
from certiscan.validate import validate_document

W, H = 1700, 2200
KSI = 6.894757

# All company names below are fictional.
MILLS = [
    ("Narmada Tube Works Pvt. Ltd.", "Plot 14, Industrial Area, Pithampur, MP, India"),
    ("Great Lakes Pipe & Tube Co.", "2200 Harbor Road, Cleveland, OH 44114, USA"),
    ("Rheinland Rohrwerk GmbH", "Industriestrasse 8, 47057 Duisburg, Germany"),
    ("Coastal Alloy Mills Inc.", "915 Gulf Freeway, Houston, TX 77017, USA"),
    ("Northern Plate Industries Ltd.", "48 Mill Lane, Sheffield S9 1XY, United Kingdom"),
    ("Sierra Seamless Pipe S.A.", "Av. Industrial 300, Monterrey, NL, Mexico"),
    ("Kaveri Steel & Alloys Ltd.", "Survey No. 112, Hosur Road, Karnataka, India"),
]
CUSTOMERS = [
    "Midwest Industrial Supply", "Apex Fabricators LLC", "Bharat Process Engineering",
    "Delta Piping Contractors", "Summit Pressure Vessels", "Keystone Metals Dist.",
    "Orion Energy Projects", "Lakeside Structural Inc.",
]
TITLES = ["MILL TEST CERTIFICATE", "MATERIAL TEST REPORT", "INSPECTION CERTIFICATE",
          "CERTIFIED MILL TEST REPORT", "TEST CERTIFICATE EN 10204 3.1"]
LABELS: Dict[str, List[str]] = {
    "cert_no": ["Certificate No.", "Cert. No.", "MTC No.", "Test Certificate No.", "Report No."],
    "date": ["Date", "Issue Date", "Date of Issue"],
    "customer": ["Customer", "Purchaser", "Sold To"],
    "po_no": ["P.O. No.", "PO No.", "Purchase Order", "Customer PO"],
    "heat_list": ["Heat No.", "Heat No(s).", "Heat Number", "Melt No."],
    "spec": ["Specification", "Spec.", "Material Spec."],
    "grade": ["Grade", "Steel Grade", "Material Grade"],
    "size": ["Size", "Dimensions", "Product Size"],
    "quantity": ["Quantity", "Qty."],
}
SPEC_TEXT = {
    "A106-B": (["ASTM A106/A106M-19a", "ASTM A106", "ASME SA106", "ASTM A106/ASME SA106"], "B", "pipe"),
    "A53-B": (["ASTM A53/A53M-22", "ASTM A53", "ASME SA53"], "B", "pipe"),
    "A36": (["ASTM A36/A36M-19", "ASTM A36", "ASME SA36"], None, "plate"),
    "A516-70": (["ASTM A516/A516M-17", "ASTM A516", "ASME SA516"], "70", "plate"),
}
SIZES = {
    "pipe": ['2" SCH 40 x 21 FT', '4" SCH 80 x 6 M', '6" SCH STD x 20 FT', "114.3 x 6.02 mm x 6 M", '8" SCH 40 x 40 FT'],
    "plate": ["10 mm x 2000 x 6000 mm", "12 mm x 2500 x 8000 mm", "8 mm x 1500 x 6000 mm", '3/8" x 96" x 240"'],
}
ELEMENT_ORDER = ["C", "Mn", "P", "S", "Si", "Cu", "Ni", "Cr", "Mo", "V", "Nb", "Ti", "Al"]
DECIMALS = {"C": 2, "Mn": 2, "Si": 2, "Cu": 2, "Ni": 2, "Cr": 2, "Mo": 2,
            "P": 3, "S": 3, "V": 3, "Nb": 3, "Ti": 3, "Al": 3}
EXTRA_RANGE = {"Si": (0.15, 0.35), "Cu": (0.02, 0.25), "Ni": (0.01, 0.15), "Cr": (0.02, 0.20),
               "Mo": (0.006, 0.05), "V": (0.001, 0.010), "Nb": (0.001, 0.020), "Ti": (0.001, 0.020),
               "Al": (0.015, 0.045)}
MECH_HEADERS = {
    "yield_mpa": {"mpa": ["Yield Strength (MPa)", "YS (MPa)"], "nmm": ["0.2% Proof (N/mm2)", "Yield (N/mm2)"],
                  "ksi": ["Yield Strength (ksi)", "YS (ksi)"]},
    "tensile_mpa": {"mpa": ["Tensile Strength (MPa)", "UTS (MPa)"], "nmm": ["Tensile (N/mm2)", "UTS (N/mm2)"],
                    "ksi": ["Tensile Strength (ksi)", "UTS (ksi)"]},
    "elongation_pct": ["Elongation (%)", "El. (%)", "Elong. % (2in)"],
}


# ----------------------------------------------------------------------------- values

def _rand_id(rng: random.Random, pattern: str) -> str:
    return "".join(rng.choice(string.ascii_uppercase) if c == "A" else str(rng.randint(0, 9)) if c == "9" else c
                   for c in pattern)


def _in_spec(rng: random.Random, lim: dict) -> float:
    mn, mx = lim.get("min"), lim.get("max")
    if mn is not None and mx is not None:
        return rng.uniform(mn + 0.1 * (mx - mn), mx - 0.1 * (mx - mn))
    if mx is not None:
        return rng.uniform(0.3 * mx, 0.9 * mx)
    return rng.uniform(mn * 1.15, mn * 2.5)


def _out_of_spec(rng: random.Random, lim: dict) -> float:
    if lim.get("max") is not None and (lim.get("min") is None or rng.random() < 0.6):
        return lim["max"] * rng.uniform(1.1, 1.35)
    return lim["min"] * rng.uniform(0.6, 0.88)


def _fmt_chem(el: str, v: float) -> str:
    return f"{v:.{DECIMALS.get(el, 3)}f}"


def make_record(rng: random.Random, specs: dict) -> dict:
    """Sample the content of one certificate (no rendering yet)."""
    key = rng.choice(sorted(specs))
    spec = specs[key]
    bases, grade, kind = SPEC_TEXT[key]
    n_heats = rng.choice([1, 1, 1, 2])
    heat_pat = rng.choice(["A99999", "99A9999", "H999999", "9999A99"])
    heats = []
    while len(heats) < n_heats:
        h = _rand_id(rng, heat_pat)
        if h not in heats:
            heats.append(h)

    chem_lims = spec["chemistry"]
    required = [e for e, l in chem_lims.items() if l.get("required")]
    optional = [e for e in chem_lims if e not in required]
    extras = [e for e in EXTRA_RANGE if e not in chem_lims]
    els = set(required) | set(rng.sample(optional, rng.randint(0, len(optional))))
    els |= set(rng.sample(extras, rng.randint(0, min(2, len(extras)))))
    elements = [e for e in ELEMENT_ORDER if e in els]

    mech_lims = spec["mechanical"]
    unit = rng.choice(["mpa", "mpa", "nmm", "ksi"])
    elong_dec = rng.choice([0, 1])

    heat_rows = []
    for h in heats:
        chem_raw = {e: (_in_spec(rng, chem_lims[e]) if e in chem_lims else rng.uniform(*EXTRA_RANGE[e])) for e in elements}
        ymin = mech_lims["yield_mpa"]["min"]
        y = ymin * rng.uniform(1.05, 1.4)
        tl = mech_lims["tensile_mpa"]
        lo = max(tl["min"] * 1.03, y * 1.15)
        hi = tl["max"] * 0.97 if "max" in tl else tl["min"] * 1.45
        t = rng.uniform(min(lo, hi * 0.97), hi)
        el_min = mech_lims.get("elongation_pct", {}).get("min", 18)
        e = rng.uniform(max(el_min * 1.1, 22), 38)
        heat_rows.append({"heat_no": h, "chem_raw": chem_raw, "mech_raw": {"yield_mpa": y, "tensile_mpa": t, "elongation_pct": e}})

    if rng.random() < 0.25:  # inject a non-conformance
        row = rng.choice(heat_rows)
        targets = [("chem", e) for e in elements if e in chem_lims] + [("mech", m) for m in mech_lims]
        section, name = rng.choice(targets)
        lims = chem_lims if section == "chem" else mech_lims
        row["chem_raw" if section == "chem" else "mech_raw"][name] = _out_of_spec(rng, lims[name])

    # render strings and derive the exact ground truth from what is printed
    out_heats = []
    for row in heat_rows:
        chem_txt = {e: _fmt_chem(e, v) for e, v in row["chem_raw"].items()}
        mech_txt, mech_gt = {}, {}
        for m, v in row["mech_raw"].items():
            if m == "elongation_pct":
                s = f"{v:.{elong_dec}f}"
                mech_gt[m] = float(s)
            elif unit == "ksi":
                s = f"{v / KSI:.1f}"
                mech_gt[m] = round(float(s) * KSI, 1)
            else:
                s = str(int(round(v)))
                mech_gt[m] = float(s)
            mech_txt[m] = s
        out_heats.append({"heat_no": row["heat_no"], "chem_txt": chem_txt, "mech_txt": mech_txt,
                          "chemistry": {e: float(s) for e, s in chem_txt.items()}, "mechanical": mech_gt})

    base = rng.choice(bases)
    combined = grade is None or rng.random() < 0.6
    grade_style = rng.choice(["Gr. {}", "Grade {}", "GR {}"]) if grade else None
    spec_value = f"{base} {grade_style.format(grade)}" if grade and combined else base
    grade_value = None if combined else rng.choice(["{}", "Gr. {}", "Grade {}"]).format(grade)

    day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2024, 2026)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    date = rng.choice([f"{day:02d}-{months[month - 1]}-{year}", f"{year}-{month:02d}-{day:02d}",
                       f"{month:02d}/{day:02d}/{year}", f"{day:02d}.{month:02d}.{year}"])

    fields = {
        "cert_no": _rand_id(rng, rng.choice(["MTC-2025-99999", "AA999999", "TC/9999/99", "99999-AA"])),
        "date": date,
        "customer": rng.choice(CUSTOMERS),
        "po_no": _rand_id(rng, rng.choice(["PO-999999", "4500999999", "AAA/99999"])),
        "spec": spec_value,
        "grade": grade,
        "spec_key": key,
        "size": rng.choice(SIZES[kind]),
        "quantity": f"{rng.randint(4, 120)} PCS / {rng.randint(800, 24000)} KG",
    }
    gt = dict(fields)
    gt["heats"] = [{"heat_no": h["heat_no"], "chemistry": h["chemistry"], "mechanical": h["mechanical"]} for h in out_heats]
    gt["compliance"] = validate_document(gt, specs)
    return {"gt": gt, "fields": fields, "grade_value": grade_value, "heats": out_heats,
            "elements": elements, "unit": unit, "spec_limits": spec}


# ----------------------------------------------------------------------------- rendering

class Canvas:
    def __init__(self, rng: random.Random, family: str):
        self.rng = rng
        self.family = family
        self.img = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.img)
        self.words: List[dict] = []
        self.regions: List[dict] = []
        self._stack: List[list] = []

    def font(self, size: int, bold: bool = False):
        return get_font(self.family + ("-bold" if bold else ""), size)

    def _track(self, box) -> None:
        for boxes in self._stack:
            boxes.append(box)

    def textlen(self, s: str, size: int, bold: bool = False) -> float:
        return self.d.textlength(s, font=self.font(size, bold))

    def text(self, x: float, y: float, s: str, size: int, bold: bool = False, center: bool = False,
             fill=(25, 25, 25), record: bool = True) -> float:
        """Draw word by word so each word's exact box is known. Returns the right edge."""
        f = self.font(size, bold)
        total = self.d.textlength(s, font=f)
        if center:
            x -= total / 2
        cursor = x
        for tok in s.split(" "):
            if tok:
                box = self.d.textbbox((cursor, y), tok, font=f)
                self.d.text((cursor, y), tok, font=f, fill=fill)
                if record:
                    self.words.append({"text": tok, "box": [float(v) for v in box]})
                self._track(box)
            cursor += self.d.textlength(tok + " ", font=f)
        return x + total

    def line(self, x0, y0, x1, y1, width: int = 2, fill=(40, 40, 40)) -> None:
        self.d.line([(x0, y0), (x1, y1)], fill=fill, width=width)
        self._track((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    @contextmanager
    def region(self, cls: str, pad: int = 8):
        boxes: list = []
        self._stack.append(boxes)
        yield
        self._stack.pop()
        if boxes:
            x0 = max(0, min(b[0] for b in boxes) - pad)
            y0 = max(0, min(b[1] for b in boxes) - pad)
            x1 = min(W, max(b[2] for b in boxes) + pad)
            y1 = min(H, max(b[3] for b in boxes) + pad)
            self.regions.append({"cls": cls, "box": [float(x0), float(y0), float(x1), float(y1)]})
            for outer in self._stack:  # nested regions count toward their parent
                outer.append((x0, y0, x1, y1))


def _table(cv: Canvas, x: float, y: float, width: float, title: str, header: List[str], rows: List[List[str]],
           fs: int, style: str) -> float:
    """Render a ruled/unruled table with centred cells. Returns the bottom y."""
    rng = cv.rng
    cv.text(x, y, title, fs + 2, bold=True)
    y += int(fs * 2.0)
    natural = [max(cv.textlen(h, fs, True), *(cv.textlen(r[i], fs) for r in rows)) for i, h in enumerate(header)]
    pad = 2.4 * fs
    if sum(natural) + pad * len(natural) + fs > width:  # wide table: tighten (but keep gaps > 1 text height)
        pad = max(1.5 * fs, (width - fs - sum(natural)) / len(natural))
    widths = [w + pad for w in natural]
    widths[0] += fs
    extra = width - sum(widths)
    if extra > 0:
        widths = [w + extra / len(widths) for w in widths]
    row_h = int(fs * rng.uniform(1.8, 2.2))
    xs = [x]
    for w in widths:
        xs.append(xs[-1] + w)
    n = len(rows) + 1
    top, bottom = y, y + n * row_h
    if style == "full":
        for i in range(n + 1):
            cv.line(xs[0], top + i * row_h, xs[-1], top + i * row_h)
        for xv in xs:
            cv.line(xv, top, xv, bottom)
    elif style == "hlines":
        cv.line(xs[0], top, xs[-1], top, 3)
        cv.line(xs[0], top + row_h, xs[-1], top + row_h)
        cv.line(xs[0], bottom, xs[-1], bottom, 3)
    for r, cells in enumerate([header] + rows):
        ty = top + r * row_h + (row_h - fs) / 2 - 2
        for i, cell in enumerate(cells):
            cv.text((xs[i] + xs[i + 1]) / 2, ty, cell, fs, bold=(r == 0), center=True)
    return bottom


def _scribble(cv: Canvas, x: float, y: float) -> None:
    rng = cv.rng
    pts = []
    a1, a2, f1, f2 = rng.uniform(10, 25), rng.uniform(5, 15), rng.uniform(0.02, 0.05), rng.uniform(0.08, 0.15)
    for i in range(0, 260, 3):
        pts.append((x + i, y + 40 + a1 * math.sin(i * f1) + a2 * math.sin(i * f2 + 1.3)))
    cv.d.line(pts, fill=(20, 35, 120), width=3, joint="curve")
    cv._track((x, y, x + 260, y + 80))


def _stamp(cv: Canvas, cx: float, cy: float) -> None:
    rng = cv.rng
    r = rng.randint(80, 110)
    color = rng.choice([(190, 30, 30), (30, 60, 170), (110, 30, 140)])
    layer = Image.new("RGBA", (2 * r + 20, 2 * r + 20), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    c = r + 10
    ld.ellipse([c - r, c - r, c + r, c + r], outline=color + (200,), width=5)
    ld.ellipse([c - r + 12, c - r + 12, c + r - 12, c + r - 12], outline=color + (200,), width=2)
    f = cv.font(int(r * 0.28), bold=True)
    for txt, dy in (("QA DEPT", -r * 0.35), ("APPROVED", 0), (str(rng.randint(10, 99)), r * 0.35)):
        tw = ld.textlength(txt, font=f)
        ld.text((c - tw / 2, c + dy - r * 0.14), txt, font=f, fill=color + (210,))
    layer = layer.rotate(rng.uniform(-25, 25), resample=Image.BICUBIC)
    cv.img.paste(layer, (int(cx - c), int(cy - c)), layer)
    cv._track((cx - r, cy - r, cx + r, cy + r))


def render(rng: random.Random, rec: dict) -> Canvas:
    family = rng.choice(available_families() or ["sans"])
    cv = Canvas(rng, family)
    m = rng.randint(70, 110)
    y = rng.randint(60, 110)
    mill, address = rng.choice(MILLS)
    f = rec["fields"]

    with cv.region("title_block"):
        x = m
        if rng.random() < 0.7:
            initials = "".join(wd[0] for wd in mill.split()[:2]).upper()
            box = [m, y, m + 96, y + 96]
            (cv.d.ellipse if rng.random() < 0.5 else cv.d.rectangle)(box, outline=(30, 60, 120), width=5)
            cv._track(box)
            cv.text(m + 48, y + 30, initials, 34, bold=True, center=True, fill=(30, 60, 120))
            x = m + 120
        cv.text(x, y + 4, mill, rng.randint(36, 46), bold=True)
        cv.text(x, y + 58, address, rng.randint(19, 23))
        y += 130
        title = rng.choice(TITLES)
        if rng.random() < 0.6:
            cv.text(W / 2, y, title, rng.randint(34, 40), bold=True, center=True)
        else:
            cv.text(m, y, title, rng.randint(34, 40), bold=True)
        y += 70

    # ---- info block
    order = ["cert_no", "date", "customer", "po_no", "heat_list", "spec"]
    if rec["grade_value"]:
        order.append("grade")
    order += ["size", "quantity"]
    head, tail = order[:2], order[2:]
    rng.shuffle(tail)
    order = head + tail
    values = dict(f)
    values["heat_list"] = ", ".join(h["heat_no"] for h in rec["heats"])
    values["grade"] = rec["grade_value"]
    labels = {k: rng.choice(LABELS[k]) for k in order}
    two_col = rng.random() < 0.6
    fs = rng.randint(22, 25) if two_col else rng.randint(22, 28)
    style = rng.choice(["inline", "aligned_colon", "aligned"])
    lw = max(cv.textlen(labels[k] + ":", fs, True) for k in order) + 1.8 * fs
    row_h = int(fs * rng.uniform(1.7, 2.0))
    c2 = m + (W - 2 * m) * 0.55
    with cv.region("info_block"):
        for i, k in enumerate(order):
            col, row = (i % 2, i // 2) if two_col else (0, i)
            x = m if col == 0 else c2
            yy = y + row * row_h
            label, value = labels[k], values[k]
            if style == "inline":
                end = cv.text(x, yy, label + ":", fs, bold=True)
                cv.text(end + fs * 0.4, yy, value, fs)
            else:
                cv.text(x, yy, label + (":" if style == "aligned_colon" else ""), fs, bold=True)
                cv.text(x + lw, yy, value, fs)
        rows_used = math.ceil(len(order) / 2) if two_col else len(order)
    y += rows_used * row_h + 40

    # ---- tables
    width = W - 2 * m
    tfs = rng.randint(20, 24)
    grid = rng.choice(["full", "full", "hlines", "none"])
    heat_hdr = rng.choice(["Heat No.", "Heat", "Heat #", "Cast No."])

    def chem_table(yy: float) -> float:
        els = rec["elements"]
        rows = [[h["heat_no"]] + [h["chem_txt"][e] for e in els] for h in rec["heats"]]
        if rng.random() < 0.3:
            lims = rec["spec_limits"]["chemistry"]
            limit_row = [rng.choice(["Max", "Spec Max", "Max."])] + [
                _fmt_chem(e, lims[e]["max"]) if e in lims and "max" in lims[e] else "-" for e in els]
            rows = [limit_row] + rows if rng.random() < 0.5 else rows + [limit_row]
        title = rng.choice(["CHEMICAL COMPOSITION (%)", "CHEMICAL ANALYSIS (wt%)", "Chemical Composition - Heat Analysis (%)"])
        with cv.region("chem_table"):
            return _table(cv, m, yy, width, title, [heat_hdr] + els, rows, tfs, grid)

    def mech_table(yy: float) -> float:
        unit = rec["unit"]
        cols = ["yield_mpa", "tensile_mpa", "elongation_pct"]
        header = [heat_hdr] + [rng.choice(MECH_HEADERS[c][unit]) for c in cols[:2]] + [rng.choice(MECH_HEADERS["elongation_pct"])]
        rows = [[h["heat_no"]] + [h["mech_txt"][c] for c in cols] for h in rec["heats"]]
        if rng.random() < 0.4:
            header.append(rng.choice(["Hardness (HRB)", "HB"]))
            for r in rows:
                r.append(str(rng.randint(70, 90)))
        title = rng.choice(["MECHANICAL PROPERTIES", "TENSILE TEST RESULTS", "Mechanical Properties"])
        with cv.region("mech_table"):
            return _table(cv, m, yy, width * rng.uniform(0.75, 1.0), title, header, rows, tfs, grid)

    tables = [chem_table, mech_table] if rng.random() < 0.75 else [mech_table, chem_table]
    for tbl in tables:
        y = tbl(y) + rng.randint(45, 70)

    # ---- remarks
    with cv.region("remarks"):
        rfs = rng.randint(19, 22)
        cv.text(m, y, "Remarks:", rfs, bold=True)
        sentences = [
            "We hereby certify that the material described herein has been manufactured, sampled and tested",
            "in accordance with the specification and the results are found satisfactory.",
        ]
        if rng.random() < 0.5:
            sentences.append("Material is free from mercury contamination. No weld repair performed.")
        for i, s in enumerate(sentences):
            cv.text(m, y + (i + 1) * rfs * 1.5, s, rfs)
        y += (len(sentences) + 1) * rfs * 1.5 + 50

    # ---- signature + stamp
    sig_right = rng.random() < 0.6
    sx = W - m - 320 if sig_right else m + 20
    sy = min(y, H - 260)
    with cv.region("signature"):
        _scribble(cv, sx, sy)
        cv.text(sx, sy + 95, rng.choice(["Authorized Signatory", "QA Manager", "Surveyor / Inspector"]), 21)
    stx = (m + 220) if sig_right else (W - m - 220)
    if rng.random() < 0.3:
        stx = sx + rng.randint(120, 260)  # stamp overlapping the signature, as often happens
    with cv.region("stamp"):
        _stamp(cv, stx, sy + 70)
    return cv


# ----------------------------------------------------------------------------- augmentation

def _transform_box(box, mat) -> List[float]:
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=np.float64)
    t = pts @ mat.T
    return [float(max(0, t[:, 0].min())), float(max(0, t[:, 1].min())),
            float(min(W, t[:, 0].max())), float(min(H, t[:, 1].max()))]


def augment(rng: random.Random, img: Image.Image, regions: List[dict]):
    arr = np.asarray(img).astype(np.float32)
    tint = np.array([rng.uniform(0.93, 1.0), rng.uniform(0.92, 1.0), rng.uniform(0.86, 1.0)], dtype=np.float32)
    arr = arr * tint
    angle = rng.uniform(-2.0, 2.0) if rng.random() < 0.7 else 0.0
    mat = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    if angle:
        arr = cv2.warpAffine(arr, mat, (W, H), borderValue=(250, 250, 248))
    if rng.random() < 0.3:
        s = rng.uniform(0.55, 0.8)
        arr = cv2.resize(cv2.resize(arr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        arr = cv2.GaussianBlur(arr, (k, k), 0)
    alpha, beta = rng.uniform(0.85, 1.1), rng.uniform(-20, 12)
    arr = arr * alpha + beta
    if rng.random() < 0.6:
        arr = arr + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 10), arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    regs = [{"cls": r["cls"], "box": _transform_box(r["box"], mat)} for r in regions]
    return arr, regs, angle


# ----------------------------------------------------------------------------- dataset

def generate_one(args) -> dict:
    idx, out, split, seed, do_aug = args
    rng = random.Random(seed * 100003 + idx)
    specs = load_specs()
    rec = make_record(rng, specs)
    cv = render(rng, rec)
    name = f"mtr_{idx:05d}"
    if do_aug:
        arr, regions, angle = augment(rng, cv.img, cv.regions)
        quality = rng.randint(45, 95)
    else:
        arr, regions, angle, quality = np.asarray(cv.img), cv.regions, 0.0, 95
    out = Path(out)
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])
    with open(out / "labels" / split / f"{name}.txt", "w") as fh:
        for r in regions:
            x0, y0, x1, y1 = r["box"]
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            fh.write(f"{REGION_CLASSES.index(r['cls'])} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
    gt = {"id": name, "split": split, "size": [W, H], "skew_applied": angle, "fields": rec["gt"],
          "words_clean": cv.words, "regions_clean": cv.regions, "regions": regions}
    (out / "gt" / split / f"{name}.json").write_text(json.dumps(gt), encoding="utf-8")
    return {"id": name, "status": rec["gt"]["compliance"]["status"]}


def build_dataset(out: str, n: int, seed: int = 42, augment_images: bool = True, workers: int = 1,
                  splits=(0.7, 0.15, 0.15)) -> dict:
    out_p = Path(out).resolve()
    names = ["train", "val", "test"]
    for sub in ("images", "labels", "gt"):
        for s in names:
            (out_p / sub / s).mkdir(parents=True, exist_ok=True)
    n_train, n_val = int(n * splits[0]), int(n * splits[1])
    jobs = [(i, str(out_p), "train" if i < n_train else "val" if i < n_train + n_val else "test", seed, augment_images)
            for i in range(n)]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(generate_one, jobs, chunksize=8)
    else:
        results = [generate_one(j) for j in jobs]
    yaml = [f"path: {out_p.as_posix()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(REGION_CLASSES)]
    (out_p / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    counts: Dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"n": n, "out": str(out_p), "compliance_counts": counts}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-augment", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build_dataset(a.out, a.n, a.seed, not a.no_augment, a.workers), indent=2))


if __name__ == "__main__":
    main()
