"""Evaluate the pipeline on a synthetic split, isolating where errors come from.

    # parser only (perfect OCR + perfect layout): upper bound for the rules
    python scripts/evaluate.py --data data/synth --ocr oracle --layout gt
    # keyword layout instead of the detector
    python scripts/evaluate.py --data data/synth --ocr oracle --layout rules
    # full system
    python scripts/evaluate.py --data data/synth --ocr easyocr --layout yolo --weights runs/layout/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from certiscan.layout import RuleLayout, UltralyticsLayout  # noqa: E402
from certiscan.metrics import Scorer, cer  # noqa: E402
from certiscan.pipeline import Pipeline  # noqa: E402
from certiscan.preprocess import rotate  # noqa: E402
from certiscan.types import Region, Word  # noqa: E402

DATA_REGIONS = ("info_block", "chem_table", "mech_table")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ocr", default="oracle", choices=["oracle", "easyocr", "tesseract"])
    ap.add_argument("--layout", default="gt", choices=["gt", "rules", "yolo"])
    ap.add_argument("--weights")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    ocr = None
    if a.ocr != "oracle":
        from certiscan.ocr import get_engine

        ocr = get_engine(a.ocr)
    layout = UltralyticsLayout(a.weights) if a.layout == "yolo" else RuleLayout()
    pipe = Pipeline(layout=layout, ocr=ocr)

    files = sorted((Path(a.data) / "gt" / a.split).glob("*.json"))[: a.limit]
    scorer, details, t0 = Scorer(), [], time.perf_counter()
    for i, f in enumerate(files):
        gt = json.loads(f.read_text(encoding="utf-8"))
        gt_words = [Word(w["text"], tuple(w["box"])) for w in gt["words_clean"]]
        gt_regions = [Region(r["cls"], tuple(r["box"])) for r in gt["regions_clean"]]
        image = cv2.cvtColor(cv2.imread(str(Path(a.data) / "images" / a.split / f"{gt['id']}.jpg")), cv2.COLOR_BGR2RGB)
        if a.ocr == "oracle":
            res = pipe.run(image if a.layout == "yolo" else None, words=gt_words,
                           regions=gt_regions if a.layout == "gt" else None)
        else:
            cache = Path(a.data) / "ocr_cache" / a.ocr / f"{gt['id']}.json"
            if cache.exists():  # re-score parser changes without paying for OCR again
                c = json.loads(cache.read_text())
                image = rotate(image, c["angle"])
                res = pipe.run(image, words=[Word.from_dict(w) for w in c["words"]],
                               regions=None if a.layout != "gt" else gt_regions)
                res["skew_angle"] = c["angle"]
            else:
                res = pipe.run(image, regions=None if a.layout != "gt" else gt_regions)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"angle": res["skew_angle"], "words": res["words"]}))
            ocr_words = [Word.from_dict(w) for w in res["words"]]
            scorer.add_cer(*cer(ocr_words, gt_words, [r for r in gt_regions if r.cls in DATA_REGIONS]))
        d = scorer.add(res["document"], gt["fields"])
        d.update({"id": gt["id"], "skew_applied": gt["skew_applied"], "skew_estimated": res["skew_angle"]})
        details.append(d)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(files)}", file=sys.stderr)

    summary = scorer.summary()
    summary.update({"ocr": a.ocr, "layout": a.layout, "split": a.split,
                    "seconds_per_doc": round((time.perf_counter() - t0) / max(1, len(files)), 3)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"eval_{a.ocr}_{a.layout}_{a.split}"
    (out / f"{name}.json").write_text(json.dumps({"summary": summary, "documents": details}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
