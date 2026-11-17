"""Generate a few synthetic certificates and run the real pipeline on them (no OCR model needed)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from certiscan.layout import RuleLayout
from certiscan.metrics import Scorer
from certiscan.pipeline import Pipeline
from certiscan.preprocess import estimate_skew, rotate
from certiscan.synth.generate import build_dataset
from certiscan.types import REGION_CLASSES, Region, Word


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    build_dataset(str(out), n=20, seed=3)
    return out


def load(gt_path: Path):
    gt = json.loads(gt_path.read_text())
    words = [Word(w["text"], tuple(w["box"])) for w in gt["words_clean"]]
    regions = [Region(r["cls"], tuple(r["box"])) for r in gt["regions_clean"]]
    return gt, words, regions


def test_dataset_layout(dataset):
    assert (dataset / "data.yaml").exists()
    labels = list((dataset / "labels" / "train").glob("*.txt"))
    assert len(labels) == 14
    for lbl in labels:
        for line in lbl.read_text().splitlines():
            cls, *box = line.split()
            assert 0 <= int(cls) < len(REGION_CLASSES)
            assert all(0 <= float(v) <= 1 for v in box)


@pytest.mark.parametrize("layout", ["gt", "rules"])
def test_parser_is_exact_with_perfect_ocr(dataset, layout):
    pipe = Pipeline(layout=RuleLayout())
    scorer = Scorer()
    for f in sorted((dataset / "gt" / "train").glob("*.json")):
        gt, words, regions = load(f)
        res = pipe.run(words=words, regions=regions if layout == "gt" else None)
        scorer.add(res["document"], gt["fields"])
    s = scorer.summary()
    assert s["field_f1"] == 1.0 and s["compliance_accuracy"] == 1.0, s


def test_deskew_recovers_applied_rotation(dataset):
    f = sorted((dataset / "gt" / "train").glob("*.json"))[0]
    img = cv2.cvtColor(cv2.imread(str(f).replace("gt", "images").replace(".json", ".jpg")), cv2.COLOR_BGR2RGB)
    gt = json.loads(f.read_text())
    straight = rotate(img, -gt["skew_applied"])  # undo augmentation, then apply a known tilt
    assert abs(estimate_skew(rotate(straight, 1.5)) + 1.5) < 0.2
