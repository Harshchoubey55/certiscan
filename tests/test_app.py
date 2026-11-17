"""Flask app with an 'oracle' pipeline that returns ground-truth OCR for a synthetic page."""
import json

import pytest

from app.server import create_app
from certiscan.pipeline import Pipeline
from certiscan.synth.generate import build_dataset
from certiscan.types import Region, Word


class OraclePipeline:
    def __init__(self, gt):
        self.gt = gt

    def run(self, image):
        words = [Word(w["text"], tuple(w["box"])) for w in self.gt["words_clean"]]
        regions = [Region(r["cls"], tuple(r["box"])) for r in self.gt["regions_clean"]]
        res = Pipeline().run(words=words, regions=regions)
        res["image"] = image
        return res


@pytest.fixture()
def setup(tmp_path):
    build_dataset(str(tmp_path / "synth"), n=4, seed=11)
    gt_file = sorted((tmp_path / "synth" / "gt" / "train").glob("*.json"))[0]
    gt = json.loads(gt_file.read_text())
    img = (tmp_path / "synth" / "images" / "train" / f"{gt['id']}.jpg").read_bytes()
    app = create_app(pipeline=OraclePipeline(gt), db_url=f"sqlite:///{tmp_path / 't.db'}", upload_dir=str(tmp_path / "up"))
    app.testing = True
    return app.test_client(), gt, img


def test_extract_duplicate_and_csv(setup):
    client, gt, img = setup
    r = client.post("/api/extract", data={"file": (__import__("io").BytesIO(img), "mtr.jpg")})
    body = r.get_json()
    assert r.status_code == 200 and body["duplicate"] is False
    assert body["document"]["cert_no"] == gt["fields"]["cert_no"]
    assert body["document"]["compliance"]["status"] == gt["fields"]["compliance"]["status"]

    r2 = client.post("/api/extract", data={"file": (__import__("io").BytesIO(img), "again.jpg")})
    assert r2.get_json()["duplicate"] is True and r2.get_json()["id"] == body["id"]

    csv_text = client.get(f"/api/certs/{body['id']}.csv").get_data(as_text=True)
    assert gt["fields"]["heats"][0]["heat_no"] in csv_text

    assert client.get(f"/cert/{body['id']}").status_code == 200
    assert client.get(f"/cert/{body['id']}/overlay.png").mimetype == "image/png"


def test_review_correction_revalidates(setup):
    client, gt, img = setup
    cid = client.post("/api/extract", data={"file": (__import__("io").BytesIO(img), "mtr.jpg")}).get_json()["id"]
    doc = client.get(f"/api/certs/{cid}").get_json()["document"]
    form = {f: doc.get(f) or "" for f in ["cert_no", "date", "customer", "po_no", "spec", "grade", "size", "quantity"]}
    form["spec_key"] = doc["spec_key"]
    form["heat__0"] = doc["heats"][0]["heat_no"]
    form["chem__0__C"] = "0.99"  # far above every spec's carbon max
    client.post(f"/cert/{cid}", data=form)
    after = client.get(f"/api/certs/{cid}").get_json()
    assert after["status"] == "approved"
    assert after["document"]["compliance"]["status"] == "fail"
