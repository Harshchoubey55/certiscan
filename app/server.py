"""Flask app: upload certificates, review/correct extracted values, export for ERP import.

    flask --app app.server run            # dev server on http://127.0.0.1:5000
Env: CERTISCAN_WEIGHTS (YOLO .pt), CERTISCAN_OCR (easyocr|tesseract), DATABASE_URL.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import cv2
from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from certiscan.db import Certificate, find_by_hash, make_session_factory  # noqa: E402
from certiscan.extract import INFO_FIELDS  # noqa: E402
from certiscan.preprocess import load_image  # noqa: E402
from certiscan.specs import load_specs, resolve_spec_key  # noqa: E402
from certiscan.textlayout import parse_number  # noqa: E402
from certiscan.validate import validate_document  # noqa: E402

REGION_COLORS = {"title_block": (120, 120, 120), "info_block": (0, 120, 255), "chem_table": (0, 170, 0),
                 "mech_table": (230, 120, 0), "stamp": (200, 0, 0), "signature": (150, 0, 200), "remarks": (0, 160, 160)}
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def create_app(pipeline=None, db_url: str | None = None, upload_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = os.getenv("SECRET_KEY", "dev-only-secret")
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    Session = make_session_factory(db_url)
    uploads = Path(upload_dir or os.getenv("CERTISCAN_UPLOADS", "uploads"))
    uploads.mkdir(parents=True, exist_ok=True)
    state = {"pipeline": pipeline}

    def get_pipeline():
        if state["pipeline"] is None:
            from certiscan.pipeline import build_pipeline

            state["pipeline"] = build_pipeline()
        return state["pipeline"]

    def process(data: bytes, filename: str):
        """Returns (certificate, is_duplicate)."""
        digest = hashlib.sha256(data).hexdigest()
        with Session() as s:
            existing = find_by_hash(s, digest)
            if existing:
                return existing, True
        image = load_image(data, filename)
        result = get_pipeline().run(image)
        img_path = uploads / f"{digest}.png"
        cv2.imwrite(str(img_path), cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))
        doc = result["document"]
        cert = Certificate(sha256=digest, filename=filename, image_path=str(img_path),
                           compliance=doc["compliance"]["status"], extracted_json=json.dumps(doc),
                           regions_json=json.dumps(result["regions"]), timings_json=json.dumps(result["timings"]))
        with Session() as s:
            s.add(cert)
            s.commit()
        return cert, False

    def get_cert(cert_id: int) -> Certificate:
        with Session() as s:
            cert = s.get(Certificate, cert_id)
        if cert is None:
            abort(404)
        return cert

    def upload_ok(f) -> bool:
        return bool(f and f.filename and Path(f.filename).suffix.lower() in ALLOWED)

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        with Session() as s:
            certs = s.scalars(select(Certificate).order_by(Certificate.id.desc()).limit(100)).all()
        return render_template("index.html", certs=certs)

    @app.post("/upload")
    def upload():
        f = request.files.get("file")
        if not upload_ok(f):
            flash("Upload a PDF or image file.", "error")
            return redirect(url_for("index"))
        cert, dup = process(f.read(), f.filename)
        if dup:
            flash(f"Duplicate document: already processed as #{cert.id}.", "warn")
        return redirect(url_for("review", cert_id=cert.id))

    @app.get("/cert/<int:cert_id>")
    def review(cert_id: int):
        cert = get_cert(cert_id)
        doc = cert.document
        elements = sorted({e for h in doc.get("heats", []) for e in h.get("chemistry", {})},
                          key=lambda e: ["C", "Mn", "P", "S", "Si", "Cu", "Ni", "Cr", "Mo", "V"].index(e)
                          if e in ["C", "Mn", "P", "S", "Si", "Cu", "Ni", "Cr", "Mo", "V"] else 99)
        flagged = {(v["heat_no"], v["field"]) for v in doc.get("compliance", {}).get("violations", [])}
        return render_template("review.html", cert=cert, doc=doc, info_fields=INFO_FIELDS, elements=elements,
                               mech_fields=["yield_mpa", "tensile_mpa", "elongation_pct"], flagged=flagged,
                               timings=json.loads(cert.timings_json), specs=sorted(load_specs()))

    @app.post("/cert/<int:cert_id>")
    def save_review(cert_id: int):
        cert = get_cert(cert_id)
        doc = cert.document
        form = request.form
        for f in INFO_FIELDS:
            doc[f] = form.get(f) or None
        doc["spec_key"] = form.get("spec_key") or resolve_spec_key(doc.get("spec"), doc.get("grade"))
        for i, heat in enumerate(doc.get("heats", [])):
            heat["heat_no"] = form.get(f"heat__{i}", heat["heat_no"])
            for section, prefix in (("chemistry", "chem"), ("mechanical", "mech")):
                for key in list(form.keys()):
                    if key.startswith(f"{prefix}__{i}__"):
                        name = key.split("__", 2)[2]
                        v = parse_number(form.get(key))
                        if v is None:
                            heat[section].pop(name, None)
                        else:
                            heat[section][name] = v
        doc["compliance"] = validate_document(doc)
        with Session() as s:
            c = s.get(Certificate, cert_id)
            c.corrected_json = json.dumps(doc)
            c.compliance = doc["compliance"]["status"]
            c.status = "approved"
            s.commit()
        flash("Saved and re-validated.", "ok")
        return redirect(url_for("review", cert_id=cert_id))

    @app.get("/cert/<int:cert_id>/overlay.png")
    def overlay(cert_id: int):
        cert = get_cert(cert_id)
        img = cv2.imread(cert.image_path)
        if img is None:
            abort(404)
        for r in cert.regions:
            x0, y0, x1, y1 = map(int, r["box"])
            color = REGION_COLORS.get(r["cls"], (0, 0, 0))[::-1]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
            cv2.putText(img, f"{r['cls']} {r['conf']:.2f}", (x0 + 4, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        ok, buf = cv2.imencode(".png", img)
        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")

    # ------------------------------------------------------------------ API
    @app.post("/api/extract")
    def api_extract():
        f = request.files.get("file")
        if not upload_ok(f):
            return jsonify(error="multipart field 'file' with a PDF/image is required"), 400
        cert, dup = process(f.read(), f.filename)
        return jsonify(id=cert.id, duplicate=dup, document=cert.document, regions=cert.regions)

    @app.get("/api/certs")
    def api_list():
        with Session() as s:
            certs = s.scalars(select(Certificate).order_by(Certificate.id.desc())).all()
        return jsonify([{"id": c.id, "filename": c.filename, "compliance": c.compliance, "status": c.status,
                         "spec_key": c.document.get("spec_key"), "created_at": c.created_at.isoformat()} for c in certs])

    @app.get("/api/certs/<int:cert_id>")
    def api_get(cert_id: int):
        cert = get_cert(cert_id)
        return jsonify(id=cert.id, status=cert.status, document=cert.document)

    @app.get("/api/certs/<int:cert_id>.csv")
    def api_csv(cert_id: int):
        """One row per heat: the flat shape an ERP quality module import expects."""
        doc = get_cert(cert_id).document
        elements = sorted({e for h in doc.get("heats", []) for e in h["chemistry"]})
        mech = ["yield_mpa", "tensile_mpa", "elongation_pct"]
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(INFO_FIELDS + ["spec_key", "heat_no"] + elements + mech + ["compliance"])
        for h in doc.get("heats", []):
            w.writerow([doc.get(f) or "" for f in INFO_FIELDS] + [doc.get("spec_key") or "", h["heat_no"]]
                       + [h["chemistry"].get(e, "") for e in elements] + [h["mechanical"].get(m, "") for m in mech]
                       + [doc.get("compliance", {}).get("status", "")])
        return Response(out.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=cert_{cert_id}.csv"})

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
