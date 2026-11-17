"""Collect results/eval_*.json into a markdown table for the README."""
import json
import sys
from pathlib import Path

ORDER = [("oracle", "gt"), ("oracle", "rules"), ("tesseract", "gt"), ("easyocr", "gt"),
         ("easyocr", "rules"), ("tesseract", "yolo"), ("easyocr", "yolo")]
DESC = {
    ("oracle", "gt"): "Parser only (perfect text + perfect layout)",
    ("oracle", "rules"): "Perfect text + keyword layout",
    ("tesseract", "gt"): "Tesseract + perfect layout",
    ("easyocr", "gt"): "EasyOCR + perfect layout",
    ("easyocr", "rules"): "EasyOCR + keyword layout",
    ("tesseract", "yolo"): "Tesseract + YOLO layout",
    ("easyocr", "yolo"): "**Full system: EasyOCR + YOLO layout**",
}


def main(results_dir: str = "results") -> None:
    rows = {}
    for f in Path(results_dir).glob("eval_*.json"):
        s = json.loads(f.read_text())["summary"]
        rows[(s["ocr"], s["layout"])] = s
    lines = ["| Setting | Field F1 | Chem. values | Mech. values | Doc exact | Compliance acc. | CER | s/doc |",
             "|---|---|---|---|---|---|---|---|"]
    for key in ORDER + sorted(k for k in rows if k not in ORDER):
        if key not in rows:
            continue
        s = rows[key]
        pf = s["per_field_recall"]
        cer = f"{s['cer_data_regions']:.3f}" if "cer_data_regions" in s else "–"
        lines.append(f"| {DESC.get(key, key)} | {s['field_f1']:.3f} | {pf.get('chemistry', 0):.3f} | "
                     f"{pf.get('mechanical', 0):.3f} | {s['document_exact_match']:.3f} | "
                     f"{s['compliance_accuracy']:.3f} | {cer} | {s['seconds_per_doc']} |")
    n = next(iter(rows.values()))["documents"] if rows else 0
    text = f"Test split, {n} synthetic certificates.\n\n" + "\n".join(lines) + "\n"
    (Path(results_dir) / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
