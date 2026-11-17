Test split, 90 synthetic certificates.

| Setting | Field F1 | Chem. values | Mech. values | Doc exact | Compliance acc. | CER | s/doc |
|---|---|---|---|---|---|---|---|
| Parser only (perfect text + perfect layout) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | – | 0.022 |
| Perfect text + keyword layout | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | – | 0.032 |
| Tesseract + perfect layout | 0.749 | 0.363 | 0.556 | 0.067 | 0.200 | 0.087 | 3.15 |
| EasyOCR + perfect layout | 0.732 | 0.542 | 0.629 | 0.078 | 0.333 | 0.051 | 2.469 |
| EasyOCR + keyword layout | 0.732 | 0.542 | 0.626 | 0.078 | 0.333 | 0.051 | 0.04 |
| Tesseract + YOLO layout | 0.749 | 0.363 | 0.556 | 0.067 | 0.200 | 0.087 | 0.08 |
| **Full system: EasyOCR + YOLO layout** | 0.732 | 0.542 | 0.629 | 0.078 | 0.333 | 0.051 | 0.06 |
