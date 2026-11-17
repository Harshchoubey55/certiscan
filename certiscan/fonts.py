"""Find TrueType fonts on Colab (Liberation/DejaVu), Windows, macOS, or matplotlib's bundle."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from PIL import ImageFont

FAMILIES: Dict[str, List[str]] = {
    "sans": ["LiberationSans-Regular.ttf", "DejaVuSans.ttf", "arial.ttf", "Arial.ttf"],
    "sans-bold": ["LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"],
    "serif": ["LiberationSerif-Regular.ttf", "DejaVuSerif.ttf", "times.ttf", "Times New Roman.ttf"],
    "serif-bold": ["LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf", "timesbd.ttf"],
    "mono": ["LiberationMono-Regular.ttf", "DejaVuSansMono.ttf", "cour.ttf", "Courier New.ttf"],
    "mono-bold": ["LiberationMono-Bold.ttf", "DejaVuSansMono-Bold.ttf", "courbd.ttf", "Courier New Bold.ttf"],
}


def _dirs() -> List[Path]:
    dirs = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path("C:/Windows/Fonts"),
            Path("/Library/Fonts"), Path("/System/Library/Fonts")]
    try:
        import matplotlib

        dirs.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except Exception:
        pass
    return [d for d in dirs if d.exists()]


@lru_cache(maxsize=1)
def _index() -> Dict[str, str]:
    found: Dict[str, str] = {}
    for d in _dirs():
        pattern = "*.ttf" if str(d).lower().startswith("c:") else "**/*.ttf"
        for p in d.glob(pattern):
            found.setdefault(p.name.lower(), str(p))
    return found


def font_path(family: str) -> str | None:
    idx = _index()
    for name in FAMILIES[family]:
        if name.lower() in idx:
            return idx[name.lower()]
    return None


def available_families() -> List[str]:
    """Base families (sans/serif/mono) that have a usable regular font."""
    return [f for f in ("sans", "serif", "mono") if font_path(f)]


@lru_cache(maxsize=256)
def get_font(family: str, size: int) -> ImageFont.FreeTypeFont:
    path = font_path(family) or font_path(family.split("-")[0])
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)
