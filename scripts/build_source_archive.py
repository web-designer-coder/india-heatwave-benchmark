"""Build a versioned source-only archive without acquired data or outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.2.0"
ARCHIVE = ROOT / "dist" / f"india-heatwave-benchmark-v{VERSION}-source.zip"
INCLUDE_ROOTS = ("src", "scripts", "configs", "tests", "docs")
ROOT_FILES = (
    "README.md", "LICENSE", "CITATION.cff", "MANIFEST.in", "pyproject.toml", "uv.lock", ".gitignore",
    "paper/reproducibility-manifest.json", "paper/acquisition-provenance.json",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".venv", "superpowers"}


def files_to_archive() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for dirname in INCLUDE_ROOTS:
        base = ROOT / dirname
        if base.exists():
            files.extend(
                path for path in base.rglob("*")
                if path.is_file()
                and not (set(path.parts) & EXCLUDED_PARTS)
                and not path.parent.name.endswith(".egg-info")
            )
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    files = files_to_archive()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        prefix = f"india-heatwave-benchmark-v{VERSION}"
        for path in files:
            arcname = f"{prefix}/{path.relative_to(ROOT).as_posix()}"
            bundle.write(path, arcname)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    sums = ROOT / "dist" / f"SHA256SUMS-v{VERSION}-source.txt"
    sums.write_text(f"{digest}  {ARCHIVE.name}\n", encoding="utf-8")
    print(f"Archive: {ARCHIVE}")
    print(f"Files: {len(files)}")
    print(f"SHA-256: {digest}")
    with zipfile.ZipFile(ARCHIVE) as bundle:
        forbidden = [name for name in bundle.namelist() if name.startswith((f"{prefix}/data/", f"{prefix}/reports/"))]
        if forbidden:
            raise RuntimeError(f"source archive unexpectedly contains data or reports: {forbidden[:3]}")


if __name__ == "__main__":
    main()
