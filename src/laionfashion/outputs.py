from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4


def make_output_dir(script_path: str | Path, output_root: Path | None = None) -> Path:
    script_path = Path(script_path)
    base = output_root or script_path.resolve().parent / "outputs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / script_path.stem / f"{stamp}_{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir

