#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.config import load_data_paths, on_raven
from laionfashion.data_access import FEATURE_REGISTRY, NaturalSubsetIndex
from laionfashion.outputs import make_output_dir


def package_status() -> dict[str, bool]:
    packages = [
        "torch",
        "clip",
        "open_clip",
        "streamlit",
        "gradio",
        "faiss",
        "sklearn",
        "umap",
        "pandas",
        "pyarrow",
        "PIL",
        "cv2",
        "ultralytics",
        "segment_anything",
    ]
    return {name: importlib.util.find_spec(name) is not None for name in packages}


def gpu_status() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {
        "available": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def sample_metadata_keys(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path, "r:*", ignore_zeros=True) as tar:
        json_name = next(name for name in tar.getnames() if name.endswith(".json"))
        metadata = json.load(tar.extractfile(json_name))
    return sorted(metadata.keys())


def main() -> None:
    paths = load_data_paths()
    payload: dict = {
        "on_raven": on_raven(),
        "paths": {
            "subset_root": str(paths.subset_root),
            "memmap_root": str(paths.memmap_root),
            "metadata_path": str(paths.metadata_path),
            "subset_root_exists": paths.subset_root.exists(),
            "memmap_root_exists": paths.memmap_root.exists(),
            "metadata_path_exists": paths.metadata_path.exists(),
        },
        "python": sys.version.replace("\n", " "),
        "packages": package_status(),
        "gpu": gpu_status(),
        "feature_registry": FEATURE_REGISTRY,
    }

    if paths.metadata_path.exists():
        index = NaturalSubsetIndex.from_paths(paths)
        payload["natural_subset"] = {
            "n_shards": len(index.shards),
            "n_images": len(index),
            "first_tar": str(index.shards[0].tar_path),
            "last_tar": str(index.shards[-1].tar_path),
        }
        if index.shards[0].tar_path.exists():
            payload["natural_subset"]["sample_metadata_keys"] = sample_metadata_keys(
                index.shards[0].tar_path
            )

    if paths.memmap_root.exists():
        payload["available_memmaps"] = sorted(
            path.name for path in paths.memmap_root.glob("*.mmap")
        )

    out_dir = make_output_dir(__file__, paths.output_root)
    with (out_dir / "inventory.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote inventory to {out_dir}")


if __name__ == "__main__":
    main()

