from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SUBSET_ROOT = Path("/ptmp/rothj/cstims_laion_natural_subset")
DEFAULT_MEMMAP_ROOT = Path("/ptmp/rothj/cstims_laion_natural_subset_memmaps")


@dataclass(frozen=True)
class DataPaths:
    subset_root: Path = DEFAULT_SUBSET_ROOT
    memmap_root: Path = DEFAULT_MEMMAP_ROOT
    output_root: Path | None = None

    @property
    def metadata_path(self) -> Path:
        return self.memmap_root / "_metadata.pkl"


def path_from_env(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return default


def load_data_paths() -> DataPaths:
    return DataPaths(
        subset_root=path_from_env("LAIONFASHION_SUBSET_ROOT", DEFAULT_SUBSET_ROOT),
        memmap_root=path_from_env("LAIONFASHION_MEMMAP_ROOT", DEFAULT_MEMMAP_ROOT),
        output_root=path_from_env("LAIONFASHION_OUTPUT_ROOT"),
    )


def on_raven() -> bool:
    return DEFAULT_SUBSET_ROOT.exists() or Path("/raven/ptmp/rothj").exists()


def normalize_cluster_path(path: str | Path) -> Path:
    path = Path(path)
    text = str(path)
    if text.startswith("/raven/ptmp/"):
        return Path(text.replace("/raven/ptmp/", "/ptmp/", 1))
    return path

