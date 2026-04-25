from __future__ import annotations

import bisect
import io
import json
import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image

from laionfashion.config import DataPaths, load_data_paths, normalize_cluster_path


FEATURE_REGISTRY: dict[str, dict[str, object]] = {
    "openclip_vit_l_14_quickgelu_metaclip_fullcc.ln_post": {
        "filename": "openclip_vit_l_14_quickgelu_metaclip_fullcc.ln_post.mmap",
        "dim": 1024,
        "dtype": "float16",
    },
    "openclip_vit_l_14_quickgelu_metaclip_400m.ln_post": {
        "filename": "openclip_vit_l_14_quickgelu_metaclip_400m.ln_post.mmap",
        "dim": 1024,
        "dtype": "float16",
    },
    "openclip_vit_l_14_laion400m_e31.ln_post": {
        "filename": "openclip_vit_l_14_laion400m_e31.ln_post.mmap",
        "dim": 1024,
        "dtype": "float16",
    },
    "openclip_vit_so400m_14_siglip_webli.trunk.fc_norm": {
        "filename": "openclip_vit_so400m_14_siglip_webli.trunk.fc_norm.mmap",
        "dim": 1152,
        "dtype": "float16",
    },
    "dinov2_vitl14.head": {
        "filename": "dinov2_vitl14.head.mmap",
        "dim": 1024,
        "dtype": "float16",
    },
}


@dataclass(frozen=True)
class ShardEntry:
    tar_path: Path
    start_index: int
    end_index: int
    failed_images: tuple[int, ...] = ()
    summary_path: Path | None = None

    @property
    def n_images(self) -> int:
        return self.end_index - self.start_index


@dataclass(frozen=True)
class LocatedSample:
    global_index: int
    shard_index: int
    within_shard_index: int
    tar_path: Path


class NaturalSubsetIndex:
    def __init__(self, shards: Sequence[ShardEntry]):
        if not shards:
            raise ValueError("NaturalSubsetIndex requires at least one shard.")
        self.shards = list(shards)
        self._end_indices = [entry.end_index for entry in self.shards]

    @classmethod
    def from_metadata(cls, metadata_path: Path) -> "NaturalSubsetIndex":
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing LAION-natural memmap metadata: {metadata_path}. "
                "Set LAIONFASHION_MEMMAP_ROOT or run this on Raven."
            )
        with metadata_path.open("rb") as f:
            raw_entries = pickle.load(f)
        shards = [
            ShardEntry(
                tar_path=normalize_cluster_path(entry["tar_path"]),
                start_index=int(entry["start_index"]),
                end_index=int(entry["end_index"]),
                failed_images=tuple(entry.get("failed_images", ())),
                summary_path=(
                    normalize_cluster_path(entry["summary_path"])
                    if entry.get("summary_path") is not None
                    else None
                ),
            )
            for entry in raw_entries
        ]
        return cls(shards)

    @classmethod
    def from_paths(cls, paths: DataPaths | None = None) -> "NaturalSubsetIndex":
        paths = paths or load_data_paths()
        return cls.from_metadata(paths.metadata_path)

    def __len__(self) -> int:
        return self.shards[-1].end_index

    def locate(self, global_index: int) -> LocatedSample:
        if global_index < 0 or global_index >= len(self):
            raise IndexError(f"Global index {global_index} outside [0, {len(self)}).")
        shard_index = bisect.bisect_right(self._end_indices, global_index)
        shard = self.shards[shard_index]
        return LocatedSample(
            global_index=global_index,
            shard_index=shard_index,
            within_shard_index=global_index - shard.start_index,
            tar_path=shard.tar_path,
        )

    def iter_locations(self, global_indices: Sequence[int]) -> Iterator[LocatedSample]:
        for global_index in global_indices:
            yield self.locate(int(global_index))


class LaionTarReader:
    def __init__(self, tar_path: Path):
        self.tar_path = Path(tar_path)
        self._tar = tarfile.open(self.tar_path, "r:*", ignore_zeros=True)
        names = self._tar.getnames()
        name_set = set(names)
        self.image_ids = [
            name.removesuffix(".jpg")
            for name in names
            if name.endswith(".jpg") and f"{name.removesuffix('.jpg')}.json" in name_set
        ]

    def __len__(self) -> int:
        return len(self.image_ids)

    def close(self) -> None:
        self._tar.close()

    def __enter__(self) -> "LaionTarReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def read_metadata(self, index: int) -> dict:
        image_id = self.image_ids[index]
        member = self._tar.extractfile(f"{image_id}.json")
        if member is None:
            raise FileNotFoundError(f"Missing metadata for {image_id} in {self.tar_path}.")
        return json.load(member)

    def read_image(self, index: int) -> Image.Image:
        image_id = self.image_ids[index]
        member = self._tar.extractfile(f"{image_id}.jpg")
        if member is None:
            raise FileNotFoundError(f"Missing image for {image_id} in {self.tar_path}.")
        return Image.open(io.BytesIO(member.read())).convert("RGB")


def open_feature_memmap(
    feature_key: str,
    *,
    paths: DataPaths | None = None,
    n_images: int | None = None,
) -> np.memmap:
    paths = paths or load_data_paths()
    if feature_key not in FEATURE_REGISTRY:
        available = ", ".join(sorted(FEATURE_REGISTRY))
        raise KeyError(f"Unknown feature key {feature_key!r}. Available: {available}")
    spec = FEATURE_REGISTRY[feature_key]
    feature_path = paths.memmap_root / str(spec["filename"])
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature memmap: {feature_path}")
    if n_images is None:
        n_images = len(NaturalSubsetIndex.from_paths(paths))
    return np.memmap(
        feature_path,
        mode="r",
        dtype=np.dtype(str(spec["dtype"])),
        shape=(int(n_images), int(spec["dim"])),
    )

