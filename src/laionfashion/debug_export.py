from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from laionfashion.data_access import LaionTarReader, NaturalSubsetIndex, open_feature_memmap
from laionfashion.filtering import caption_matches_fashion


def write_thumbnail(image: Image.Image, path: Path, max_size: int) -> None:
    thumb = image.copy()
    thumb.thumbnail((max_size, max_size))
    thumb.save(path, quality=90)


def collect_caption_filtered_subset(
    *,
    index: NaturalSubsetIndex,
    rng: np.random.Generator,
    n_images: int,
    candidate_scan: int,
    thumbnail_dir: Path,
    thumbnail_size: int,
) -> pd.DataFrame:
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    scanned = 0

    with tqdm(total=candidate_scan, desc="Scanning captions") as pbar:
        shard_order = rng.permutation(len(index.shards))
        for shard_index in shard_order:
            if len(records) >= n_images:
                break
            if scanned >= candidate_scan:
                break

            shard = index.shards[int(shard_index)]
            with LaionTarReader(shard.tar_path) as reader:
                n_available = min(shard.n_images, len(reader))
                within_order = rng.permutation(n_available)
                for within_shard_index in within_order:
                    if len(records) >= n_images or scanned >= candidate_scan:
                        break

                    scanned += 1
                    pbar.update(1)
                    global_index = shard.start_index + int(within_shard_index)

                    try:
                        metadata = reader.read_metadata(int(within_shard_index))
                    except Exception:
                        continue

                    caption = metadata.get("caption", "")
                    if not caption_matches_fashion(caption):
                        continue

                    try:
                        image = reader.read_image(int(within_shard_index))
                    except Exception:
                        continue

                    thumb_name = f"{len(records):06d}_{global_index}.jpg"
                    write_thumbnail(image, thumbnail_dir / thumb_name, thumbnail_size)
                    records.append(
                        {
                            "row_id": len(records),
                            "global_index": int(global_index),
                            "tar_path": str(shard.tar_path),
                            "within_shard_index": int(within_shard_index),
                            "caption": caption,
                            "url": metadata.get("url", ""),
                            "width": metadata.get("width"),
                            "height": metadata.get("height"),
                            "original_width": metadata.get("original_width"),
                            "original_height": metadata.get("original_height"),
                            "sha256": metadata.get("sha256", ""),
                            "thumbnail_path": str(Path("thumbnails") / thumb_name),
                        }
                    )

    return pd.DataFrame.from_records(records)


def export_embeddings(
    *,
    records: pd.DataFrame,
    feature_key: str,
    output_path: Path,
    index: NaturalSubsetIndex,
) -> dict:
    features = open_feature_memmap(feature_key, n_images=len(index))
    global_indices = records["global_index"].to_numpy(dtype=np.int64)
    embedding = np.asarray(features[global_indices], dtype=np.float32)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.clip(norms, 1e-12, None)
    np.save(output_path, embedding)
    return {
        "feature_key": feature_key,
        "path": str(output_path),
        "shape": [int(v) for v in embedding.shape],
        "dtype": str(embedding.dtype),
    }

