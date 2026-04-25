from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from laionfashion.data_access import LaionTarReader, NaturalSubsetIndex, open_feature_memmap
from laionfashion.filtering import SELECTION_MODES, RejectReason, filter_caption, score_caption
from laionfashion.image_scoring import ImageScorer


def write_thumbnail(image: Image.Image, path: Path, max_size: int) -> None:
    thumb = image.copy()
    thumb.thumbnail((max_size, max_size))
    thumb.save(path, quality=90)


# Maximum caption samples to keep per reject reason (keeps output small).
MAX_SAMPLES_PER_REASON = 100


@dataclass
class FilterDiagnostics:
    """Summary of caption filtering during subset collection."""

    scanned: int = 0
    accepted: int = 0
    metadata_errors: int = 0
    image_errors: int = 0
    reject_counts: dict[str, int] = field(default_factory=dict)

    # Caption samples for review artifacts
    accepted_samples: list[dict] = field(default_factory=list)
    rejected_samples: dict[str, list[dict]] = field(default_factory=dict)

    # Score distribution tracking
    accepted_scores: list[float] = field(default_factory=list)
    selection_mode: str | None = None

    # Image-side scoring
    image_scored: int = 0
    image_rejected: int = 0
    image_scores: list[float] = field(default_factory=list)

    def record_reject(self, reason: RejectReason, caption: str, matched_term: str | None = None) -> None:
        key = reason.value
        self.reject_counts[key] = self.reject_counts.get(key, 0) + 1
        samples = self.rejected_samples.setdefault(key, [])
        if len(samples) < MAX_SAMPLES_PER_REASON:
            entry = {"caption": caption, "reason": key}
            if matched_term:
                entry["matched_term"] = matched_term
            samples.append(entry)

    def record_accept(self, caption: str, matched_term: str | None = None, score: float | None = None) -> None:
        self.accepted += 1
        if score is not None:
            self.accepted_scores.append(score)
        if len(self.accepted_samples) < MAX_SAMPLES_PER_REASON:
            entry: dict = {"caption": caption}
            if matched_term:
                entry["matched_term"] = matched_term
            if score is not None:
                entry["score"] = round(score, 2)
            self.accepted_samples.append(entry)

    def to_dict(self) -> dict:
        d: dict = {
            "scanned": self.scanned,
            "accepted": self.accepted,
            "metadata_errors": self.metadata_errors,
            "image_errors": self.image_errors,
            "reject_counts": dict(sorted(self.reject_counts.items())),
            "accept_rate": round(self.accepted / max(self.scanned, 1), 4),
        }
        if self.selection_mode:
            d["selection_mode"] = self.selection_mode
        if self.accepted_scores:
            scores = self.accepted_scores
            d["score_distribution"] = {
                "min": round(min(scores), 2),
                "max": round(max(scores), 2),
                "mean": round(sum(scores) / len(scores), 2),
                "count": len(scores),
            }
        if self.image_scored > 0:
            d["image_scoring"] = {
                "scored": self.image_scored,
                "rejected": self.image_rejected,
                "accepted": self.image_scored - self.image_rejected,
            }
            if self.image_scores:
                d["image_scoring"]["score_distribution"] = {
                    "min": round(min(self.image_scores), 4),
                    "max": round(max(self.image_scores), 4),
                    "mean": round(sum(self.image_scores) / len(self.image_scores), 4),
                }
        return d

    def write_review_artifacts(self, out_dir: Path) -> dict[str, str]:
        """Write accepted/rejected caption samples to *out_dir*.

        Returns a dict of artifact names to file paths (relative to out_dir).
        """
        artifacts: dict[str, str] = {}

        # Accepted captions
        if self.accepted_samples:
            path = out_dir / "accepted_captions.csv"
            _write_dicts_csv(path, self.accepted_samples, ["caption", "matched_term", "score"])
            artifacts["accepted_captions"] = path.name

        # Rejected captions (one file, all reasons)
        all_rejected = []
        for samples in self.rejected_samples.values():
            all_rejected.extend(samples)
        if all_rejected:
            path = out_dir / "rejected_captions.csv"
            _write_dicts_csv(path, all_rejected, ["caption", "reason", "matched_term"])
            artifacts["rejected_captions"] = path.name

        # Summary JSON
        summary = self.to_dict()
        summary["accepted_sample_count"] = len(self.accepted_samples)
        summary["rejected_sample_count"] = len(all_rejected)
        path = out_dir / "filter_summary.json"
        with path.open("w") as f:
            json.dump(summary, f, indent=2)
        artifacts["filter_summary"] = path.name

        return artifacts


def _write_dicts_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_caption_filtered_subset(
    *,
    index: NaturalSubsetIndex,
    rng: np.random.Generator,
    n_images: int,
    candidate_scan: int,
    thumbnail_dir: Path,
    thumbnail_size: int,
    require_person_context: bool = False,
    selection_mode: str | None = None,
    min_score: float | None = None,
    image_scorer: ImageScorer | None = None,
    min_image_score: float = 0.0,
) -> tuple[pd.DataFrame, FilterDiagnostics]:
    """Collect a caption-filtered subset.  Returns ``(records, diagnostics)``.

    Parameters
    ----------
    selection_mode:
        One of ``"broad"``, ``"strict"``, ``"outfit"``.  Sets a score threshold
        from :data:`SELECTION_MODES`.  Overrides *require_person_context*.
    min_score:
        Explicit score threshold.  Overrides *selection_mode* if both are set.
    image_scorer:
        Optional image-side scorer (e.g. CLIPOutfitScorer).  When provided,
        images that pass caption filtering are additionally scored and rejected
        if below *min_image_score*.
    min_image_score:
        Minimum image score to accept.  Only used when *image_scorer* is set.
    """
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    diag = FilterDiagnostics()

    # Resolve effective score threshold
    effective_min_score: float | None = None
    if min_score is not None:
        effective_min_score = min_score
        diag.selection_mode = f"custom(min_score={min_score})"
    elif selection_mode is not None:
        effective_min_score = SELECTION_MODES[selection_mode]
        diag.selection_mode = selection_mode
    elif require_person_context:
        diag.selection_mode = "strict_legacy"

    use_scoring = effective_min_score is not None

    with tqdm(total=candidate_scan, desc="Scanning captions") as pbar:
        shard_order = rng.permutation(len(index.shards))
        for shard_index in shard_order:
            if len(records) >= n_images:
                break
            if diag.scanned >= candidate_scan:
                break

            shard = index.shards[int(shard_index)]
            with LaionTarReader(shard.tar_path) as reader:
                n_available = min(shard.n_images, len(reader))
                within_order = rng.permutation(n_available)
                for within_shard_index in within_order:
                    if len(records) >= n_images or diag.scanned >= candidate_scan:
                        break

                    diag.scanned += 1
                    pbar.update(1)
                    global_index = shard.start_index + int(within_shard_index)

                    try:
                        metadata = reader.read_metadata(int(within_shard_index))
                    except Exception:
                        diag.metadata_errors += 1
                        continue

                    caption = metadata.get("caption", "")

                    if use_scoring:
                        result = filter_caption(caption, min_score=effective_min_score)
                        caption_score = score_caption(caption).score
                    else:
                        result = filter_caption(caption, require_person_context=require_person_context)
                        caption_score = None

                    if result.rejected:
                        diag.record_reject(result.reason, caption, result.matched_term)
                        continue

                    try:
                        image = reader.read_image(int(within_shard_index))
                    except Exception:
                        diag.image_errors += 1
                        continue

                    # Optional image-side scoring
                    img_score: float | None = None
                    if image_scorer is not None:
                        try:
                            img_score = image_scorer.score_image(image)
                        except Exception:
                            diag.image_errors += 1
                            continue
                        diag.image_scored += 1
                        diag.image_scores.append(img_score)
                        if img_score < min_image_score:
                            diag.image_rejected += 1
                            diag.record_reject(
                                RejectReason.BELOW_SCORE_THRESHOLD,
                                caption,
                                f"image_score={img_score:.4f}<{min_image_score}",
                            )
                            continue

                    diag.record_accept(caption, result.matched_term, score=caption_score)
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
                            **({"image_outfit_score": round(img_score, 4)} if img_score is not None else {}),
                        }
                    )

    return pd.DataFrame.from_records(records), diag


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
