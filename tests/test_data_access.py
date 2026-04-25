from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path

from PIL import Image

from laionfashion.data_access import LaionTarReader, NaturalSubsetIndex


def write_test_tar(path: Path, n_images: int) -> None:
    with tarfile.open(path, "w") as tar:
        for index in range(n_images):
            image = Image.new("RGB", (8, 8), color=(index * 30, 0, 0))
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_payload = image_bytes.getvalue()
            image_info = tarfile.TarInfo(f"{index:010d}.jpg")
            image_info.size = len(image_payload)
            tar.addfile(image_info, io.BytesIO(image_payload))

            metadata_payload = (
                f'{{"caption": "person wearing jacket {index}", "url": "https://example.com/{index}"}}'
            ).encode()
            metadata_info = tarfile.TarInfo(f"{index:010d}.json")
            metadata_info.size = len(metadata_payload)
            tar.addfile(metadata_info, io.BytesIO(metadata_payload))


def test_natural_subset_index_locates_global_indices(tmp_path: Path) -> None:
    tar_a = tmp_path / "a.tar"
    tar_b = tmp_path / "b.tar"
    write_test_tar(tar_a, 2)
    write_test_tar(tar_b, 3)
    metadata_path = tmp_path / "_metadata.pkl"
    with metadata_path.open("wb") as f:
        pickle.dump(
            [
                {"tar_path": tar_a, "start_index": 0, "end_index": 2, "failed_images": ()},
                {"tar_path": tar_b, "start_index": 2, "end_index": 5, "failed_images": ()},
            ],
            f,
        )

    index = NaturalSubsetIndex.from_metadata(metadata_path)

    assert len(index) == 5
    assert index.locate(0).tar_path == tar_a
    assert index.locate(2).tar_path == tar_b
    assert index.locate(4).within_shard_index == 2


def test_laion_tar_reader_reads_image_and_metadata(tmp_path: Path) -> None:
    tar_path = tmp_path / "sample.tar"
    write_test_tar(tar_path, 2)

    with LaionTarReader(tar_path) as reader:
        assert len(reader) == 2
        metadata = reader.read_metadata(1)
        image = reader.read_image(1)

    assert metadata["caption"] == "person wearing jacket 1"
    assert image.size == (8, 8)

