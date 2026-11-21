from pathlib import Path

import json
from pathlib import Path

from research.audit_dataset import aggregate_audits, audit_source, normalize_class_name
from research.prepare_yolo import validate_label_file
from research.download_hf_shards import REPOS
from research.prepare_hf_detection_top3 import convert_parquet_top3
from research.cross_source_audit import audit_cross_source_duplicates
from research.prepare_hf_single_class import split_parquet_single_class
from research.build_detector_experiment import build_detector_experiment
from research.audit_detector_dataset import audit_dataset
from research.pseudo_label_detector import write_candidate


def test_validate_label_file_accepts_normalized_yolo_rows(tmp_path: Path) -> None:
    label_file = tmp_path / "fruit.txt"
    label_file.write_text("0 0.5 0.5 0.4 0.6\n", encoding="utf-8")

    errors = validate_label_file(label_file, class_count=5)

    assert errors == []


def test_validate_label_file_rejects_bad_class_and_box(tmp_path: Path) -> None:
    label_file = tmp_path / "fruit.txt"
    label_file.write_text("5 0.5 0.5 0 1.2\n", encoding="utf-8")

    errors = validate_label_file(label_file, class_count=5)

    assert len(errors) == 2


def test_audit_source_counts_valid_coco_boxes_by_normalized_class(tmp_path: Path) -> None:
    image = tmp_path / "img-1.jpg"
    image.write_bytes(b"same-image")
    annotation = tmp_path / "annotations.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": image.name, "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "Tomatoes"}, {"id": 2, "name": "Potato"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                    {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 20, 0, 20]},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = audit_source(tmp_path)

    assert result.get("annotation_format") == "coco"
    assert result.get("valid_box_counts") == {"tomato": 1}
    assert result.get("invalid_box_count") == 1


def test_aggregate_audits_deduplicates_image_hashes_across_sources() -> None:
    audits = [
        {
            "source": "dataset-a",
            "valid_box_counts": {"tomato": 2},
            "valid_image_hashes": {"tomato": ["hash-1", "hash-2"]},
            "valid_box_counts_by_image": {"tomato": {"hash-1": 1, "hash-2": 1}},
        },
        {
            "source": "dataset-b",
            "valid_box_counts": {"tomato": 1},
            "valid_image_hashes": {"tomato": ["hash-2"]},
            "valid_box_counts_by_image": {"tomato": {"hash-2": 1}},
        },
    ]

    result = aggregate_audits(audits)

    assert result["top3"][0] == {
        "class_name": "tomato",
        "valid_box_count": 2,
        "unique_image_count": 2,
        "source_count": 2,
    }


def test_aggregate_audits_uses_legacy_count_when_image_breakdown_is_missing() -> None:
    result = aggregate_audits(
        [{"source": "legacy", "valid_box_counts": {"banana": 4}, "valid_image_counts": {"banana": 3}}]
    )

    assert result["top3"][0]["valid_box_count"] == 4
    assert result["top3"][0]["unique_image_count"] == 3


def test_normalize_class_name_merges_tomato_variants_but_keeps_similar_classes_separate() -> None:
    assert [normalize_class_name(value) for value in ["Tomato", "tomatoes", "tomat"]] == ["tomato"] * 3
    assert normalize_class_name("green banana") == "green banana"
    assert normalize_class_name("orange") == "orange"
    assert normalize_class_name("lemon") == "lemon"


def test_normalize_class_name_merges_green_bean_singular_and_plural() -> None:
    assert normalize_class_name("green bean") == "green beans"
    assert normalize_class_name("Green Beans") == "green beans"


def test_audit_source_reads_parallel_yolo_labels_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "test"
    label_dir = tmp_path / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("names:\n  0: Tomato\n", encoding="utf-8")
    (image_dir / "fruit.jpg").write_bytes(b"image")
    (label_dir / "fruit.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    result = audit_source(tmp_path)

    assert result["valid_box_counts"] == {"tomato": 1}


def test_hf_registry_includes_audited_smartphone_vegetable_candidate() -> None:
    assert REPOS["smartphone_vegetables"] == "LFTTim/multiple_smartphone_bangladesh_vegetable_detection"


def test_hf_registry_includes_cc0_tomato_detection_candidate() -> None:
    assert REPOS["tomato_detection_cc0"] == "Project-AgML/tomato_detection_kaggle"


def test_convert_parquet_top3_preserves_source_split_and_target_labels(tmp_path: Path) -> None:
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    image_buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "green").save(image_buffer, format="JPEG")
    source = tmp_path / "source.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "image": {"bytes": image_buffer.getvalue(), "path": "a.jpg"},
                "objects": {"bbox": [[10.0, 8.0, 30.0, 20.0]], "categories": [0]},
                "split": "test",
            }
        ],
        schema=pa.schema(
            [
                pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
                pa.field("objects", pa.struct([pa.field("bbox", pa.list_(pa.list_(pa.float64()))), pa.field("categories", pa.list_(pa.int64()))])),
                pa.field("split", pa.string()),
            ]
        ),
    )
    pq.write_table(table, source)

    result = convert_parquet_top3(source, tmp_path / "out", {0: "Cucumber"}, "smartphone")

    assert result["per_split"]["test"]["cucumber"] == 1
    assert (tmp_path / "out" / "images" / "test").is_dir()
    assert len(list((tmp_path / "out" / "labels" / "test").glob("*.txt"))) == 1


def test_cross_source_audit_counts_exact_hashes_and_split_collisions(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    for root, split, name, content in [
        (old, "train", "old-a.jpg", b"same"),
        (old, "test", "old-b.jpg", b"different"),
        (new, "validation", "new-a.jpg", b"same"),
        (new, "test", "new-c.jpg", b"new"),
    ]:
        path = root / "images" / split / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    result = audit_cross_source_duplicates(old, new)

    assert result["old_image_count"] == 2
    assert result["new_image_count"] == 2
    assert result["cross_exact_duplicates"] == 1
    assert result["split_collisions"] == {"train->validation": 1}


def test_split_parquet_single_class_creates_deterministic_train_and_holdout(tmp_path: Path) -> None:
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    rows = []
    for index in range(5):
        image_buffer = io.BytesIO()
        Image.new("RGB", (100, 80), (index * 20, 100, 20)).save(image_buffer, format="JPEG")
        rows.append(
            {
                "image": {"bytes": image_buffer.getvalue(), "path": f"{index}.jpg"},
                "objects": {"bbox": [[10.0, 8.0, 30.0, 20.0]], "categories": [0]},
            }
        )
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=pa.schema(
                [
                    pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
                    pa.field("objects", pa.struct([pa.field("bbox", pa.list_(pa.list_(pa.float64()))), pa.field("categories", pa.list_(pa.int64()))])),
                ]
            ),
        ),
        source,
    )

    result = split_parquet_single_class(source, tmp_path / "out", class_name="tomato", holdout_fraction=0.4, seed=7)

    assert result["train_images"] == 3
    assert result["holdout_images"] == 2
    assert len(list((tmp_path / "out" / "images" / "train").glob("*.jpg"))) == 3
    assert len(list((tmp_path / "out" / "images" / "holdout").glob("*.jpg"))) == 2


def test_build_detector_experiment_adds_only_extra_train_and_preserves_holdouts(tmp_path: Path) -> None:
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    for root, split, name, label in [
        (base, "train", "base-train.jpg", "0 0.5 0.5 0.2 0.2\n"),
        (base, "val", "base-val.jpg", "1 0.5 0.5 0.2 0.2\n"),
        (base, "test", "base-test.jpg", "2 0.5 0.5 0.2 0.2\n"),
        (extra, "train", "extra-train.jpg", "0 0.5 0.5 0.2 0.2\n"),
        (extra, "holdout", "extra-holdout.jpg", "0 0.5 0.5 0.2 0.2\n"),
    ]:
        image = root / "images" / split / name
        label_path = root / "labels" / split / name.replace(".jpg", ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(name.encode())
        label_path.write_text(label, encoding="utf-8")
    (base / "data.yaml").write_text("names:\n- cucumber\n- tomato\n- green beans\n", encoding="utf-8")

    result = build_detector_experiment(base, extra, tmp_path / "out")

    assert result == {"base_train": 1, "extra_train": 1, "base_val": 1, "base_test": 1, "extra_holdout": 1}
    assert len(list((tmp_path / "out" / "images" / "train").glob("*.jpg"))) == 2
    assert not (tmp_path / "out" / "images" / "holdout").exists()
    assert (tmp_path / "out" / "images" / "test" / "base-test.jpg").is_file()
    assert f"path: {str((tmp_path / 'out').resolve()).replace(chr(92), '/') }" in (tmp_path / "out" / "data.yaml").read_text(encoding="utf-8")


def test_build_detector_experiment_renames_extra_file_when_filename_collides(tmp_path: Path) -> None:
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    for root in (base, extra):
        image = root / "images" / "train" / "same.jpg"
        label = root / "labels" / "train" / "same.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(root.name.encode())
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (base / "data.yaml").write_text("names:\n- tomato\n", encoding="utf-8")

    result = build_detector_experiment(base, extra, tmp_path / "out")

    assert result["base_train"] == 1
    assert result["extra_train"] == 1
    train_images = sorted(path.name for path in (tmp_path / "out" / "images" / "train").glob("*.jpg"))
    assert train_images == ["extra-same.jpg", "same.jpg"]
    assert (tmp_path / "out" / "labels" / "train" / "extra-same.txt").is_file()


def test_build_detector_experiment_remaps_extra_class_ids_to_combined_taxonomy(tmp_path: Path) -> None:
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    image = extra / "images" / "train" / "tomato.jpg"
    label = extra / "labels" / "train" / "tomato.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"tomato")
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (extra / "classes.json").write_text("[\"tomato\"]", encoding="utf-8")
    base.mkdir(parents=True, exist_ok=True)
    (base / "data.yaml").write_text("names:\n- cucumber\n- tomato\n- green beans\n", encoding="utf-8")

    build_detector_experiment(base, extra, tmp_path / "out")

    assert (tmp_path / "out" / "labels" / "train" / "extra-tomato.txt").read_text(encoding="utf-8") == "1 0.5 0.5 0.2 0.2\n"


def test_build_detector_experiment_removes_stale_generated_split_files(tmp_path: Path) -> None:
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    (base / "images" / "train").mkdir(parents=True)
    (base / "labels" / "train").mkdir(parents=True)
    (base / "data.yaml").write_text("names:\n- tomato\n", encoding="utf-8")
    (extra / "classes.json").parent.mkdir(parents=True)
    (extra / "classes.json").write_text("[\"tomato\"]", encoding="utf-8")
    output = tmp_path / "out"
    stale = output / "images" / "external_holdout" / "stale.jpg"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")
    (output / "labels" / "external_holdout").mkdir(parents=True, exist_ok=True)
    (output / "labels" / "external_holdout" / "stale.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    build_detector_experiment(base, extra, output)

    assert not stale.exists()
    assert not (output / "labels" / "external_holdout" / "stale.txt").exists()


def test_pseudo_label_is_review_only_and_keeps_provenance(tmp_path: Path) -> None:
    image = tmp_path / "candidate.jpg"
    image.write_bytes(b"image")
    output = tmp_path / "pseudo"

    assert write_candidate(
        image,
        [{"class_name": "tomato", "confidence": 0.91, "box": [0.1, 0.1, 0.8, 0.8]}],
        output,
        model="detector-v1",
        source="public-source",
        min_confidence=0.85,
    ) is True

    record = json.loads((output / "candidates.jsonl").read_text(encoding="utf-8"))
    assert record["status"] == "review"
    assert record["model"] == "detector-v1"
    assert record["source"] == "public-source"
    assert not (output / "candidate.txt").exists()


def test_audit_detector_dataset_rejects_cross_split_duplicate(tmp_path: Path) -> None:
    for split in ("train", "test"):
        image = tmp_path / "images" / split / "same.jpg"
        label = tmp_path / "labels" / split / "same.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"same")
        label.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    report = audit_dataset(tmp_path)

    assert report["status"] == "review_required"
    assert len(report["cross_split_duplicate_groups"]) == 1
