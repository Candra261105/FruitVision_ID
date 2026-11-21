import io
import json
from pathlib import Path

from PIL import Image

from research.convert_hf_detection import convert_parquet, read_parquet_class_names


def test_convert_parquet_writes_yolo_box_from_image_dimensions(tmp_path: Path, monkeypatch) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    buffer = io.BytesIO()
    Image.new("RGB", (100, 50), "red").save(buffer, format="JPEG")
    table = pa.table(
        {
            "image": [{"bytes": buffer.getvalue(), "path": "one.jpg"}],
            "objects": [{"bbox": [[10.0, 5.0, 20.0, 10.0]], "categories": [0]}],
        }
    )
    source = tmp_path / "source.parquet"
    parquet.write_table(table, source)

    counts = convert_parquet(source, tmp_path / "out", "sample", {0: "tomato"})

    assert counts == {"tomato": 1}
    assert (tmp_path / "out" / "images" / "sample_000000.jpg").is_file()
    fields = (tmp_path / "out" / "labels" / "sample_000000.txt").read_text(encoding="utf-8").split()
    assert fields[0] == "0"
    assert [float(value) for value in fields[1:]] == [0.2, 0.2, 0.2, 0.2]


def test_read_parquet_class_names_reads_huggingface_classlabel_metadata(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    table = pa.table(
        {
            "image": [{"bytes": b"x", "path": "one.jpg"}],
            "objects": [{"bbox": [[]], "categories": [0]}],
        }
    )
    metadata = {
        "info": {
            "features": {
                "objects": {
                    "categories": {"feature": {"names": ["Tomato", "Green Banana"]}}
                }
            }
        }
    }
    table = table.replace_schema_metadata({b"huggingface": json.dumps(metadata).encode("utf-8")})
    source = tmp_path / "source.parquet"
    parquet.write_table(table, source)

    assert read_parquet_class_names(source) == {0: "Tomato", 1: "Green Banana"}


def test_convert_parquet_normalizes_tomato_variants_to_one_label(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(buffer, format="PNG")
    table = pa.table(
        {
            "image": [{"bytes": buffer.getvalue(), "path": "one.png"}],
            "objects": [{"bbox": [[0, 0, 10, 10], [10, 0, 10, 10]], "categories": [0, 1]}],
        }
    )
    source = tmp_path / "source.parquet"
    parquet.write_table(table, source)

    counts = convert_parquet(source, tmp_path / "out", "sample", {0: "Tomato", 1: "tomatoes"})

    assert counts == {"tomato": 2}
    assert json.loads((tmp_path / "out" / "classes.json").read_text(encoding="utf-8")) == ["tomato"]
