import json
from pathlib import Path
from research.convert_hf_detection import read_parquet_class_names, convert_parquet

jobs=[
 ('data/raw/bangladesh-smartphone-parquet/bangladesh.parquet','data/processed/bangladesh'),
 ('data/raw/fruit-worldwide-00000/fruit-worldwide-00000.parquet','data/processed/fruit-worldwide-00000'),
 ('data/raw/fruit-worldwide-00001/fruit-worldwide-00001.parquet','data/processed/fruit-worldwide-00001'),
]
for source_text, output_text in jobs:
 source=Path(source_text); output=Path(output_text)
 classes=read_parquet_class_names(source)
 (output/'source_classes.json').parent.mkdir(parents=True,exist_ok=True)
 (output/'source_classes.json').write_text(json.dumps(classes,ensure_ascii=False,indent=2),encoding='utf-8')
 counts=convert_parquet(source,output,source.stem,classes)
 print(json.dumps({'source':str(source),'classes':classes,'counts':counts},ensure_ascii=False))
