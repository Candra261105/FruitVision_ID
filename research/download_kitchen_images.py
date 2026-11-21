import json
import urllib.request
from pathlib import Path

root=Path('data/raw/kitchen')
annotation=json.loads(Path('data/raw/kitchen-annotations/annotations.json').read_text(encoding='utf-8'))
root.mkdir(parents=True,exist_ok=True)
for index,image in enumerate(annotation.get('images',[]),1):
    relative=Path(str(image['file_name']))
    target=root/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size == int(image.get('size',0) or 0):
        continue
    url='https://huggingface.co/datasets/omorros/kitchen-fruit-vegetable-detection/resolve/main/'+str(image['file_name'])
    urllib.request.urlretrieve(url,target)
    print(f'{index}/{len(annotation["images"])} {target.name}',flush=True)
