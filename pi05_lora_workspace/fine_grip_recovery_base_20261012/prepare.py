"""Frozen data inputs for this training run; no conversion or collection."""
from pathlib import Path
import json
RUN=Path(__file__).resolve().parent
INPUT=json.loads((RUN/'training_input.json').read_text())
MERGED=Path(INPUT['dataset_root']);REPO='lp531/z1_bottle_with_fragments_20261012'
def write(name,value):
 (RUN/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
