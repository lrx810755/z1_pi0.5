import json
from pathlib import Path
RUN=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/act_cup_recovery_rgb224_cache_20261020')
CAMERAS=['observation.images.'+x for x in ['wrist','global','global_right']]
def write(name,value):
 p=RUN/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str));t.replace(p)
