"""Fail before collection if the running scene did not report the render fix."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime


def main():
    p=argparse.ArgumentParser();p.add_argument('--logs',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    observations=[]
    for line in (a.logs/'scene.log').read_text().splitlines():
        if 'Close-range glass occlusion:' in line:
            observations.append(ast.literal_eval(line.split('Close-range glass occlusion:',1)[1].strip()))
    if not observations or observations[-1].get('/rtx/translucency/worldEps')!=.0001:
        raise RuntimeError('Scene did not confirm worldEps=0.0001; collection refused')
    startup=json.loads((a.logs/'startup_verified.json').read_text())
    if Path(startup['dataset_root']).resolve()!=a.data.resolve() or startup['right_topic']!='/global_right_camera/color/image_raw':
        raise RuntimeError('Startup recorder root/cameras mismatch')
    root=Path(__file__).resolve().parent
    result=dict(verified=True,time=datetime.now().astimezone().isoformat(),
        runtime_render_setting=observations[-1],data_root=str(a.data.resolve()),
        policy='column0 original; columns1..4 horizontal',goal=160,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
    (a.logs/'rerecord_start_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
