"""Verify the final160 successful episodes; leave every raw file untouched."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--logs',type=Path,required=True);a=p.parse_args()
    start=json.loads((a.logs/'rerecord_start_verified.json').read_text())
    assert start['verified'] and start['runtime_render_setting']['/rtx/translucency/worldEps']==.0001
    grid=Counter();episodes=[]
    for f in sorted(a.data.glob('episode_*/episode_summary.json')):
        s=json.loads(f.read_text());assert s['target_object']=='beverage_bottle'
        assert all(s.get(k) is True for k in ['task_success','physical_grasp_ok','bilateral_contact_ever','object_lifted_clear_of_table_ever','release_sequence_complete','placement_stable'])
        assert s['dropped_frames']==0 and s['fps']==10
        cell=tuple(s['scene_randomization']['target_grid']);assert cell[0] in range(5) and cell[1] in range(4);grid[cell]+=1
        provenance=json.loads((f.parent/'recording_provenance.json').read_text())
        assert provenance['column']==cell[0] and provenance['approach_policy']==('original' if cell[0]==0 else 'horizontal')
        raw=(f.parent/'frames.jsonl').read_bytes();rows=[json.loads(line) for line in raw.splitlines()];assert len(rows)==s['frame_count']
        timing=json.loads((f.parent/'time_screen.json').read_text())
        assert timing['passed'] and timing['longest_candidate_wait_s']<8
        assert timing['frames_sha256']==hashlib.sha256(raw).hexdigest()
        for key in ['observation.state','action']:
            values=np.asarray([r[key] for r in rows],float);assert values.shape==(len(rows),7) and np.isfinite(values).all()
        assert np.allclose(np.diff([r['simulation_time'] for r in rows]),.1,atol=1e-6)
        assert all(r['action_stamp']<=r['simulation_time']+1e-7 for r in rows)
        for row in rows:
            for camera,shape in [('wrist',(640,360)),('global',(640,480)),('global_right',(640,480))]:
                with Image.open(f.parent/row['observation.images.'+camera]) as image:
                    assert image.size==shape;image.load()
        episodes.append(dict(path=str(f.parent),grid=list(cell),frames=len(rows),frames_sha256=hashlib.sha256(raw).hexdigest()))
    assert len(episodes)==160 and grid==Counter({(c,r):8 for c in range(5) for r in range(4)})
    old_hashes=json.loads((Path(__file__).resolve().parent/'old160_metadata_hashes.json').read_text())
    for path,expected in old_hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    result=dict(verified=True,completed_at=datetime.now().astimezone().isoformat(),episodes=160,
        grid_counts={f'{c},{r}':n for (c,r),n in sorted(grid.items())},old160_lowdim_metadata_unchanged=True,
        renderer_world_eps=.0001,first_column='original approach',other_columns='horizontal approach',training_started=False)
    (a.logs/'saved_integrity.json').write_text(json.dumps(dict(result,episode_details=episodes),ensure_ascii=False,indent=2))
    (a.logs/'completion_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
