"""User-approved retention applies only to the exact audited episode content."""
import json
from pathlib import Path

def accepted_reverse(run,path,frames_sha256):
    source=Path(run)/'quality_waivers.json'
    if not source.exists():return False
    entries=json.loads(source.read_text()).get('episodes',[])
    return any(Path(e['path']).resolve()==Path(path).resolve()
        and e['frames_sha256']==frames_sha256 and e['decision']=='include_without_rerecord'
        for e in entries)
