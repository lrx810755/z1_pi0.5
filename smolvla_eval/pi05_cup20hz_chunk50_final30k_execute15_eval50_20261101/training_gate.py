"""Never evaluate an incomplete, unverified or still-GPU-resident training run."""
import json
from pathlib import Path
def training_ready(root,checkpoint,training_service_active):
 root=Path(root);checkpoint=Path(checkpoint)
 p=root/'workflow_status.json'
 if not p.exists():return False
 status=json.loads(p.read_text())
 if status.get('stage') in ['failed','blocked']:raise RuntimeError('Training did not complete successfully: '+str(status))
 if status.get('stage')!='completed' or training_service_active:return False
 complete=json.loads((root/'completed.json').read_text());reload=json.loads((root/'final_checkpoint_reload_verified.json').read_text())
 if complete.get('optimizer_updates')!=30000 or complete.get('microsteps')!=240000:raise ValueError('Not final30000-update model')
 if reload.get('passed') is not True or Path(reload.get('checkpoint','')).resolve()!=checkpoint.resolve():raise ValueError('Final checkpoint reload gate failed/mismatched')
 if not reload.get('tests') or not all(t.get('prediction_shape')==[1,50,7] and t.get('finite') is True for t in reload['tests']):raise ValueError('Expected finite50-step prediction verification')
 config=json.loads((checkpoint/'config.json').read_text())
 if config.get('chunk_size')!=50 or not (checkpoint/'adapter_model.safetensors').is_file():raise ValueError('Checkpoint horizon/weights mismatch')
 return True
