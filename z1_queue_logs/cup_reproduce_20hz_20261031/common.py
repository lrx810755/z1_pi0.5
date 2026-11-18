"""Shared20Hz audits and persistent faults; all grasp behavior stays in original collectors."""
import json
from pathlib import Path
from basic.audit import audit_rows,audit_episode
from basic.fault_latch import latch_fault,check_fault
ROOT=Path(__file__).resolve().parent

def guarded_recorder(Base,component):
 fault=ROOT/component/'sync_fault.json'
 class GuardedRecorder(Base):
  def _sample_callback(self):
   with self._lock:
    try:super()._sample_callback()
    finally:latch_fault(fault,self._sync_failure,self._episode_index)
  def _save_callback(self,req,res):
   with self._lock:
    try:
     check_fault(fault)
     if self._frames_file:self._frames_file.flush()
     rows=[json.loads(x) for x in (self._episode_dir/'frames.jsonl').read_text().splitlines()]
     audit_rows(rows)
     saved=self.dataset_root/f'episode_{self._episode_index:06d}'
     res=super()._save_callback(req,res)
     if res.success:
      # Derive action-window masks from expert flags at the actual20Hz frame rate.
      maskfile=saved/'training_mask.json'
      if maskfile.exists():
       mask=json.loads(maskfile.read_text());flags=mask['expert_frame_mask'];assert len(flags)==len(rows)
       for horizon in [30,60]:mask[f'valid_action_horizon{horizon}_anchors']=[i+horizon<=len(flags) and all(flags[i:i+horizon]) for i in range(len(flags))]
       maskfile.write_text(json.dumps(mask,indent=2)+'\n')
      report=audit_episode(saved)
      path=ROOT/component/'episode_audits.json';reports=json.loads(path.read_text()) if path.exists() else []
      assert report['episode'] not in {x['episode'] for x in reports}
      reports.append(report);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(reports,indent=2)+'\n');temp.replace(path)
     return res
    except Exception as e:
     latch_fault(fault,'Independent20Hz audit failed: '+str(e),self._episode_index)
     res.success=False;res.message='Independent20Hz audit failed: '+str(e);return res
 return GuardedRecorder
