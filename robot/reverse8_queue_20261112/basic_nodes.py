#!/usr/bin/python3
"""Fixed-source scene setup and provenance for native basic experts."""
import json
import os
from pathlib import Path
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent



def collection_ready(live, target):
    assistant=live.get('assistant',{})
    return bool(live.get('connected') and assistant.get('target_object')==target
        and all(live.get(k,{}).get('ready') for k in
            ['recording','episode_reset','collision_scene','object_collisions']))


def context():return json.loads(Path(os.environ['Z1_REVERSE8_CURRENT']).read_text())


def gui_main():
    import z1_isaac_bridge.grasp_gui_bridge as module
    from std_msgs.msg import String
    Base=module.GraspGuiBridge
    module.GRID_LAYOUT_RESAMPLE_ATTEMPTS=2
    class RetakeGui(Base):
        def snapshot(self):
            status=super().snapshot()
            with self._lock:
                status['assistant']=dict(self._pregrasp_status)
            return status
        def randomize_episode(self,curriculum_level='full',target_grid=None):
            task=context()['task']
            self._call_trigger(self._episode_reset_client,'/smolvla/reset_episode',timeout_s=60.)
            with self._lock:self._scene_randomization_status={}
            message=String();message.data=json.dumps(dict(seed=task['seed'],level='full',
                target_grid=task['grid'],settle_duration_s=2.,non_target_max_angular_speed_rad_s=.35))
            self._randomization_command_pub.publish(message)
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                with self._lock:state=dict(self._scene_randomization_status)
                if state.get('phase')=='ERROR':raise RuntimeError('Source-layout reset failed: '+str(state))
                if state.get('phase')=='RANDOMIZED' and state.get('seed')==task['seed']:
                    if state.get('target_grid')!=task['grid']:raise RuntimeError('Source grid changed')
                    return dict(success=True,message='Pinned source seed/grid restored with current rendering/geometry')
                if self._force_stop_event.is_set():raise RuntimeError('Retake stopped during reset')
                time.sleep(.05)
            raise RuntimeError('Source-layout reset timed out')
        def start_recording(self):
            result=super().start_recording()
            deadline=time.monotonic()+10.
            expected=context()['task']
            while time.monotonic()<deadline:
                with self._lock:live=dict(self._recording_status)
                if live.get('sync_failure'):
                    raise RuntimeError('Recording synchronization failed before movement: '+live['sync_failure'])
                if (live.get('recording') and live.get('frame_count',0)>=3
                        and live.get('randomization_seed')==expected['seed']):
                    return result
                if self._force_stop_event.is_set():raise RuntimeError('Stopped before first synchronized frames')
                time.sleep(.05)
            raise RuntimeError('Recorder did not produce three synchronized frames before movement')

        def _generate_pregrasp_for_auto(self):
            timeout=1800. if context()['task']['object']=='beverage_bottle' else 600.
            return self._call_trigger(self._generate_client,'/smolvla/auto_generate_pregrasp',timeout_s=timeout)
    module.GraspGuiBridge=RetakeGui;module.main()


def recorder_main():
    import hashlib
    import z1_isaac_bridge.episode_recorder as module
    from quality_gate import screen
    Base=module.EpisodeRecorder
    class RetakeRecorder(Base):
        def _start_callback(self,req,res):
            self.ctx=context()
            task=self.ctx['task']
            if hashlib.sha256((Path(task['source'])/'frames.jsonl').read_bytes()).hexdigest()!=task['source_frames_sha256']:
                res.success=False;res.message='Original source changed; refusing stale retake';return res
            self.require_task_success=True
            return super()._start_callback(req,res)
        def _save_callback(self,req,res):
            with self._lock:
                if not self._recording or self._episode_dir is None:return super()._save_callback(req,res)
                task=self.ctx['task']
                scene=self._episode_randomization
                if scene.get('seed')!=task['seed'] or scene.get('target_grid')!=task['grid']:
                    res.success=False;res.message='Saved layout differs from retake task';return res
                self._frames_file.flush()
                try:quality=screen(self._episode_dir)
                except (ValueError,OSError,KeyError,AssertionError) as exc:
                    res.success=False;res.message='Reverse8 retake gate: '+str(exc);return res
                proof=dict(accepted=True,task_id=task['id'],mark_id=task['mark_id'],
                    plan_hash=self.ctx['plan_hash'],source=task['source'],
                    source_frames_sha256=task['source_frames_sha256'],object=task['object'],
                    original_seed=task.get('original_seed',task['seed']),layout_reference=task.get('layout_reference'),
                    layout_reference_frames_sha256=task.get('layout_reference_frames_sha256'),
                    source_pool=task['source_pool'],grid=task['grid'],seed=task['seed'],
                    frames_sha256=quality['frames_sha256'],attempt_id=self.ctx['attempt_id'],
                    approach_policy='apple_native' if task['object']=='apple' else 'original' if task['grid'][0]==0 else 'horizontal',
                    quality_gate='reverse<8cm and wait<8s')
                from review import atomic_json
                atomic_json(self._episode_dir/'rerecord_provenance.json',proof)
                return super()._save_callback(req,res)
        def _discard_callback(self,req,res):
            with self._lock:
                if self._recording and self._episode_dir is not None and self._episode_dir.exists():
                    self._close_frames_file_locked()
                    archive=Path(context()['logs'])/'rejected_attempts';archive.mkdir(parents=True,exist_ok=True)
                    self._episode_dir.rename(archive/(str(time.time_ns())+'_'+uuid.uuid4().hex[:8]))
                return super()._discard_callback(req,res)
    module.EpisodeRecorder=RetakeRecorder;module.main()


if __name__=='__main__':
    role=sys.argv.pop(1)
    {'gui':gui_main,'recorder':recorder_main}[role]()
