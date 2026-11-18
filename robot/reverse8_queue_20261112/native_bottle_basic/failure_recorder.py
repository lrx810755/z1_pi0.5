"""Retain failed raw clips separately; only verified successes enter training."""
import hashlib,json,os,time,uuid
from pathlib import Path
ARCHIVE=Path('/home/lp531/smolvla_eval/bottle_rerecord160_renderfix_20261111_failures')


def archive_failed_episode(source,archive,metadata):
    source=Path(source);archive=Path(archive)
    if not source.is_dir():raise RuntimeError('失败回合原始目录不存在')
    archive.mkdir(parents=True,exist_ok=True)
    metadata=dict(metadata,archived_at_unix_s=time.time(),source_directory=str(source))
    temp=source/'failure_summary.json.tmp'
    with temp.open('w') as stream:
        json.dump(metadata,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    temp.replace(source/'failure_summary.json')
    destination=archive/f'failed_{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    source.rename(destination)
    return destination


def main():
    from z1_isaac_bridge import episode_recorder as module
    Base=module.EpisodeRecorder
    class FailurePreservingRecorder(Base):
        def _save_callback(self,request,response):
            with self._lock:
                if self._recording and self._episode_dir is not None:
                    grid=self._episode_randomization.get('target_grid',[])
                    if len(grid)!=2 or grid[0] not in range(5) or grid[1] not in range(4):
                        response.success=False;response.message='重录160：缺少有效目标网格，拒绝保存';return response
                    code_root=Path(__file__).resolve().parent
                    self._frames_file.flush()
                    try:
                        from time_gate import screen
                        screen(self._episode_dir)
                    except (ValueError,OSError,KeyError) as exc:
                        response.success=False;response.message='重录时间验收失败：'+str(exc);return response
                    proof=dict(collection='bottle_rerecord160_renderfix_20261111',grid=list(grid),
                        column=int(grid[0]),approach_policy='original' if grid[0]==0 else 'horizontal',
                        grasp_config_sha256=hashlib.sha256((code_root/'grasp_physics.json').read_bytes()).hexdigest(),
                        render_start_verification='/home/lp531/z1_queue_logs/reverse8_bottle_basic_17_20261112/rerecord_start_verified.json')
                    temp=self._episode_dir/'recording_provenance.json.tmp'
                    with temp.open('w') as stream:
                        json.dump(proof,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
                    temp.replace(self._episode_dir/'recording_provenance.json')
                return super()._save_callback(request,response)
        def _discard_callback(self,request,response):
            with self._lock:
                if self._recording and self._episode_dir is not None and self._episode_dir.exists():
                    self._close_frames_file_locked()
                    metadata=dict(target_object=self._target_object,episode_index=self._episode_index,
                                  frame_count=self._frame_count,task_success=False,discarded=True,
                                  final_task_status=dict(self._task_status),reason='failed_or_discarded_recording')
                    try:
                        destination=archive_failed_episode(self._episode_dir,ARCHIVE,metadata)
                    except Exception as exc:
                        response.success=False;response.message='失败片段归档失败，已保留原目录且未删除：'+str(exc)
                        return response
                    self.get_logger().info('失败片段已独立保存：'+str(destination))
                # Original path no longer exists: reset recorder state without deleting archived images.
                return super()._discard_callback(request,response)
    module.EpisodeRecorder=FailurePreservingRecorder
    module.main()

if __name__=='__main__':main()
