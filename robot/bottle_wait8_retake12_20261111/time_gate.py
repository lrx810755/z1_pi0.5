"""Reject long stationary waits before a newly recorded episode can be saved."""
import json
from pathlib import Path
import sys

sys.path.insert(0,'/home/lp531/robot/dataset_trajectory_review')
from review import Settings, analyze, chain_from_urdf, DEFAULT_URDF

_CHAIN=None


def screen(folder,maximum_wait_s=8.):
    global _CHAIN
    if _CHAIN is None:_CHAIN=chain_from_urdf(DEFAULT_URDF)
    if _CHAIN is None:raise ValueError('Z1 FK unavailable; cannot certify waiting-time gate')
    result=analyze(Path(folder)/'frames.jsonl',_CHAIN,Settings())
    waits=[f for f in result['findings'] if f['type'] in ('长停顿候选','末尾停留偏长')]
    longest=max((f['recorded_s'] for f in waits),default=0.)
    report=dict(passed=longest<maximum_wait_s,maximum_allowed_wait_s=maximum_wait_s,
        longest_candidate_wait_s=longest,frames_sha256=result['frames_sha256'],intervals=waits,
        deliberate_interventions_excluded=True)
    (Path(folder)/'time_screen.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:raise ValueError(f'连续近静止等待{longest:.2f}s达到{maximum_wait_s:g}s，拒收并重新录制')
    return report
