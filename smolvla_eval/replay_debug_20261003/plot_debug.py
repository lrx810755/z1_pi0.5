from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BASE=Path(__file__).resolve().parent

def load(relative):
    return [json.loads(line) for line in (BASE/relative).read_text().splitlines()]

fig, axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
for name, label in [('approach_collision_on','Collisions on'),('approach_collision_off','Collisions off')]:
    rows=load(name+'_trace.jsonl')
    axes[0].plot([r['scheduled_s'] for r in rows], [max(abs(np.array(r['source_state'][:6])-np.array(r['joint_state'][:6]))) for r in rows],label=label)
axes[0].set(title='Old episode 89: approach',xlabel='Source time (s)',ylabel='State difference (rad)')
for name,label in [('raw_teacher_10hz','10 Hz hold'),('raw_teacher_60hz','60 Hz commands')]:
    rows=load(name+'_trace.jsonl')
    axes[1].plot([r['scheduled_s'] for r in rows],[r['status'].get('object_position_m',[0,0,0])[2] for r in rows],label=label)
axes[1].set(title='Collision-compatible teacher commands',xlabel='Source time (s)',ylabel='Bottle origin height (m)',ylim=(-.02,.25))
for name,label in [('verified_sparse_repeat_1','Slipped'),('verified_sparse_repeat_2','Successful')]:
    rows=load('verified_replay_20261003_225919/'+name+'_trace.jsonl');rows=[r for r in rows if 11<=r['scheduled_s']<=16]
    axes[2].plot([r['scheduled_s'] for r in rows],[r['joint_state'][-1] for r in rows],label=label)
axes[2].set(title='Remaining grasp sensitivity',xlabel='Source time (s)',ylabel='Actual gripper joint (rad)')
for ax in axes:ax.grid(alpha=.2);ax.legend()
fig.savefig(BASE/'replay_debug_comparison.png',dpi=170)
