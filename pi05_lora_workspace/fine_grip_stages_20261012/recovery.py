"""Overlay logged pre-close recovery at weight 2, without multiplying phase weights."""
import numpy as np
STAGES={'lift_clear','correct_orientation','align_above','approach'}

def apply_recovery_weight(rows,stages,close,weights):
    weights=np.array(weights,dtype=float,copy=True)
    if weights.shape!=(len(rows),):raise ValueError('Weight/frame length mismatch')
    mask=np.zeros(len(rows),dtype=bool)
    for stage in stages:
        if stage.get('stage') not in STAGES:continue
        start=stage['start_sim_time'];end=stage['end_sim_time']
        if end<start:raise ValueError('Reversed recovery interval')
        mask|=np.array([i<close and start<=r['simulation_time']<=end for i,r in enumerate(rows)])
    weights[mask]=np.maximum(weights[mask],2.)
    return weights,mask
