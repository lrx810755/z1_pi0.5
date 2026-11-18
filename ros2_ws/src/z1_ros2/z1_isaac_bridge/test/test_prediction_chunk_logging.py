import json
from types import SimpleNamespace
import numpy as np
import torch
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout


def test_logs_future_close_beyond_execution_window(tmp_path):
    chunk=torch.zeros(1,16,7);chunk[0,12:,-1]=1
    node=SimpleNamespace(use_amp=False,args=SimpleNamespace(execute_steps=8),
        policy=SimpleNamespace(predict_action_chunk=lambda batch:chunk),
        _prediction_trace_path=tmp_path/'prediction_chunks.jsonl',_joint_sim_stamp=100.)
    result=SmolVLASimRollout._predict(node,{'observation.state':torch.zeros(1,7)})
    assert result.shape==(8,7)
    logged=json.loads(node._prediction_trace_path.read_text())
    assert logged['first_close_step_06']==12
    assert logged['close_in_execution_window'] is False
    assert len(logged['gripper_prediction'])==16
    np.testing.assert_array_equal(logged['input_state'],[0]*7)
