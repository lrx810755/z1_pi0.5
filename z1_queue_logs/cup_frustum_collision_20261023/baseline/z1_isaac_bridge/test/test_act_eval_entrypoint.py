from pathlib import Path
import subprocess
import pytest
ENTRY=Path('/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh')

@pytest.mark.parametrize('model',['9','act'])
def test_act_selection_preserves_binary_contract_and_execution_window(model):
    result=subprocess.run(['bash',str(ENTRY),'--model',model,'--dry-run'],text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    for expected in ['--policy-type act','--execute-steps 8','--binary-gripper','--gripper-action-encoding binary',
                     'act_correction_chunk16_20261013/checkpoints/050000/pretrained_model',
                     '--objects beverage_bottle','--no-freeze-target-until-close']:
        assert expected in result.stdout

def test_pi05_selection_is_preserved():
    result=subprocess.run(['bash',str(ENTRY),'--model','1','--dry-run'],text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert '--policy-type pi05' in result.stdout
    assert 'pi05_fine_grip_recovery_base_20261012/checkpoints/120000/pretrained_model' in result.stdout

def test_act_shell_command_passes_actual_rollout_parser():
    import shlex
    from z1_isaac_bridge.smolvla_sim_rollout import parse_args
    result=subprocess.run(['bash',str(ENTRY),'--model','9','--dry-run'],text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    command=shlex.split(result.stdout.splitlines()[-1])
    args=parse_args(command[1:])
    assert args.policy_type=='act'
    assert args.target_xy_bounds==(0.35,0.70,-0.27,0.16)
    assert args.execute_steps==8
    assert args.rate==10
    assert args.execution_rate==60
