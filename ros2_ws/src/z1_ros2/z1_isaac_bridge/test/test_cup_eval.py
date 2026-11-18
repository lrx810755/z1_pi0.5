from z1_isaac_bridge.smolvla_sim_rollout import parse_args

def test_cup_act_checkpoint_accepts_single_cup_target_and_fifteen_steps():
    a=parse_args(['--policy-type','act','--objects','cup',
        '--checkpoint','/home/lp531/smolvla_training/act_cup_gentle_cached_chunk16_20261019/checkpoints/050000/pretrained_model',
        '--target-xy-bounds','0.36','0.69','-0.27','0.074',
        '--execute-steps','15','--episodes-per-object','30',
        '--binary-gripper','--gripper-action-encoding','binary',
        '--gripper-min-close-hold-s','1','--rate','10','--execution-rate','60'])
    assert a.objects==['cup'] and a.execute_steps==15 and a.episodes_per_object==30
    assert a.target_xy_bounds==(0.36,0.69,-0.27,0.074)
