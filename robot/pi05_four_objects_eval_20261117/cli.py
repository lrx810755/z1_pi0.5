"""Manual launcher with side-effect-free check and command preview."""
import json
import os
import shlex
import subprocess
import sys

from contracts import (HERE, RUN, FINAL, ROS_PYTHON, MODEL_PYTHON, check_final,
                       check_pins, check_bridge_verification, parse_args, read, rollout_arguments,
                       training_coverage, validate_seeds)


def scene_command(args):
    return ['ros2', 'launch', str(HERE / 'scene.launch.py'), 'external_camera_layout:=dual',
            'headless:=' + str(args.headless).lower(), 'rviz:=false',
            'target_object:=' + args.objects[0], 'task_gui:=true', 'task_gui_port:=8868',
            'run_smolvla_policy:=false', 'freeze_target_keep_collisions:=true',
            'wrist_rgb_topic:=/left_wrist_camera/color/image_raw', 'vla_wrist_rgb_topic:=none',
            'dataset_root:=' + str(args.output / 'inactive_recorder'), 'recording_fps:=10.0']


def command_environment(mode):
    env = os.environ.copy()
    if mode == 'run':
        # The established Python3.10 evaluation environment uses user-site Torch
        # dependencies; keep system ROS launch processes isolated as before.
        env.pop('PYTHONNOUSERSITE', None)
    return env


def main(argv=None):
    args = parse_args(argv)
    check_pins()
    coverage = training_coverage(RUN / 'split_manifest.json', HERE / 'grasp_physics.json')
    validate_seeds(args.eval_seeds, coverage)
    try:
        checkpoint = check_final(require_reload=args.mode in ('scene', 'run'))
        status = dict(final_checkpoint_ready=True, checkpoint=str(checkpoint))
    except (ValueError, KeyError) as exc:
        status = dict(final_checkpoint_ready=False, expected_checkpoint=str(FINAL), reason=str(exc))
        if args.mode not in ('check', 'dry-run'):
            print(json.dumps(status, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        checkpoint = FINAL
    if args.mode == 'check':
        status.update(interface_ready=True, objects=args.objects, source_episodes=coverage['records'],
                      training_seed_count=len(coverage['training_seeds']), launched=False,
                      latest_training=read(RUN / 'live_training.json'),
                      final_reload_verified=(RUN / 'final_checkpoint_reload_verified.json').is_file())
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status['final_checkpoint_ready'] else 2
    rollout = [ROS_PYTHON, str(HERE / 'entry.py'), *rollout_arguments(args, checkpoint)]
    if args.mode == 'dry-run':
        print(json.dumps(dict(**status, launched=False, scene=shlex.join(scene_command(args)),
                              evaluation=shlex.join(rollout), output=str(args.output)), ensure_ascii=False, indent=2))
        return 0
    if args.mode == 'verify-model':
        env = os.environ.copy()
        for key in ['PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE']:
            env.pop(key, None)
        command = [MODEL_PYTHON, str(RUN / 'verify.py'), '--reload-final']
        result = subprocess.run(command, env=env)
        if result.returncode:
            return result.returncode
        check_final(require_reload=True)
        # Verify the deployed raw RGB bridge separately from the training Dataset path.
        return subprocess.run([MODEL_PYTHON, str(HERE / 'verify_bridge.py')], env=env).returncode
    if args.mode == 'run':
        check_bridge_verification(checkpoint)
    command = scene_command(args) if args.mode == 'scene' else rollout
    os.execvpe(command[0], command, command_environment(args.mode))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f'评估接口未就绪: {exc}', file=sys.stderr)
        raise SystemExit(2)
