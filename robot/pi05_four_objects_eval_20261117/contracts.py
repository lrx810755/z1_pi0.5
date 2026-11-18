"""Read-only contracts; importing this module never loads ROS or a GPU model."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = Path('/home/lp531/pi05_lora_workspace/pi05_current_round_20261111')
OUTPUT = Path('/home/lp531/smolvla_training/pi05_current_round_20261111')
FINAL = OUTPUT / 'checkpoints/800000/pretrained_model'
ROS_SOURCE = Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
ROS_PYTHON = '/home/lp531/miniconda3/envs/unifolm-vla/bin/python'
MODEL_PYTHON = '/home/lp531/pi05_lora_workspace/.venv/bin/python'
OBJECTS = ['apple', 'peach', 'cup', 'beverage_bottle']
CAMERAS = ['wrist', 'global', 'global_right']
TASKS = {o: f'Physically grasp the {"bottle" if o == "beverage_bottle" else o}, lift it clear, and place it in the box.' for o in OBJECTS}
CROPS = {'observation.images.' + c: [-140 if c == 'wrist' else -80, 0, 640, 640] for c in CAMERAS}


def read(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'missing {path}')
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_pins():
    pins = read(HERE / 'source_snapshot.json')
    for path, digest in pins['files'].items():
        if sha(path) != digest:
            raise ValueError(f'pinned artifact changed: {path}')
    return pins


def check_final(run=RUN, output=OUTPUT, *, require_reload=False):
    run, output = Path(run), Path(output)
    done = read(run / 'completed.json')
    latest = read(run / 'latest_checkpoint.json')
    checkpoint = output / 'checkpoints/800000/pretrained_model'
    for marker in (done, latest):
        if (marker.get('microsteps'), marker.get('optimizer_updates')) != (800000, 100000):
            raise ValueError('final model requires 800000 microsteps / 100000 optimizer updates')
    if Path(done.get('output', '')).resolve() != output.resolve() or Path(latest.get('checkpoint', '')).resolve() != checkpoint.resolve():
        raise ValueError('final checkpoint belongs to a different training round')
    cfg = read(checkpoint / 'config.json')
    expected = dict(type='pi05', chunk_size=30, n_action_steps=15, num_inference_steps=10, use_relative_actions=False)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('checkpoint config does not match approved action contract')
    inputs = cfg.get('input_features', {})
    shapes = {'observation.state': [7], **{'observation.images.' + c: [3, 360 if c == 'wrist' else 480, 640] for c in CAMERAS}}
    if set(inputs) != set(shapes) or any(inputs[k]['shape'] != s for k, s in shapes.items()) or cfg['output_features']['action']['shape'] != [7]:
        raise ValueError('checkpoint config does not match seven joints and three raw cameras')
    adapter = read(checkpoint / 'adapter_config.json')
    if adapter.get('r') != 32 or adapter.get('lora_alpha') != 32 or adapter.get('lora_dropout') != 0:
        raise ValueError('adapter config differs from approved LoRA')
    base = Path(adapter['base_model_name_or_path'])
    if base.resolve() != (RUN.parent / 'weights/pi05_base').resolve() or not (base / 'model.safetensors').is_file():
        raise ValueError('official local base is missing or different')
    required = ['adapter_model.safetensors', 'tokenizer/tokenizer.json']
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        processor = read(checkpoint / name)
        reference_path = run / 'processor_verification' / name
        if reference_path.is_file():
            reference_processor = read(reference_path)
            def canonical(value):
                value = copy.deepcopy(value)
                for step in value['steps']:
                    if step['registry_name'] == 'device_processor':
                        step['config']['device'] = '<runtime device>'
                return value
            if canonical(processor) != canonical(reference_processor):
                raise ValueError(f'saved processor behavior differs from frozen preparation: {name}')
        for step in processor['steps']:
            if step.get('state_file'):
                required.append(step['state_file'])
        if name == 'policy_preprocessor.json':
            crops = [s['config'] for s in processor['steps'] if s['registry_name'] == 'image_crop_resize_processor']
            if crops != [dict(crop_params_dict=CROPS, resize_size=[224, 224])]:
                raise ValueError('saved image letterbox processor differs')
        normals = [s for s in processor['steps'] if s['registry_name'] in ('normalizer_processor', 'unnormalizer_processor')]
        if len(normals) != 1 or any(normals[0]['config']['norm_map'].get(k) != 'QUANTILES' for k in ('STATE', 'ACTION')):
            raise ValueError('saved normalization must use frozen quantiles')
    for name in required:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f'missing checkpoint state/artifact: {name}')
    # Compare deployed quantiles to the serialized, verified preparation processors.
    reference = run / 'processor_verification'
    if reference.exists():
        for name in required:
            if name.startswith('policy_') and (reference / name).is_file() and sha(reference / name) != sha(checkpoint / name):
                raise ValueError(f'final normalization state differs from frozen preparation: {name}')
    if require_reload:
        evidence = read(run / 'final_checkpoint_reload_verified.json')
        if not evidence.get('passed') or Path(evidence.get('checkpoint', '')).resolve() != checkpoint.resolve():
            raise ValueError('final checkpoint reload verification missing or mismatched')
        if evidence.get('preparation_sha256') != sha(run / 'preparation_complete.json'):
            raise ValueError('final reload verification preparation hash differs')
    return checkpoint


def checkpoint_fingerprints(checkpoint):
    checkpoint = Path(checkpoint)
    return {str(p.relative_to(checkpoint)): sha(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()}


def file_identity(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def check_bridge_verification(checkpoint, evidence_path=HERE / 'final_bridge_verified.json'):
    evidence = read(evidence_path)
    if not evidence.get('passed') or evidence.get('checkpoint') != str(checkpoint) or evidence.get('pins') != read(HERE / 'source_snapshot.json'):
        raise ValueError('Run SH/pi05_verify_final.sh: deployed final bridge verification missing or stale')
    if evidence.get('checkpoint_files') != checkpoint_fingerprints(checkpoint):
        raise ValueError('Final checkpoint files changed since deployment verification')
    base_file = Path(read(Path(checkpoint) / 'adapter_config.json')['base_model_name_or_path']) / 'model.safetensors'
    if evidence.get('base_file_identity') != file_identity(base_file):
        raise ValueError('Base model file changed since deployment verification')
    return evidence


def training_coverage(manifest, physics):
    scene = read(physics)['scene_randomization']
    bounds = (scene['workspace_x_m'], scene['workspace_y_m'])
    shape = scene['levels']['full']['target_grid_shape']
    objects = {o: dict(cells=set(), positions=[]) for o in OBJECTS}
    seeds = set()
    records = read(manifest)['source_records']
    for record in records:
        target = 'beverage_bottle' if record['object'] == 'bottle' else record['object']
        if target not in objects or record['task'] != TASKS[target]:
            raise ValueError('frozen object/task text mismatch')
        if record.get('seed') is not None:
            seeds.add(int(record['seed']))
        summary_path = Path(record['source_path']) / 'episode_summary.json'
        expected_summary_hash = record.get('source_hashes', {}).get('episode_summary.json')
        if expected_summary_hash and sha(summary_path) != expected_summary_hash:
            raise ValueError(f'training layout metadata changed: {summary_path}')
        summary = read(summary_path)
        randomization = summary.get('scene_randomization', {})
        if randomization.get('seed') is not None:
            seeds.add(int(randomization['seed']))
        position = randomization.get('object_positions_m', {}).get(target)
        if position is None or not all(math.isfinite(float(v)) for v in position[:2]):
            raise ValueError(f'missing training layout: {record["source_path"]}')
        xy = [float(v) for v in position[:2]]
        cell = tuple(min(n - 1, max(0, int((v - b[0]) / (b[1] - b[0]) * n))) for v, b, n in zip(xy, bounds, shape))
        objects[target]['cells'].add(cell)
        objects[target]['positions'].append(xy)
    for target, data in objects.items():
        if not data['positions']:
            raise ValueError(f'missing training coverage: {target}')
        data['cells'] = [list(c) for c in sorted(data['cells'])]
    return dict(records=len(records), training_seeds=sorted(seeds), objects=objects,
                manifest_sha256=sha(manifest), physics_sha256=sha(physics))


def validate_seeds(seeds, coverage):
    if seeds and set(seeds) & set(coverage['training_seeds']):
        raise ValueError('evaluation seed was used for training')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='本轮PI0.5最终模型四物体评估接口；默认只检查，不启动场景或评估。')
    modes = parser.add_mutually_exclusive_group()
    for name in ['check', 'dry-run', 'scene', 'run', 'verify-model']:
        modes.add_argument('--' + name, dest='mode', action='store_const', const=name)
    parser.set_defaults(mode='check')
    parser.add_argument('--objects', nargs='+', choices=OBJECTS + ['bottle', 'all'], default=['all'])
    parser.add_argument('--episodes-per-object', type=int, default=20)
    parser.add_argument('--execute-steps', type=int, choices=[10, 15, 20, 25, 30], default=15,
                        help='Execute this many of the model 30 predicted actions before observing again')
    parser.add_argument('--eval-seeds', nargs='+', type=int)
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('/home/lp531/smolvla_eval/pi05_four_objects_final_20261117'))
    args = parser.parse_args(argv)
    args.objects = list(dict.fromkeys(OBJECTS if 'all' in args.objects else ['beverage_bottle' if o == 'bottle' else o for o in args.objects]))
    if args.episodes_per_object < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('episode count and timeout must be positive')
    if args.eval_seeds and (len(args.eval_seeds) != args.episodes_per_object or len(set(args.eval_seeds)) != len(args.eval_seeds) or min(args.eval_seeds) < 0):
        parser.error('provide one distinct nonnegative seed per episode (same list reused for each target)')
    args.output = args.output.expanduser().resolve()
    return args


def validate_rollout_contract(args):
    if (args.rate, args.execution_rate, args.gripper_action_encoding,
            args.binary_gripper, args.freeze_target_until_close) != (10, 60, 'binary', True, False) or args.execute_steps not in (10, 15, 20, 25, 30):
        raise ValueError('Evaluation parameters differ from the approved deployment contract')


def rollout_arguments(args, checkpoint):
    result = ['--policy-type', 'pi05', '--checkpoint', str(checkpoint), '--objects', *args.objects,
              '--episodes-per-object', str(args.episodes_per_object), '--timeout', str(args.timeout),
              '--rate', '10', '--execution-rate', '60', '--execute-steps', str(args.execute_steps),
              '--binary-gripper', '--gripper-action-encoding', 'binary', '--gripper-min-close-hold-s', '0',
              '--no-freeze-target-until-close', '--prelift-contact-stable-s', '0',
              '--wrist-rgb-topic', '/left_wrist_camera/color/image_raw', '--reject-blank-cameras',
              '--input-sync-tolerance', '0.045', '--output', str(args.output)]
    if args.eval_seeds:
        result += ['--eval-seeds', *map(str, args.eval_seeds)]
    return result
