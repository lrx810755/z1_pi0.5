"""Independent accepted-subset audit; never overwrite the original19 completion gate."""
import copy
import datetime
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from config import RUN, WORK

SOURCE = Path('/home/lp531/z1_queue_logs/cup_policy_push_recovery20_20261028_134320')
OLD = WORK/'act_cup_renderfix500_chunk30_batch64_20261028_003937'
NEW = Path('/home/lp531/smolvla_data/z1_cup_policy_push_recovery20_20261028')
CAMERAS = ['observation.images.'+x for x in ['wrist', 'global', 'global_right']]
CROPS = {CAMERAS[0]: (-140, 0, 640, 640), CAMERAS[1]: (-80, 0, 640, 640), CAMERAS[2]: (-80, 0, 640, 640)}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def helpers():
    # Original preparation helpers are read-only; their collection/count gate is not used.
    sys.path.insert(0, str(SOURCE/'training'))
    try:
        return module('original_recovery_prepare', SOURCE/'training/prepare.py')
    finally:
        sys.path.pop(0)


def write(name, data):
    path = RUN/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str)+'\n')
    temp.replace(path)


def complete_windows(expert):
    expert = np.asarray(expert, dtype=bool)
    out = np.zeros(len(expert), dtype=bool)
    if len(expert) >= 30:
        bad = np.r_[0, np.cumsum(~expert)]
        out[:len(expert)-29] = bad[30:] == bad[:-30]
    return out


def quantile_stats(values, mask):
    x = np.asarray(values)[mask].astype(np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('No finite training expert values')
    return dict(mean=x.mean(0).tolist(), std=x.std(0).tolist(), min=x.min(0).tolist(),
                max=x.max(0).tolist(), q01=np.quantile(x, .01, axis=0).tolist(),
                q99=np.quantile(x, .99, axis=0).tolist(), count=[len(x)])


def subset_identities(actual, expected):
    if not actual or len(actual) != len(set(actual)) or not set(actual) <= set(expected):
        raise ValueError('Empty, duplicate, or unknown accepted source identity')
    return set(actual)


class NumericURDFFK:
    """Read the six literal joint origins/axes, matching original Geometry.link_fk.

    No xacro expansion is needed for these six numeric elements; expressions fail closed.
    The source URDF hash is stored with this experiment's audit.
    """
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.chain = []
        for i in range(1, 7):
            joint = root.find(f"joint[@name='joint{i}']")
            origin = joint.find('origin')
            xyz, rpy, axis = [np.array([float(v) for v in s.split()]) for s in
                              [origin.get('xyz'), origin.get('rpy'), joint.find('axis').get('xyz')]]
            if any(a.shape != (3,) or not np.isfinite(a).all() for a in [xyz, rpy, axis]):
                raise ValueError('Joint FK inputs must be finite literal numeric3 vectors')
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
            transform[:3, 3] = xyz
            self.chain.append((transform, axis))

    def link_fk(self, joints):
        transform = np.eye(4)
        for angle, (origin, axis) in zip(joints, self.chain):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(float(angle)*axis).as_matrix()
            transform = transform @ origin @ rotation
        return transform[:3, 3], transform[:3, :3]


def audit_subset():
    from PIL import Image
    audit = module('original_episode_audit', SOURCE/'collector/audit_saved.py')
    expected = audit.expected_sources()
    pins = audit.load(audit.PINS)
    audit.verify_pins(pins['files'])
    audit.validate_identity_set(pins['source_identities'], expected)
    urdf = Path('/home/lp531/ros2_ws/src/z1_ros2/z1_description/urdf/z1.urdf.xacro')
    geometry = NumericURDFFK(urdf)
    protocol = audit.load_module('cup_original_protocol', audit.OLD/'protocol.py')
    minus = audit.load_module('cup_minus_protocol', audit.MINUS/'protocol.py')
    report = dict(accepted=[], rejected=[], original_candidate_count=19,
                  selection='current saved subset authorized by user; original collection incomplete',
                  urdf_path=str(urdf), urdf_sha256=audit.sha(urdf), source_pins_sha256=audit.sha(audit.PINS))
    identities = []
    for folder in sorted(NEW.glob('episode_[0-9]*')):
        try:
            detail = audit.audit_episode(folder, expected, geometry, protocol, minus)
            image_hashes = {}
            # Decode every JPEG as well as the original SOF/EOI structural audit.
            for row in audit.lines(folder/'frames.jsonl'):
                for camera in CAMERAS:
                    image_hashes[row[camera]] = audit.sha(folder/row[camera])
                    with Image.open(folder/row[camera]) as im:
                        im.load()
                        if im.mode != 'RGB':
                            raise ValueError('Expected RGB JPEG')
            identities.append(detail['source_identity'])
            detail['image_hashes'] = image_hashes
            report['accepted'].append(detail)
            print('AUDITED', folder.name, detail['frames'], flush=True)
        except Exception as exc:
            report['rejected'].append(dict(path=str(folder), error=str(exc)))
            print('REJECTED', folder.name, str(exc), flush=True)
    subset_identities(identities, expected)
    report.update(passed=True, accepted_count=len(identities), image_validation='all JPEGs fully decoded',
                  original_collection_complete=False,
                  audited_at=datetime.datetime.now().astimezone().isoformat())
    audit.verify_pins(pins['files'])
    write('accepted_subset_audit.json', report)
    return report


def main():
    h = helpers()
    report = audit_subset()
    records = copy.deepcopy(h.load(OLD/'split_manifest.json')['source_records'])
    h.validate_old_records(records)
    h.assert_pins(records)
    parts = h.old_arrays_and_annotations(records)
    start = records[-1]['dataset_to_index']
    for item in report['accepted']:
        record, values = h.new_record(Path(item['path']), len(records), start)
        records.append(record)
        for name, value in values.items():
            parts[name].append(value)
        start += record['frames']
    arrays, train, val, audit, training = h.assemble(records, parts)
    for record in records:
        a, b = record['dataset_from_index'], record['dataset_to_index']
        if not np.array_equal(arrays['valid'][a:b], complete_windows(arrays['expert'][a:b])):
            raise ValueError('Expert-window alignment mismatch')
    stats = {name: quantile_stats(arrays[key], training & arrays['expert'])
             for name, key in [('action', 'actions'), ('observation.state', 'states')]}
    if stats['action']['q01'][-1] != 0 or stats['action']['q99'][-1] != 1:
        raise ValueError('Binary gripper quantiles must be0/1')
    for camera in CAMERAS:
        stats[camera] = dict(mean=[[[.5]]]*3, std=[[[.5]]]*3, min=[[[0.]]]*3, max=[[[1.]]]*3)
    np.savez(RUN/'lowdim.npz', **{k: arrays[k] for k in ['actions', 'states', 'episode', 'frame']})
    np.savez_compressed(RUN/'sampling_arrays.npz', **{k: arrays[k] for k in
        ['episode', 'frame', 'expert', 'key', 'recovery', 'source_pool', 'eligible_train', 'eligible_validation', 'weights']})
    split = dict(train_episodes=train, validation_episodes=val, source_records=records,
                 holdout_cells=[[1, 1], [2, 3]], subset_audit_sha256=h.sha(RUN/'accepted_subset_audit.json'))
    write('split_manifest.json', split)
    write('train_stats.json', stats)
    write('sampling_audit.json', audit)
    pins = {str(path): h.sha(path) for path in [RUN/'accepted_subset_audit.json', RUN/'split_manifest.json',
            RUN/'train_stats.json', RUN/'sampling_arrays.npz', RUN/'lowdim.npz']}
    write('preparation_complete.json', dict(passed=True, episodes=len(records), accepted_new=len(report['accepted']),
          train_episodes=len(train), validation_episodes=len(val), files=pins))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
