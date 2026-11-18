from pathlib import Path
import hashlib,json,datetime
paths=[Path('/home/lp531/robot/cup_cylinder_horizontal200_20261108/base_assistant_snapshot.py'),Path('/home/lp531/robot/cup_horizontal_demo_20261108/base_assistant_snapshot.py'),Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/grasp_assistant.py')]
L=Path(__file__).parent/'direct_release_20261109';L.mkdir(exist_ok=True);records=[]
helper='''CUP_DIRECT_RELEASE_MIN_ROOT_Z_M = 0.115


def _validate_cup_direct_box_release(position, box_center, xy_tolerance):
    """Cup direct release uses a fresh world-root pose, strictly above115mm."""
    if len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
        raise RuntimeError("杯子直接释放前世界位置无效")
    xy_error = math.hypot(position[0] - box_center[0], position[1] - box_center[1])
    if xy_error > xy_tolerance:
        raise RuntimeError(f"杯子直接释放前水平对中超限：{xy_error:.4f} m > {xy_tolerance:.4f} m")
    if position[2] <= CUP_DIRECT_RELEASE_MIN_ROOT_Z_M:
        raise RuntimeError(f"杯子直接释放前高度不足：杯底 z={position[2]:.6f} m，必须严格大于0.115 m")


'''
for index,p in enumerate(paths):
 original=p.read_text();assert '_validate_cup_direct_box_release' not in original
 (L/f'before_{index}.py').write_text(original)
 text=original.replace('BOX_FALLBACK_RELEASE_MAX_OBJECT_Z_M = 0.225\n','BOX_FALLBACK_RELEASE_MAX_OBJECT_Z_M = 0.225\n'+helper,1)
 anchor='''            # The verified loaded carry has already delivered the object into the
'''
 assert text.count(anchor)==1
 text=text.replace(anchor,'''            # Keep the original carry clearance; cup alignment/release can finish
            # above the explicitly requested115mm world-root floor.
            if target_object == "cup":
                minimum_clearance_z = CUP_DIRECT_RELEASE_MIN_ROOT_Z_M
'''+anchor,1)
 start=text.index('            # Descend from the converged position instead of re-targeting the box');end=text.index('        self._require_loaded_physical_grasp(\n            target_object,\n            profile_envelope,\n            "高位回退释放前"',start)
 old=text[start:end]
 # Other objects keep their previous descent and no-IK fallback unchanged.
 new='''            fallback_release = False
            if target_object == "cup":
                _validate_cup_direct_box_release(position, self.box_center, release_xy_tolerance)
            else:
'''+''.join('    '+line+'\n' for line in old.splitlines())
 text=text[:start]+new+text[end:]
 text=text.replace('''            "高位回退释放前" if fallback_release else "到达蓝盒释放位后",
        )
        self._set_status(
            "GUIDED_RELEASE",''','''            "杯子对中后直接释放前" if target_object == "cup" else (
                "高位回退释放前" if fallback_release else "到达蓝盒释放位后"
            ),
        )
        if target_object == "cup":
            _validate_cup_direct_box_release(
                self._current_object_position(target_object), self.box_center,
                BOX_RELEASE_XY_TOLERANCE_M.get(target_object, 0.020),
            )
        self._set_status(
            "GUIDED_RELEASE",''',1)
 assert text!=original and '_validate_cup_direct_box_release(\n                self._current_object_position' in text
 compile(text,str(p),'exec');p.write_text(text)
 records.append(dict(path=str(p),before_sha256=hashlib.sha256(original.encode()).hexdigest(),after_sha256=hashlib.sha256(text.encode()).hexdigest()))
(L/'changes.json').write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),cup_only=True,skip_descent=True,release_root_z_strictly_greater_than_m=.115,xy_tolerance_m=.03,files=records),ensure_ascii=False,indent=2))
print('Cup-only direct release updated in current, original cup snapshot and production assistant')
