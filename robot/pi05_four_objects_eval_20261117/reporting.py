"""Summarize observed physical stages without inventing missing evidence."""
STAGES = {
    'close_commanded': ('gripper_commanded_closed', 'gripper_closed_once'),
    'bilateral_contact': ('bilateral_contact_ever', 'bilateral_contact'),
    'lifted': ('object_lifted_clear_of_table_ever', 'lifted'),
    'stable_grasp': ('physical_grasp_ok_ever', 'physical_grasp_ok'),
    'transport': ('object_transported_to_box', 'object_above_box_ever'),
    'release': ('release_sequence_complete',),
    'placement_stable': ('placement_stable',),
}


def stage_summary(results):
    report = {}
    for result in results:
        target = result['target_object']
        counts = report.setdefault(target, {key: dict(true=0, false=0, unknown=0) for key in [*STAGES, 'task_success']})
        evidence = {**result.get('final_status', {}), **result.get('physical_predicates', {})}
        for name, keys in STAGES.items():
            known = [evidence[k] for k in keys if isinstance(evidence.get(k), bool)]
            value = 'true' if any(known) else 'false' if known else 'unknown'
            counts[name][value] += 1
        counts['task_success']['true' if result.get('success') is True else 'false'] += 1
    return report
