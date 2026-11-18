"""Pure geometry checks for Isaac scene layout validation."""

import math


def _rotate_vector_wxyz(quaternion, vector):
    w, x, y, z = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    dot = x * vx + y * vy + z * vz
    cross = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    norm = x * x + y * y + z * z
    return (
        2.0 * dot * x + (w * w - norm) * vx + 2.0 * w * cross[0],
        2.0 * dot * y + (w * w - norm) * vy + 2.0 * w * cross[1],
        2.0 * dot * z + (w * w - norm) * vz + 2.0 * w * cross[2],
    )


def target_approach_direction(
    target_object, target_position, target_orientation_wxyz, sweep_config
):
    local_axis = sweep_config.get("object_approach_local_axes", {}).get(
        target_object
    )
    if local_axis is None:
        axis = (float(target_position[0]), float(target_position[1]))
    else:
        rotated = _rotate_vector_wxyz(target_orientation_wxyz, local_axis)
        axis = (rotated[0], rotated[1])
    length = math.hypot(*axis)
    if length <= 1.0e-9:
        raise ValueError("target approach axis has no horizontal projection")
    direction = (axis[0] / length, axis[1] / length)
    if sum(direction[i] * float(target_position[i]) for i in range(2)) < 0:
        direction = (-direction[0], -direction[1])
    if target_object in sweep_config.get("reverse_object_axis_approach", []):
        direction = (-direction[0], -direction[1])
    return direction


def _point_to_aabb_distance(point, x_bounds, y_bounds):
    x, y = float(point[0]), float(point[1])
    dx = max(float(x_bounds[0]) - x, 0.0, x - float(x_bounds[1]))
    dy = max(float(y_bounds[0]) - y, 0.0, y - float(y_bounds[1]))
    return math.hypot(dx, dy)


def _segment_intersects_expanded_aabb(
    start, end, x_bounds, y_bounds, expansion
):
    minimums = (float(x_bounds[0]) - expansion, float(y_bounds[0]) - expansion)
    maximums = (float(x_bounds[1]) + expansion, float(y_bounds[1]) + expansion)
    t_min, t_max = 0.0, 1.0
    for axis in range(2):
        delta = float(end[axis]) - float(start[axis])
        if abs(delta) <= 1.0e-12:
            if not minimums[axis] <= float(start[axis]) <= maximums[axis]:
                return False
            continue
        enter = (minimums[axis] - float(start[axis])) / delta
        leave = (maximums[axis] - float(start[axis])) / delta
        if enter > leave:
            enter, leave = leave, enter
        t_min, t_max = max(t_min, enter), min(t_max, leave)
        if t_min > t_max:
            return False
    return True


def layout_sweep_blockers(
    target_object,
    target_position,
    target_orientation_wxyz,
    object_positions,
    randomization_config,
    _direction=None,
):
    """Return blockers for the actual per-target horizontal approach axis."""
    sweep = randomization_config["target_sweep_clearance"]
    if _direction is None:
        directions = [target_approach_direction(
            target_object, target_position, target_orientation_wxyz, sweep
        )]
        directions.extend(
            (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
            for angle in sweep.get("extra_world_azimuths_deg_by_object", {}).get(
                target_object, ()
            )
        )
        if len(directions) > 1:
            failures = [layout_sweep_blockers(
                target_object, target_position, target_orientation_wxyz,
                object_positions, randomization_config, direction
            ) for direction in directions]
            return next((blockers for blockers in failures if not blockers),
                        min(failures, key=len))
        _direction = directions[0]
    radii = randomization_config["object_footprint_radii_m"]
    box_x = randomization_config["box_exclusion_x_m"]
    box_y = randomization_config["box_exclusion_y_m"]
    direction = _direction
    before = float(sweep["approach_before_target_m"])
    beyond = float(sweep.get("approach_beyond_target_by_object_m", {}).get(
        target_object, sweep["approach_beyond_target_m"]
    ))
    closing_radius = float(sweep["closing_radius_m"])
    lift_radius = float(sweep["lift_radius_m"])
    target_xy = (float(target_position[0]), float(target_position[1]))
    blockers = {}

    box_distance = _point_to_aabb_distance(target_xy, box_x, box_y)
    if box_distance < closing_radius:
        blockers["smolvla_box"] = "蓝盒闭爪空间"
    elif box_distance < lift_radius:
        blockers["smolvla_box"] = "蓝盒初始抬升空间"
    else:
        robot_start = tuple(target_xy[i] - before * direction[i] for i in range(2))
        target_end = tuple(target_xy[i] + beyond * direction[i] for i in range(2))
        if (
            _segment_intersects_expanded_aabb(
                robot_start, target_xy, box_x, box_y,
                float(sweep["robot_side_half_width_m"]),
            )
            or _segment_intersects_expanded_aabb(
                target_xy, target_end, box_x, box_y,
                float(sweep["target_side_half_width_m"]),
            )
        ):
            blockers["smolvla_box"] = "蓝盒夹爪接近扫掠通道"

    for name, position in object_positions.items():
        if name == target_object or name not in radii:
            continue
        if len(position) >= 3 and float(position[2]) < -0.10:
            continue
        radius = float(radii[name])
        relative = (
            float(position[0]) - target_xy[0],
            float(position[1]) - target_xy[1],
        )
        longitudinal = relative[0] * direction[0] + relative[1] * direction[1]
        lateral = abs(relative[0] * -direction[1] + relative[1] * direction[0])
        center_distance = math.hypot(*relative)
        if center_distance < closing_radius + radius:
            blockers[name] = "闭爪空间"
            continue
        if center_distance < lift_radius + radius:
            blockers[name] = "初始抬升空间"
            continue
        robot_limit = float(sweep["robot_side_half_width_m"]) + radius
        target_limit = float(sweep["target_side_half_width_m"]) + radius
        if longitudinal < -before:
            distance, limit = math.hypot(longitudinal + before, lateral), robot_limit
        elif longitudinal <= 0.0:
            distance, limit = lateral, robot_limit
        elif longitudinal <= beyond:
            distance, limit = lateral, target_limit
        else:
            distance, limit = math.hypot(longitudinal - beyond, lateral), target_limit
        if distance < limit:
            blockers[name] = "夹爪接近扫掠通道"
    return blockers
