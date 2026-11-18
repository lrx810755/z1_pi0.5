"""Shared cup collision geometry for PhysX and MoveIt; no ROS/USD imports."""

import math


CUP_COLLISION = {
    "shape": "frustum",
    "bottom_radius_m": 0.018,
    "top_radius_m": 0.0252,
    "height_m": 0.061,
    "sides": 64,
}


def cup_frustum_mesh():
    """Return a closed, outward-wound convex mesh, in cup-root coordinates."""
    sides = CUP_COLLISION["sides"]
    points = []
    for radius, z in (
        (CUP_COLLISION["bottom_radius_m"], 0.0),
        (CUP_COLLISION["top_radius_m"], CUP_COLLISION["height_m"]),
    ):
        for i in range(sides):
            angle = 2.0 * math.pi * i / sides
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    faces = [list(reversed(range(sides))), list(range(sides, 2 * sides))]
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, sides + j, sides + i])
    return points, [len(face) for face in faces], [i for face in faces for i in face]

