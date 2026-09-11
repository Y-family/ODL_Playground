#!/usr/bin/env python3
"""Replace both five-axis legs with ZP10D in the original 27-item tree.

The negative-Y leg is the locally validated source.  Only its existing
printed parts are mirrored to the positive-Y side.  Every positive-Y servo
is a fresh rigid placement of the supplied ZP10D STEP, never mirrored vendor
geometry.  No assembly occurrence is added.
"""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step

import build_zp10d_left_leg_complete as m


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "cad/local_fix/open_duck_mini_14_servos_numbered_fixed.step"
OUTPUT = PROJECT_ROOT / "cad/local_fix/open_duck_mini_10_zp10d_numbered.step"
EXPECTED_ROOT_OCCURRENCES = 27
EXPECTED_LEAF_OCCURRENCES = 145


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _one(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r}, found {len(matches)}")
    return matches[0]


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


def _compound_like(source: bd.Shape, children: list[bd.Shape]) -> bd.Compound:
    result = bd.Compound(children=children)
    result.label = source.label
    result.location = source.location
    return result


def _replace_by_label(
    source: bd.Shape, replacements: dict[str, bd.Shape]
) -> bd.Compound:
    source_children = _children(source)
    labels = {child.label for child in source_children}
    missing = set(replacements) - labels
    if missing:
        raise ValueError(f"{source.label} lacks {sorted(missing)!r}")
    return _compound_like(
        source,
        [replacements.get(child.label, child) for child in source_children],
    )


def _mirror_printed(source: bd.Shape, label: str | None = None) -> bd.Shape:
    result = source.mirror(bd.Plane.XZ)
    result.label = label or source.label
    result.color = source.color
    if not result.is_valid or len(result.solids()) != 1 or result.volume <= 0.0:
        raise ValueError(f"mirrored {result.label} is not one valid solid")
    return result


def _servo_compound(placement: m.ServoPlacement, label: str) -> bd.Compound:
    result = m._servo_compound(placement, label)
    return result


def _horn_center(servo: bd.Shape) -> bd.Vector:
    passive = _one(servo, "ZP10D_vendor_passive_horn")
    active = _one(servo, "ZP10D_vendor_active_horn")
    a = passive.bounding_box().center()
    b = active.bounding_box().center()
    return bd.Vector(
        (a.X + b.X) / 2.0,
        (a.Y + b.Y) / 2.0,
        (a.Z + b.Z) / 2.0,
    )


def _moved_placement(
    placement: m.ServoPlacement, transform: bd.Location
) -> m.ServoPlacement:
    center_vertex = bd.Vertex(
        placement.center.X, placement.center.Y, placement.center.Z
    ).moved(transform)
    axis_tip_vertex = bd.Vertex(
        placement.center.X + placement.axis.X,
        placement.center.Y + placement.axis.Y,
        placement.center.Z + placement.axis.Z,
    ).moved(transform)
    center = center_vertex.center()
    axis = (axis_tip_vertex.center() - center).normalized()
    return m.ServoPlacement(
        pose=transform * placement.pose,
        axis=axis,
        center=center,
        body=transform * placement.body,
        passive_horn=transform * placement.passive_horn,
        active_horn=transform * placement.active_horn,
    )


def _component_centers_match(
    right_servo: bd.Shape, placement: m.ServoPlacement
) -> bool:
    left_parts = _children(m._servo_compound(placement))
    for left_part in left_parts:
        expected = []
        for right_part in _children(right_servo):
            if abs(left_part.volume - right_part.volume) > m.VOLUME_TOLERANCE_MM3:
                continue
            center = right_part.bounding_box().center()
            expected.append(bd.Vector(center.X, -center.Y, center.Z))
        actual = left_part.bounding_box().center()
        if not expected or min((actual - point).length for point in expected) > 0.001:
            return False
    return True


def _align_body_center(
    placement: m.ServoPlacement, mirrored_from: bd.Shape
) -> m.ServoPlacement:
    source_body = max(_children(mirrored_from), key=lambda part: part.volume)
    source_center = source_body.bounding_box().center()
    desired = bd.Vector(source_center.X, -source_center.Y, source_center.Z)
    actual = placement.body.bounding_box().center()
    delta = desired - actual
    return _moved_placement(placement, bd.Pos(delta.X, delta.Y, delta.Z))


def _aligned_left_servo(
    old_servo: bd.Shape,
    mirrored_from: bd.Shape,
    name: str,
) -> m.ServoPlacement:
    placement = m._servo_placement(old_servo, name)
    placement = _align_body_center(placement, mirrored_from)
    if _component_centers_match(mirrored_from, placement):
        return placement
    if abs(placement.axis.Y) < 0.999:
        raise ValueError(f"{name} requires an unsupported vendor-body reorientation")
    center = placement.center
    flip = (
        bd.Pos(center.X, center.Y, center.Z)
        * bd.Rot(0.0, 0.0, 180.0)
        * bd.Pos(-center.X, -center.Y, -center.Z)
    )
    placement = _moved_placement(placement, flip)
    placement = _align_body_center(placement, mirrored_from)
    if not _component_centers_match(mirrored_from, placement):
        raise ValueError(f"{name} cannot be rigidly aligned to the mirrored interface")
    return placement


def _require_mirrored_servo_placement(
    label: str,
    right_servo: bd.Shape,
    left_servo: bd.Shape,
) -> None:
    right_parts = _children(right_servo)
    left_parts = _children(left_servo)
    if len(right_parts) != 3 or len(left_parts) != 3:
        raise ValueError(f"{label} no longer has three vendor components")
    for left_part in left_parts:
        expected_centers = []
        for right_part in right_parts:
            if abs(left_part.volume - right_part.volume) > m.VOLUME_TOLERANCE_MM3:
                continue
            center = right_part.bounding_box().center()
            expected_centers.append(bd.Vector(center.X, -center.Y, center.Z))
        actual = left_part.bounding_box().center()
        if not expected_centers or min((actual - point).length for point in expected_centers) > 0.001:
            raise ValueError(f"{label} component {left_part.label} is not mirror-aligned")


def _axis_hole_faces(
    shape: bd.Shape, axis: str, center: bd.Vector
) -> list[tuple[bd.Face, float]]:
    radii = (m.HORN_HOLE_DIAMETER_MM / 2.0, m.HORN_COUNTERBORE_DIAMETER_MM / 2.0)
    result = []
    for face in shape.faces():
        if "CYLINDER" not in str(face.geom_type):
            continue
        feature_axis = face.axis_of_rotation
        direction = getattr(feature_axis.direction, axis.upper())
        if abs(abs(direction) - 1.0) > 0.001:
            continue
        position = feature_axis.position
        if axis == "x":
            radial = ((position.Y - center.Y) ** 2 + (position.Z - center.Z) ** 2) ** 0.5
        elif axis == "y":
            radial = ((position.X - center.X) ** 2 + (position.Z - center.Z) ** 2) ** 0.5
        else:
            raise ValueError(f"unsupported hole axis {axis!r}")
        if abs(radial - m.HORN_PATTERN_RADIUS_MM) > 0.01:
            continue
        if face.radius is None:
            continue
        radius = float(face.radius)
        if any(abs(radius - candidate) <= 0.001 for candidate in radii):
            result.append((face, radius))
    return result


def _remap_y_horn_host(
    host: bd.Shape,
    horn: bd.Shape,
    center: bd.Vector,
) -> bd.Shape:
    old_faces = _axis_hole_faces(host, "y", center)
    through_radius = m.HORN_HOLE_DIAMETER_MM / 2.0
    counterbore_radius = m.HORN_COUNTERBORE_DIAMETER_MM / 2.0
    if len(old_faces) != 8:
        raise ValueError(
            f"{host.label} expected 8 old horn-hole faces, found {len(old_faces)}"
        )
    counterbore_depths = []
    fillers = []
    for face, radius in old_faces:
        box = face.bounding_box()
        point = face.axis_of_rotation.position
        fillers.append(
            m._local_y_cylinder(
                box.min.Y,
                box.max.Y,
                point.X,
                point.Z,
                radius + m.PITCH_SHEET_HOLE_FILL_OVERLAP_MM,
            )
        )
        if abs(radius - counterbore_radius) <= 0.001:
            counterbore_depths.append(box.size.Y)
    if len(counterbore_depths) != 4:
        raise ValueError(f"{host.label} does not expose four old counterbores")
    depth = sum(counterbore_depths) / len(counterbore_depths)
    filled = m._require_one_solid(host.fuse(*fillers), host.label)
    new_centers = m._y_axis_pattern_centers(
        horn, center, m.HORN_PATTERN_RADIUS_MM
    )
    if len(new_centers) != 4:
        raise ValueError(f"{horn.label} does not expose four Y-axis horn holes")
    box = host.bounding_box()
    cutters = [
        m._local_y_cylinder(
            box.min.Y - 1.0,
            box.max.Y + 1.0,
            point.X,
            point.Z,
            through_radius,
        )
        for point in new_centers
    ]
    if box.center().Y > center.Y:
        cb_start, cb_end = box.max.Y + 1.0, box.max.Y - depth
    else:
        cb_start, cb_end = box.min.Y - 1.0, box.min.Y + depth
    cutters.extend(
        m._local_y_cylinder(
            cb_start,
            cb_end,
            point.X,
            point.Z,
            counterbore_radius,
        )
        for point in new_centers
    )
    colour = host.color
    revised = m._require_one_solid(m._cut_sequential(filled, cutters), host.label)
    revised.color = colour
    return revised


def _remap_x_roll_link_holes(
    link: bd.Shape, placement: m.ServoPlacement
) -> bd.Shape:
    center = placement.center
    old_faces = _axis_hole_faces(link, "x", center)
    through_radius = m.HORN_HOLE_DIAMETER_MM / 2.0
    counterbore_radius = m.HORN_COUNTERBORE_DIAMETER_MM / 2.0
    if not old_faces:
        raise ValueError(f"{link.label} has no prior X-axis horn-hole faces")
    old_holes: dict[tuple[float, float], list[tuple[bd.Face, float]]] = {}
    for face, radius in old_faces:
        position = face.axis_of_rotation.position
        key = (round(position.Y, 6), round(position.Z, 6))
        old_holes.setdefault(key, []).append((face, radius))
    if len(old_holes) != 8:
        raise ValueError(
            f"{link.label} expected 8 old horn-hole axes, found {len(old_holes)}"
        )

    fillers = []
    counterbore_depths = []
    for (y, z), features in sorted(old_holes.items()):
        boxes = [face.bounding_box() for face, _ in features]
        fill_radius = max(radius for _, radius in features)
        fillers.append(
            m._local_x_cylinder(
                min(box.min.X for box in boxes),
                max(box.max.X for box in boxes),
                y,
                z,
                fill_radius + m.PITCH_SHEET_HOLE_FILL_OVERLAP_MM,
            )
        )
        counterbore_depths.extend(
            face.bounding_box().size.X
            for face, radius in features
            if abs(radius - counterbore_radius) <= 0.001
        )
    if not counterbore_depths:
        raise ValueError(f"{link.label} has no prior X-axis counterbores")
    depth = sum(counterbore_depths) / len(counterbore_depths)
    filled = link
    for filler in fillers:
        filled = filled.fuse(filler).clean()
    filled = m._require_one_solid(filled, link.label)
    box = link.bounding_box()
    cutters = []
    for horn in (placement.passive_horn, placement.active_horn):
        centers = m._x_axis_pattern_centers(
            horn, center, m.HORN_PATTERN_RADIUS_MM
        )
        if len(centers) != 4:
            raise ValueError(f"{horn.label} does not expose four X-axis horn holes")
        horn_box = horn.bounding_box()
        if horn_box.center().X < center.X:
            start, end = box.min.X - 1.0, horn_box.min.X + 1.0
            cb_start, cb_end = box.min.X - 1.0, box.min.X + depth
        else:
            start, end = horn_box.max.X - 1.0, box.max.X + 1.0
            cb_start, cb_end = box.max.X + 1.0, box.max.X - depth
        cutters.extend(
            m._local_x_cylinder(
                start, end, point.Y, point.Z, through_radius
            )
            for point in centers
        )
        cutters.extend(
            m._local_x_cylinder(
                cb_start, cb_end, point.Y, point.Z, counterbore_radius
            )
            for point in centers
        )
    colour = link.color
    revised = m._require_one_solid(m._cut_sequential(filled, cutters), link.label)
    revised.color = colour
    return revised


@step(out="../../cad/local_fix/open_duck_mini_10_zp10d_numbered.step")
def open_duck_mini_10_zp10d_numbered():
    robot = read_step(SOURCE)
    if len(_children(robot)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError("source root occurrence count changed")

    right_trunk = _one(robot, "trunk_assembly <1>")
    right_hip = _one(robot, "hip_roll_assembly <1>")
    right_pitch = _one(robot, "right_roll_to_pitch_assembly <1>")
    right_upper = _one(robot, "knee_and_ankle_assembly <1>")
    right_lower = _one(robot, "knee_and_ankle_assembly <3>")
    right_foot = _one(robot, "foot_assembly <1>")

    left_trunk_old = _one(right_trunk, "S06_LEFT_HIP_YAW_STS3215")
    left_hip_source = _one(robot, "hip_roll_assembly <2>")
    left_pitch_source = _one(robot, "left_roll_to_pitch_assembly <1>")
    left_upper_source = _one(robot, "knee_and_ankle_assembly <2>")
    left_lower_source = _one(robot, "knee_and_ankle_assembly <4>")
    left_foot_source = _one(robot, "foot_assembly <2>")

    right_servos = {
        "yaw": _one(right_trunk, "S01_RIGHT_HIP_YAW_ZP10D"),
        "roll": _one(right_hip, "S02_RIGHT_HIP_ROLL_ZP10D"),
        "pitch": _one(right_pitch, "S03_RIGHT_HIP_PITCH_ZP10D"),
        "knee": _one(right_upper, "S04_RIGHT_KNEE_ZP10D"),
        "ankle": _one(right_lower, "S05_RIGHT_ANKLE_ZP10D"),
    }
    left_placements = {
        "yaw": _aligned_left_servo(left_trunk_old, right_servos["yaw"], "left_yaw"),
        "roll": _aligned_left_servo(
            _one(left_hip_source, "S07_LEFT_HIP_ROLL_STS3215"),
            right_servos["roll"],
            "left_roll",
        ),
        "pitch": _aligned_left_servo(
            _one(left_pitch_source, "S08_LEFT_HIP_PITCH_STS3215"),
            right_servos["pitch"],
            "left_pitch",
        ),
        "knee": _aligned_left_servo(
            _one(left_upper_source, "S09_LEFT_KNEE_STS3215"),
            right_servos["knee"],
            "left_knee",
        ),
        "ankle": _aligned_left_servo(
            _one(left_lower_source, "S10_LEFT_ANKLE_STS3215"),
            right_servos["ankle"],
            "left_ankle",
        ),
    }
    left_servos = {
        "yaw": _servo_compound(left_placements["yaw"], "S06_LEFT_HIP_YAW_ZP10D"),
        "roll": _servo_compound(left_placements["roll"], "S07_LEFT_HIP_ROLL_ZP10D"),
        "pitch": _servo_compound(left_placements["pitch"], "S08_LEFT_HIP_PITCH_ZP10D"),
        "knee": _servo_compound(left_placements["knee"], "S09_LEFT_KNEE_ZP10D"),
        "ankle": _servo_compound(left_placements["ankle"], "S10_LEFT_ANKLE_ZP10D"),
    }
    for name in left_servos:
        _require_mirrored_servo_placement(
            name, right_servos[name], left_servos[name]
        )

    right_knee_passive = _one(
        right_servos["knee"], "ZP10D_vendor_passive_horn"
    )
    right_knee_active = _one(
        right_servos["knee"], "ZP10D_vendor_active_horn"
    )
    right_knee_center = _horn_center(right_servos["knee"])
    right_lower_left = _remap_y_horn_host(
        _one(right_lower, "left_knee_to_ankle_left_sheet"),
        right_knee_active,
        right_knee_center,
    )
    right_lower_right = _remap_y_horn_host(
        _one(right_lower, "left_knee_to_ankle_right_sheet"),
        right_knee_passive,
        right_knee_center,
    )
    revised_right_lower = _replace_by_label(
        right_lower,
        {
            "left_knee_to_ankle_left_sheet": right_lower_left,
            "left_knee_to_ankle_right_sheet": right_lower_right,
        },
    )

    right_ankle_passive = _one(
        right_servos["ankle"], "ZP10D_vendor_passive_horn"
    )
    right_ankle_active = _one(
        right_servos["ankle"], "ZP10D_vendor_active_horn"
    )
    right_ankle_center = _horn_center(right_servos["ankle"])
    right_foot_top = _remap_y_horn_host(
        _one(right_foot, "foot_top"),
        right_ankle_active,
        right_ankle_center,
    )
    right_foot_side = _remap_y_horn_host(
        _one(right_foot, "foot_side"),
        right_ankle_passive,
        right_ankle_center,
    )
    revised_right_foot = _replace_by_label(
        right_foot,
        {"foot_top": right_foot_top, "foot_side": right_foot_side},
    )

    trunk_top = _one(right_trunk, "trunk_top")
    trunk_bottom = _one(right_trunk, "trunk_bottom")
    trunk_top, trunk_bottom = m._mount_servo_in_original_parts(
        [trunk_top, trunk_bottom], left_placements["yaw"]
    )
    m._colour(trunk_top, m.COLOURS["trunk"])
    m._colour(trunk_bottom, m.COLOURS["trunk"])
    trunk = _replace_by_label(
        right_trunk,
        {
            "trunk_top": trunk_top,
            "trunk_bottom": trunk_bottom,
            "S06_LEFT_HIP_YAW_STS3215": left_servos["yaw"],
        },
    )

    left_roll_top = _mirror_printed(_one(right_hip, "roll_motor_top"))
    left_roll_bottom = _mirror_printed(_one(right_hip, "roll_motor_bottom"))
    left_roll_top, left_roll_bottom = m._mount_servo_in_original_parts(
        [left_roll_top, left_roll_bottom],
        left_placements["roll"],
        bridge_roll_far_side=True,
        require_all_mounts=True,
    )
    left_roll_top, left_roll_bottom = m._mount_horns_in_original_parts(
        [left_roll_top, left_roll_bottom], left_placements["yaw"]
    )
    left_roll_top = m._seat_yaw_active_horn_outside_roll_top(
        left_roll_top, left_placements["yaw"]
    )
    left_roll_obstacles = [
        left_placements["roll"].body,
        left_placements["roll"].passive_horn,
        left_placements["roll"].active_horn,
    ]
    left_roll_top = m._require_one_solid(
        left_roll_top.cut(*left_roll_obstacles), left_roll_top.label
    )
    left_roll_bottom = m._require_one_solid(
        left_roll_bottom.cut(*left_roll_obstacles),
        left_roll_bottom.label,
        max_disconnected_trim=100.0,
    )
    m._colour(left_roll_top, m.COLOURS["roll_top"])
    m._colour(left_roll_bottom, m.COLOURS["roll_bottom"])
    left_hip = _replace_by_label(
        left_hip_source,
        {
            "roll_motor_top": left_roll_top,
            "roll_motor_bottom": left_roll_bottom,
            "S07_LEFT_HIP_ROLL_STS3215": left_servos["roll"],
        },
    )

    left_link = _mirror_printed(
        _one(right_pitch, "right_roll_to_pitch"), "left_roll_to_pitch"
    )
    left_link = _remap_x_roll_link_holes(
        left_link, left_placements["roll"]
    )
    left_link = m._mount_servo_in_original_parts(
        [left_link], left_placements["pitch"]
    )[0]
    left_link = m._clear_pitch_horns_from_roll_link(
        left_link, left_placements["pitch"]
    )
    m._colour(left_link, m.COLOURS["roll_link"])
    left_pitch = _replace_by_label(
        left_pitch_source,
        {
            "left_roll_to_pitch": left_link,
            "S08_LEFT_HIP_PITCH_STS3215": left_servos["pitch"],
        },
    )

    left_upper_left = _mirror_printed(
        _one(right_upper, "left_knee_to_ankle_left_sheet")
    )
    left_upper_right = _mirror_printed(
        _one(right_upper, "left_knee_to_ankle_right_sheet")
    )
    left_upper_left, left_upper_right = m._mount_servo_in_original_parts(
        [left_upper_left, left_upper_right],
        left_placements["knee"],
        require_all_mounts=True,
    )
    left_pitch_passive = _one(
        left_servos["pitch"], "ZP10D_vendor_passive_horn"
    )
    left_pitch_active = _one(
        left_servos["pitch"], "ZP10D_vendor_active_horn"
    )
    left_pitch_center = _horn_center(left_servos["pitch"])
    left_upper_left = _remap_y_horn_host(
        left_upper_left, left_pitch_passive, left_pitch_center
    )
    left_upper_right = _remap_y_horn_host(
        left_upper_right, left_pitch_active, left_pitch_center
    )
    left_upper = _replace_by_label(
        left_upper_source,
        {
            "left_knee_to_ankle_left_sheet": left_upper_left,
            "left_knee_to_ankle_right_sheet": left_upper_right,
            "leg_spacer": _mirror_printed(_one(right_upper, "leg_spacer")),
            "S09_LEFT_KNEE_STS3215": left_servos["knee"],
        },
    )

    left_lower_left = _mirror_printed(right_lower_left)
    left_lower_right = _mirror_printed(right_lower_right)
    left_lower_left, left_lower_right = m._mount_servo_in_original_parts(
        [left_lower_left, left_lower_right],
        left_placements["ankle"],
        require_all_mounts=True,
    )
    left_knee_passive = _one(
        left_servos["knee"], "ZP10D_vendor_passive_horn"
    )
    left_knee_active = _one(
        left_servos["knee"], "ZP10D_vendor_active_horn"
    )
    left_knee_center = _horn_center(left_servos["knee"])
    left_lower_left = _remap_y_horn_host(
        left_lower_left, left_knee_active, left_knee_center
    )
    left_lower_right = _remap_y_horn_host(
        left_lower_right, left_knee_passive, left_knee_center
    )
    left_lower = _replace_by_label(
        left_lower_source,
        {
            "left_knee_to_ankle_left_sheet": left_lower_left,
            "left_knee_to_ankle_right_sheet": left_lower_right,
            "leg_spacer": _mirror_printed(
                _one(revised_right_lower, "leg_spacer")
            ),
            "S10_LEFT_ANKLE_STS3215": left_servos["ankle"],
        },
    )

    left_ankle_passive = _one(
        left_servos["ankle"], "ZP10D_vendor_passive_horn"
    )
    left_ankle_active = _one(
        left_servos["ankle"], "ZP10D_vendor_active_horn"
    )
    left_ankle_center = _horn_center(left_servos["ankle"])
    left_foot_top = _remap_y_horn_host(
        _mirror_printed(right_foot_top),
        left_ankle_active,
        left_ankle_center,
    )
    left_foot_side = _remap_y_horn_host(
        _mirror_printed(right_foot_side),
        left_ankle_passive,
        left_ankle_center,
    )
    left_foot = _replace_by_label(
        left_foot_source,
        {
            "foot_top": left_foot_top,
            "foot_side": left_foot_side,
            "foot_bottom_pla": _mirror_printed(
                _one(revised_right_foot, "foot_bottom_pla")
            ),
            "foot_bottom_tpu": _mirror_printed(
                _one(revised_right_foot, "foot_bottom_tpu")
            ),
        },
    )
    left_cache = _mirror_printed(_one(robot, "right_cache"), "left_cache")

    root_replacements = {
        "trunk_assembly <1>": trunk,
        "left_cache": left_cache,
        "hip_roll_assembly <2>": left_hip,
        "left_roll_to_pitch_assembly <1>": left_pitch,
        "knee_and_ankle_assembly <2>": left_upper,
        "knee_and_ankle_assembly <4>": left_lower,
        "foot_assembly <2>": left_foot,
        "knee_and_ankle_assembly <3>": revised_right_lower,
        "foot_assembly <1>": revised_right_foot,
    }
    result = _replace_by_label(robot, root_replacements)
    result.label = robot.label

    nodes = list(_walk(result))
    servo_ids = sorted(node.label for node in nodes if node.label.startswith("S"))
    expected_ids = [f"S{index:02}" for index in range(1, 15)]
    if [label[:3] for label in servo_ids] != expected_ids:
        raise ValueError(f"servo IDs are not unique and complete: {servo_ids!r}")
    zp10d_ids = [label for label in servo_ids if label.endswith("_ZP10D")]
    sts_ids = [label for label in servo_ids if label.endswith("_STS3215")]
    if len(zp10d_ids) != 10 or len(sts_ids) != 4:
        raise ValueError(f"expected 10 ZP10D and 4 STS3215, got {len(zp10d_ids)}/{len(sts_ids)}")
    leaves = [node for node in nodes if not _children(node)]
    if len(leaves) != EXPECTED_LEAF_OCCURRENCES:
        raise ValueError(f"leaf occurrence count changed to {len(leaves)}")
    return result


if __name__ == "__main__":
    open_duck_mini_10_zp10d_numbered()
