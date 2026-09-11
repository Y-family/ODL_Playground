#!/usr/bin/env python3
"""Complete the numbered 14-servo ZP10D robot from the validated leg model."""

from __future__ import annotations

from pathlib import Path

import cadgen
from cadgen import build123d as bd
from cadgen import read_step, report, step

import build_zp10d_left_leg_complete as m
from build_zp10d_head_internal_original_part import (
    head_internal_zp10d_no_new_parts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROBOT = PROJECT_ROOT / "cad/local_fix/open_duck_mini_10_zp10d_current_lower.step"
EXPECTED_ROOT_OCCURRENCES = 27
EXPECTED_LEAF_OCCURRENCES = 137
ORIGINAL_ROOT_LABEL = "Open_Duck_Mini_v2"
MAX_HORN_BRIDGE_GAP_MM = 8.0
CROSS_AXIS_BODY_CLEARANCE_MM = 0.20
CROSS_AXIS_FUSE_TOLERANCES_MM = (0.001, 0.01, 0.05, 0.10)
CASE_BOSS_OVERLAP_MM = 1.0
MAX_CASE_BOSS_GAP_MM = 5.0
HEAD_PLACEMENT_POSITION = (-20.6876888093, 147.4272365477, 33.06911715999)
HEAD_PLACEMENT_ORIENTATION = (-8.793931720976, -22.86107399046, -8.381437692179)
HEAD_ROLL_CLEARANCE_MM = 0.30
HEAD_BOTTOM_SLOT_X_RANGE_MM = (-13.0, 15.0)
HEAD_BOTTOM_SLOT_Y_RANGES_MM = (
    (-58.5588, -42.6678),
    (42.8678, 60.2022),
)
HEAD_YAW_HEIGHT_INCREASE_MM = 3.0
CROSS_LINK_SPLICE_HALF_HEIGHT_MM = 0.5
MOTION_CLEARANCE_MM = 0.30


KINEMATICS = {
    "mates": [
        cadgen.revolute(
            "right_hip_yaw",
            parent="#o1.25",
            child="#o1.13",
            origin=(-19.0, -35.0, 64.365938812),
            direction=(0.0, 0.0, -1.0),
            limits=(-30.0, 30.0),
        ),
        cadgen.revolute(
            "right_hip_roll",
            parent="#o1.13",
            child="#o1.17",
            origin=(-18.375, -35.0, -0.074061188),
            direction=(1.0, 0.0, 0.0),
            limits=(-25.0, 25.0),
        ),
        cadgen.revolute(
            "right_hip_pitch",
            parent="#o1.17",
            child="#o1.20",
            origin=(-35.06, -90.725, -0.074061188),
            direction=(0.0, -1.0, 0.0),
            limits=(-30.0, 70.0),
        ),
        cadgen.revolute(
            "right_knee",
            parent="#o1.20",
            child="#o1.14",
            origin=(-35.06, -90.6, -78.724061188),
            direction=(0.0, 1.0, 0.0),
            limits=(-90.0, 90.0),
        ),
        cadgen.revolute(
            "right_ankle",
            parent="#o1.14",
            child="#o1.18",
            origin=(-35.06, -90.7, -157.374061188),
            direction=(0.0, 1.0, 0.0),
            limits=(-90.0, 90.0),
        ),
        cadgen.revolute(
            "left_hip_yaw",
            parent="#o1.25",
            child="#o1.24",
            origin=(-19.0, 35.0, 64.365938812),
            direction=(0.0, 0.0, -1.0),
            limits=(-30.0, 30.0),
        ),
        cadgen.revolute(
            "left_hip_roll",
            parent="#o1.24",
            child="#o1.26",
            origin=(-18.375, 35.0, -0.074061188),
            direction=(1.0, 0.0, 0.0),
            limits=(-25.0, 25.0),
        ),
        cadgen.revolute(
            "left_hip_pitch",
            parent="#o1.26",
            child="#o1.27",
            origin=(-35.06, 90.725, -0.074061188),
            direction=(0.0, 1.0, 0.0),
            limits=(-70.0, 30.0),
        ),
        cadgen.revolute(
            "left_knee",
            parent="#o1.27",
            child="#o1.21",
            origin=(-35.06, 90.6, -78.724061188),
            direction=(0.0, -1.0, 0.0),
            limits=(-90.0, 90.0),
        ),
        cadgen.revolute(
            "left_ankle",
            parent="#o1.21",
            child="#o1.23",
            origin=(-35.06, 90.7, -157.374061188),
            direction=(0.0, -1.0, 0.0),
            limits=(-90.0, 90.0),
        ),
        cadgen.revolute(
            "neck_pitch",
            parent="#o1.25",
            child="#o1.9",
            origin=(1.0, 0.725, 90.000938812),
            direction=(0.0, 1.0, 0.0),
            limits=(-20.0, 65.0),
        ),
        cadgen.revolute(
            "head_pitch",
            parent="#o1.9",
            child="#o1.7",
            origin=(1.0, 0.725, 156.000938812),
            direction=(0.0, 1.0, 0.0),
            limits=(-45.0, 45.0),
        ),
        cadgen.revolute(
            "head_yaw",
            parent="#o1.7",
            child="#o1.16",
            origin=(1.0, 0.1, 194.575938812 + HEAD_YAW_HEIGHT_INCREASE_MM),
            direction=(0.0, 0.0, 1.0),
            limits=(-135.0, 135.0),
        ),
        cadgen.revolute(
            "head_roll",
            parent="#o1.16",
            child="#o1.22",
            origin=(67.95, 0.1, 193.850938812 + HEAD_YAW_HEIGHT_INCREASE_MM),
            direction=(-1.0, 0.0, 0.0),
            limits=(-30.0, 30.0),
        ),
        cadgen.fastened(
            "head_right_antenna_holder",
            parent="#o1.22",
            child="#o1.2",
        ),
        cadgen.fastened(
            "head_right_antenna",
            parent="#o1.22",
            child="#o1.5",
        ),
        cadgen.fastened(
            "head_left_antenna_holder",
            parent="#o1.22",
            child="#o1.10",
        ),
        cadgen.fastened(
            "head_left_antenna",
            parent="#o1.22",
            child="#o1.11",
        ),
    ],
    "poses": {"neutral": {}},
}


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _one(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label == label]
    if len(matches) != 1:
        labels = [child.label for child in _children(parent)]
        raise ValueError(
            f"expected one {label!r}, found {len(matches)}; children={labels!r}"
        )
    return matches[0]


def _one_prefix(parent: bd.Shape, prefix: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix!r} child, found {len(matches)}")
    return matches[0]


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


def _is_zp10d(shape: bd.Shape) -> bool:
    parts = _children(shape)
    if len(parts) != 3 or any(_children(part) for part in parts):
        return False
    volumes = sorted(part.volume for part in parts)
    expected = sorted(m.EXPECTED_VENDOR_VOLUMES_MM3)
    return all(
        abs(actual - target) <= m.VOLUME_TOLERANCE_MM3
        for actual, target in zip(volumes, expected, strict=True)
    )


def _one_zp10d(parent: bd.Shape) -> bd.Shape:
    matches = [child for child in _children(parent) if _is_zp10d(child)]
    if len(matches) != 1:
        raise ValueError(f"{parent.label} has {len(matches)} ZP10D candidates")
    return matches[0]


def _normalised_servo(source: bd.Shape, label: str) -> bd.Compound:
    parts = _children(source)
    body = max(parts, key=lambda part: part.volume)
    horns = sorted(
        (part for part in parts if part is not body), key=lambda part: part.volume
    )
    passive, active = horns
    body.label = "ZP10D_vendor_body"
    passive.label = "ZP10D_vendor_passive_horn"
    active.label = "ZP10D_vendor_active_horn"
    m._colour(body, m.COLOURS["servo_body"])
    m._colour(passive, m.COLOURS["servo_horn"])
    m._colour(active, m.COLOURS["servo_horn"])
    result = bd.Compound(children=[body, passive, active])
    result.label = label
    return result


def _compound_like(source: bd.Shape, children: list[bd.Shape]) -> bd.Compound:
    result = bd.Compound(children=children)
    result.label = source.label
    result.location = source.location
    return result


def _replace_by_label(
    source: bd.Shape, replacements: dict[str, bd.Shape]
) -> bd.Compound:
    labels = {child.label for child in _children(source)}
    missing = set(replacements) - labels
    if missing:
        raise ValueError(f"{source.label} lacks {sorted(missing)!r}")
    return _compound_like(
        source,
        [replacements.get(child.label, child) for child in _children(source)],
    )


def _component_order_key(label: str) -> str:
    if label.startswith("S14_"):
        return "S14"
    return label


def _reorder_children_like(reference: bd.Shape, candidate: bd.Shape) -> bd.Compound:
    pools: dict[str, list[bd.Shape]] = {}
    for child in _children(candidate):
        pools.setdefault(_component_order_key(child.label), []).append(child)

    ordered = []
    for child in _children(reference):
        key = _component_order_key(child.label)
        matches = pools.get(key, [])
        if not matches:
            raise ValueError(f"{candidate.label} lacks original tree item {child.label!r}")
        ordered.append(matches.pop(0))

    leftovers = [child.label for matches in pools.values() for child in matches]
    if leftovers:
        raise ValueError(f"{candidate.label} has extra tree items {leftovers!r}")
    return _compound_like(candidate, ordered)


def _center_delta(left: bd.Shape, right: bd.Shape) -> float:
    return (left.bounding_box().center() - right.bounding_box().center()).length


def _placement_from_servo(
    servo: bd.Shape, vendor_body: bd.Shape
) -> m.ServoPlacement:
    body = _one(servo, "ZP10D_vendor_body")
    passive = _one(servo, "ZP10D_vendor_passive_horn")
    active = _one(servo, "ZP10D_vendor_active_horn")
    passive_center = passive.bounding_box().center()
    active_center = active.bounding_box().center()
    return m.ServoPlacement(
        pose=body.location * vendor_body.location.inverse(),
        axis=(active_center - passive_center).normalized(),
        center=(passive_center + active_center) * 0.5,
        body=body,
        passive_horn=passive,
        active_horn=active,
    )


def _body_box_keepout(placement: m.ServoPlacement) -> bd.Shape:
    local_body = placement.pose.inverse() * placement.body
    box = local_body.bounding_box()
    center = box.center()
    local_keepout = bd.Box(
        box.size.X + 2.0 * CROSS_AXIS_BODY_CLEARANCE_MM,
        box.size.Y + 2.0 * CROSS_AXIS_BODY_CLEARANCE_MM,
        box.size.Z + 2.0 * CROSS_AXIS_BODY_CLEARANCE_MM,
        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER),
    ).moved(bd.Pos(center.X, center.Y, center.Z))
    return placement.pose * local_keepout


def _mount_cross_axis_horns_to_existing_part(
    part: bd.Shape, placements: tuple[m.ServoPlacement, ...]
) -> bd.Shape:
    working = part
    prior_cutters = []
    for placement_index, placement in enumerate(placements, start=1):
        pads, holes, clearances = m._outer_horn_bracket_tools(placement)
        bridges = []
        center_y = m.ZP10D_HORN_FACE_MIDPOINT[1]
        center_z = m.ZP10D_HORN_FACE_MIDPOINT[2]
        for horn_x, _ in m.ZP10D_HORN_PATTERNS:
            start = horn_x - m.HORN_PAD_REACH_MM
            end = horn_x + m.HORN_PAD_REACH_MM
            outer = m._local_x_cylinder(
                start,
                end,
                center_y,
                center_z,
                m.HORN_BRACKET_PAD_RADIUS_MM,
            )
            inner = m._local_x_cylinder(
                start - 1.0,
                end + 1.0,
                center_y,
                center_z,
                m.HORN_CLEARANCE_RADIUS_MM,
            )
            bridges.append(placement.pose * outer.cut(inner))

        additions = [
            bridge.fuse(pad, tol=0.001).clean()
            for bridge, pad in zip(bridges, pads, strict=True)
        ]
        for addition_index, addition in enumerate(additions, start=1):
            if prior_cutters:
                addition = m._cut_sequential(addition, prior_cutters).clean()
            fragments = sorted(
                addition.solids(), key=lambda solid: solid.volume, reverse=True
            )
            for fragment_index, fragment in enumerate(fragments, start=1):
                gap = working.distance_to(fragment)
                if gap > MAX_HORN_BRIDGE_GAP_MM:
                    raise ValueError(
                        f"{part.label} axis {placement_index} horn addition "
                        f"{addition_index}.{fragment_index} remains {gap:.3f} mm away"
                    )
                tolerances = (None, *CROSS_AXIS_FUSE_TOLERANCES_MM)
                for tolerance in tolerances:
                    candidate = (
                        working.fuse(fragment).clean()
                        if tolerance is None
                        else working.fuse(fragment, tol=tolerance).clean()
                    )
                    variants = (candidate,) if candidate.is_valid else (candidate.fix(),)
                    for variant in variants:
                        if (
                            variant.is_valid
                            and len(variant.solids()) == 1
                            and variant.volume > 0
                        ):
                            working = variant
                            working.label = part.label
                            break
                    else:
                        continue
                    break
                else:
                    raise ValueError(
                        f"{part.label} axis {placement_index} horn addition "
                        f"{addition_index}.{fragment_index} cannot form one valid solid"
                    )

        body_keepout = _body_box_keepout(placement)
        current_cutters = [body_keepout, *clearances, *holes]
        working = m._cut_sequential(working, current_cutters)
        try:
            working = m._require_one_solid(working, part.label)
        except ValueError as error:
            raise ValueError(
                f"{part.label} axis {placement_index} interface cuts produced "
                "invalid geometry"
            ) from error
        prior_cutters.append(body_keepout)

    for placement_index, placement in enumerate(placements, start=1):
        overlap = m._intersection_volume(working, placement.body)
        if overlap > m.VOLUME_TOLERANCE_MM3:
            raise ValueError(
                f"{part.label} axis {placement_index} body overlap remains "
                f"{overlap:.3f} mm3"
            )
    colour = part.color
    result = m._require_one_solid(working, part.label)
    result.color = colour
    return result


def _wide_case_mount_tools(
    part: bd.Shape, placement: m.ServoPlacement
) -> tuple[list[bd.Shape], list[bd.Shape]]:
    pads = []
    holes = []
    for x, y, z in m.ZP10D_CASE_HOLES:
        pad = (
            placement.pose
            * m._local_x_cylinder(
                x - m.CASE_PAD_EXTRA_REACH_MM,
                x + m.CASE_PAD_EXTRA_REACH_MM,
                y,
                z,
                m.ROLL_CASE_FAR_PAD_RADIUS_MM,
            )
        )
        gap = part.distance_to(pad)
        if gap > MAX_CASE_BOSS_GAP_MM:
            raise ValueError(
                f"{part.label} case boss remains {gap:.3f} mm away"
            )
        if gap > 0.001:
            pad = (
                placement.pose
                * m._local_x_cylinder(
                    x - m.CASE_PAD_EXTRA_REACH_MM,
                    x + m.CASE_PAD_EXTRA_REACH_MM,
                    y,
                    z,
                    m.ROLL_CASE_FAR_PAD_RADIUS_MM
                    + gap
                    + CASE_BOSS_OVERLAP_MM,
                )
            )
        pads.append(pad)
        holes.append(
            placement.pose
            * m._local_x_cylinder(
                x - m.CASE_PAD_EXTRA_REACH_MM - 1.0,
                x + m.CASE_PAD_EXTRA_REACH_MM + 1.0,
                y,
                z,
                m.CASE_HOLE_DIAMETER_MM / 2.0,
            )
        )
    return pads, holes


def _mount_case_with_wide_bosses(
    part: bd.Shape, placement: m.ServoPlacement
) -> bd.Shape:
    pads, holes = _wide_case_mount_tools(part, placement)
    working = part
    for index, pad in enumerate(pads, start=1):
        gap = working.distance_to(pad)
        if gap > 0.001:
            raise ValueError(
                f"{part.label} wide case boss {index} remains {gap:.3f} mm away"
            )
        working = working.fuse(pad).clean()
    working = m._cut_sequential(
        working,
        [
            placement.body,
            placement.passive_horn,
            placement.active_horn,
            *holes,
        ],
    )
    colour = part.color
    result = m._require_one_solid(working, part.label)
    result.color = colour
    return result


def _mount_shared_yaw_link(
    part: bd.Shape,
    case_placement: m.ServoPlacement,
    horn_placement: m.ServoPlacement,
) -> bd.Shape:
    colour = part.color
    case_pads, case_holes = _wide_case_mount_tools(part, case_placement)
    horn_pads, horn_holes = m._horn_mount_tools([part], horn_placement)
    if len(horn_pads) != 4 or len(horn_holes) != 4:
        raise ValueError("head_yaw_to_roll does not expose four S14 horn mounts")

    working = part
    for index, pad in enumerate(case_pads, start=1):
        if working.distance_to(pad) > 0.001:
            raise ValueError(f"head_yaw_to_roll case boss {index} is disconnected")
        working = m._require_one_solid(
            working.fuse(pad).clean(),
            part.label,
        )

    case_body_keepout = _body_box_keepout(case_placement)
    for index, pad in enumerate(horn_pads, start=1):
        trimmed = pad.cut(case_body_keepout).clean()
        fragments = sorted(
            trimmed.solids(), key=lambda solid: solid.volume, reverse=True
        )
        connected = False
        for fragment in fragments:
            if working.distance_to(fragment) > 0.001:
                continue
            candidate = working.fuse(fragment).clean()
            if candidate.is_valid and len(candidate.solids()) == 1:
                working = candidate
                working.label = part.label
                connected = True
        if not connected:
            raise ValueError(
                f"head_yaw_to_roll S14 horn boss {index} is disconnected"
            )

    working = m._cut_sequential(
        working,
        [
            case_placement.body,
            case_placement.passive_horn,
            case_placement.active_horn,
            *case_holes,
            horn_placement.body,
            horn_placement.passive_horn,
            horn_placement.active_horn,
            *horn_holes,
        ],
    )
    result = m._require_one_solid(working, part.label)
    result.color = colour
    return result


def _recut_case_interfaces(
    parts: list[bd.Shape],
    placements: tuple[m.ServoPlacement, ...],
) -> list[bd.Shape]:
    cutters = []
    for placement in placements:
        _, holes = m._case_mount_tools(placement)
        cutters.extend(
            (
                placement.body,
                placement.passive_horn,
                placement.active_horn,
                *holes,
            )
        )

    revised = []
    for part in parts:
        colour = part.color
        working = m._cut_sequential(part, cutters)
        result = m._require_one_solid(working, part.label)
        result.color = colour
        revised.append(result)
    return revised


def _box_from_bounds(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> bd.Solid:
    return bd.Box(
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0],
        align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN),
    ).moved(bd.Pos(x_range[0], y_range[0], z_range[0]))


def _translated_shape(shape: bd.Shape, dz: float) -> bd.Shape:
    result = bd.Pos(0.0, 0.0, dz) * shape
    result.label = shape.label
    result.color = shape.color
    return result


def _translated_placement(
    placement: m.ServoPlacement,
    dz: float,
) -> m.ServoPlacement:
    offset = bd.Pos(0.0, 0.0, dz)
    return m.ServoPlacement(
        pose=offset * placement.pose,
        axis=placement.axis,
        center=placement.center + bd.Vector(0.0, 0.0, dz),
        body=_translated_shape(placement.body, dz),
        passive_horn=_translated_shape(placement.passive_horn, dz),
        active_horn=_translated_shape(placement.active_horn, dz),
    )


def _lengthen_head_pitch_to_yaw(
    part: bd.Shape,
    lower_placement: m.ServoPlacement,
    increase: float,
) -> bd.Shape:
    """Lengthen only the feature-free web between the S12 and S13 mounts."""
    half = CROSS_LINK_SPLICE_HALF_HEIGHT_MM
    splice_z = (
        lower_placement.center.Z
        + m.HORN_BRACKET_PAD_RADIUS_MM
        + 1.0
    )
    box = part.bounding_box()
    margin = 1.0
    lower_clip = _box_from_bounds(
        (box.min.X - margin, box.max.X + margin),
        (box.min.Y - margin, box.max.Y + margin),
        (box.min.Z - margin, splice_z + half),
    )
    upper_clip = _box_from_bounds(
        (box.min.X - margin, box.max.X + margin),
        (box.min.Y - margin, box.max.Y + margin),
        (splice_z - half, box.max.Z + margin),
    )
    middle_clip = _box_from_bounds(
        (box.min.X - margin, box.max.X + margin),
        (box.min.Y - margin, box.max.Y + margin),
        (splice_z - half, splice_z + half),
    )
    lower = m._intersect_solids(part, lower_clip)
    upper = m._intersect_solids(part, upper_clip)
    middle = m._intersect_solids(part, middle_clip)
    if not lower or not upper or not middle:
        raise ValueError("head_pitch_to_yaw splice does not cross one continuous web")

    lower_boundary = splice_z - half
    factors = (1.0, 1.0, (2.0 * half + increase) / (2.0 * half))
    pieces = list(lower)
    pieces.extend(
        piece.scale(factors, about=(0.0, 0.0, lower_boundary))
        for piece in middle
    )
    pieces.extend(_translated_shape(piece, increase) for piece in upper)
    colour = part.color
    result = m._require_one_solid(
        pieces[0].fuse(*pieces[1:]).clean(),
        part.label,
    )
    result.color = colour
    return result


def _motion_clearance_box(
    fixed_part: bd.Shape,
    moving_part: bd.Shape,
    axis: bd.Axis,
    angles: tuple[float, ...],
) -> bd.Solid | None:
    intersections = []
    for angle in angles:
        moved = moving_part.rotate(axis, angle)
        if not m._bounding_boxes_overlap(fixed_part, moved):
            continue
        intersections.extend(m._intersect_solids(fixed_part, moved))
    intersections = [
        item for item in intersections if item.volume > m.VOLUME_TOLERANCE_MM3
    ]
    if not intersections:
        # The approved lower body may already clear this sampled envelope.
        # In that case there is no material to remove from trunk_bottom.
        return None
    box = bd.Compound(children=intersections).bounding_box()
    clearance = MOTION_CLEARANCE_MM
    return _box_from_bounds(
        (box.min.X - clearance, box.max.X + clearance),
        (box.min.Y - clearance, box.max.Y + clearance),
        (box.min.Z - clearance, box.max.Z + clearance),
    )


def _trim_trunk_bottom_motion_clearances(
    trunk_bottom: bd.Shape,
    robot: bd.Shape,
) -> bd.Shape:
    right_roll_link = _one(
        _one(robot, "right_roll_to_pitch_assembly <1>"),
        "right_roll_to_pitch",
    )
    left_roll_link = _one(
        _one(robot, "left_roll_to_pitch_assembly <1>"),
        "left_roll_to_pitch",
    )
    right_upper_sheet = _one(
        _one(robot, "knee_and_ankle_assembly <1>"),
        "left_knee_to_ankle_left_sheet",
    )
    left_upper_sheet = _one(
        _one(robot, "knee_and_ankle_assembly <2>"),
        "left_knee_to_ankle_left_sheet",
    )
    candidates = [
        _motion_clearance_box(
            trunk_bottom,
            right_roll_link,
            bd.Axis((-19.0, -35.0, 64.365938812), (0.0, 0.0, -1.0)),
            (-30.0, -29.0),
        ),
        _motion_clearance_box(
            trunk_bottom,
            right_roll_link,
            bd.Axis((-19.0, -35.0, 64.365938812), (0.0, 0.0, -1.0)),
            (28.0, 29.0, 30.0),
        ),
        _motion_clearance_box(
            trunk_bottom,
            left_roll_link,
            bd.Axis((-19.0, 35.0, 64.365938812), (0.0, 0.0, -1.0)),
            (-30.0, -29.0, -28.0),
        ),
        _motion_clearance_box(
            trunk_bottom,
            left_roll_link,
            bd.Axis((-19.0, 35.0, 64.365938812), (0.0, 0.0, -1.0)),
            (29.0, 30.0),
        ),
        _motion_clearance_box(
            trunk_bottom,
            right_upper_sheet,
            bd.Axis((-18.375, -35.0, -0.074061188), (1.0, 0.0, 0.0)),
            (20.0, 21.0, 22.0, 23.0, 24.0, 25.0),
        ),
        _motion_clearance_box(
            trunk_bottom,
            left_upper_sheet,
            bd.Axis((-18.375, 35.0, -0.074061188), (1.0, 0.0, 0.0)),
            (-25.0, -24.0, -23.0, -22.0, -21.0, -20.0),
        ),
    ]
    cutters = [cutter for cutter in candidates if cutter is not None]
    if not cutters:
        return trunk_bottom
    colour = trunk_bottom.color
    result = m._require_one_solid(
        m._cut_sequential(trunk_bottom, cutters).clean(),
        trunk_bottom.label,
    )
    result.color = colour
    return result


def _trim_s13_upper_bridge(
    part: bd.Shape,
    placement: m.ServoPlacement,
) -> bd.Shape:
    """End the upper S13 bridge at its required outer horn mounting pad."""
    pads, _, _ = m._outer_horn_bracket_tools(placement)
    upper_pad = max(pads, key=lambda pad: pad.bounding_box().center().Z)
    bridge_limit_z = upper_pad.bounding_box().max.Z
    box = part.bounding_box()
    margin = 1.0
    cutter = _box_from_bounds(
        (box.min.X - margin, box.max.X + margin),
        (box.min.Y - margin, box.max.Y + margin),
        (bridge_limit_z, box.max.Z + margin),
    )
    colour = part.color
    result = m._require_one_solid(part.cut(cutter).clean(), part.label)
    result.color = colour
    return result


def _cut_head_bottom_roll_slots(
    local_head: bd.Shape,
    head_placement: bd.Location,
) -> bd.Shape:
    """Open only the two head-bottom areas swept by the wider S12 mounts."""
    head_bottom = _one(local_head, "head_bot_sheet")
    world_head_bottom = head_placement * head_bottom
    world_box = world_head_bottom.bounding_box()
    clearance = HEAD_ROLL_CLEARANCE_MM
    world_cutters = [
        _box_from_bounds(
            (
                HEAD_BOTTOM_SLOT_X_RANGE_MM[0] - clearance,
                HEAD_BOTTOM_SLOT_X_RANGE_MM[1] + clearance,
            ),
            (y_min - clearance, y_max + clearance),
            (world_box.min.Z - clearance, world_box.max.Z + clearance),
        )
        for y_min, y_max in HEAD_BOTTOM_SLOT_Y_RANGES_MM
    ]
    local_cutters = [head_placement.inverse() * cutter for cutter in world_cutters]
    colour = head_bottom.color
    revised_bottom = m._require_one_solid(
        m._cut_sequential(head_bottom, local_cutters).clean(),
        head_bottom.label,
    )
    revised_bottom.color = colour
    return _replace_by_label(
        local_head,
        {"head_bot_sheet": revised_bottom},
    )


@step(
    out="../../cad/local_fix/open_duck_mini_complete_14_zp10d_numbered.step",
    kinematics=KINEMATICS,
)
def open_duck_mini_14_zp10d_numbered():
    report("load validated ten-servo robot")
    robot = read_step(SOURCE_ROBOT)
    if len(_children(robot)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError("ten-servo source root occurrence count changed")
    vendor = read_step(m.VENDOR_SERVO)
    vendor_body = max(_children(vendor), key=lambda child: child.volume)

    trunk = _one(robot, "trunk_assembly <1>")
    neck_pitch = _one(robot, "neck_pitch_assembly <1>")
    neck_yaw = _one(robot, "neck_yaw_assembly <1>")
    head = _one(robot, "head_assembly <1>")

    report("build S12 head-pitch interface from the original joint axis")
    old_s12 = _one(neck_pitch, "S12_HEAD_PITCH_STS3215")
    s12_placement = m._servo_placement(old_s12, "S12_HEAD_PITCH")
    s12 = m._servo_compound(s12_placement, "S12_HEAD_PITCH_ZP10D")
    neck_left = _one(neck_pitch, "neck_left_sheet")
    neck_right = _one(neck_pitch, "neck_right_sheet")
    neck_left, neck_right = m._mount_servo_in_original_parts(
        [neck_left, neck_right],
        s12_placement,
        require_all_mounts=True,
    )

    report("build S11 neck-pitch interface")
    old_s11 = _one(trunk, "S11_NECK_PITCH_STS3215")
    s11_placement = m._servo_placement(old_s11, "S11_NECK_PITCH")
    s11 = m._servo_compound(s11_placement, "S11_NECK_PITCH_ZP10D")

    neck_left, neck_right = m._mount_horns_in_original_parts(
        [neck_left, neck_right], s11_placement
    )
    m._colour(neck_left, m.COLOURS["sheet"])
    m._colour(neck_right, m.COLOURS["sheet"])

    trunk_top = _one(trunk, "trunk_top")
    trunk_bottom = _one(trunk, "trunk_bottom")
    trunk_top, trunk_bottom = m._mount_servo_in_original_parts(
        [trunk_top, trunk_bottom],
        s11_placement,
        require_all_mounts=True,
    )
    s01_placement = _placement_from_servo(
        _one(trunk, "S01_RIGHT_HIP_YAW_ZP10D"), vendor_body
    )
    s06_placement = _placement_from_servo(
        _one(trunk, "S06_LEFT_HIP_YAW_ZP10D"), vendor_body
    )
    trunk_top, trunk_bottom = _recut_case_interfaces(
        [trunk_top, trunk_bottom],
        (s01_placement, s06_placement),
    )
    trunk_bottom = _trim_trunk_bottom_motion_clearances(
        trunk_bottom,
        robot,
    )
    m._colour(trunk_top, m.COLOURS["trunk"])
    m._colour(trunk_bottom, m.COLOURS["trunk"])
    revised_trunk = _replace_by_label(
        trunk,
        {
            "trunk_top": trunk_top,
            "trunk_bottom": trunk_bottom,
            "S11_NECK_PITCH_STS3215": s11,
        },
    )
    revised_neck_pitch = _replace_by_label(
        neck_pitch,
        {
            "neck_left_sheet": neck_left,
            "neck_right_sheet": neck_right,
            "S12_HEAD_PITCH_STS3215": s12,
        },
    )

    report("build S13 head-yaw interface from the original joint axis")
    old_s13 = _one(neck_yaw, "S13_HEAD_YAW_STS3215")
    s13_placement = _translated_placement(
        m._servo_placement(old_s13, "S13_HEAD_YAW"),
        HEAD_YAW_HEIGHT_INCREASE_MM,
    )
    s13 = m._servo_compound(s13_placement, "S13_HEAD_YAW_ZP10D")
    yaw_link = _translated_shape(
        _one(neck_yaw, "head_yaw_to_roll"),
        HEAD_YAW_HEIGHT_INCREASE_MM,
    )

    report("package S14 head-roll candidate")
    local_head = head_internal_zp10d_no_new_parts()
    local_head.bounding_box()
    if len(_children(local_head)) == 1 and _children(_children(local_head)[0]):
        local_head = _children(local_head)[0]
    head_placement = bd.Pos(0.0, 0.0, HEAD_YAW_HEIGHT_INCREASE_MM) * bd.Location(
        HEAD_PLACEMENT_POSITION,
        HEAD_PLACEMENT_ORIENTATION,
    )
    local_servo = _one_zp10d(local_head)
    s14 = _normalised_servo(local_servo, "S14_HEAD_ROLL_ZP10D")
    numbered_local_head = _replace_by_label(
        local_head,
        {local_servo.label: s14},
    )
    numbered_local_head = _reorder_children_like(head, numbered_local_head)
    revised_head = numbered_local_head.moved(head_placement)
    revised_head.label = head.label
    expected_head_center = (
        head.bounding_box().center()
        + bd.Vector(0.0, 0.0, HEAD_YAW_HEIGHT_INCREASE_MM)
    )
    if (revised_head.bounding_box().center() - expected_head_center).length > 0.001:
        raise ValueError("S14 candidate head is not in the current robot frame")

    s14_world = bd.Compound(
        children=[child.moved(head_placement) for child in _children(s14)]
    )
    s14_world.label = "S14_HEAD_ROLL_ZP10D"
    s14_placement = _placement_from_servo(s14_world, vendor_body)
    yaw_link = _mount_shared_yaw_link(
        yaw_link,
        s13_placement,
        s14_placement,
    )
    revised_neck_yaw = _replace_by_label(
        neck_yaw,
        {
            "S13_HEAD_YAW_STS3215": s13,
            "head_yaw_to_roll": yaw_link,
        },
    )

    report("build S12 and S13 cross-axis horn interface")
    head_pitch_to_yaw = _one(robot, "head_pitch_to_yaw")
    head_pitch_to_yaw = _lengthen_head_pitch_to_yaw(
        head_pitch_to_yaw,
        s12_placement,
        HEAD_YAW_HEIGHT_INCREASE_MM,
    )
    head_pitch_to_yaw = _mount_cross_axis_horns_to_existing_part(
        head_pitch_to_yaw,
        (s12_placement, s13_placement),
    )
    head_pitch_to_yaw = _trim_s13_upper_bridge(
        head_pitch_to_yaw,
        s13_placement,
    )
    numbered_local_head = _cut_head_bottom_roll_slots(
        numbered_local_head,
        head_placement,
    )
    revised_head = numbered_local_head.moved(head_placement)
    revised_head.label = head.label
    if (revised_head.bounding_box().center() - expected_head_center).length > 0.001:
        raise ValueError("head-roll clearance changed the head placement")

    report("package original root tree")
    root_replacements = {
        id(trunk): revised_trunk,
        id(neck_pitch): revised_neck_pitch,
        id(neck_yaw): revised_neck_yaw,
        id(head): revised_head,
        id(_one(robot, "head_pitch_to_yaw")): head_pitch_to_yaw,
    }
    for child in _children(robot):
        if child.label in {
            "right_antenna_holder",
            "left_antenna_holder",
            "antenna",
        }:
            root_replacements[id(child)] = _translated_shape(
                child,
                HEAD_YAW_HEIGHT_INCREASE_MM,
            )
    result = _compound_like(
        robot,
        [root_replacements.get(id(child), child) for child in _children(robot)],
    )
    result.label = ORIGINAL_ROOT_LABEL

    nodes = list(_walk(result))
    servo_ids = sorted(node.label for node in nodes if node.label.startswith("S"))
    expected_ids = [f"S{index:02}" for index in range(1, 15)]
    if [label[:3] for label in servo_ids] != expected_ids:
        raise ValueError(f"servo IDs are not unique and complete: {servo_ids!r}")
    if not all(label.endswith("_ZP10D") for label in servo_ids):
        raise ValueError(f"non-ZP10D numbered servo remains: {servo_ids!r}")
    if len(_children(result)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError("root occurrence count changed")
    leaves = [node for node in nodes if not _children(node)]
    if len(leaves) != EXPECTED_LEAF_OCCURRENCES:
        raise ValueError(f"leaf occurrence count changed to {len(leaves)}")
    return result


if __name__ == "__main__":
    open_duck_mini_14_zp10d_numbered()
