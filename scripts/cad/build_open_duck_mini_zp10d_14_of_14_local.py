#!/usr/bin/env python3
"""Build a local 14-of-14 ZP10D robot from the audited 10-of-14 export.

The two remaining source subassemblies are replaced one-for-one.  Their
placements are derived from common unchanged printed parts rather than from
visual offsets, and the root occurrence count is preserved.
"""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step

from build_zp10d_head_internal_original_part import (
    head_internal_zp10d_no_new_parts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROBOT = PROJECT_ROOT / "cad/source_exports/open_duck_top_10_of_14.step"
TRUNK_INPUT = PROJECT_ROOT / "cad/remaining/trunk_internal_zp10d_no_new_parts.step"
EXPECTED_ROOT_OCCURRENCES = 27
EXPECTED_ZP10D_COUNT = 14
EXPECTED_SOURCE_ZP10D_COUNT = 10
EXPECTED_SOURCE_STS3215_COUNT = 4
VENDOR_COMPONENT_VOLUMES_MM3 = (709.120294, 750.525630, 31788.686334)
VOLUME_TOLERANCE_MM3 = 0.05
PRINT_CLEARANCE_MM = 0.30
M3_CLEARANCE_DIAMETER_MM = 3.40
HORN_BOLT_RADIUS_MM = 7.0
HORN_HOLE_CUT_LENGTH_MM = 12.0
SERVO_SHAFT_CENTER_YZ = (10.4198318473, 51.0)
PASSIVE_HORN_X = -16.9434386542
DRIVE_HORN_X = 30.5565613458
DRIVE_HORN_ANGLE_DEG = -5.7105931375
HEAD_PLACEMENT_POSITION = (-20.6876888093, 147.4272365477, 33.06911715999)
HEAD_PLACEMENT_ORIENTATION = (-8.793931720976, -22.86107399046, -8.381437692179)
TRUNK_PLACEMENT_POSITION = (80.4911065536, 8.525977843465, -110.831058654)
TRUNK_PLACEMENT_ORIENTATION = (0.0, 0.0, 0.0)


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _single_prefix_child(assembly: bd.Shape, prefix: str) -> bd.Shape:
    matches = [child for child in _children(assembly) if child.label.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix!r} child, found {len(matches)}")
    return matches[0]


def _single_label_child(assembly: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(assembly) if child.label == label]
    if len(matches) != 1:
        labels = [child.label for child in _children(assembly)]
        raise ValueError(
            f"expected one {label!r} child, found {len(matches)}; children={labels!r}"
        )
    return matches[0]


def _replace_children(
    assembly: bd.Shape, replacements: dict[int, bd.Shape]
) -> bd.Compound:
    children = [replacements.get(id(child), child) for child in _children(assembly)]
    result = bd.Compound(children=children)
    result.label = assembly.label
    result.location = assembly.location
    return result


def _intersection_volume(left: bd.Shape, right: bd.Shape) -> float:
    result = left.intersect(right)
    if result is None:
        return 0.0
    if isinstance(result, list):
        return sum(item.volume for item in result)
    return result.volume


def _require_printable_solid(shape: bd.Shape, label: str) -> bd.Shape:
    if not shape.is_valid or len(shape.solids()) != 1 or shape.volume <= 0:
        raise ValueError(f"{label} is not one valid positive-volume solid")
    shape.label = label
    return shape


def _expanded_keepout(shape: bd.Shape) -> bd.Shape:
    expanded = [
        solid.offset_3d([], PRINT_CLEARANCE_MM)
        for solid in shape.solids()
    ]
    if len(expanded) == 1:
        return expanded[0]
    return bd.Compound(children=expanded)


def _servo_location(servo: bd.Shape, vendor_body: bd.Shape) -> bd.Location:
    servo_body = max(_children(servo), key=lambda child: child.volume)
    return servo_body.location * vendor_body.location.inverse()


def _horn_hole_pattern(servo_location: bd.Location) -> list[bd.Shape]:
    center_y, center_z = SERVO_SHAFT_CENTER_YZ
    holes = []
    for horn_x, angle in (
        (PASSIVE_HORN_X, 0.0),
        (DRIVE_HORN_X, DRIVE_HORN_ANGLE_DEG),
    ):
        for base_angle in (0.0, 90.0, 180.0, 270.0):
            point = bd.Vector(0, HORN_BOLT_RADIUS_MM, 0).rotate(
                bd.Axis.X, base_angle + angle
            )
            hole = bd.Cylinder(
                M3_CLEARANCE_DIAMETER_MM / 2,
                HORN_HOLE_CUT_LENGTH_MM,
                rotation=(0, 90, 0),
            ).moved(
                bd.Pos(
                    horn_x,
                    center_y + point.Y,
                    center_z + point.Z,
                )
            )
            holes.append(servo_location * hole)
    return holes


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


def _is_zp10d(shape: bd.Shape) -> bool:
    children = _children(shape)
    if len(children) != 3 or any(_children(child) for child in children):
        return False
    volumes = sorted(child.volume for child in children)
    return all(
        abs(actual - expected) <= VOLUME_TOLERANCE_MM3
        for actual, expected in zip(volumes, VENDOR_COMPONENT_VOLUMES_MM3, strict=True)
    )


@step(out="../../cad/remaining/open_duck_mini_zp10d_14_of_14_local.step")
def open_duck_mini_zp10d_14_of_14_local():
    robot = read_step(SOURCE_ROBOT)
    if len(_children(robot)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError(
            f"source root count is {len(_children(robot))}, expected "
            f"{EXPECTED_ROOT_OCCURRENCES}"
        )

    old_head = _single_prefix_child(robot, "head_assembly")
    old_trunk = _single_prefix_child(robot, "trunk_assembly")
    old_body = _single_prefix_child(robot, "body_assembly")
    old_neck_pitch = _single_prefix_child(robot, "neck_pitch_assembly")
    source_zp10d_count = sum(_is_zp10d(item) for item in _walk(robot))
    source_sts_count = sum(item.label.startswith("STS32155") for item in _walk(robot))
    if source_zp10d_count != EXPECTED_SOURCE_ZP10D_COUNT:
        raise ValueError(f"source contains {source_zp10d_count} ZP10D units, expected 10")
    if source_sts_count != EXPECTED_SOURCE_STS3215_COUNT:
        raise ValueError(f"source contains {source_sts_count} STS3215 units, expected 4")

    new_head = head_internal_zp10d_no_new_parts()
    # This assembly is read as an explicit input because its original
    # trunk_top child is modified again in whole-robot context below.
    new_trunk = read_step(TRUNK_INPUT)

    head_placement = bd.Location(HEAD_PLACEMENT_POSITION, HEAD_PLACEMENT_ORIENTATION)
    trunk_placement = bd.Location(TRUNK_PLACEMENT_POSITION, TRUNK_PLACEMENT_ORIENTATION)
    placed_head = new_head.moved(head_placement)

    local_trunk_top = _single_label_child(new_trunk, "trunk_top")
    local_trunk_servos = [child for child in _children(new_trunk) if _is_zp10d(child)]
    if len(local_trunk_servos) != 3:
        raise ValueError(
            f"expected three local trunk ZP10D units, found {len(local_trunk_servos)}"
        )
    placed_trunk_children = [child.moved(trunk_placement) for child in _children(new_trunk)]
    placed_trunk = bd.Compound(children=placed_trunk_children)
    placed_trunk.label = new_trunk.label
    trunk_top = placed_trunk_children[_children(new_trunk).index(local_trunk_top)]
    trunk_servos = [
        placed_trunk_children[_children(new_trunk).index(servo)]
        for servo in local_trunk_servos
    ]

    body_middle_top = _single_label_child(old_body, "body_middle_top")
    neck_left = _single_label_child(old_neck_pitch, "neck_left_sheet")
    neck_right = _single_label_child(old_neck_pitch, "neck_right_sheet")
    # The new retaining sleeves may only occupy free volume inside the original
    # robot envelope.  Trim them at the unchanged body and neck surfaces.
    revised_trunk_top = trunk_top.cut(body_middle_top, neck_left, neck_right)
    revised_trunk_top = _require_printable_solid(revised_trunk_top, "trunk_top")

    vendor = read_step(Path("/root/ZP10D舵机/金属舵机最小..双轴.STEP"))
    vendor_body = max(_children(vendor), key=lambda child: child.volume)
    component_keepouts = []
    screw_holes = []
    world_servo_components: list[list[bd.Shape]] = []
    for servo in local_trunk_servos:
        servo_world_location = trunk_placement * servo.location
        components = [
            component.moved(servo_world_location)
            for component in _children(servo)
        ]
        world_servo_components.append(components)
        component_keepouts.extend(
            _expanded_keepout(component) for component in components
        )
        screw_holes.extend(
            _horn_hole_pattern(
                trunk_placement * _servo_location(servo, vendor_body)
            )
        )

    # Existing printed parts receive only the material removals required by
    # the larger actuator envelope and its actual M3/R7 horn screw pattern.
    revised_body_middle_top = body_middle_top.cut(*component_keepouts, *screw_holes)
    revised_body_middle_top = _require_printable_solid(
        revised_body_middle_top, "body_middle_top"
    )
    revised_neck_left = neck_left.cut(*component_keepouts, *screw_holes)
    revised_neck_left = _require_printable_solid(revised_neck_left, "neck_left_sheet")
    revised_neck_right = neck_right.cut(*component_keepouts, *screw_holes)
    revised_neck_right = _require_printable_solid(revised_neck_right, "neck_right_sheet")

    placed_trunk = _replace_children(
        placed_trunk, {id(trunk_top): revised_trunk_top}
    )
    revised_body = _replace_children(
        old_body, {id(body_middle_top): revised_body_middle_top}
    )
    revised_neck_pitch = _replace_children(
        old_neck_pitch,
        {
            id(neck_left): revised_neck_left,
            id(neck_right): revised_neck_right,
        },
    )

    for printed_part in (
        revised_trunk_top,
        revised_body_middle_top,
        revised_neck_left,
        revised_neck_right,
    ):
        for components in world_servo_components:
            for component in components:
                overlap = _intersection_volume(printed_part, component)
                if overlap > VOLUME_TOLERANCE_MM3:
                    raise ValueError(
                        f"{printed_part.label} intersects {component.label}: "
                        f"{overlap:.3f} mm3"
                    )

    children = []
    for child in _children(robot):
        if child is old_head:
            children.append(placed_head)
        elif child is old_trunk:
            children.append(placed_trunk)
        elif child is old_body:
            children.append(revised_body)
        elif child is old_neck_pitch:
            children.append(revised_neck_pitch)
        else:
            children.append(child)

    result = bd.Compound(children=children)
    result.label = "Open_Duck_Mini_v2_ZP10D_14_of_14_local"

    root_count = len(_children(result))
    if root_count != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError(f"root occurrence count changed to {root_count}")
    return result


if __name__ == "__main__":
    open_duck_mini_zp10d_14_of_14_local()
