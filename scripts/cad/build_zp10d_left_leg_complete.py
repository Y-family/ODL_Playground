#!/usr/bin/env python3
"""Build one complete ZP10D leg while retaining the original robot parts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, report, srgb, step

from build_zp10d_left_ankle_step1 import (
    ANKLE_AXIS_X_MM,
    ANKLE_AXIS_Y_MM,
    ANKLE_AXIS_Z_MM,
    HALF_WIDTH_INCREASE_MM,
    zp10d_left_ankle_step1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROBOT = PROJECT_ROOT / "cad/source_exports/open_duck_original_isolated_branch.step"
VENDOR_SERVO = Path("/root/ZP10D舵机/金属舵机最小..双轴.STEP")

OLD_ROLL_HORN_OUTER_FACE_SPAN_MM = 36.850
ZP10D_ROLL_HORN_OUTER_FACE_SPAN_MM = 52.000
ROLL_LINK_SPLICE_HALF_WIDTH_MM = 0.500
KNEE_INSTANCE_OFFSET = (0.0, 0.100, 78.650)
CASE_HOLE_DIAMETER_MM = 2.000
CASE_PAD_RADIUS_MM = 4.000
ROLL_CASE_FAR_PAD_RADIUS_MM = 6.100
CASE_PAD_EXTRA_REACH_MM = 12.000
HORN_PAD_REACH_MM = 18.000
HORN_PAD_RADIUS_MM = 4.000
HORN_HOLE_DIAMETER_MM = 3.200
HORN_PATTERN_RADIUS_MM = 7.000
HORN_COUNTERBORE_DIAMETER_MM = 6.500
HORN_COUNTERBORE_DEPTH_MM = 2.000
HORN_OLD_HOLE_FILL_DIAMETER_MM = 6.800
PITCH_SHEET_COUNTERBORE_DEPTH_MM = 1.000
PITCH_SHEET_HOLE_FILL_OVERLAP_MM = 0.050
HORN_BRACKET_PAD_RADIUS_MM = 14.000
HORN_BRACKET_INNER_OFFSET_MM = 1.250
HORN_BRACKET_OUTER_REACH_MM = 7.000
HORN_CENTER_HOLE_DIAMETER_MM = 4.200
HORN_CLEARANCE_RADIUS_MM = 10.300
HORN_CLEARANCE_HALF_WIDTH_MM = 2.260
ZP10D_HORN_OUTER_RADIUS_MM = 10.000
# Measured on both original STS3215 roll-to-pitch assemblies.  The pitch
# servo horns rotate with the thigh and must not touch the fixed orange link.
ORIGINAL_PITCH_ACTIVE_HORN_CLEARANCE_MM = 1.438
ORIGINAL_PITCH_PASSIVE_HORN_CLEARANCE_MM = 2.389
PITCH_HORN_CLEARANCE_TOLERANCE_MM = 0.010
PITCH_HORN_RADIAL_CLEARANCE_MM = ORIGINAL_PITCH_PASSIVE_HORN_CLEARANCE_MM
MAX_HORN_HOST_GAP_MM = 5.000
VOLUME_TOLERANCE_MM3 = 0.05
MAX_DISCONNECTED_TRIM_MM3 = 25.0
MAX_FASTENER_INTERFACE_GAP_MM = 0.25
MAX_HORN_FACE_GAP_MM = 0.001
MAX_PASSIVE_BEARING_GAP_MM = 0.10
ROLL_BOTTOM_JOURNAL_RADIUS_MM = 9.950
ROLL_BOTTOM_JOURNAL_HEIGHT_MM = 7.000
ROLL_BOTTOM_BEARING_GAP_MM = 0.700
YAW_RANGE_DEG = (-30.0, 30.0)
ROLL_RANGE_DEG = (-25.0, 25.0)
# Onshape's right-hip-pitch mate is -30..+70 deg about world +Y.  The
# imported servo case exposes the same shaft as world -Y, so angles expressed
# about ServoPlacement.axis have the opposite signs.
PITCH_RANGE_DEG = (-70.0, 30.0)
MOTION_CHECK_STEP_DEG = 5.0

OLD_SHAFT_POINT = (-25.500, 4.400, 0.000)
ZP10D_HORN_FACE_MIDPOINT = (6.8065613458, 10.4198318473, 51.000)
ZP10D_CASE_HOLES = (
    (-13.6934386542, -16.5801681527, 43.5427693399),
    (-13.6934386542, -16.5801681527, 58.5427693399),
    (26.3065613458, -9.5801681527, 44.0000000000),
    (26.3065613458, -9.5801681527, 58.0000000000),
)
ZP10D_HORN_PATTERNS = (
    (-16.9434386542, 0.155095),
    (30.5565613458, -5.710593),
)
EXPECTED_VENDOR_VOLUMES_MM3 = (709.120294, 750.525630, 31788.686497)

COLOURS = {
    "trunk": "#2E2E2E",
    "roll_bottom": "#A5A5A5",
    "roll_top": "#E6E6E6",
    "roll_link": "#E8922A",
    "cache": "#EAEAEA",
    "spacer": "#A5A5A5",
    "sheet": "#393838",
    "foot_yellow": "#FAB601",
    "foot_bottom": "#4E4C47",
    "servo_body": "#34383D",
    "servo_horn": "#A9AFB5",
}


@dataclass(frozen=True)
class ServoPlacement:
    pose: bd.Location
    axis: bd.Vector
    center: bd.Vector
    body: bd.Shape
    passive_horn: bd.Shape
    active_horn: bd.Shape


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _one_child(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r}, found {len(matches)}")
    return matches[0]


def _one_prefix_child(parent: bd.Shape, prefix: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one child starting {prefix!r}, found {len(matches)}")
    return matches[0]


def _colour(shape: bd.Shape, hex_colour: str) -> bd.Shape:
    shape.color = srgb(hex_colour)
    return shape


def _require_one_solid(
    shape: bd.Shape,
    label: str,
    *,
    max_disconnected_trim: float = MAX_DISCONNECTED_TRIM_MM3,
) -> bd.Shape:
    result = shape.clean()
    solids = list(result.solids())
    if len(solids) > 1:
        solids.sort(key=lambda solid: solid.volume, reverse=True)
        discarded = sum(solid.volume for solid in solids[1:])
        if discarded <= max_disconnected_trim:
            result = solids[0]
            solids = [result]
    if not result.is_valid or len(solids) != 1 or result.volume <= 0:
        raise ValueError(f"{label} is not one valid positive-volume solid")
    result.label = label
    return result


def _intersection_volume(left: bd.Shape, right: bd.Shape) -> float:
    result = left.intersect(right)
    if result is None:
        return 0.0
    if isinstance(result, list):
        return sum(item.volume for item in result)
    return result.volume


def _cut_sequential(shape: bd.Shape, cutters: list[bd.Shape]) -> bd.Shape:
    result = shape
    for index, cutter in enumerate(cutters):
        try:
            result = result.cut(cutter)
        except ValueError as error:
            raise ValueError(
                f"cut {index} ({cutter.label or type(cutter).__name__}) failed"
            ) from error
    return result


def _basis_vector(location: bd.Location, point: tuple[float, float, float]) -> bd.Vector:
    origin = bd.Vertex(0, 0, 0).moved(location).center()
    tip = bd.Vertex(*point).moved(location).center()
    return (tip - origin).normalized()


def _transform_point(
    location: bd.Location, point: tuple[float, float, float]
) -> bd.Vector:
    return bd.Vertex(*point).moved(location).center()


def _bbox_projection(shape: bd.Shape, axis: bd.Vector) -> tuple[float, float]:
    box = shape.bounding_box()
    values = []
    for x in (box.min.X, box.max.X):
        for y in (box.min.Y, box.max.Y):
            for z in (box.min.Z, box.max.Z):
                values.append(bd.Vector(x, y, z).dot(axis))
    return min(values), max(values)


def _servo_placement(old_servo: bd.Shape, name: str) -> ServoPlacement:
    old_case = _one_prefix_child(old_servo, "WJ-WK00-0123MIDDLECASE")
    old_drive = _one_child(old_servo, "drive_palonier")
    old_passive = _one_child(old_servo, "passive_palonier")

    old_axis = _basis_vector(old_case.location, (0, 1, 0))
    old_z = _basis_vector(old_case.location, (0, 0, 1))
    vendor_x = -old_axis
    vendor_z = -old_z
    orientation = bd.Plane(
        origin=(0, 0, 0),
        x_dir=vendor_x,
        z_dir=vendor_z,
    ).location

    drive_range = _bbox_projection(old_drive, old_axis)
    passive_range = _bbox_projection(old_passive, old_axis)
    axial_midpoint = (min(*drive_range, *passive_range) + max(*drive_range, *passive_range)) / 2
    old_line_point = _transform_point(old_case.location, OLD_SHAFT_POINT)
    center = old_line_point + old_axis * (axial_midpoint - old_line_point.dot(old_axis))

    rotated_vendor_midpoint = _transform_point(orientation, ZP10D_HORN_FACE_MIDPOINT)
    translation = center - rotated_vendor_midpoint
    pose = bd.Pos(translation.X, translation.Y, translation.Z) * orientation

    vendor = read_step(VENDOR_SERVO)
    source_parts = _children(vendor)
    volumes = sorted(part.volume for part in source_parts)
    if len(source_parts) != 3 or any(
        abs(actual - expected) > VOLUME_TOLERANCE_MM3
        for actual, expected in zip(volumes, EXPECTED_VENDOR_VOLUMES_MM3, strict=True)
    ):
        raise ValueError("the supplied ZP10D STEP no longer matches the audited source")

    placed = [pose * part for part in source_parts]
    body = max(placed, key=lambda part: part.volume)
    horns = sorted(
        (part for part in placed if part is not body),
        key=lambda part: part.bounding_box().center().dot(old_axis),
    )
    body.label = "ZP10D_vendor_body"
    horns[0].label = "ZP10D_vendor_passive_horn"
    horns[1].label = "ZP10D_vendor_active_horn"
    _colour(body, COLOURS["servo_body"])
    _colour(horns[0], COLOURS["servo_horn"])
    _colour(horns[1], COLOURS["servo_horn"])
    return ServoPlacement(pose, old_axis, center, body, horns[0], horns[1])


def _servo_compound(placement: ServoPlacement, label: str = "ZP10D <1>") -> bd.Compound:
    result = bd.Compound(
        children=[placement.body, placement.passive_horn, placement.active_horn]
    )
    result.label = label
    return result


def _intersect_solids(shape: bd.Shape, cutter: bd.Shape) -> list[bd.Shape]:
    result = shape.intersect(cutter)
    if result is None:
        return []
    if isinstance(result, (list, bd.ShapeList)):
        return list(result)
    return [result]


def _axis_aligned_clip_box(
    shape: bd.Shape,
    axis: bd.Vector,
    lower: float,
    upper: float,
) -> bd.Shape:
    box = shape.bounding_box()
    margin = 1.0
    if abs(axis.X) > 0.999:
        return bd.Box(
            upper - lower,
            box.size.Y + 2.0 * margin,
            box.size.Z + 2.0 * margin,
            align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN),
        ).moved(bd.Pos(lower, box.min.Y - margin, box.min.Z - margin))
    if abs(axis.Y) > 0.999:
        return bd.Box(
            box.size.X + 2.0 * margin,
            upper - lower,
            box.size.Z + 2.0 * margin,
            align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN),
        ).moved(bd.Pos(box.min.X - margin, lower, box.min.Z - margin))
    if abs(axis.Z) > 0.999:
        return bd.Box(
            box.size.X + 2.0 * margin,
            box.size.Y + 2.0 * margin,
            upper - lower,
            align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN),
        ).moved(bd.Pos(box.min.X - margin, box.min.Y - margin, lower))
    raise ValueError("servo axis is not aligned with a world coordinate axis")


def _stretch_original_roll_link(
    shape: bd.Shape,
    axis: bd.Vector,
    center: bd.Vector,
) -> bd.Shape:
    """Move the two original end plates outward and extend only the middle.

    The original link already has the correct single-layer plate geometry,
    circular bores and mounting pattern.  A whole-part non-uniform scale would
    turn circular interfaces into ellipses.  Instead, split through a plain
    middle band, rigidly translate both original ends and stretch only that
    band to reconnect them.
    """
    original_range = _bbox_projection(shape, axis)
    midpoint = center.dot(axis)
    half_splice = ROLL_LINK_SPLICE_HALF_WIDTH_MM
    half_increase = (
        ZP10D_ROLL_HORN_OUTER_FACE_SPAN_MM
        - OLD_ROLL_HORN_OUTER_FACE_SPAN_MM
    ) / 2.0
    if half_increase <= 0.0:
        raise ValueError("ZP10D roll horn span must exceed the original span")

    lower_clip = _axis_aligned_clip_box(
        shape,
        axis,
        original_range[0] - 1.0,
        midpoint + half_splice,
    )
    upper_clip = _axis_aligned_clip_box(
        shape,
        axis,
        midpoint - half_splice,
        original_range[1] + 1.0,
    )
    middle_clip = _axis_aligned_clip_box(
        shape,
        axis,
        midpoint - half_splice,
        midpoint + half_splice,
    )

    lower = _intersect_solids(shape, lower_clip)
    upper = _intersect_solids(shape, upper_clip)
    middle = _intersect_solids(shape, middle_clip)
    if not lower or not upper or not middle:
        raise ValueError("right_roll_to_pitch splice did not cross the original solid")

    lower_offset = axis * -half_increase
    upper_offset = axis * half_increase
    pieces = [
        bd.Pos(lower_offset.X, lower_offset.Y, lower_offset.Z) * piece
        for piece in lower
    ]
    components = [abs(axis.X), abs(axis.Y), abs(axis.Z)]
    dominant = components.index(max(components))
    factors = [1.0, 1.0, 1.0]
    factors[dominant] = (
        2.0 * half_splice + 2.0 * half_increase
    ) / (2.0 * half_splice)
    pieces.extend(
        piece.scale(
            tuple(factors),
            about=(center.X, center.Y, center.Z),
        )
        for piece in middle
    )
    pieces.extend(
        bd.Pos(upper_offset.X, upper_offset.Y, upper_offset.Z) * piece
        for piece in upper
    )
    return _require_one_solid(pieces[0].fuse(*pieces[1:]), shape.label)


def _x_axis_pattern_centers(
    shape: bd.Shape,
    center: bd.Vector,
    radius: float,
) -> list[bd.Vector]:
    """Read unique X-axis cylindrical feature centers at a radial distance."""
    found: dict[tuple[float, float], bd.Vector] = {}
    for face in shape.faces():
        if "CYLINDER" not in str(face.geom_type):
            continue
        feature_axis = face.axis_of_rotation
        if abs(abs(feature_axis.direction.X) - 1.0) > 0.001:
            continue
        position = feature_axis.position
        radial_distance = ((position.Y - center.Y) ** 2 + (position.Z - center.Z) ** 2) ** 0.5
        if abs(radial_distance - radius) > 0.01:
            continue
        key = (round(position.Y, 6), round(position.Z, 6))
        found[key] = bd.Vector(center.X, position.Y, position.Z)
    return [found[key] for key in sorted(found)]


def _require_x_axis_hole(
    shape: bd.Shape,
    center: bd.Vector,
    radius: float,
    label: str,
) -> None:
    circumference = 2.0 * 3.141592653589793 * radius
    for edge in shape.edges():
        if "CIRCLE" not in str(edge.geom_type):
            continue
        if abs(float(edge.radius) - radius) > 0.001:
            continue
        if abs(edge.length - circumference) > 0.001:
            continue
        arc_center = edge.arc_center
        if (
            abs(arc_center.Y - center.Y) <= 0.001
            and abs(arc_center.Z - center.Z) <= 0.001
        ):
            return
    raise ValueError(f"required circular hole {label} is missing")


def _remap_roll_horn_mount_holes(
    shape: bd.Shape,
    placement: ServoPlacement,
) -> bd.Shape:
    """Replace only the original horn screw holes with the ZP10D pattern.

    The center bores and both original single-layer end plates remain intact.
    Hole axes come from the imported source solids instead of drawing values.
    """
    axis = placement.axis
    if abs(axis.X - 1.0) > 0.001 or abs(axis.Y) > 0.001 or abs(axis.Z) > 0.001:
        raise ValueError("right_roll_to_pitch horn remap expects world +X axis")

    old_centers = _x_axis_pattern_centers(shape, placement.center, HORN_PATTERN_RADIUS_MM)
    passive_centers = _x_axis_pattern_centers(
        placement.passive_horn, placement.center, HORN_PATTERN_RADIUS_MM
    )
    active_centers = _x_axis_pattern_centers(
        placement.active_horn, placement.center, HORN_PATTERN_RADIUS_MM
    )
    if len(old_centers) != 4:
        raise ValueError(f"expected four original horn-hole axes, found {len(old_centers)}")
    if len(passive_centers) != 4 or len(active_centers) != 4:
        raise ValueError(
            "the audited ZP10D horns no longer expose four mounting-hole axes per side"
        )

    shape_range = _bbox_projection(shape, axis)
    passive_range = _bbox_projection(placement.passive_horn, axis)
    active_range = _bbox_projection(placement.active_horn, axis)
    passive_plate = (shape_range[0], passive_range[0])
    active_plate = (active_range[1], shape_range[1])
    if passive_plate[1] <= passive_plate[0] or active_plate[1] <= active_plate[0]:
        raise ValueError("ZP10D horns are not seated between the original end plates")

    fill_radius = HORN_OLD_HOLE_FILL_DIAMETER_MM / 2.0
    fillers = [
        _local_x_cylinder(x_start, x_end, point.Y, point.Z, fill_radius)
        for x_start, x_end in (passive_plate, active_plate)
        for point in old_centers
    ]
    filled = _require_one_solid(shape.fuse(*fillers), shape.label)

    through_radius = HORN_HOLE_DIAMETER_MM / 2.0
    counterbore_radius = HORN_COUNTERBORE_DIAMETER_MM / 2.0
    cutters: list[bd.Shape] = []
    for point in passive_centers:
        cutters.append(
            _local_x_cylinder(
                passive_plate[0] - 1.0,
                passive_plate[1] + 1.0,
                point.Y,
                point.Z,
                through_radius,
            )
        )
        cutters.append(
            _local_x_cylinder(
                passive_plate[0] - 1.0,
                passive_plate[0] + HORN_COUNTERBORE_DEPTH_MM,
                point.Y,
                point.Z,
                counterbore_radius,
            )
        )
    for point in active_centers:
        cutters.append(
            _local_x_cylinder(
                active_plate[0] - 1.0,
                active_plate[1] + 1.0,
                point.Y,
                point.Z,
                through_radius,
            )
        )
        cutters.append(
            _local_x_cylinder(
                active_plate[1] + 1.0,
                active_plate[1] - HORN_COUNTERBORE_DEPTH_MM,
                point.Y,
                point.Z,
                counterbore_radius,
            )
        )

    revised = _require_one_solid(_cut_sequential(filled, cutters), shape.label)
    for side, centers in (("passive", passive_centers), ("active", active_centers)):
        for index, point in enumerate(centers, start=1):
            _require_x_axis_hole(
                revised,
                point,
                through_radius,
                f"ZP10D {side} M3 clearance {index}",
            )
            _require_x_axis_hole(
                revised,
                point,
                counterbore_radius,
                f"ZP10D {side} counterbore {index}",
            )
    return revised


def _local_x_cylinder(
    x_start: float,
    x_end: float,
    y: float,
    z: float,
    radius: float,
) -> bd.Solid:
    direction = 1.0 if x_end > x_start else -1.0
    return bd.Solid.make_cylinder(
        radius,
        abs(x_end - x_start),
        plane=bd.Plane(
            origin=(x_start, y, z),
            x_dir=(0, 1, 0),
            z_dir=(direction, 0, 0),
        ),
    )


def _local_y_cylinder(
    y_start: float,
    y_end: float,
    x: float,
    z: float,
    radius: float,
) -> bd.Solid:
    direction = 1.0 if y_end > y_start else -1.0
    return bd.Solid.make_cylinder(
        radius,
        abs(y_end - y_start),
        plane=bd.Plane(
            origin=(x, y_start, z),
            x_dir=(1, 0, 0),
            z_dir=(0, direction, 0),
        ),
    )


def _y_axis_pattern_centers(
    shape: bd.Shape,
    center: bd.Vector,
    radius: float,
) -> list[bd.Vector]:
    found: dict[tuple[float, float], bd.Vector] = {}
    for face in shape.faces():
        if "CYLINDER" not in str(face.geom_type):
            continue
        feature_axis = face.axis_of_rotation
        if abs(abs(feature_axis.direction.Y) - 1.0) > 0.001:
            continue
        position = feature_axis.position
        radial_distance = (
            (position.X - center.X) ** 2 + (position.Z - center.Z) ** 2
        ) ** 0.5
        if abs(radial_distance - radius) > 0.01:
            continue
        key = (round(position.X, 6), round(position.Z, 6))
        found[key] = bd.Vector(position.X, center.Y, position.Z)
    return [found[key] for key in sorted(found)]


def _pitch_sheet_hole_faces(
    shape: bd.Shape,
    center: bd.Vector,
) -> list[tuple[bd.Face, float]]:
    radii = (HORN_HOLE_DIAMETER_MM / 2.0, HORN_COUNTERBORE_DIAMETER_MM / 2.0)
    result = []
    for face in shape.faces():
        if "CYLINDER" not in str(face.geom_type):
            continue
        feature_axis = face.axis_of_rotation
        if abs(abs(feature_axis.direction.Y) - 1.0) > 0.001:
            continue
        position = feature_axis.position
        radial_distance = (
            (position.X - center.X) ** 2 + (position.Z - center.Z) ** 2
        ) ** 0.5
        if abs(radial_distance - HORN_PATTERN_RADIUS_MM) > 0.01:
            continue
        surface_radii = (
            [float(face.radius)]
            if face.radius is not None
            else [
                float(edge.radius)
                for edge in face.edges()
                if "CIRCLE" in str(edge.geom_type) and edge.radius is not None
            ]
        )
        surface_radius = next(
            (
                candidate
                for candidate in surface_radii
                if any(abs(candidate - radius) <= 0.001 for radius in radii)
            ),
            None,
        )
        if surface_radius is None:
            continue
        result.append((face, surface_radius))
    return result


def _require_y_axis_hole(
    shape: bd.Shape,
    center: bd.Vector,
    radius: float,
    label: str,
) -> None:
    circumference = 2.0 * 3.141592653589793 * radius
    for edge in shape.edges():
        if "CIRCLE" not in str(edge.geom_type):
            continue
        if abs(float(edge.radius) - radius) > 0.001:
            continue
        if abs(edge.length - circumference) > 0.001:
            continue
        arc_center = edge.arc_center
        if (
            abs(arc_center.X - center.X) <= 0.001
            and abs(arc_center.Z - center.Z) <= 0.001
        ):
            return
    raise ValueError(f"required circular hole {label} is missing")


def _remap_pitch_sheet_horn_holes(
    sheet: bd.Shape,
    horn: bd.Shape,
    joint_center: bd.Vector,
) -> bd.Shape:
    """Move the original sheet holes onto the imported S03 horn axes."""
    old_faces = _pitch_sheet_hole_faces(sheet, joint_center)
    if len(old_faces) != 8:
        raise ValueError(
            f"{sheet.label} expected four through holes and four counterbores, "
            f"found {len(old_faces)} cylindrical faces"
        )

    fillers = []
    for face, radius in old_faces:
        face_box = face.bounding_box()
        axis_position = face.axis_of_rotation.position
        fillers.append(
            _local_y_cylinder(
                face_box.min.Y,
                face_box.max.Y,
                axis_position.X,
                axis_position.Z,
                radius + PITCH_SHEET_HOLE_FILL_OVERLAP_MM,
            )
        )
    filled = _require_one_solid(sheet.fuse(*fillers), sheet.label)

    new_centers = _y_axis_pattern_centers(
        horn, joint_center, HORN_PATTERN_RADIUS_MM
    )
    if len(new_centers) != 4:
        raise ValueError(
            f"{horn.label} expected four S03 mounting-hole axes, "
            f"found {len(new_centers)}"
        )

    sheet_box = sheet.bounding_box()
    through_radius = HORN_HOLE_DIAMETER_MM / 2.0
    counterbore_radius = HORN_COUNTERBORE_DIAMETER_MM / 2.0
    cutters = [
        _local_y_cylinder(
            sheet_box.min.Y - 1.0,
            sheet_box.max.Y + 1.0,
            point.X,
            point.Z,
            through_radius,
        )
        for point in new_centers
    ]
    if sheet_box.center().Y > joint_center.Y:
        counterbore_start = sheet_box.max.Y + 1.0
        counterbore_end = sheet_box.max.Y - PITCH_SHEET_COUNTERBORE_DEPTH_MM
    else:
        counterbore_start = sheet_box.min.Y - 1.0
        counterbore_end = sheet_box.min.Y + PITCH_SHEET_COUNTERBORE_DEPTH_MM
    cutters.extend(
        _local_y_cylinder(
            counterbore_start,
            counterbore_end,
            point.X,
            point.Z,
            counterbore_radius,
        )
        for point in new_centers
    )

    colour = sheet.color
    revised = _require_one_solid(_cut_sequential(filled, cutters), sheet.label)
    revised.color = colour
    for index, point in enumerate(new_centers, start=1):
        _require_y_axis_hole(
            revised,
            point,
            through_radius,
            f"S03 {sheet.label} M3 clearance {index}",
        )
        _require_y_axis_hole(
            revised,
            point,
            counterbore_radius,
            f"S03 {sheet.label} counterbore {index}",
        )
    return revised


def _case_mount_tools(
    placement: ServoPlacement, *, bridge_roll_far_side: bool = False
) -> tuple[list[bd.Shape], list[bd.Shape]]:
    pads = []
    holes = []
    for x, y, z in ZP10D_CASE_HOLES:
        pad_radius = CASE_PAD_RADIUS_MM
        if bridge_roll_far_side and x > ZP10D_HORN_FACE_MIDPOINT[0]:
            pad_radius = ROLL_CASE_FAR_PAD_RADIUS_MM
        pads.append(
            placement.pose
            * _local_x_cylinder(
                x - CASE_PAD_EXTRA_REACH_MM,
                x + CASE_PAD_EXTRA_REACH_MM,
                y,
                z,
                pad_radius,
            )
        )
        holes.append(
            placement.pose
            * _local_x_cylinder(
                x - CASE_PAD_EXTRA_REACH_MM - 1.0,
                x + CASE_PAD_EXTRA_REACH_MM + 1.0,
                y,
                z,
                CASE_HOLE_DIAMETER_MM / 2,
            )
        )
    return pads, holes


def _horn_mount_tools(
    parts: list[bd.Shape], placement: ServoPlacement
) -> tuple[list[bd.Shape], list[bd.Shape]]:
    pads = []
    holes = []
    center_y = 10.4198318473
    center_z = 51.0
    for horn_x, clocking in ZP10D_HORN_PATTERNS:
        for base_angle in (0.0, 90.0, 180.0, 270.0):
            radial = bd.Vector(0, HORN_PATTERN_RADIUS_MM, 0).rotate(
                bd.Axis.X, base_angle + clocking
            )
            y = center_y + radial.Y
            z = center_z + radial.Z
            candidates = (
                (
                    _local_x_cylinder(
                        horn_x - HORN_PAD_REACH_MM,
                        horn_x,
                        y,
                        z,
                        HORN_PAD_RADIUS_MM,
                    ),
                    horn_x - HORN_PAD_REACH_MM,
                    horn_x,
                ),
                (
                    _local_x_cylinder(
                        horn_x,
                        horn_x + HORN_PAD_REACH_MM,
                        y,
                        z,
                        HORN_PAD_RADIUS_MM,
                    ),
                    horn_x,
                    horn_x + HORN_PAD_REACH_MM,
                ),
            )
            placed = [
                (placement.pose * pad, start, end) for pad, start, end in candidates
            ]
            pad, start, end = min(
                placed,
                key=lambda item: min(part.distance_to(item[0]) for part in parts),
            )
            if min(part.distance_to(pad) for part in parts) > MAX_HORN_HOST_GAP_MM:
                continue
            pads.append(pad)
            holes.append(
                placement.pose
                * _local_x_cylinder(
                    start - 1.0,
                    end + 1.0,
                    y,
                    z,
                    HORN_HOLE_DIAMETER_MM / 2,
                )
            )
    return pads, holes


def _outer_horn_bracket_tools(
    placement: ServoPlacement,
) -> tuple[list[bd.Shape], list[bd.Shape], list[bd.Shape]]:
    pads = []
    holes = []
    clearances = []
    center_y = 10.4198318473
    center_z = 51.0
    for horn_x, clocking in ZP10D_HORN_PATTERNS:
        if horn_x < ZP10D_HORN_FACE_MIDPOINT[0]:
            start = horn_x - HORN_BRACKET_OUTER_REACH_MM
            end = horn_x - HORN_BRACKET_INNER_OFFSET_MM
        else:
            start = horn_x + HORN_BRACKET_INNER_OFFSET_MM
            end = horn_x + HORN_BRACKET_OUTER_REACH_MM
        pads.append(
            placement.pose
            * _local_x_cylinder(
                start,
                end,
                center_y,
                center_z,
                HORN_BRACKET_PAD_RADIUS_MM,
            )
        )
        clearances.append(
            placement.pose
            * _local_x_cylinder(
                horn_x - HORN_CLEARANCE_HALF_WIDTH_MM,
                horn_x + HORN_CLEARANCE_HALF_WIDTH_MM,
                center_y,
                center_z,
                HORN_CLEARANCE_RADIUS_MM,
            )
        )
        holes.append(
            placement.pose
            * _local_x_cylinder(
                min(start, end) - 1.0,
                max(start, end) + 1.0,
                center_y,
                center_z,
                HORN_CENTER_HOLE_DIAMETER_MM / 2,
            )
        )
        for base_angle in (0.0, 90.0, 180.0, 270.0):
            radial = bd.Vector(0, HORN_PATTERN_RADIUS_MM, 0).rotate(
                bd.Axis.X, base_angle + clocking
            )
            holes.append(
                placement.pose
                * _local_x_cylinder(
                    min(start, end) - 1.0,
                    max(start, end) + 1.0,
                    center_y + radial.Y,
                    center_z + radial.Z,
                    HORN_HOLE_DIAMETER_MM / 2,
                )
            )
    return pads, holes, clearances


def _mount_servo_in_original_parts(
    parts: list[bd.Shape],
    placement: ServoPlacement,
    *,
    bridge_roll_far_side: bool = False,
    require_all_mounts: bool = False,
) -> list[bd.Shape]:
    pads, holes = _case_mount_tools(
        placement, bridge_roll_far_side=bridge_roll_far_side
    )
    assignments: list[list[bd.Shape]] = [[] for _ in parts]
    active_holes = []
    for pad, hole in zip(pads, holes, strict=True):
        closest = min(range(len(parts)), key=lambda index: parts[index].distance_to(pad))
        if parts[closest].distance_to(pad) > 0.001:
            if require_all_mounts:
                raise ValueError(
                    f"ZP10D case mount {len(active_holes) + 1} is not connected "
                    f"to {', '.join(part.label for part in parts)}"
                )
            continue
        assignments[closest].append(pad)
        active_holes.append(hole)

    revised = []
    obstacles = [placement.body, placement.passive_horn, placement.active_horn]
    for part, assigned in zip(parts, assignments, strict=True):
        working = part.fuse(*assigned) if assigned else part
        working = working.cut(*obstacles, *active_holes)
        revised_part = _require_one_solid(working, part.label)
        for obstacle in obstacles:
            overlap = _intersection_volume(revised_part, obstacle)
            if overlap > VOLUME_TOLERANCE_MM3:
                raise ValueError(
                    f"{part.label} intersects {obstacle.label} by {overlap:.3f} mm3"
                )
        revised.append(revised_part)
    return revised


def _shorten_pitch_horn_feet_from_roll_link(
    roll_link: bd.Shape,
    placement: ServoPlacement,
) -> bd.Shape:
    """Shorten the two fixed orange feet before the S03 passive horn.

    The original STS3215 horn sits farther outboard than the ZP10D horn.  Keep
    the original straight-ended support layout: open the existing servo pocket
    to a new planar stop before the horn instead of surrounding it with a
    circular clearance bore.
    """
    axis = placement.axis.normalized()
    if abs(abs(axis.Y) - 1.0) > 0.001:
        raise ValueError("hip-pitch foot shortening expects a world Y axis")

    horn_box = placement.passive_horn.bounding_box()
    lower, upper = _bbox_projection(placement.passive_horn, axis)
    stop = lower - ORIGINAL_PITCH_PASSIVE_HORN_CLEARANCE_MM
    overlap = upper + 1.0
    start_point = placement.center + axis * (
        stop - placement.center.dot(axis)
    )
    end_point = placement.center + axis * (
        overlap - placement.center.dot(axis)
    )
    cutter = bd.Box(
        horn_box.size.X + 2.0 * PITCH_HORN_RADIAL_CLEARANCE_MM,
        abs(end_point.Y - start_point.Y),
        horn_box.size.Z + 2.0 * PITCH_HORN_RADIAL_CLEARANCE_MM,
        align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN),
    ).moved(
        bd.Pos(
            horn_box.min.X - PITCH_HORN_RADIAL_CLEARANCE_MM,
            min(start_point.Y, end_point.Y),
            horn_box.min.Z - PITCH_HORN_RADIAL_CLEARANCE_MM,
        )
    )

    result = _require_one_solid(roll_link.cut(cutter), roll_link.label)
    for side, horn, required in (
        ("active", placement.active_horn, ORIGINAL_PITCH_ACTIVE_HORN_CLEARANCE_MM),
        ("passive", placement.passive_horn, ORIGINAL_PITCH_PASSIVE_HORN_CLEARANCE_MM),
    ):
        actual = horn.distance_to(result)
        if actual < required - PITCH_HORN_CLEARANCE_TOLERANCE_MM:
            raise ValueError(
                f"hip-pitch {side} horn touches right_roll_to_pitch: "
                f"clearance is {actual:.3f} mm, expected at least "
                f"{required:.3f} mm"
            )
    return result


def _trim_servo_clearance(
    part: bd.Shape,
    placement: ServoPlacement,
) -> bd.Shape:
    result = part.cut(
        placement.body,
        placement.passive_horn,
        placement.active_horn,
    )
    return _require_one_solid(result, part.label)


def _mount_horns_in_original_parts(
    parts: list[bd.Shape], placement: ServoPlacement
) -> list[bd.Shape]:
    pads, holes = _horn_mount_tools(parts, placement)
    assignments: list[list[bd.Shape]] = [[] for _ in parts]
    for pad in pads:
        closest = min(range(len(parts)), key=lambda index: parts[index].distance_to(pad))
        assignments[closest].append(pad)

    revised = []
    for part, assigned in zip(parts, assignments, strict=True):
        working = part.fuse(*assigned) if assigned else part
        working = working.cut(
            placement.body,
            placement.passive_horn,
            placement.active_horn,
            *holes,
        )
        revised.append(_require_one_solid(working, part.label))
    return revised


def _seat_yaw_active_horn_outside_roll_top(
    roll_top: bd.Shape, placement: ServoPlacement
) -> bd.Shape:
    """Lower the original mounting face so the drive horn remains external.

    The original STS3215 drive horn sits wholly above roll_motor_top and only
    its outer face contacts the printed part.  The wider ZP10D horn reaches
    farther down the same joint axis, so retaining the old top face would
    bury it in a pocket.  Removing only the material above the new outer horn
    face preserves the original one-piece part and restores the same external
    face-mount relationship.
    """
    axis = placement.axis
    if abs(axis.Z + 1.0) > 0.001 or abs(axis.X) > 0.001 or abs(axis.Y) > 0.001:
        raise ValueError("hip-yaw axis is no longer world -Z")

    horn_box = placement.active_horn.bounding_box()
    support_z = horn_box.min.Z
    part_box = roll_top.bounding_box()
    if not (part_box.min.Z < support_z < part_box.max.Z):
        raise ValueError("ZP10D yaw drive face does not cross roll_motor_top")

    keeper = bd.Box(
        part_box.size.X + 2.0,
        part_box.size.Y + 2.0,
        support_z - part_box.min.Z + 1.0,
        align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN),
    ).moved(bd.Pos(part_box.min.X - 1.0, part_box.min.Y - 1.0, part_box.min.Z - 1.0))
    clipped = roll_top.intersect(keeper)
    if isinstance(clipped, list):
        if len(clipped) != 1:
            raise ValueError(
                "lowering the yaw mounting face split roll_motor_top into "
                f"{len(clipped)} solids"
            )
        clipped = clipped[0]
    result = _require_one_solid(clipped, roll_top.label)
    if abs(result.bounding_box().max.Z - support_z) > 0.001:
        raise ValueError("roll_motor_top did not terminate at the yaw drive face")
    return result


def _mount_horns_outside_original_bracket(
    part: bd.Shape, placement: ServoPlacement
) -> bd.Shape:
    pads, holes, clearances = _outer_horn_bracket_tools(placement)
    working = part.fuse(*pads)
    working = _cut_sequential(
        working,
        [
            placement.body,
            *clearances,
            *holes,
        ],
    )
    return _require_one_solid(working, part.label)


def _translated_leaf(shape: bd.Shape, offset: tuple[float, float, float]) -> bd.Shape:
    result = bd.Pos(*offset) * shape
    result.label = shape.label
    return result


def _translated_servo(servo: bd.Shape, offset: tuple[float, float, float]) -> bd.Compound:
    parts = []
    for child in _children(servo):
        moved = bd.Pos(*offset) * child
        moved.label = child.label
        moved.color = child.color
        parts.append(moved)
    result = bd.Compound(children=parts)
    result.label = "ZP10D <1>"
    return result


def _recolour_accepted_ankle(ankle: bd.Shape) -> dict[str, bd.Shape]:
    ankle.bounding_box()
    if len(_children(ankle)) == 1 and _children(_children(ankle)[0]):
        ankle = _children(ankle)[0]
    leaves = {child.label: child for child in _children(ankle)}
    palette = {
        "left_knee_to_ankle_left_sheet": COLOURS["sheet"],
        "left_knee_to_ankle_right_sheet": COLOURS["sheet"],
        "leg_spacer": COLOURS["spacer"],
        "foot_top": COLOURS["foot_yellow"],
        "foot_side": COLOURS["foot_yellow"],
        "foot_bottom_pla": COLOURS["foot_bottom"],
        "foot_bottom_tpu": COLOURS["foot_bottom"],
    }
    for label, colour in palette.items():
        if label not in leaves:
            raise ValueError(f"accepted ankle children are {sorted(leaves)!r}")
        _colour(leaves[label], colour)
    servo = leaves["ZP10D_vendor_servo <1>"]
    for child in _children(servo):
        _colour(
            child,
            COLOURS["servo_body"] if child.label.endswith("body") else COLOURS["servo_horn"],
        )
    servo.label = "ZP10D <1>"
    return leaves


def _tree_labels(shape: bd.Shape) -> list[str]:
    labels = [shape.label]
    for child in _children(shape):
        labels.extend(_tree_labels(child))
    return labels


def _bounding_boxes_overlap(left: bd.Shape, right: bd.Shape) -> bool:
    a = left.bounding_box()
    b = right.bounding_box()
    return (
        a.min.X <= b.max.X
        and b.min.X <= a.max.X
        and a.min.Y <= b.max.Y
        and b.min.Y <= a.max.Y
        and a.min.Z <= b.max.Z
        and b.min.Z <= a.max.Z
    )


def _require_full_circle(
    shape: bd.Shape,
    radius: float,
    center: tuple[float, float, float],
    label: str,
) -> None:
    for edge in shape.edges():
        if "CIRCLE" not in str(edge.geom_type):
            continue
        edge_center = edge.center()
        if (
            abs(float(edge.radius) - radius) <= 0.001
            and abs(edge.length - 2.0 * 3.141592653589793 * radius) <= 0.001
            and abs(edge_center.X - center[0]) <= 0.001
            and abs(edge_center.Y - center[1]) <= 0.001
            and abs(edge_center.Z - center[2]) <= 0.001
        ):
            return
    raise ValueError(f"required circular mechanical interface {label} is missing")


def _require_servo_interface(
    label: str,
    servo_parts: list[bd.Shape],
    printed_parts: list[bd.Shape],
    max_gap: float,
) -> None:
    for servo_part in servo_parts:
        gap = min(part.distance_to(servo_part) for part in printed_parts)
        if gap > max_gap:
            raise ValueError(
                f"{label}: {servo_part.label} is floating by {gap:.3f} mm"
            )
        for printed_part in printed_parts:
            if not _bounding_boxes_overlap(servo_part, printed_part):
                continue
            overlap = _intersection_volume(servo_part, printed_part)
            if overlap > VOLUME_TOLERANCE_MM3:
                raise ValueError(
                    f"{label}: {servo_part.label} intersects "
                    f"{printed_part.label} by {overlap:.3f} mm3"
                )


def _require_no_servo_host_intersection(
    label: str,
    servo_parts: list[bd.Shape],
    printed_parts: list[bd.Shape],
) -> None:
    for servo_part in servo_parts:
        for printed_part in printed_parts:
            if not _bounding_boxes_overlap(servo_part, printed_part):
                continue
            overlap = _intersection_volume(servo_part, printed_part)
            if overlap > VOLUME_TOLERANCE_MM3:
                raise ValueError(
                    f"{label}: {servo_part.label} intersects "
                    f"{printed_part.label} by {overlap:.3f} mm3"
                )


def _rotation_angles(start: float, end: float) -> list[float]:
    count = round((end - start) / MOTION_CHECK_STEP_DEG)
    return [start + index * MOTION_CHECK_STEP_DEG for index in range(count + 1)]


def _rotate_about_servo_axis(
    shape: bd.Shape, placement: ServoPlacement, angle_deg: float
) -> bd.Shape:
    axis = placement.axis
    center = placement.center
    if abs(axis.X) > 0.999:
        rotation = bd.Rot(angle_deg * axis.X, 0, 0)
    elif abs(axis.Y) > 0.999:
        rotation = bd.Rot(0, angle_deg * axis.Y, 0)
    elif abs(axis.Z) > 0.999:
        rotation = bd.Rot(0, 0, angle_deg * axis.Z)
    else:
        raise ValueError("servo axis is not aligned with a world coordinate axis")
    transform = (
        bd.Pos(center.X, center.Y, center.Z)
        * rotation
        * bd.Pos(-center.X, -center.Y, -center.Z)
    )
    return transform * shape


def _require_rotation_clearance(
    label: str,
    moving_parts: list[bd.Shape],
    fixed_parts: list[bd.Shape],
    placement: ServoPlacement,
    angle_range: tuple[float, float],
) -> None:
    for angle in _rotation_angles(*angle_range):
        for moving_part in moving_parts:
            moved = _rotate_about_servo_axis(moving_part, placement, angle)
            for fixed_part in fixed_parts:
                if not _bounding_boxes_overlap(moved, fixed_part):
                    continue
                overlap = _intersection_volume(moved, fixed_part)
                if overlap > VOLUME_TOLERANCE_MM3:
                    raise ValueError(
                        f"{label}: {moving_part.label} collides with "
                        f"{fixed_part.label} at {angle:.1f} deg by "
                        f"{overlap:.3f} mm3"
                    )


def _trim_moving_part_for_rotation(
    moving_part: bd.Shape,
    fixed_parts: list[bd.Shape],
    placement: ServoPlacement,
    angle_range: tuple[float, float],
) -> bd.Shape:
    """Remove only fixed-side volume encountered across the joint range."""
    cutters = []
    for angle in _rotation_angles(*angle_range):
        if abs(angle) < 0.001:
            continue
        for fixed_part in fixed_parts:
            cutter = _rotate_about_servo_axis(fixed_part, placement, -angle)
            if not _bounding_boxes_overlap(moving_part, cutter):
                continue
            collision = moving_part.intersect(cutter)
            if collision is None:
                continue
            collisions = (
                list(collision)
                if isinstance(collision, (list, bd.ShapeList))
                else [collision]
            )
            cutters.extend(
                item for item in collisions if item.volume > VOLUME_TOLERANCE_MM3
            )
    if not cutters:
        return moving_part
    try:
        trimmed = moving_part.cut(*cutters)
    except ValueError as error:
        raise ValueError(
            f"rotation-envelope cut failed for {moving_part.label}"
        ) from error
    return _require_one_solid(trimmed, moving_part.label)


def _trim_fixed_part_for_rotation(
    fixed_part: bd.Shape,
    moving_parts: list[bd.Shape],
    placement: ServoPlacement,
    angle_range: tuple[float, float],
) -> bd.Shape:
    """Pocket the fixed host around moving original parts without thinning them."""
    cutters = []
    for angle in _rotation_angles(*angle_range):
        if abs(angle) < 0.001:
            continue
        for moving_part in moving_parts:
            moved = _rotate_about_servo_axis(moving_part, placement, angle)
            if not _bounding_boxes_overlap(fixed_part, moved):
                continue
            collision = fixed_part.intersect(moved)
            if collision is None:
                continue
            collisions = (
                list(collision)
                if isinstance(collision, (list, bd.ShapeList))
                else [collision]
            )
            cutters.extend(
                item for item in collisions if item.volume > VOLUME_TOLERANCE_MM3
            )
    if not cutters:
        return fixed_part
    try:
        trimmed = fixed_part.cut(*cutters)
    except ValueError as error:
        raise ValueError(
            f"rotation-envelope cut failed for fixed {fixed_part.label}"
        ) from error
    return _require_one_solid(trimmed, fixed_part.label)


def _enforce_mechanical_interface_rules(
    roll_link: bd.Shape,
    roll_servo: ServoPlacement,
    pitch_servo: ServoPlacement,
    yaw_servo: ServoPlacement,
    roll_top: bd.Shape,
    roll_bottom: bd.Shape,
    trunk_top: bd.Shape,
    trunk_bottom: bd.Shape,
    upper_left_sheet: bd.Shape,
    upper_right_sheet: bd.Shape,
) -> None:
    """Mandatory mechanical rule: every servo must connect and still rotate.

    A servo case must be retained by an original printed host, its drive horn
    must face-contact the intended moving link, and its passive side must have
    the original bearing/support relationship.  Servo solids may never occupy
    positive volume inside a printed part.  The connected printed parts must
    also clear the fixed side throughout the full MJCF joint range.
    """
    _require_servo_interface(
        "hip-roll horn/right_roll_to_pitch",
        [roll_servo.passive_horn, roll_servo.active_horn],
        [roll_link],
        MAX_HORN_FACE_GAP_MM,
    )
    _require_servo_interface(
        "hip-roll case/roll housing",
        [roll_servo.body],
        [roll_top, roll_bottom],
        MAX_FASTENER_INTERFACE_GAP_MM,
    )
    _require_servo_interface(
        "hip-pitch case/right_roll_to_pitch",
        [pitch_servo.body],
        [roll_link],
        MAX_FASTENER_INTERFACE_GAP_MM,
    )
    pitch_horn_clearances = (
        (
            "active",
            pitch_servo.active_horn,
            ORIGINAL_PITCH_ACTIVE_HORN_CLEARANCE_MM,
        ),
        (
            "passive",
            pitch_servo.passive_horn,
            ORIGINAL_PITCH_PASSIVE_HORN_CLEARANCE_MM,
        ),
    )
    for side, horn, required in pitch_horn_clearances:
        actual = horn.distance_to(roll_link)
        if actual < required - PITCH_HORN_CLEARANCE_TOLERANCE_MM:
            raise ValueError(
                f"hip-pitch {side} horn/right_roll_to_pitch clearance is "
                f"{actual:.3f} mm, expected at least {required:.3f} mm"
            )
    _require_servo_interface(
        "hip-pitch passive horn/left thigh sheet",
        [pitch_servo.passive_horn],
        [upper_left_sheet],
        MAX_FASTENER_INTERFACE_GAP_MM,
    )
    _require_servo_interface(
        "hip-pitch active horn/right thigh sheet",
        [pitch_servo.active_horn],
        [upper_right_sheet],
        MAX_HORN_FACE_GAP_MM,
    )
    _require_servo_interface(
        "hip-yaw case/trunk",
        [yaw_servo.body],
        [trunk_top, trunk_bottom],
        MAX_FASTENER_INTERFACE_GAP_MM,
    )
    _require_servo_interface(
        "hip-yaw passive horn/trunk bearing support",
        [yaw_servo.passive_horn],
        [trunk_top],
        MAX_PASSIVE_BEARING_GAP_MM,
    )
    _require_servo_interface(
        "hip-yaw active horn/roll_motor_top",
        [yaw_servo.active_horn],
        [roll_top],
        MAX_HORN_FACE_GAP_MM,
    )

    all_servo_parts = [
        roll_servo.body,
        roll_servo.passive_horn,
        roll_servo.active_horn,
        pitch_servo.body,
        pitch_servo.passive_horn,
        pitch_servo.active_horn,
        yaw_servo.body,
        yaw_servo.passive_horn,
        yaw_servo.active_horn,
    ]
    all_hosts = [
        roll_link,
        roll_top,
        roll_bottom,
        trunk_top,
        trunk_bottom,
        upper_left_sheet,
        upper_right_sheet,
    ]
    _require_no_servo_host_intersection(
        "hip servo/printed-part keepout", all_servo_parts, all_hosts
    )

    _require_rotation_clearance(
        "hip-yaw rotation envelope",
        [roll_top, roll_bottom],
        [yaw_servo.body, yaw_servo.passive_horn, trunk_top, trunk_bottom],
        yaw_servo,
        YAW_RANGE_DEG,
    )
    _require_rotation_clearance(
        "hip-roll rotation envelope",
        [roll_link],
        [roll_servo.body, roll_top, roll_bottom],
        roll_servo,
        ROLL_RANGE_DEG,
    )
    _require_rotation_clearance(
        "hip-pitch rotation envelope",
        [upper_left_sheet, upper_right_sheet],
        [pitch_servo.body, roll_link],
        pitch_servo,
        PITCH_RANGE_DEG,
    )

    _require_full_circle(
        roll_bottom,
        ROLL_BOTTOM_JOURNAL_RADIUS_MM,
        (-28.950, -35.000, -13.1840611876),
        "roll_motor_bottom journal top",
    )
    _require_full_circle(
        roll_bottom,
        ROLL_BOTTOM_JOURNAL_RADIUS_MM,
        (-28.950, -35.000, -20.1840611876),
        "roll_motor_bottom journal bottom",
    )
    journal_height = -13.1840611876 - -20.1840611876
    if abs(journal_height - ROLL_BOTTOM_JOURNAL_HEIGHT_MM) > 0.001:
        raise ValueError("roll_motor_bottom bearing journal height changed")
    bearing_gap = roll_bottom.distance_to(trunk_bottom)
    if abs(bearing_gap - ROLL_BOTTOM_BEARING_GAP_MM) > 0.001:
        raise ValueError(
            f"roll bearing gap is {bearing_gap:.3f} mm, expected "
            f"{ROLL_BOTTOM_BEARING_GAP_MM:.3f} mm"
        )


@step(out="../../cad/left_leg_complete/zp10d_left_leg_complete.step")
def zp10d_left_leg_complete():
    report("load original robot")
    robot = read_step(SOURCE_ROBOT)

    # Reuse the user-approved ankle geometry, then restore original colours and
    # regroup its leaves under the original assembly names.
    report("load accepted ankle")
    ankle = zp10d_left_ankle_step1()
    ankle_parts = _recolour_accepted_ankle(ankle)
    ankle_servo = ankle_parts["ZP10D_vendor_servo <1>"]
    ankle_center = bd.Vector(
        ANKLE_AXIS_X_MM,
        ANKLE_AXIS_Y_MM,
        ANKLE_AXIS_Z_MM,
    )
    foot_side = _remap_pitch_sheet_horn_holes(
        ankle_parts["foot_side"],
        _one_child(ankle_servo, "ZP10D_vendor_passive_horn"),
        ankle_center,
    )
    foot_top = _remap_pitch_sheet_horn_holes(
        ankle_parts["foot_top"],
        _one_child(ankle_servo, "ZP10D_vendor_active_horn"),
        ankle_center,
    )

    upper_servo = _translated_servo(ankle_servo, KNEE_INSTANCE_OFFSET)
    knee_center = ankle_center + bd.Vector(*KNEE_INSTANCE_OFFSET)
    lower_left_sheet = _remap_pitch_sheet_horn_holes(
        ankle_parts["left_knee_to_ankle_left_sheet"],
        _one_child(upper_servo, "ZP10D_vendor_active_horn"),
        knee_center,
    )
    lower_right_sheet = _remap_pitch_sheet_horn_holes(
        ankle_parts["left_knee_to_ankle_right_sheet"],
        _one_child(upper_servo, "ZP10D_vendor_passive_horn"),
        knee_center,
    )
    foot = bd.Compound(
        children=[
            foot_side,
            ankle_parts["foot_bottom_tpu"],
            ankle_parts["foot_bottom_pla"],
            foot_top,
        ]
    )
    foot.label = "foot_assembly <1>"
    lower_leg = bd.Compound(
        children=[
            ankle_parts["leg_spacer"],
            lower_left_sheet,
            ankle_servo,
            lower_right_sheet,
            foot,
        ]
    )
    lower_leg.label = "knee_and_ankle_assembly <3>"

    upper_left_sheet = _translated_leaf(
        ankle_parts["left_knee_to_ankle_left_sheet"], KNEE_INSTANCE_OFFSET
    )
    upper_right_sheet = _translated_leaf(
        ankle_parts["left_knee_to_ankle_right_sheet"], KNEE_INSTANCE_OFFSET
    )
    upper_spacer = _translated_leaf(ankle_parts["leg_spacer"], KNEE_INSTANCE_OFFSET)
    right_cache = _one_child(robot, "right_cache")
    right_cache = bd.Pos(0, -HALF_WIDTH_INCREASE_MM, 0) * right_cache
    right_cache.label = "right_cache"
    _colour(right_cache, COLOURS["cache"])
    report("build hip-pitch interface")
    roll_to_pitch_source = _one_child(robot, "right_roll_to_pitch_assembly <1>")
    roll_link = _one_child(roll_to_pitch_source, "right_roll_to_pitch")
    old_pitch_servo = _one_prefix_child(roll_to_pitch_source, "STS32155")
    pitch_servo = _servo_placement(old_pitch_servo, "hip_pitch")

    upper_left_sheet = _trim_servo_clearance(upper_left_sheet, pitch_servo)
    upper_right_sheet = _trim_servo_clearance(upper_right_sheet, pitch_servo)
    upper_left_sheet = _remap_pitch_sheet_horn_holes(
        upper_left_sheet,
        pitch_servo.passive_horn,
        pitch_servo.center,
    )
    upper_right_sheet = _remap_pitch_sheet_horn_holes(
        upper_right_sheet,
        pitch_servo.active_horn,
        pitch_servo.center,
    )
    _colour(upper_left_sheet, COLOURS["sheet"])
    _colour(upper_right_sheet, COLOURS["sheet"])
    upper_leg = bd.Compound(
        children=[
            right_cache,
            upper_spacer,
            upper_right_sheet,
            upper_left_sheet,
            upper_servo,
            lower_leg,
        ]
    )
    upper_leg.label = "knee_and_ankle_assembly <1>"

    report("build hip-roll horn interface")
    hip_roll_source = _one_child(robot, "hip_roll_assembly <1>")
    old_roll_servo = _one_prefix_child(hip_roll_source, "STS32155")
    roll_servo = _servo_placement(old_roll_servo, "hip_roll")
    roll_link = _stretch_original_roll_link(
        roll_link, roll_servo.axis, roll_servo.center
    )
    roll_link = _remap_roll_horn_mount_holes(roll_link, roll_servo)
    roll_link = _mount_servo_in_original_parts([roll_link], pitch_servo)[0]
    roll_link = _shorten_pitch_horn_feet_from_roll_link(roll_link, pitch_servo)
    roll_link = _require_one_solid(
        roll_link.cut(
            right_cache,
            upper_spacer,
            upper_left_sheet,
            upper_right_sheet,
        ),
        roll_link.label,
    )
    _colour(roll_link, COLOURS["roll_link"])
    roll_to_pitch = bd.Compound(
        children=[roll_link, _servo_compound(pitch_servo), upper_leg]
    )
    roll_to_pitch.label = "right_roll_to_pitch_assembly <1>"

    report("build hip-roll case housing")
    roll_top = _one_child(hip_roll_source, "roll_motor_top")
    roll_bottom = _one_child(hip_roll_source, "roll_motor_bottom")

    report("build hip-yaw interface")
    trunk_source = _one_child(robot, "trunk_assembly <1>")
    trunk_servos = [
        child for child in _children(trunk_source) if child.label.startswith("STS32155")
    ]
    old_yaw_servo = min(
        trunk_servos,
        key=lambda servo: servo.bounding_box().center().Y,
    )
    yaw_servo = _servo_placement(old_yaw_servo, "hip_yaw")

    roll_top, roll_bottom = _mount_servo_in_original_parts(
        [roll_top, roll_bottom],
        roll_servo,
        bridge_roll_far_side=True,
        require_all_mounts=True,
    )
    roll_top, roll_bottom = _mount_horns_in_original_parts(
        [roll_top, roll_bottom], yaw_servo
    )
    roll_top = _seat_yaw_active_horn_outside_roll_top(roll_top, yaw_servo)
    trunk_top = _one_child(trunk_source, "trunk_top")
    trunk_bottom = _one_child(trunk_source, "trunk_bottom")
    trunk_top, trunk_bottom = _mount_servo_in_original_parts(
        [trunk_top, trunk_bottom], yaw_servo
    )
    roll_obstacles = [
        roll_servo.body,
        roll_servo.passive_horn,
        roll_servo.active_horn,
        roll_link,
        trunk_top,
        trunk_bottom,
    ]
    roll_top = _require_one_solid(roll_top.cut(*roll_obstacles), roll_top.label)
    roll_bottom = _require_one_solid(
        roll_bottom.cut(*roll_obstacles),
        roll_bottom.label,
        max_disconnected_trim=100.0,
    )
    report("trim hip-yaw rotation envelope")
    roll_top = _trim_moving_part_for_rotation(
        roll_top,
        [trunk_top, trunk_bottom],
        yaw_servo,
        YAW_RANGE_DEG,
    )
    report("trim hip-roll rotation envelope")
    roll_link = _trim_moving_part_for_rotation(
        roll_link,
        [roll_top, roll_bottom],
        roll_servo,
        ROLL_RANGE_DEG,
    )
    report("trim hip-pitch rotation envelope")
    roll_link = _trim_fixed_part_for_rotation(
        roll_link,
        [upper_left_sheet, upper_right_sheet],
        pitch_servo,
        PITCH_RANGE_DEG,
    )
    _colour(upper_left_sheet, COLOURS["sheet"])
    _colour(upper_right_sheet, COLOURS["sheet"])
    upper_leg = bd.Compound(
        children=[
            right_cache,
            upper_spacer,
            upper_right_sheet,
            upper_left_sheet,
            upper_servo,
            lower_leg,
        ]
    )
    upper_leg.label = "knee_and_ankle_assembly <1>"
    _colour(roll_link, COLOURS["roll_link"])
    roll_to_pitch = bd.Compound(
        children=[roll_link, _servo_compound(pitch_servo), upper_leg]
    )
    roll_to_pitch.label = "right_roll_to_pitch_assembly <1>"
    _colour(roll_top, COLOURS["roll_top"])
    _colour(roll_bottom, COLOURS["roll_bottom"])
    hip_roll = bd.Compound(
        children=[
            roll_top,
            roll_bottom,
            _servo_compound(roll_servo),
            roll_to_pitch,
        ]
    )
    hip_roll.label = "hip_roll_assembly <1>"

    _colour(trunk_top, COLOURS["trunk"])
    _colour(trunk_bottom, COLOURS["trunk"])
    trunk = bd.Compound(
        children=[
            trunk_top,
            trunk_bottom,
            _servo_compound(yaw_servo, "ZP10D <2>"),
            hip_roll,
        ]
    )
    trunk.label = "trunk_assembly <1>"

    result = bd.Compound(children=[trunk])
    result.label = "Open_Duck_Mini_v2"

    report("validate mechanical interfaces and motion")
    _enforce_mechanical_interface_rules(
        roll_link,
        roll_servo,
        pitch_servo,
        yaw_servo,
        roll_top,
        roll_bottom,
        trunk_top,
        trunk_bottom,
        upper_left_sheet,
        upper_right_sheet,
    )

    labels = _tree_labels(result)
    if sum(label.startswith("ZP10D <") for label in labels) != 5:
        raise ValueError("complete leg does not contain exactly five ZP10D assemblies")
    if any(label.startswith("STS32155") for label in labels):
        raise ValueError("complete leg still contains an STS3215 assembly")
    return result


if __name__ == "__main__":
    zp10d_left_leg_complete()
