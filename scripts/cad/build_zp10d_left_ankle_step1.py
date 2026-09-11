#!/usr/bin/env python3
"""Build the isolated left-ankle ZP10D conversion without new robot parts."""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROBOT = PROJECT_ROOT / "cad/source_exports/open_duck_original_isolated_branch.step"
VENDOR_SERVO = Path("/root/ZP10D舵机/金属舵机最小..双轴.STEP")

ANKLE_AXIS_X_MM = -35.060
ANKLE_AXIS_Y_MM = -90.700
ANKLE_AXIS_Z_MM = -157.3740611876
ORIGINAL_HORN_FACE_SPAN_MM = 36.900
ZP10D_HORN_FACE_SPAN_MM = 52.000
HALF_WIDTH_INCREASE_MM = (
    ZP10D_HORN_FACE_SPAN_MM - ORIGINAL_HORN_FACE_SPAN_MM
) / 2
CASE_CLEARANCE_MM = 0.250
FOOT_WIDTH_SCALE = ZP10D_HORN_FACE_SPAN_MM / ORIGINAL_HORN_FACE_SPAN_MM
ACTIVE_FOOT_WALL_INNER_Y_MM = -72.250
PASSIVE_FOOT_WALL_INNER_Y_MM = -109.150
SERVO_MOUNT_HOLE_DIAMETER_MM = 2.000
SERVO_MOUNT_PAD_RADIUS_MM = 4.000
OLD_MOUNT_HOLE_FILL_RADIUS_MM = 3.000

# The old STS3215 fastener pockets are replaced in the same printed sheets.
LEFT_OLD_MOUNT_CENTERS_XZ_MM = (
    (-45.310, -128.3740611876),
    (-24.810, -128.3740611876),
)
RIGHT_OLD_MOUNT_CENTERS_XZ_MM = (
    (-45.310, -124.6240611876),
    (-24.810, -124.6240611876),
)

# Measured directly from the supplied ZP10D STEP after applying _servo_pose().
LEFT_ZP10D_MOUNT_CENTERS_XZ_MM = (
    (-42.5172306601, -130.3740611876),
    (-27.5172306601, -130.3740611876),
)
RIGHT_ZP10D_MOUNT_CENTERS_XZ_MM = (
    (-42.060, -137.3740611876),
    (-28.060, -137.3740611876),
)
ZP10D_LEFT_MOUNT_FACE_Y_MM = -70.200
ZP10D_RIGHT_MOUNT_FACE_Y_MM = -110.200

# Measured in the supplied STEP assembly coordinate system.
ZP10D_SHAFT_Y_MM = 10.4198318473
ZP10D_SHAFT_Z_MM = 51.000
ZP10D_ACTIVE_OUTER_X_MM = -19.1934386542
ZP10D_CASE_ACTIVE_X_MM = -13.6934386542
ZP10D_CASE_PASSIVE_X_MM = 26.3065613458

EXPECTED_VENDOR_VOLUMES_MM3 = (709.120294, 750.525630, 31788.686497)
VOLUME_TOLERANCE_MM3 = 0.05


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _one_child(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r}, found {len(matches)}")
    return matches[0]


def _intersection_volume(left: bd.Shape, right: bd.Shape) -> float:
    result = left.intersect(right)
    if result is None:
        return 0.0
    if isinstance(result, list):
        return sum(item.volume for item in result)
    return result.volume


def _y_cylinder(
    x: float,
    z: float,
    start_y: float,
    end_y: float,
    radius: float,
) -> bd.Solid:
    direction = 1.0 if end_y > start_y else -1.0
    return bd.Solid.make_cylinder(
        radius,
        abs(end_y - start_y),
        plane=bd.Plane(
            origin=(x, start_y, z),
            x_dir=(1.0, 0.0, 0.0),
            z_dir=(0.0, direction, 0.0),
        ),
    )


def _revise_servo_mount(
    sheet: bd.Shape,
    *,
    positive_side: bool,
    old_centers_xz: tuple[tuple[float, float], ...],
    new_centers_xz: tuple[tuple[float, float], ...],
    mount_face_y: float,
) -> bd.Shape:
    bbox = sheet.bounding_box()
    outer_y = bbox.max.Y if positive_side else bbox.min.Y

    old_hole_fillers = [
        _y_cylinder(
            x,
            z,
            bbox.min.Y,
            bbox.max.Y,
            OLD_MOUNT_HOLE_FILL_RADIUS_MM,
        )
        for x, z in old_centers_xz
    ]
    mount_pads = [
        _y_cylinder(
            x,
            z,
            mount_face_y,
            outer_y,
            SERVO_MOUNT_PAD_RADIUS_MM,
        )
        for x, z in new_centers_xz
    ]
    mounted_sheet = _require_one_solid(
        sheet.fuse(*old_hole_fillers, *mount_pads),
        sheet.label,
    )

    hole_start_y = min(outer_y, mount_face_y) - 1.0
    hole_end_y = max(outer_y, mount_face_y) + 1.0
    new_holes = [
        _y_cylinder(
            x,
            z,
            hole_start_y,
            hole_end_y,
            SERVO_MOUNT_HOLE_DIAMETER_MM / 2,
        )
        for x, z in new_centers_xz
    ]
    return _require_one_solid(
        mounted_sheet.cut(*new_holes),
        sheet.label,
    )


def _require_one_solid(shape: bd.Shape, label: str) -> bd.Shape:
    result = shape.clean()
    if not result.is_valid or len(result.solids()) != 1 or result.volume <= 0:
        raise ValueError(f"{label} is not one valid positive-volume solid")
    result.label = label
    return result


def _planar_y_faces(
    shape: bd.Shape,
    *,
    below: float | None = None,
    above: float | None = None,
) -> list[bd.Face]:
    selected = []
    for face in shape.faces():
        if face.geom_type != bd.GeomType.PLANE:
            continue
        normal = face.normal_at()
        if abs(normal.Y) < 0.98:
            continue
        y = face.center().Y
        if below is not None and y >= below:
            continue
        if above is not None and y <= above:
            continue
        selected.append(face)
    if not selected:
        raise ValueError(f"no Y-planar faces selected on {shape.label}")
    return selected


def _extend_existing_faces(
    shape: bd.Shape,
    faces: list[bd.Face],
    direction: tuple[float, float, float],
    amount: float,
    label: str,
) -> bd.Shape:
    additions = [
        bd.extrude(face, amount=amount, dir=direction)
        for face in faces
    ]
    return _require_one_solid(shape.fuse(*additions), label)


def _scale_original_foot_width(shape: bd.Shape) -> bd.Shape:
    resized = shape.scale(
        (1.0, FOOT_WIDTH_SCALE, 1.0),
        about=(0.0, ANKLE_AXIS_Y_MM, 0.0),
    )
    return _require_one_solid(resized, shape.label)


def _y_slice(shape: bd.Shape, min_y: float, max_y: float) -> bd.Shape:
    bbox = shape.bounding_box()
    cutter = bd.Box(
        bbox.size.X + 4.0,
        max_y - min_y,
        bbox.size.Z + 4.0,
        align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN),
    )
    cutter = bd.Pos(bbox.min.X - 2.0, min_y, bbox.min.Z - 2.0) * cutter
    result = shape.intersect(cutter)
    if result is None:
        raise ValueError(f"empty Y slice for {shape.label}")
    if isinstance(result, list):
        result = bd.Compound(children=result)
    return result


def _widen_foot_with_original_wall(
    shape: bd.Shape,
    *,
    wall_inner_y: float,
    positive_side: bool,
) -> bd.Shape:
    """Stretch the foot span while translating the original wall unchanged."""
    bbox = shape.bounding_box()
    if positive_side:
        core = _y_slice(shape, bbox.min.Y - 1.0, wall_inner_y)
        wall = _y_slice(shape, wall_inner_y, bbox.max.Y + 1.0)
        wall = bd.Pos(0.0, HALF_WIDTH_INCREASE_MM, 0.0) * wall
    else:
        wall = _y_slice(shape, bbox.min.Y - 1.0, wall_inner_y)
        core = _y_slice(shape, wall_inner_y, bbox.max.Y + 1.0)
        wall = bd.Pos(0.0, -HALF_WIDTH_INCREASE_MM, 0.0) * wall
    resized_core = core.scale(
        (1.0, FOOT_WIDTH_SCALE, 1.0),
        about=(0.0, ANKLE_AXIS_Y_MM, 0.0),
    )
    return _require_one_solid(resized_core.fuse(wall), shape.label)


def _lengthen_leg_spacer(shape: bd.Shape) -> bd.Shape:
    bbox = shape.bounding_box()
    positive_faces = _planar_y_faces(shape, above=bbox.max.Y - 0.01)
    negative_faces = _planar_y_faces(shape, below=bbox.min.Y + 0.01)
    additions = [
        *(bd.extrude(face, amount=HALF_WIDTH_INCREASE_MM, dir=(0, 1, 0)) for face in positive_faces),
        *(bd.extrude(face, amount=HALF_WIDTH_INCREASE_MM, dir=(0, -1, 0)) for face in negative_faces),
    ]
    return _require_one_solid(shape.fuse(*additions), shape.label)


def _servo_pose() -> bd.Location:
    # Local X is the shaft axis. Local Y is pointed up the leg and local Z is
    # transverse, matching the original actuator's orientation.
    rotation = bd.Rot(0, 90, 0) * bd.Rot(0, 0, -90)
    translation = bd.Pos(
        ANKLE_AXIS_X_MM - ZP10D_SHAFT_Z_MM,
        ANKLE_AXIS_Y_MM + ZP10D_ACTIVE_OUTER_X_MM
        + ZP10D_HORN_FACE_SPAN_MM / 2,
        ANKLE_AXIS_Z_MM + ZP10D_SHAFT_Y_MM,
    )
    return translation * rotation


def _thicken_case_seat(
    shifted_sheet: bd.Shape,
    *,
    positive_side: bool,
    case_active_y: float,
    case_passive_y: float,
) -> bd.Shape:
    bbox = shifted_sheet.bounding_box()
    if positive_side:
        seat_y = bbox.min.Y
        target_y = case_active_y + CASE_CLEARANCE_MM
        amount = seat_y - target_y
        faces = _planar_y_faces(shifted_sheet, below=seat_y + 0.01)
        direction = (0.0, -1.0, 0.0)
    else:
        seat_y = bbox.max.Y
        target_y = case_passive_y - CASE_CLEARANCE_MM
        amount = target_y - seat_y
        faces = _planar_y_faces(shifted_sheet, above=seat_y - 0.01)
        direction = (0.0, 1.0, 0.0)
    if amount <= 0:
        raise ValueError(f"non-positive case-seat thickening for {shifted_sheet.label}")
    return _extend_existing_faces(
        shifted_sheet,
        faces,
        direction,
        amount,
        shifted_sheet.label,
    )


@step(out="../../cad/left_leg_step1/zp10d_left_ankle_step1.step")
def zp10d_left_ankle_step1():
    robot = read_step(SOURCE_ROBOT)
    lower_leg = _one_child(robot, "knee_and_ankle_assembly <3>")
    foot = _one_child(robot, "foot_assembly <1>")

    left_sheet = _one_child(lower_leg, "left_knee_to_ankle_left_sheet")
    right_sheet = _one_child(lower_leg, "left_knee_to_ankle_right_sheet")
    leg_spacer = _one_child(lower_leg, "leg_spacer")
    foot_top = _one_child(foot, "foot_top")
    foot_side = _one_child(foot, "foot_side")
    foot_bottom_pla = _one_child(foot, "foot_bottom_pla")
    foot_bottom_tpu = _one_child(foot, "foot_bottom_tpu")

    vendor = read_step(VENDOR_SERVO)
    vendor_children = _children(vendor)
    volumes = sorted(child.volume for child in vendor_children)
    if len(vendor_children) != 3 or any(
        abs(actual - expected) > VOLUME_TOLERANCE_MM3
        for actual, expected in zip(volumes, EXPECTED_VENDOR_VOLUMES_MM3, strict=True)
    ):
        raise ValueError("the supplied ZP10D STEP no longer matches the audited source")

    pose = _servo_pose()
    servo_parts = [pose * child for child in vendor_children]
    servo_body = max(servo_parts, key=lambda child: child.volume)
    servo_horns = [child for child in servo_parts if child is not servo_body]
    servo_body.label = "ZP10D_vendor_body"
    servo_horns[0].label = "ZP10D_vendor_active_horn"
    servo_horns[1].label = "ZP10D_vendor_passive_horn"
    servo_assembly = bd.Compound(children=[servo_body, *servo_horns])
    servo_assembly.label = "ZP10D_vendor_servo <1>"
    case_active_y = -ZP10D_CASE_ACTIVE_X_MM + (
        ANKLE_AXIS_Y_MM + ZP10D_ACTIVE_OUTER_X_MM
        + ZP10D_HORN_FACE_SPAN_MM / 2
    )
    case_passive_y = -ZP10D_CASE_PASSIVE_X_MM + (
        ANKLE_AXIS_Y_MM + ZP10D_ACTIVE_OUTER_X_MM
        + ZP10D_HORN_FACE_SPAN_MM / 2
    )

    shifted_left_sheet = bd.Pos(0, HALF_WIDTH_INCREASE_MM, 0) * left_sheet
    shifted_right_sheet = bd.Pos(0, -HALF_WIDTH_INCREASE_MM, 0) * right_sheet
    shifted_left_sheet.label = left_sheet.label
    shifted_right_sheet.label = right_sheet.label
    revised_left_sheet = _thicken_case_seat(
        shifted_left_sheet,
        positive_side=True,
        case_active_y=case_active_y,
        case_passive_y=case_passive_y,
    )
    revised_right_sheet = _thicken_case_seat(
        shifted_right_sheet,
        positive_side=False,
        case_active_y=case_active_y,
        case_passive_y=case_passive_y,
    )
    revised_left_sheet = _revise_servo_mount(
        revised_left_sheet,
        positive_side=True,
        old_centers_xz=LEFT_OLD_MOUNT_CENTERS_XZ_MM,
        new_centers_xz=LEFT_ZP10D_MOUNT_CENTERS_XZ_MM,
        mount_face_y=ZP10D_LEFT_MOUNT_FACE_Y_MM,
    )
    revised_right_sheet = _revise_servo_mount(
        revised_right_sheet,
        positive_side=False,
        old_centers_xz=RIGHT_OLD_MOUNT_CENTERS_XZ_MM,
        new_centers_xz=RIGHT_ZP10D_MOUNT_CENTERS_XZ_MM,
        mount_face_y=ZP10D_RIGHT_MOUNT_FACE_Y_MM,
    )
    revised_spacer = _lengthen_leg_spacer(leg_spacer)
    revised_foot_top = _widen_foot_with_original_wall(
        foot_top,
        wall_inner_y=ACTIVE_FOOT_WALL_INNER_Y_MM,
        positive_side=True,
    )
    revised_foot_side = _widen_foot_with_original_wall(
        foot_side,
        wall_inner_y=PASSIVE_FOOT_WALL_INNER_Y_MM,
        positive_side=False,
    )
    revised_foot_bottom_pla = _scale_original_foot_width(foot_bottom_pla)
    revised_foot_bottom_tpu = _scale_original_foot_width(foot_bottom_tpu)

    for printed_part in (
        revised_left_sheet,
        revised_right_sheet,
        revised_spacer,
        revised_foot_top,
        revised_foot_side,
        revised_foot_bottom_pla,
        revised_foot_bottom_tpu,
    ):
        if not printed_part.is_valid or len(printed_part.solids()) != 1:
            raise ValueError(f"{printed_part.label} is not one valid solid")

    for sheet in (revised_left_sheet, revised_right_sheet):
        overlap = _intersection_volume(sheet, servo_body)
        if overlap > VOLUME_TOLERANCE_MM3:
            raise ValueError(
                f"{sheet.label} intersects the ZP10D body by {overlap:.3f} mm3"
            )
        if sheet.distance_to(servo_body) > 0.001:
            raise ValueError(
                f"{sheet.label} does not contact the ZP10D mounting face"
            )
    for foot_part, horn in zip(
        (revised_foot_top, revised_foot_side), servo_horns, strict=True
    ):
        overlap = _intersection_volume(foot_part, horn)
        if overlap > VOLUME_TOLERANCE_MM3:
            raise ValueError(
                f"{foot_part.label} intersects {horn.label} by {overlap:.3f} mm3"
            )
        body_overlap = _intersection_volume(foot_part, servo_body)
        if body_overlap > VOLUME_TOLERANCE_MM3:
            raise ValueError(
                f"{foot_part.label} intersects the ZP10D body by "
                f"{body_overlap:.3f} mm3"
            )

    result = bd.Compound(
        children=[
            revised_left_sheet,
            revised_right_sheet,
            revised_spacer,
            servo_assembly,
            revised_foot_top,
            revised_foot_side,
            revised_foot_bottom_pla,
            revised_foot_bottom_tpu,
        ]
    )
    result.label = "ZP10D_left_ankle_step1_no_new_robot_parts"
    return result


if __name__ == "__main__":
    zp10d_left_ankle_step1()
