#!/usr/bin/env python3
"""Validate static mechanical interfaces for one or both ZP10D legs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step

import build_zp10d_left_leg_complete as m


ROBOT = m.PROJECT_ROOT / "cad/local_fix/open_duck_mini_10_zp10d_numbered.step"
RIGHT_LEG_ROBOT = (
    m.PROJECT_ROOT
    / "cad/local_fix/open_duck_mini_14_servos_numbered.step"
)
VENDOR = Path("/root/ZP10D舵机/金属舵机最小..双轴.STEP")
AXIS_TOLERANCE_MM = 0.01
CONTACT_TOLERANCE_MM = 0.25
INTERFERENCE_TOLERANCE_MM3 = 0.05
VENDOR_HORN_HOLE_RADIUS_MM = 1.5
HOST_HORN_HOLE_RADIUS_MM = m.HORN_HOLE_DIAMETER_MM / 2.0
HOST_CASE_HOLE_RADIUS_MM = m.CASE_HOLE_DIAMETER_MM / 2.0


@dataclass(frozen=True)
class JointSpec:
    servo: bd.Shape
    case_hosts: tuple[bd.Shape, ...]
    horn_hosts: tuple[bd.Shape, ...]
    active_host: bd.Shape | None = None
    passive_support: bd.Shape | None = None


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _one(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r}, found {len(matches)}")
    return matches[0]


SINGLE_OUTPUT_SERVOS = {"S01_RIGHT_HIP_YAW_ZP10D", "S06_LEFT_HIP_YAW_ZP10D"}
ZP10D_HORN_TO_JOINT_DATUM_MM = 23.75


def _servo_parts(servo: bd.Shape) -> tuple[bd.Shape, bd.Shape | None, bd.Shape]:
    body = _one(servo, "ZP10D_vendor_body")
    passive_horns = [child for child in _children(servo)
                     if child.label == "ZP10D_vendor_passive_horn"]
    if not passive_horns and servo.label in SINGLE_OUTPUT_SERVOS:
        passive = None
    else:
        passive = _one(servo, "ZP10D_vendor_passive_horn")
    active = _one(servo, "ZP10D_vendor_active_horn")
    return body, passive, active


def _axis(shape: bd.Shape) -> tuple[bd.Vector, bd.Vector]:
    body, passive, active = _servo_parts(shape)
    active_center = active.bounding_box().center()
    if passive is None:
        # The retained R10 output cylinder supplies the axis; the datum offset
        # is verified against both unmodified vendor placements before removal.
        cylinders = [face for face in active.faces()
                     if face.geom_type == bd.GeomType.CYLINDER
                     and face.radius is not None and abs(face.radius - 10.0) < 0.001]
        if not cylinders:
            raise ValueError(f"{shape.label}: missing output-axis cylinder")
        direction = cylinders[0].axis_of_rotation.direction.normalized()
        if direction.dot(active_center - body.bounding_box().center()) < 0:
            direction = -direction
        return direction, active_center - direction * ZP10D_HORN_TO_JOINT_DATUM_MM
    passive_center = passive.bounding_box().center()
    return (active_center - passive_center).normalized(), (
        passive_center + active_center
    ) * 0.5


def _line_distance(
    point_a: bd.Vector,
    direction_a: bd.Vector,
    point_b: bd.Vector,
    direction_b: bd.Vector,
) -> float:
    if abs(direction_a.dot(direction_b)) < 0.999:
        return float("inf")
    delta = point_b - point_a
    return (delta - direction_a * delta.dot(direction_a)).length


def _cylinder_lines(
    shape: bd.Shape,
    radius: float,
    direction: bd.Vector,
) -> list[tuple[bd.Vector, bd.Vector]]:
    lines: list[tuple[bd.Vector, bd.Vector]] = []
    for face in shape.faces():
        if "CYLINDER" not in str(face.geom_type):
            continue
        face_radii = (
            [float(face.radius)]
            if face.radius is not None
            else [
                float(edge.radius)
                for edge in face.edges()
                if "CIRCLE" in str(edge.geom_type) and edge.radius is not None
            ]
        )
        if not any(abs(candidate - radius) <= 0.001 for candidate in face_radii):
            continue
        feature_axis = face.axis_of_rotation
        axis_direction = feature_axis.direction.normalized()
        if abs(axis_direction.dot(direction)) < 0.999:
            continue
        line = (feature_axis.position, axis_direction)
        if any(
            _line_distance(line[0], line[1], old[0], old[1])
            <= AXIS_TOLERANCE_MM
            for old in lines
        ):
            continue
        lines.append(line)
    return lines


def _horn_pattern_lines(
    horn: bd.Shape,
    joint_axis: bd.Vector,
    joint_center: bd.Vector,
) -> list[tuple[bd.Vector, bd.Vector]]:
    candidates = _cylinder_lines(
        horn, VENDOR_HORN_HOLE_RADIUS_MM, joint_axis
    )
    result = []
    for point, direction in candidates:
        delta = point - joint_center
        radial = (delta - joint_axis * delta.dot(joint_axis)).length
        if abs(radial - m.HORN_PATTERN_RADIUS_MM) <= AXIS_TOLERANCE_MM:
            result.append((point, direction))
    if len(result) != 4:
        raise ValueError(
            f"{horn.label} exposes {len(result)} R7 horn mounting axes, expected 4"
        )
    return result


def _host_has_line(
    hosts: tuple[bd.Shape, ...],
    line: tuple[bd.Vector, bd.Vector],
    radius: float,
) -> bool:
    for host in hosts:
        for candidate in _cylinder_lines(host, radius, line[1]):
            if _line_distance(line[0], line[1], candidate[0], candidate[1]) <= AXIS_TOLERANCE_MM:
                return True
    return False


def _require_host_lines(
    label: str,
    hosts: tuple[bd.Shape, ...],
    lines: list[tuple[bd.Vector, bd.Vector]],
    radius: float,
) -> None:
    missing = [
        index
        for index, line in enumerate(lines, start=1)
        if not _host_has_line(hosts, line, radius)
    ]
    if missing:
        raise ValueError(f"{label} missing matching hole axes {missing}")


def _transform_point(location: bd.Location, point: tuple[float, float, float]) -> bd.Vector:
    return bd.Vertex(*point).moved(location).center()


def _case_mount_lines(
    body: bd.Shape, vendor_body: bd.Shape
) -> list[tuple[bd.Vector, bd.Vector]]:
    location = body.location * vendor_body.location.inverse()
    origin = _transform_point(location, (0.0, 0.0, 0.0))
    x_tip = _transform_point(location, (1.0, 0.0, 0.0))
    direction = (x_tip - origin).normalized()
    return [
        (_transform_point(location, point), direction)
        for point in m.ZP10D_CASE_HOLES
    ]


def _require_contact(label: str, left: bd.Shape, hosts: tuple[bd.Shape, ...]) -> float:
    gap = min(left.distance_to(host) for host in hosts)
    if gap > CONTACT_TOLERANCE_MM:
        raise ValueError(
            f"{label} gap is {gap:.3f} mm, maximum is {CONTACT_TOLERANCE_MM:.3f} mm"
        )
    return gap


def _require_no_interference(
    label: str, component: bd.Shape, hosts: tuple[bd.Shape, ...]
) -> None:
    for host in hosts:
        overlap = m._intersection_volume(component, host)
        if overlap > INTERFERENCE_TOLERANCE_MM3:
            raise ValueError(
                f"{label} intersects {host.label} by {overlap:.3f} mm3"
            )


def _nearest_host(component: bd.Shape, hosts: tuple[bd.Shape, ...]) -> bd.Shape:
    return min(hosts, key=component.distance_to)


def _validate_joint(
    servo_id: str,
    spec: JointSpec,
    vendor_body: bd.Shape,
) -> None:
    body, passive, active = _servo_parts(spec.servo)
    joint_axis, joint_center = _axis(spec.servo)

    case_lines = _case_mount_lines(body, vendor_body)
    _require_host_lines(
        f"{servo_id} case mount",
        spec.case_hosts,
        case_lines,
        HOST_CASE_HOLE_RADIUS_MM,
    )
    case_gap = _require_contact(f"{servo_id} case/host", body, spec.case_hosts)
    _require_no_interference(f"{servo_id} body", body, spec.case_hosts)

    horn_gaps = []
    if spec.active_host is not None:
        horn_lines = _horn_pattern_lines(active, joint_axis, joint_center)
        _require_host_lines(
            f"{servo_id} active horn",
            (spec.active_host,),
            horn_lines,
            HOST_HORN_HOLE_RADIUS_MM,
        )
        horn_gaps.append(
            _require_contact(
                f"{servo_id} active horn/host", active, (spec.active_host,)
            )
        )
        _require_no_interference(
            f"{servo_id} active horn", active, (spec.active_host,)
        )
    else:
        for horn in (passive, active):
            if horn is None:
                continue
            host = _nearest_host(horn, spec.horn_hosts)
            horn_lines = _horn_pattern_lines(horn, joint_axis, joint_center)
            _require_host_lines(
                f"{servo_id} {horn.label}",
                (host,),
                horn_lines,
                HOST_HORN_HOLE_RADIUS_MM,
            )
            horn_gaps.append(
                _require_contact(
                    f"{servo_id} {horn.label}/host", horn, (host,)
                )
            )
            _require_no_interference(
                f"{servo_id} {horn.label}", horn, (host,)
            )

    support_gap = None
    if spec.passive_support is not None and passive is not None:
        support_gap = _require_contact(
            f"{servo_id} passive horn/support",
            passive,
            (spec.passive_support,),
        )
        _require_no_interference(
            f"{servo_id} passive horn", passive, (spec.passive_support,)
        )

    horn_text = ",".join(f"{gap:.3f}" for gap in horn_gaps)
    support_text = "" if support_gap is None else f" support_gap={support_gap:.3f}"
    print(
        f"{servo_id} PASS case_holes=4 case_gap={case_gap:.3f} "
        f"horn_holes={4 * len(horn_gaps)} horn_gaps={horn_text}{support_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=Path, default=ROBOT)
    parser.add_argument(
        "--right-only",
        action="store_true",
        help="validate S01-S05 without requiring the left leg to use ZP10D",
    )
    args = parser.parse_args()

    robot = read_step(args.robot)
    vendor = read_step(VENDOR)
    vendor_body = max(_children(vendor), key=lambda child: child.volume)

    trunk = _one(robot, "trunk_assembly <1>")
    right_hip = _one(robot, "hip_roll_assembly <1>")
    right_pitch = _one(robot, "right_roll_to_pitch_assembly <1>")
    right_upper = _one(robot, "knee_and_ankle_assembly <1>")
    right_lower = _one(robot, "knee_and_ankle_assembly <3>")
    right_foot = _one(robot, "foot_assembly <1>")
    trunk_parts = (_one(trunk, "trunk_top"), _one(trunk, "trunk_bottom"))
    right_roll_parts = (
        _one(right_hip, "roll_motor_top"),
        _one(right_hip, "roll_motor_bottom"),
    )
    right_link = _one(right_pitch, "right_roll_to_pitch")
    right_upper_sheets = (
        _one(right_upper, "left_knee_to_ankle_left_sheet"),
        _one(right_upper, "left_knee_to_ankle_right_sheet"),
    )
    right_lower_sheets = (
        _one(right_lower, "left_knee_to_ankle_left_sheet"),
        _one(right_lower, "left_knee_to_ankle_right_sheet"),
    )
    right_foot_parts = (_one(right_foot, "foot_top"), _one(right_foot, "foot_side"))

    specs = {
        "S01": JointSpec(
            _one(trunk, "S01_RIGHT_HIP_YAW_ZP10D"),
            trunk_parts,
            (),
            active_host=right_roll_parts[0],
            passive_support=trunk_parts[0],
        ),
        "S02": JointSpec(
            _one(right_hip, "S02_RIGHT_HIP_ROLL_ZP10D"),
            right_roll_parts,
            (right_link,),
        ),
        "S03": JointSpec(
            _one(right_pitch, "S03_RIGHT_HIP_PITCH_ZP10D"),
            (right_link,),
            right_upper_sheets,
        ),
        "S04": JointSpec(
            _one(right_upper, "S04_RIGHT_KNEE_ZP10D"),
            right_upper_sheets,
            right_lower_sheets,
        ),
        "S05": JointSpec(
            _one(right_lower, "S05_RIGHT_ANKLE_ZP10D"),
            right_lower_sheets,
            right_foot_parts,
        ),
    }

    if not args.right_only:
        left_hip = _one(robot, "hip_roll_assembly <2>")
        left_pitch = _one(robot, "left_roll_to_pitch_assembly <1>")
        left_upper = _one(robot, "knee_and_ankle_assembly <2>")
        left_lower = _one(robot, "knee_and_ankle_assembly <4>")
        left_foot = _one(robot, "foot_assembly <2>")
        left_roll_parts = (
            _one(left_hip, "roll_motor_top"),
            _one(left_hip, "roll_motor_bottom"),
        )
        left_link = _one(left_pitch, "left_roll_to_pitch")
        left_upper_sheets = (
            _one(left_upper, "left_knee_to_ankle_left_sheet"),
            _one(left_upper, "left_knee_to_ankle_right_sheet"),
        )
        left_lower_sheets = (
            _one(left_lower, "left_knee_to_ankle_left_sheet"),
            _one(left_lower, "left_knee_to_ankle_right_sheet"),
        )
        left_foot_parts = (
            _one(left_foot, "foot_top"),
            _one(left_foot, "foot_side"),
        )
        specs.update(
            {
                "S06": JointSpec(
                    _one(trunk, "S06_LEFT_HIP_YAW_ZP10D"),
                    trunk_parts,
                    (),
                    active_host=left_roll_parts[0],
                    passive_support=trunk_parts[0],
                ),
                "S07": JointSpec(
                    _one(left_hip, "S07_LEFT_HIP_ROLL_ZP10D"),
                    left_roll_parts,
                    (left_link,),
                ),
                "S08": JointSpec(
                    _one(left_pitch, "S08_LEFT_HIP_PITCH_ZP10D"),
                    (left_link,),
                    left_upper_sheets,
                ),
                "S09": JointSpec(
                    _one(left_upper, "S09_LEFT_KNEE_ZP10D"),
                    left_upper_sheets,
                    left_lower_sheets,
                ),
                "S10": JointSpec(
                    _one(left_lower, "S10_LEFT_ANKLE_ZP10D"),
                    left_lower_sheets,
                    left_foot_parts,
                ),
            }
        )

    for servo_id, spec in specs.items():
        _validate_joint(servo_id, spec, vendor_body)


if __name__ == "__main__":
    main()
