#!/usr/bin/env python3
"""Validate the numbered S03 hip-pitch interface in the local robot STEP."""

from __future__ import annotations

from cadgen import read_step

import build_zp10d_left_leg_complete as m


ROBOT = m.PROJECT_ROOT / "cad/local_fix/open_duck_mini_14_servos_numbered_fixed.step"
JOINT_CENTER = m.bd.Vector(-35.060, -90.725, -0.0740611876)
AXIS_TOLERANCE_MM = 0.001


def _axis_set(shape: m.bd.Shape) -> set[tuple[float, float]]:
    return {
        (round(point.X, 6), round(point.Z, 6))
        for point in m._y_axis_pattern_centers(
            shape, JOINT_CENTER, m.HORN_PATTERN_RADIUS_MM
        )
    }


def _walk(shape: m.bd.Shape):
    yield shape
    for child in m._children(shape):
        yield from _walk(child)


def _require_matching_axes(horn: m.bd.Shape, sheet: m.bd.Shape) -> None:
    horn_axes = _axis_set(horn)
    sheet_axes = _axis_set(sheet)
    if len(horn_axes) != 4:
        raise ValueError(f"{horn.label} exposes {len(horn_axes)} horn axes")
    if not horn_axes.issubset(sheet_axes):
        raise ValueError(
            f"{sheet.label} axes do not contain {horn.label} axes: "
            f"missing {sorted(horn_axes - sheet_axes)!r}"
        )


def main() -> None:
    robot = read_step(ROBOT)
    pitch_branch = m._one_child(robot, "right_roll_to_pitch_assembly <1>")
    link = m._one_child(pitch_branch, "right_roll_to_pitch")
    servo = m._one_child(pitch_branch, "S03_RIGHT_HIP_PITCH_ZP10D")
    body = m._one_child(servo, "ZP10D_vendor_body")
    passive_horn = m._one_child(servo, "ZP10D_vendor_passive_horn")
    active_horn = m._one_child(servo, "ZP10D_vendor_active_horn")

    upper_leg = m._one_child(robot, "knee_and_ankle_assembly <1>")
    left_sheet = m._one_child(upper_leg, "left_knee_to_ankle_left_sheet")
    right_sheet = m._one_child(upper_leg, "left_knee_to_ankle_right_sheet")

    _require_matching_axes(passive_horn, left_sheet)
    _require_matching_axes(active_horn, right_sheet)
    m._require_servo_interface(
        "S03 case/right_roll_to_pitch",
        [body],
        [link],
        m.MAX_FASTENER_INTERFACE_GAP_MM,
    )
    m._require_servo_interface(
        "S03 passive horn/left thigh sheet",
        [passive_horn],
        [left_sheet],
        m.MAX_FASTENER_INTERFACE_GAP_MM,
    )
    m._require_servo_interface(
        "S03 active horn/right thigh sheet",
        [active_horn],
        [right_sheet],
        m.MAX_HORN_FACE_GAP_MM,
    )

    placement = m.ServoPlacement(
        pose=m.bd.Location(),
        axis=m.bd.Vector(0, -1, 0),
        center=JOINT_CENTER,
        body=body,
        passive_horn=passive_horn,
        active_horn=active_horn,
    )
    m._require_rotation_clearance(
        "S03 thigh-sheet rotation envelope",
        [left_sheet, right_sheet],
        [body, link],
        placement,
        m.PITCH_RANGE_DEG,
    )
    m._require_rotation_clearance(
        "S03 horn/right_roll_to_pitch rotation envelope",
        [passive_horn, active_horn],
        [link],
        placement,
        m.PITCH_RANGE_DEG,
    )

    ids = sorted(
        node.label
        for node in _walk(robot)
        if node.label.startswith("S")
    )
    print(f"servo_ids={len(ids)}")
    print(
        "passive_horn_to_link_mm="
        f"{passive_horn.distance_to(link):.6f}"
    )
    print(f"active_horn_to_link_mm={active_horn.distance_to(link):.6f}")
    print(
        "passive_horn_to_sheet_mm="
        f"{passive_horn.distance_to(left_sheet):.6f}"
    )
    print(
        "active_horn_to_sheet_mm="
        f"{active_horn.distance_to(right_sheet):.6f}"
    )
    print("pitch_rotation_deg=-70..30 step=5 PASS")


if __name__ == "__main__":
    main()
