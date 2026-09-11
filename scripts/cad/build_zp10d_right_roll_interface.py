#!/usr/bin/env python3
"""Build and validate the ZP10D right hip-roll horn interface only."""

from __future__ import annotations

from cadgen import build123d as bd
from cadgen import read_step, step

from build_zp10d_left_leg_complete import (
    COLOURS,
    MAX_HORN_FACE_GAP_MM,
    SOURCE_ROBOT,
    VOLUME_TOLERANCE_MM3,
    ZP10D_ROLL_HORN_OUTER_FACE_SPAN_MM,
    _bbox_projection,
    _children,
    _colour,
    _intersection_volume,
    _one_child,
    _one_prefix_child,
    _remap_roll_horn_mount_holes,
    _require_servo_interface,
    _servo_compound,
    _servo_placement,
    _stretch_original_roll_link,
)


@step(out="../../cad/roll_interface/zp10d_right_roll_interface.step")
def zp10d_right_roll_interface():
    robot = read_step(SOURCE_ROBOT)
    roll_to_pitch = _one_child(robot, "right_roll_to_pitch_assembly <1>")
    original_link = _one_child(roll_to_pitch, "right_roll_to_pitch")

    hip_roll = _one_child(robot, "hip_roll_assembly <1>")
    original_servo = _one_prefix_child(hip_roll, "STS32155")
    servo = _servo_placement(original_servo, "hip_roll")

    link = _stretch_original_roll_link(original_link, servo.axis, servo.center)
    link = _remap_roll_horn_mount_holes(link, servo)
    _colour(link, COLOURS["roll_link"])

    solids = list(link.solids())
    if len(solids) != 1 or not link.is_valid or link.volume <= 0.0:
        raise ValueError("right_roll_to_pitch must remain one valid original part")
    _require_servo_interface(
        "hip-roll horn/right_roll_to_pitch",
        [servo.passive_horn, servo.active_horn],
        [link],
        MAX_HORN_FACE_GAP_MM,
    )
    for horn in (servo.passive_horn, servo.active_horn):
        overlap = _intersection_volume(horn, link)
        if overlap > VOLUME_TOLERANCE_MM3:
            raise ValueError(f"{horn.label} overlaps the link by {overlap:.3f} mm3")

    horn_span = (
        _bbox_projection(servo.active_horn, servo.axis)[1]
        - _bbox_projection(servo.passive_horn, servo.axis)[0]
    )
    if abs(horn_span - ZP10D_ROLL_HORN_OUTER_FACE_SPAN_MM) > 0.001:
        raise ValueError(f"ZP10D horn outer-face span changed to {horn_span:.3f} mm")

    servo_node = _servo_compound(servo, "S02_RIGHT_HIP_ROLL_ZP10D")
    result = bd.Compound(children=[link, servo_node])
    result.label = "right_roll_to_pitch_assembly <1>"
    if len(_children(result)) != 2:
        raise ValueError("interface output must contain one original link and one servo")
    return result


if __name__ == "__main__":
    zp10d_right_roll_interface()
