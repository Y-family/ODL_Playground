#!/usr/bin/env python3
"""Build the untrimmed right hip-pitch interface for focused review."""

from __future__ import annotations

from cadgen import build123d as bd
from cadgen import read_step, step

import build_zp10d_left_leg_complete as m


ACCEPTED_ANKLE = m.PROJECT_ROOT / "cad/left_leg_step1/zp10d_left_ankle_step1.step"


@step(out="../../cad/hip_pitch_interface/zp10d_right_hip_pitch_interface.step")
def zp10d_right_hip_pitch_interface():
    robot = read_step(m.SOURCE_ROBOT)
    source = m._one_child(robot, "right_roll_to_pitch_assembly <1>")
    link = m._one_child(source, "right_roll_to_pitch")
    pitch_servo = m._servo_placement(
        m._one_prefix_child(source, "STS32155"), "hip_pitch"
    )

    hip_roll = m._one_child(robot, "hip_roll_assembly <1>")
    roll_servo = m._servo_placement(
        m._one_prefix_child(hip_roll, "STS32155"), "hip_roll"
    )
    link = m._stretch_original_roll_link(link, roll_servo.axis, roll_servo.center)
    link = m._remap_roll_horn_mount_holes(link, roll_servo)
    link = m._mount_servo_in_original_parts([link], pitch_servo)[0]

    ankle = read_step(ACCEPTED_ANKLE)
    ankle_parts = m._recolour_accepted_ankle(ankle)
    left_sheet = m._translated_leaf(
        ankle_parts["left_knee_to_ankle_left_sheet"], m.KNEE_INSTANCE_OFFSET
    )
    right_sheet = m._translated_leaf(
        ankle_parts["left_knee_to_ankle_right_sheet"], m.KNEE_INSTANCE_OFFSET
    )
    left_sheet = m._trim_servo_clearance(left_sheet, pitch_servo)
    right_sheet = m._trim_servo_clearance(right_sheet, pitch_servo)
    left_sheet = m._remap_pitch_sheet_horn_holes(
        left_sheet, pitch_servo.passive_horn, pitch_servo.center
    )
    right_sheet = m._remap_pitch_sheet_horn_holes(
        right_sheet, pitch_servo.active_horn, pitch_servo.center
    )
    link = m._require_one_solid(
        link.cut(left_sheet, right_sheet), link.label
    )

    m._colour(link, m.COLOURS["roll_link"])
    m._colour(left_sheet, m.COLOURS["sheet"])
    m._colour(right_sheet, m.COLOURS["sheet"])
    m._require_servo_interface(
        "hip-pitch case/right_roll_to_pitch",
        [pitch_servo.body],
        [link],
        m.MAX_FASTENER_INTERFACE_GAP_MM,
    )
    m._require_servo_interface(
        "hip-pitch passive horn/left thigh sheet",
        [pitch_servo.passive_horn],
        [left_sheet],
        m.MAX_FASTENER_INTERFACE_GAP_MM,
    )
    m._require_servo_interface(
        "hip-pitch active horn/right thigh sheet",
        [pitch_servo.active_horn],
        [right_sheet],
        m.MAX_HORN_FACE_GAP_MM,
    )

    result = bd.Compound(
        children=[
            link,
            m._servo_compound(pitch_servo, "S03_RIGHT_HIP_PITCH_ZP10D"),
            left_sheet,
            right_sheet,
        ]
    )
    result.label = "right_roll_to_pitch_assembly <1>"
    return result


if __name__ == "__main__":
    zp10d_right_hip_pitch_interface()
