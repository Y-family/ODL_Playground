#!/usr/bin/env python3
"""Extract the original left hip-roll interface for local comparison."""

from __future__ import annotations

from cadgen import build123d as bd
from cadgen import read_step, step

from build_zp10d_left_leg_complete import (
    SOURCE_ROBOT,
    _children,
    _one_child,
    _one_prefix_child,
)


@step(out="../../cad/local_fix/original_left_hip_roll_reference.step")
def original_left_hip_roll_reference():
    robot = read_step(SOURCE_ROBOT)
    hip_roll = _one_child(robot, "hip_roll_assembly <2>")
    servo = _one_prefix_child(hip_roll, "STS32155")
    roll_to_pitch = _one_child(robot, "left_roll_to_pitch_assembly <1>")
    link = _one_child(roll_to_pitch, "left_roll_to_pitch")

    reference_servo = bd.Compound(children=_children(servo))
    reference_servo.label = "S07_LEFT_HIP_ROLL_STS3215"
    result = bd.Compound(children=[link, reference_servo])
    result.label = "original_left_hip_roll_reference"
    return result


if __name__ == "__main__":
    original_left_hip_roll_reference()
