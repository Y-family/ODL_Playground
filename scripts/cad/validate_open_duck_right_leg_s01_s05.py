#!/usr/bin/env python3
"""Validate only the local S01-S05 right-leg assembly."""

from __future__ import annotations

from cadgen import read_step

import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as interfaces
from package_open_duck_right_leg_s01_s05 import PROJECT_ROOT, _one_child


RIGHT_LEG = PROJECT_ROOT / "cad/local_fix/open_duck_mini_right_leg_s01_s05.step"
VOLUME_TOLERANCE_MM3 = 0.05


def main() -> None:
    root = read_step(RIGHT_LEG)
    vendor = read_step(interfaces.VENDOR)
    vendor_body = max(interfaces._children(vendor), key=lambda child: child.volume)

    trunk = _one_child(root, "trunk_assembly <1>")
    hip = _one_child(trunk, "hip_roll_assembly <1>")
    pitch = _one_child(hip, "right_roll_to_pitch_assembly <1>")
    upper = _one_child(pitch, "knee_and_ankle_assembly <1>")
    lower = _one_child(upper, "knee_and_ankle_assembly <3>")
    foot = _one_child(lower, "foot_assembly <1>")

    trunk_parts = (_one_child(trunk, "trunk_top"), _one_child(trunk, "trunk_bottom"))
    roll_parts = (_one_child(hip, "roll_motor_top"), _one_child(hip, "roll_motor_bottom"))
    link = _one_child(pitch, "right_roll_to_pitch")
    upper_sheets = (
        _one_child(upper, "left_knee_to_ankle_left_sheet"),
        _one_child(upper, "left_knee_to_ankle_right_sheet"),
    )
    lower_sheets = (
        _one_child(lower, "left_knee_to_ankle_left_sheet"),
        _one_child(lower, "left_knee_to_ankle_right_sheet"),
    )
    foot_parts = (_one_child(foot, "foot_top"), _one_child(foot, "foot_side"))

    specs = {
        "S01": interfaces.JointSpec(
            _one_child(trunk, "S01_RIGHT_HIP_YAW_ZP10D"),
            trunk_parts,
            (),
            active_host=roll_parts[0],
            passive_support=trunk_parts[0],
        ),
        "S02": interfaces.JointSpec(
            _one_child(hip, "S02_RIGHT_HIP_ROLL_ZP10D"),
            roll_parts,
            (link,),
        ),
        "S03": interfaces.JointSpec(
            _one_child(pitch, "S03_RIGHT_HIP_PITCH_ZP10D"),
            (link,),
            upper_sheets,
        ),
        "S04": interfaces.JointSpec(
            _one_child(upper, "S04_RIGHT_KNEE_ZP10D"),
            upper_sheets,
            lower_sheets,
        ),
        "S05": interfaces.JointSpec(
            _one_child(lower, "S05_RIGHT_ANKLE_ZP10D"),
            lower_sheets,
            foot_parts,
        ),
    }

    for servo_id, spec in specs.items():
        interfaces._validate_joint(servo_id, spec, vendor_body)

    _, passive_horn, active_horn = interfaces._servo_parts(specs["S03"].servo)
    for side, horn in (("passive", passive_horn), ("active", active_horn)):
        gap = horn.distance_to(link)
        overlap = m._intersection_volume(horn, link)
        if overlap > VOLUME_TOLERANCE_MM3:
            raise ValueError(f"S03 {side} horn/link overlap is {overlap:.3f} mm3")
        print(f"S03 {side}_horn_to_link_gap={gap:.3f} overlap={overlap:.3f}")


if __name__ == "__main__":
    main()
