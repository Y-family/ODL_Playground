#!/usr/bin/env python3
"""Shorten only the local right_roll_to_pitch feet before the S03 horn."""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step

from build_zp10d_left_leg_complete import (
    ORIGINAL_PITCH_ACTIVE_HORN_CLEARANCE_MM,
    ORIGINAL_PITCH_PASSIVE_HORN_CLEARANCE_MM,
    PITCH_HORN_CLEARANCE_TOLERANCE_MM,
    ServoPlacement,
    _children,
    _intersection_volume,
    _one_child,
    _shorten_pitch_horn_feet_from_roll_link,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    PROJECT_ROOT
    / "cad/onshape_staging/right_roll_to_pitch_assembly_1_zp10d.step"
)
JOINT_AXIS = bd.Vector(0, -1, 0)
JOINT_CENTER = bd.Vector(-35.060, -90.725, -0.0740611876)


@step(
    out="../../cad/local_fix/"
    "right_roll_to_pitch_assembly_1_zp10d_clearance.step"
)
def right_roll_to_pitch_assembly_1_zp10d_clearance():
    source = read_step(SOURCE)
    roll_link = _one_child(source, "right_roll_to_pitch")
    servo = _one_child(source, "ZP10D <1>")
    servo.label = "S03_RIGHT_HIP_PITCH_ZP10D"
    body = _one_child(servo, "ZP10D_vendor_body")
    passive_horn = _one_child(servo, "ZP10D_vendor_passive_horn")
    active_horn = _one_child(servo, "ZP10D_vendor_active_horn")

    placement = ServoPlacement(
        pose=bd.Location(),
        axis=JOINT_AXIS,
        center=JOINT_CENTER,
        body=body,
        passive_horn=passive_horn,
        active_horn=active_horn,
    )
    revised_link = _shorten_pitch_horn_feet_from_roll_link(roll_link, placement)

    checks = (
        (
            "active",
            active_horn,
            ORIGINAL_PITCH_ACTIVE_HORN_CLEARANCE_MM,
        ),
        (
            "passive",
            passive_horn,
            ORIGINAL_PITCH_PASSIVE_HORN_CLEARANCE_MM,
        ),
    )
    for side, horn, required in checks:
        actual = horn.distance_to(revised_link)
        if actual < required - PITCH_HORN_CLEARANCE_TOLERANCE_MM:
            raise ValueError(
                f"{side} horn clearance {actual:.3f} mm is below "
                f"{required:.3f} mm"
            )
        overlap = _intersection_volume(horn, revised_link)
        if overlap > 0.001:
            raise ValueError(
                f"{side} horn intersects right_roll_to_pitch by "
                f"{overlap:.6f} mm3"
            )

    if len(_children(source)) != 2:
        raise ValueError("source subassembly tree changed")
    result = bd.Compound(children=[revised_link, servo])
    result.label = source.label
    return result


if __name__ == "__main__":
    right_roll_to_pitch_assembly_1_zp10d_clearance()
