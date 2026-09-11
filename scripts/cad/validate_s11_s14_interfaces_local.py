#!/usr/bin/env python3
"""Validate static mechanical interfaces for the four upper ZP10D servos."""

from __future__ import annotations

import argparse
from pathlib import Path

from cadgen import read_step

import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v


ROBOT = (
    m.PROJECT_ROOT
    / "cad/local_fix/open_duck_mini_complete_14_zp10d_numbered.step"
)
MIN_FREE_HORN_CLEARANCE_MM = 0.25


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=Path, default=ROBOT)
    args = parser.parse_args()

    robot = read_step(args.robot)
    vendor = read_step(m.VENDOR_SERVO)
    vendor_body = max(v._children(vendor), key=lambda child: child.volume)

    trunk = v._one(robot, "trunk_assembly <1>")
    neck_pitch = v._one(robot, "neck_pitch_assembly <1>")
    neck_yaw = v._one(robot, "neck_yaw_assembly <1>")
    head = v._one(robot, "head_assembly <1>")
    cross_link = v._one(robot, "head_pitch_to_yaw")

    trunk_parts = (
        v._one(trunk, "trunk_top"),
        v._one(trunk, "trunk_bottom"),
    )
    neck_parts = (
        v._one(neck_pitch, "neck_left_sheet"),
        v._one(neck_pitch, "neck_right_sheet"),
    )
    yaw_link = v._one(neck_yaw, "head_yaw_to_roll")
    roll_mount = v._one(head, "head_roll_mount")

    specs = {
        "S11": v.JointSpec(
            v._one(trunk, "S11_NECK_PITCH_ZP10D"),
            trunk_parts,
            neck_parts,
        ),
        "S12": v.JointSpec(
            v._one(neck_pitch, "S12_HEAD_PITCH_ZP10D"),
            neck_parts,
            (cross_link,),
        ),
        "S13": v.JointSpec(
            v._one(neck_yaw, "S13_HEAD_YAW_ZP10D"),
            (yaw_link,),
            (cross_link,),
        ),
        "S14": v.JointSpec(
            v._one(head, "S14_HEAD_ROLL_ZP10D"),
            (roll_mount,),
            (),
            active_host=yaw_link,
        ),
    }

    for servo_id, spec in specs.items():
        v._validate_joint(servo_id, spec, vendor_body)

    _, passive_horn, _ = v._servo_parts(specs["S14"].servo)
    free_gap = passive_horn.distance_to(roll_mount)
    if free_gap < MIN_FREE_HORN_CLEARANCE_MM:
        raise ValueError(
            f"S14 passive horn clearance is {free_gap:.3f} mm, expected at least "
            f"{MIN_FREE_HORN_CLEARANCE_MM:.3f} mm"
        )
    v._require_no_interference(
        "S14 passive horn",
        passive_horn,
        (roll_mount,),
    )
    print(f"S14 PASS passive_horn_free_gap={free_gap:.3f}")


if __name__ == "__main__":
    main()
