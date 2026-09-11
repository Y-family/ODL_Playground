#!/usr/bin/env python3
"""Time each right-hip geometry operation without exporting an assembly."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

from cadgen import read_step

import build_zp10d_left_leg_complete as m


ACCEPTED_ANKLE = m.PROJECT_ROOT / "cad/left_leg_step1/zp10d_left_ankle_step1.step"


@contextmanager
def timed(label: str):
    started = perf_counter()
    print(f"START {label}", flush=True)
    try:
        yield
    finally:
        print(f"DONE  {label} {perf_counter() - started:.3f}s", flush=True)


def main() -> None:
    with timed("load original robot"):
        robot = read_step(m.SOURCE_ROBOT)

    roll_to_pitch_source = m._one_child(robot, "right_roll_to_pitch_assembly <1>")
    roll_link = m._one_child(roll_to_pitch_source, "right_roll_to_pitch")
    old_pitch_servo = m._one_prefix_child(roll_to_pitch_source, "STS32155")
    with timed("place hip-pitch ZP10D"):
        pitch_servo = m._servo_placement(old_pitch_servo, "hip_pitch")

    hip_roll_source = m._one_child(robot, "hip_roll_assembly <1>")
    old_roll_servo = m._one_prefix_child(hip_roll_source, "STS32155")
    with timed("place hip-roll ZP10D"):
        roll_servo = m._servo_placement(old_roll_servo, "hip_roll")
    with timed("stretch right_roll_to_pitch"):
        roll_link = m._stretch_original_roll_link(
            roll_link, roll_servo.axis, roll_servo.center
        )
    with timed("remap hip-roll horn holes"):
        roll_link = m._remap_roll_horn_mount_holes(roll_link, roll_servo)
    with timed("mount hip-pitch case in right_roll_to_pitch"):
        roll_link = m._mount_servo_in_original_parts([roll_link], pitch_servo)[0]

    roll_top = m._one_child(hip_roll_source, "roll_motor_top")
    roll_bottom = m._one_child(hip_roll_source, "roll_motor_bottom")
    trunk_source = m._one_child(robot, "trunk_assembly <1>")
    trunk_servos = [
        child
        for child in m._children(trunk_source)
        if child.label.startswith("STS32155")
    ]
    old_yaw_servo = min(
        trunk_servos,
        key=lambda servo: servo.bounding_box().center().Y,
    )
    with timed("place hip-yaw ZP10D"):
        yaw_servo = m._servo_placement(old_yaw_servo, "hip_yaw")

    with timed("mount hip-roll case in original housing"):
        roll_top, roll_bottom = m._mount_servo_in_original_parts(
            [roll_top, roll_bottom],
            roll_servo,
            bridge_roll_far_side=True,
            require_all_mounts=True,
        )
    with timed("mount hip-yaw horns in original housing"):
        roll_top, roll_bottom = m._mount_horns_in_original_parts(
            [roll_top, roll_bottom], yaw_servo
        )
    with timed("seat hip-yaw active horn"):
        roll_top = m._seat_yaw_active_horn_outside_roll_top(roll_top, yaw_servo)

    trunk_top = m._one_child(trunk_source, "trunk_top")
    trunk_bottom = m._one_child(trunk_source, "trunk_bottom")
    with timed("mount hip-yaw case in trunk"):
        trunk_top, trunk_bottom = m._mount_servo_in_original_parts(
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
    with timed("clear roll_motor_top"):
        roll_top = m._require_one_solid(
            roll_top.cut(*roll_obstacles), roll_top.label
        )
    with timed("clear roll_motor_bottom"):
        roll_bottom = m._require_one_solid(
            roll_bottom.cut(*roll_obstacles),
            roll_bottom.label,
            max_disconnected_trim=100.0,
        )

    with timed("trim hip-yaw rotation envelope"):
        roll_top = m._trim_moving_part_for_rotation(
            roll_top,
            [trunk_top, trunk_bottom],
            yaw_servo,
            m.YAW_RANGE_DEG,
        )
    with timed("trim hip-roll rotation envelope"):
        roll_link = m._trim_moving_part_for_rotation(
            roll_link,
            [roll_top, roll_bottom],
            roll_servo,
            m.ROLL_RANGE_DEG,
        )
    print("DONE right-hip pipeline", flush=True)

    with timed("load accepted ankle"):
        ankle = read_step(ACCEPTED_ANKLE)
        ankle_parts = m._recolour_accepted_ankle(ankle)
    upper_left_sheet = m._translated_leaf(
        ankle_parts["left_knee_to_ankle_left_sheet"], m.KNEE_INSTANCE_OFFSET
    )
    upper_right_sheet = m._translated_leaf(
        ankle_parts["left_knee_to_ankle_right_sheet"], m.KNEE_INSTANCE_OFFSET
    )
    with timed("clear hip-pitch servo from left thigh sheet"):
        upper_left_sheet = m._trim_servo_clearance(upper_left_sheet, pitch_servo)
    with timed("clear hip-pitch servo from right thigh sheet"):
        upper_right_sheet = m._trim_servo_clearance(upper_right_sheet, pitch_servo)
    with timed("trim fixed hip link around both thigh sheets"):
        roll_link = m._trim_fixed_part_for_rotation(
            roll_link,
            [upper_left_sheet, upper_right_sheet],
            pitch_servo,
            m.PITCH_RANGE_DEG,
        )
    print("DONE hip-pitch pipeline", flush=True)


if __name__ == "__main__":
    main()
