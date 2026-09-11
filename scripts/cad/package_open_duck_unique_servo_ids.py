#!/usr/bin/env python3
"""Assign stable, unique functional IDs to all 14 servo occurrences."""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROBOT = (
    PROJECT_ROOT
    / "cad/local_fix/open_duck_mini_one_leg_zp10d_pitch_clearance_full.step"
)
EXPECTED_ROOT_OCCURRENCES = 27
EXPECTED_LEAF_OCCURRENCES = 155

SERVO_IDS = {
    "trunk_assembly <1>": {
        "ZP10D <2>": "S01_RIGHT_HIP_YAW_ZP10D",
        "STS32155 <1>": "S06_LEFT_HIP_YAW_STS3215",
        "STS32155 <3>": "S11_NECK_PITCH_STS3215",
    },
    "hip_roll_assembly <1>": {
        "ZP10D <1>": "S02_RIGHT_HIP_ROLL_ZP10D",
    },
    "right_roll_to_pitch_assembly <1>": {
        "ZP10D <1>": "S03_RIGHT_HIP_PITCH_ZP10D",
    },
    "knee_and_ankle_assembly <1>": {
        "ZP10D <1>": "S04_RIGHT_KNEE_ZP10D",
    },
    "knee_and_ankle_assembly <3>": {
        "ZP10D <1>": "S05_RIGHT_ANKLE_ZP10D",
    },
    "hip_roll_assembly <2>": {
        "STS32155 <1>": "S07_LEFT_HIP_ROLL_STS3215",
    },
    "left_roll_to_pitch_assembly <1>": {
        "STS32155 <1>": "S08_LEFT_HIP_PITCH_STS3215",
    },
    "knee_and_ankle_assembly <2>": {
        "STS32155 <1>": "S09_LEFT_KNEE_STS3215",
    },
    "knee_and_ankle_assembly <4>": {
        "STS32155 <1>": "S10_LEFT_ANKLE_STS3215",
    },
    "neck_pitch_assembly <1>": {
        "STS32155 <1>": "S12_HEAD_PITCH_STS3215",
    },
    "neck_yaw_assembly <1>": {
        "STS32155 <1>": "S13_HEAD_YAW_STS3215",
    },
    "head_assembly <1>": {
        "STS32155 <1>": "S14_HEAD_ROLL_STS3215",
    },
}


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _compound_like(source: bd.Shape, children: list[bd.Shape]) -> bd.Compound:
    result = bd.Compound(children=children)
    result.label = source.label
    result.location = source.location
    return result


def _renamed_servo(source: bd.Shape, new_label: str) -> bd.Compound:
    result = _compound_like(source, _children(source))
    result.label = new_label
    return result


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


@step(out="../../cad/local_fix/open_duck_mini_14_servos_numbered.step")
def open_duck_mini_14_servos_numbered():
    robot = read_step(SOURCE_ROBOT)
    if len(_children(robot)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError("complete robot no longer has 27 top-level instances")

    revised_root = []
    for branch in _children(robot):
        replacements = SERVO_IDS.get(branch.label)
        if replacements is None:
            revised_root.append(branch)
            continue

        found = {
            child.label for child in _children(branch) if child.label in replacements
        }
        if found != set(replacements):
            raise ValueError(
                f"{branch.label} servo labels changed: found {sorted(found)!r}"
            )
        revised_root.append(
            _compound_like(
                branch,
                [
                    _renamed_servo(child, replacements[child.label])
                    if child.label in replacements
                    else child
                    for child in _children(branch)
                ],
            )
        )

    result = _compound_like(robot, revised_root)
    nodes = list(_walk(result))
    numbered = sorted(
        item.label for item in nodes if item.label.startswith("S")
    )
    expected = sorted(label for group in SERVO_IDS.values() for label in group.values())
    if numbered != expected:
        raise ValueError(f"servo numbering mismatch: {numbered!r}")
    if len([item for item in nodes if not _children(item)]) != EXPECTED_LEAF_OCCURRENCES:
        raise ValueError("servo renaming changed the leaf occurrence count")
    return result


if __name__ == "__main__":
    open_duck_mini_14_servos_numbered()
