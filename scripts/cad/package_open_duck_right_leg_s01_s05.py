#!/usr/bin/env python3
"""Package the locally repaired, numbered S01-S05 right leg only."""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_LEG = PROJECT_ROOT / "cad/left_leg_complete/zp10d_left_leg_complete.step"
SOURCE_REPAIRED_PITCH = (
    PROJECT_ROOT
    / "cad/local_fix/right_roll_to_pitch_assembly_1_zp10d_clearance.step"
)


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _one_child(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r}, found {len(matches)}")
    return matches[0]


def _compound_like(
    source: bd.Shape,
    children: list[bd.Shape],
    *,
    label: str | None = None,
) -> bd.Compound:
    result = bd.Compound(children=children)
    result.label = source.label if label is None else label
    result.location = source.location
    return result


def _replace_children(
    source: bd.Shape,
    replacements: dict[str, bd.Shape],
) -> bd.Compound:
    source_children = _children(source)
    found = {child.label for child in source_children if child.label in replacements}
    if found != set(replacements):
        raise ValueError(
            f"{source.label} replacement mismatch: "
            f"expected {sorted(replacements)!r}, found {sorted(found)!r}"
        )
    return _compound_like(
        source,
        [replacements.get(child.label, child) for child in source_children],
    )


def _renamed_servo(source: bd.Shape, label: str) -> bd.Compound:
    return _compound_like(source, _children(source), label=label)


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


@step(out="../../cad/local_fix/open_duck_mini_right_leg_s01_s05.step")
def open_duck_mini_right_leg_s01_s05():
    source = read_step(SOURCE_LEG)
    repaired_pitch = read_step(SOURCE_REPAIRED_PITCH)
    repaired_link = _one_child(repaired_pitch, "right_roll_to_pitch")

    trunk = _one_child(source, "trunk_assembly <1>")
    hip = _one_child(trunk, "hip_roll_assembly <1>")
    pitch = _one_child(hip, "right_roll_to_pitch_assembly <1>")
    upper = _one_child(pitch, "knee_and_ankle_assembly <1>")
    lower = _one_child(upper, "knee_and_ankle_assembly <3>")

    revised_lower = _replace_children(
        lower,
        {
            "ZP10D <1>": _renamed_servo(
                _one_child(lower, "ZP10D <1>"),
                "S05_RIGHT_ANKLE_ZP10D",
            )
        },
    )
    revised_upper = _replace_children(
        upper,
        {
            "ZP10D <1>": _renamed_servo(
                _one_child(upper, "ZP10D <1>"),
                "S04_RIGHT_KNEE_ZP10D",
            ),
            "knee_and_ankle_assembly <3>": revised_lower,
        },
    )
    revised_pitch = _replace_children(
        pitch,
        {
            "right_roll_to_pitch": repaired_link,
            "ZP10D <1>": _renamed_servo(
                _one_child(pitch, "ZP10D <1>"),
                "S03_RIGHT_HIP_PITCH_ZP10D",
            ),
            "knee_and_ankle_assembly <1>": revised_upper,
        },
    )
    revised_hip = _replace_children(
        hip,
        {
            "ZP10D <1>": _renamed_servo(
                _one_child(hip, "ZP10D <1>"),
                "S02_RIGHT_HIP_ROLL_ZP10D",
            ),
            "right_roll_to_pitch_assembly <1>": revised_pitch,
        },
    )
    revised_trunk = _replace_children(
        trunk,
        {
            "ZP10D <2>": _renamed_servo(
                _one_child(trunk, "ZP10D <2>"),
                "S01_RIGHT_HIP_YAW_ZP10D",
            ),
            "hip_roll_assembly <1>": revised_hip,
        },
    )
    result = _compound_like(
        source,
        [revised_trunk],
        label="OPEN_DUCK_MINI_RIGHT_LEG_S01_S05",
    )

    labels = [node.label for node in _walk(result)]
    expected_ids = [
        "S01_RIGHT_HIP_YAW_ZP10D",
        "S02_RIGHT_HIP_ROLL_ZP10D",
        "S03_RIGHT_HIP_PITCH_ZP10D",
        "S04_RIGHT_KNEE_ZP10D",
        "S05_RIGHT_ANKLE_ZP10D",
    ]
    actual_ids = sorted(label for label in labels if label.startswith("S"))
    if actual_ids != expected_ids:
        raise ValueError(f"right-leg servo numbering mismatch: {actual_ids!r}")
    if any(label.startswith("STS32155") for label in labels):
        raise ValueError("right-leg assembly still contains an STS3215 servo")
    return result


if __name__ == "__main__":
    open_duck_mini_right_leg_s01_s05()
