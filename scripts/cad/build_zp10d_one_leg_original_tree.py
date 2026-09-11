#!/usr/bin/env python3
"""Package the validated ZP10D leg into the original 27-item robot tree."""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, report, step

from build_zp10d_left_leg_complete import zp10d_left_leg_complete


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROBOT = PROJECT_ROOT / "cad/source_exports/open_duck_original_isolated_branch.step"
EXPECTED_ROOT_OCCURRENCES = 27
EXPECTED_SOURCE_LEAVES = 165
EXPECTED_RESULT_LEAVES = 155
EXPECTED_SOURCE_STS3215 = 14
EXPECTED_RESULT_STS3215 = 9
EXPECTED_RESULT_ZP10D = 5


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _one_child(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r}, found {len(matches)}")
    return matches[0]


def _one_descendant(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [item for item in _walk(parent) if item.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one descendant {label!r}, found {len(matches)}")
    return matches[0]


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


def _leaf_count(shape: bd.Shape) -> int:
    return sum(not _children(item) for item in _walk(shape))


def _actuator_count(shape: bd.Shape, prefix: str) -> int:
    return sum(item.label.startswith(prefix) for item in _walk(shape))


def _compound_like(source: bd.Shape, children: list[bd.Shape]) -> bd.Compound:
    result = bd.Compound(children=children)
    result.label = source.label
    result.location = source.location
    return result


def _replace_by_label(
    source: bd.Shape,
    replacements: dict[str, bd.Shape],
) -> bd.Compound:
    source_children = _children(source)
    source_labels = [child.label for child in source_children]
    missing = sorted(set(replacements) - set(source_labels))
    if missing:
        raise ValueError(f"{source.label} lacks replacement labels {missing}")
    return _compound_like(
        source,
        [replacements.get(child.label, child) for child in source_children],
    )


@step(out="../../cad/left_leg_complete/open_duck_mini_one_leg_zp10d_original_tree.step")
def open_duck_mini_one_leg_zp10d_original_tree():
    report("load original 27-item robot tree")
    source = read_step(SOURCE_ROBOT)
    if len(_children(source)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError("original root occurrence count changed")
    if _leaf_count(source) != EXPECTED_SOURCE_LEAVES:
        raise ValueError("original leaf occurrence count changed")
    if _actuator_count(source, "STS32155") != EXPECTED_SOURCE_STS3215:
        raise ValueError("original STS3215 occurrence count changed")
    source_root_labels = [child.label for child in _children(source)]

    report("load validated converted leg")
    converted_root = zp10d_left_leg_complete()
    converted_root.bounding_box()
    converted_trunk = _one_descendant(converted_root, "trunk_assembly <1>")
    converted_hip = _one_child(converted_trunk, "hip_roll_assembly <1>")
    converted_roll_to_pitch = _one_child(
        converted_hip, "right_roll_to_pitch_assembly <1>"
    )
    converted_upper_leg = _one_child(
        converted_roll_to_pitch, "knee_and_ankle_assembly <1>"
    )
    converted_lower_leg = _one_child(
        converted_upper_leg, "knee_and_ankle_assembly <3>"
    )
    converted_foot = _one_child(converted_lower_leg, "foot_assembly <1>")

    report("restore original subassembly membership")
    source_trunk = _one_child(source, "trunk_assembly <1>")
    trunk = _replace_by_label(
        source_trunk,
        {
            "trunk_top": _one_child(converted_trunk, "trunk_top"),
            "trunk_bottom": _one_child(converted_trunk, "trunk_bottom"),
            "STS32155 <2>": _one_child(converted_trunk, "ZP10D <2>"),
        },
    )

    source_hip = _one_child(source, "hip_roll_assembly <1>")
    hip = _replace_by_label(
        source_hip,
        {
            "roll_motor_top": _one_child(converted_hip, "roll_motor_top"),
            "roll_motor_bottom": _one_child(converted_hip, "roll_motor_bottom"),
            "STS32155 <1>": _one_child(converted_hip, "ZP10D <1>"),
        },
    )

    source_roll_to_pitch = _one_child(source, "right_roll_to_pitch_assembly <1>")
    roll_to_pitch = _replace_by_label(
        source_roll_to_pitch,
        {
            "right_roll_to_pitch": _one_child(
                converted_roll_to_pitch, "right_roll_to_pitch"
            ),
            "STS32155 <1>": _one_child(converted_roll_to_pitch, "ZP10D <1>"),
        },
    )

    source_upper_leg = _one_child(source, "knee_and_ankle_assembly <1>")
    upper_leg = _replace_by_label(
        source_upper_leg,
        {
            "left_knee_to_ankle_left_sheet": _one_child(
                converted_upper_leg, "left_knee_to_ankle_left_sheet"
            ),
            "left_knee_to_ankle_right_sheet": _one_child(
                converted_upper_leg, "left_knee_to_ankle_right_sheet"
            ),
            "STS32155 <1>": _one_child(converted_upper_leg, "ZP10D <1>"),
            "leg_spacer": _one_child(converted_upper_leg, "leg_spacer"),
        },
    )

    source_lower_leg = _one_child(source, "knee_and_ankle_assembly <3>")
    lower_leg = _replace_by_label(
        source_lower_leg,
        {
            "leg_spacer": _one_child(converted_lower_leg, "leg_spacer"),
            "left_knee_to_ankle_left_sheet": _one_child(
                converted_lower_leg, "left_knee_to_ankle_left_sheet"
            ),
            "STS32155 <1>": _one_child(converted_lower_leg, "ZP10D <1>"),
            "left_knee_to_ankle_right_sheet": _one_child(
                converted_lower_leg, "left_knee_to_ankle_right_sheet"
            ),
        },
    )

    source_foot = _one_child(source, "foot_assembly <1>")
    foot = _replace_by_label(
        source_foot,
        {child.label: child for child in _children(converted_foot)},
    )
    converted_cache = _one_child(converted_upper_leg, "right_cache")

    root_replacements = {
        "right_cache": converted_cache,
        "hip_roll_assembly <1>": hip,
        "knee_and_ankle_assembly <3>": lower_leg,
        "right_roll_to_pitch_assembly <1>": roll_to_pitch,
        "foot_assembly <1>": foot,
        "knee_and_ankle_assembly <1>": upper_leg,
        "trunk_assembly <1>": trunk,
    }
    result = _replace_by_label(source, root_replacements)
    result.label = source.label

    report("validate original tree and one-leg replacement counts")
    result_root_labels = [child.label for child in _children(result)]
    if result_root_labels != source_root_labels:
        raise ValueError(
            "root assembly order or labels changed: "
            f"source={source_root_labels!r}, result={result_root_labels!r}"
        )
    if len(_children(result)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError("result root occurrence count changed")
    if _leaf_count(result) != EXPECTED_RESULT_LEAVES:
        raise ValueError("result leaf occurrence count changed")
    if _actuator_count(result, "STS32155") != EXPECTED_RESULT_STS3215:
        raise ValueError("result STS3215 occurrence count is not nine")
    if _actuator_count(result, "ZP10D <") != EXPECTED_RESULT_ZP10D:
        raise ValueError("result ZP10D occurrence count is not five")
    return result


if __name__ == "__main__":
    open_duck_mini_one_leg_zp10d_original_tree()
