#!/usr/bin/env python3
"""Package the locally repaired hip-pitch link into the complete robot STEP."""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step

from build_zp10d_left_leg_complete import (
    MAX_FASTENER_INTERFACE_GAP_MM,
    MAX_HORN_FACE_GAP_MM,
    ORIGINAL_PITCH_ACTIVE_HORN_CLEARANCE_MM,
    ORIGINAL_PITCH_PASSIVE_HORN_CLEARANCE_MM,
    PITCH_HORN_CLEARANCE_TOLERANCE_MM,
    VOLUME_TOLERANCE_MM3,
    _intersection_volume,
    _remap_pitch_sheet_horn_holes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROBOT = (
    PROJECT_ROOT
    / "cad/left_leg_complete/open_duck_mini_one_leg_zp10d_original_tree.step"
)
SOURCE_REPAIRED_SUBASSEMBLY = (
    PROJECT_ROOT
    / "cad/local_fix/right_roll_to_pitch_assembly_1_zp10d_clearance.step"
)
EXPECTED_ROOT_OCCURRENCES = 27
EXPECTED_LEAF_OCCURRENCES = 155
S03_JOINT_CENTER = bd.Vector(-35.060, -90.725, -0.0740611876)


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _one_child(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r}, found {len(matches)}")
    return matches[0]


def _compound_like(source: bd.Shape, children: list[bd.Shape]) -> bd.Compound:
    result = bd.Compound(children=children)
    result.label = source.label
    result.location = source.location
    return result


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


@step(
    out="../../cad/local_fix/"
    "open_duck_mini_one_leg_zp10d_pitch_clearance_full.step"
)
def open_duck_mini_one_leg_zp10d_pitch_clearance_full():
    robot = read_step(SOURCE_ROBOT)
    repaired = read_step(SOURCE_REPAIRED_SUBASSEMBLY)
    if len(_children(robot)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError("complete robot no longer has 27 top-level instances")

    source_branch = _one_child(robot, "right_roll_to_pitch_assembly <1>")
    source_servo = _one_child(source_branch, "ZP10D <1>")
    servo_body = _one_child(source_servo, "ZP10D_vendor_body")
    passive_horn = _one_child(source_servo, "ZP10D_vendor_passive_horn")
    active_horn = _one_child(source_servo, "ZP10D_vendor_active_horn")
    repaired_link = _one_child(repaired, "right_roll_to_pitch")
    revised_branch = _compound_like(
        source_branch,
        [
            repaired_link if child.label == "right_roll_to_pitch" else child
            for child in _children(source_branch)
        ],
    )

    source_upper_leg = _one_child(robot, "knee_and_ankle_assembly <1>")
    left_sheet = _one_child(
        source_upper_leg, "left_knee_to_ankle_left_sheet"
    )
    right_sheet = _one_child(
        source_upper_leg, "left_knee_to_ankle_right_sheet"
    )
    revised_left_sheet = _remap_pitch_sheet_horn_holes(
        left_sheet, passive_horn, S03_JOINT_CENTER
    )
    revised_right_sheet = _remap_pitch_sheet_horn_holes(
        right_sheet, active_horn, S03_JOINT_CENTER
    )
    revised_upper_leg = _compound_like(
        source_upper_leg,
        [
            revised_left_sheet
            if child.label == "left_knee_to_ankle_left_sheet"
            else revised_right_sheet
            if child.label == "left_knee_to_ankle_right_sheet"
            else child
            for child in _children(source_upper_leg)
        ],
    )

    link_clearance_checks = (
        (
            "S03 passive horn/right_roll_to_pitch",
            passive_horn,
            ORIGINAL_PITCH_PASSIVE_HORN_CLEARANCE_MM,
        ),
        (
            "S03 active horn/right_roll_to_pitch",
            active_horn,
            ORIGINAL_PITCH_ACTIVE_HORN_CLEARANCE_MM,
        ),
    )
    for label, horn, required_clearance in link_clearance_checks:
        gap = horn.distance_to(repaired_link)
        if gap < required_clearance - PITCH_HORN_CLEARANCE_TOLERANCE_MM:
            raise ValueError(
                f"{label} clearance is {gap:.3f} mm; "
                f"requires {required_clearance:.3f} mm"
            )
        overlap = _intersection_volume(horn, repaired_link)
        if overlap > VOLUME_TOLERANCE_MM3:
            raise ValueError(f"{label} overlaps by {overlap:.3f} mm3")

    fastened_interface_checks = (
        (
            "S03 body/right_roll_to_pitch",
            servo_body,
            repaired_link,
            MAX_FASTENER_INTERFACE_GAP_MM,
        ),
        (
            "S03 passive horn/left thigh sheet",
            passive_horn,
            revised_left_sheet,
            MAX_FASTENER_INTERFACE_GAP_MM,
        ),
        (
            "S03 active horn/right thigh sheet",
            active_horn,
            revised_right_sheet,
            MAX_HORN_FACE_GAP_MM,
        ),
    )
    for label, servo_part, printed_part, max_gap in fastened_interface_checks:
        gap = servo_part.distance_to(printed_part)
        if gap > max_gap:
            raise ValueError(f"{label} gap is {gap:.3f} mm")
        overlap = _intersection_volume(servo_part, printed_part)
        if overlap > VOLUME_TOLERANCE_MM3:
            raise ValueError(f"{label} overlaps by {overlap:.3f} mm3")

    result = _compound_like(
        robot,
        [
            revised_branch
            if child.label == "right_roll_to_pitch_assembly <1>"
            else revised_upper_leg
            if child.label == "knee_and_ankle_assembly <1>"
            else child
            for child in _children(robot)
        ],
    )

    leaves = [item for item in _walk(result) if not _children(item)]
    if len(leaves) != EXPECTED_LEAF_OCCURRENCES:
        raise ValueError(
            f"complete robot leaf count changed: {len(leaves)}"
        )
    if sum(item.label.startswith("ZP10D_vendor_body") for item in leaves) != 5:
        raise ValueError("complete robot no longer contains five ZP10D servos")
    return result


if __name__ == "__main__":
    open_duck_mini_one_leg_zp10d_pitch_clearance_full()
