#!/usr/bin/env python3
"""Restore the original 27-item robot tree around the approved lower body."""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_TREE = PROJECT_ROOT / "cad/local_fix/open_duck_mini_10_zp10d_numbered.step"
APPROVED_LOWER = PROJECT_ROOT / "cad/local_fix/open_duck_mini_trunk_inward_candidate.step"
EXPECTED_ROOT_OCCURRENCES = 27
EXPECTED_LEAF_OCCURRENCES = 145
EXPECTED_LOWER_ROOT_OCCURRENCES = 13


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


def _one(parent: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(parent) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"{parent.label}: expected one {label!r}, found {len(matches)}")
    return matches[0]


def _merged_trunk(reference: bd.Shape, approved: bd.Shape) -> bd.Compound:
    approved_unique = {
        child.label: child
        for child in _children(approved)
        if child.label != "roll_bearing"
    }
    approved_bearings = sorted(
        (child for child in _children(approved) if child.label == "roll_bearing"),
        key=lambda child: child.bounding_box().center().Y,
    )
    if len(approved_bearings) != 2:
        raise ValueError("approved trunk must contain exactly two roll bearings")

    merged = []
    for child in _children(reference):
        if child.label == "roll_bearing":
            sign = -1 if child.bounding_box().center().Y < 0 else 1
            matches = [
                bearing
                for bearing in approved_bearings
                if (-1 if bearing.bounding_box().center().Y < 0 else 1) == sign
            ]
            if len(matches) != 1:
                raise ValueError(f"approved trunk lacks the {sign:+d}Y roll bearing")
            merged.append(matches[0])
        elif child.label in approved_unique:
            merged.append(approved_unique[child.label])
        else:
            # Keep BNO055, Board and the original S11 occurrence for the
            # one-for-one ZP10D conversion in the complete-robot builder.
            merged.append(child)

    if [child.label for child in merged] != [child.label for child in _children(reference)]:
        raise ValueError("trunk child order changed")
    result = bd.Compound(children=merged, label=reference.label)
    return result


@step(out="../../cad/local_fix/open_duck_mini_10_zp10d_current_lower.step")
def current_lower_in_original_tree():
    reference = read_step(ORIGINAL_TREE)
    approved = read_step(APPROVED_LOWER)
    if len(_children(reference)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError("original tree root count changed")
    if len(_children(approved)) != EXPECTED_LOWER_ROOT_OCCURRENCES:
        raise ValueError("approved lower-body root count changed")

    approved_by_label = {child.label: child for child in _children(approved)}
    if len(approved_by_label) != EXPECTED_LOWER_ROOT_OCCURRENCES:
        raise ValueError("approved lower-body labels are not unique")
    reference_labels = [child.label for child in _children(reference)]
    missing = set(approved_by_label) - set(reference_labels)
    if missing:
        raise ValueError(f"original tree lacks approved occurrences: {sorted(missing)!r}")

    reference_trunk = _one(reference, "trunk_assembly <1>")
    approved_trunk = _one(approved, "trunk_assembly <1>")
    children = []
    for child in _children(reference):
        replacement = approved_by_label.get(child.label)
        if child is reference_trunk:
            replacement = _merged_trunk(reference_trunk, approved_trunk)
        children.append(replacement or child)

    result = bd.Compound(children=children, label=reference.label)
    if [child.label for child in _children(result)] != reference_labels:
        raise ValueError("original 27-item root order changed")
    nodes = list(_walk(result))
    leaves = [node for node in nodes if not _children(node)]
    if len(_children(result)) != EXPECTED_ROOT_OCCURRENCES:
        raise ValueError("root occurrence count changed")
    if len(leaves) != EXPECTED_LEAF_OCCURRENCES:
        raise ValueError(f"leaf occurrence count changed to {len(leaves)}")
    servo_labels = sorted(node.label for node in nodes if node.label.startswith("S"))
    if [label[:3] for label in servo_labels] != [f"S{i:02}" for i in range(1, 15)]:
        raise ValueError(f"servo numbering is not unique: {servo_labels!r}")
    if sum(label.endswith("_ZP10D") for label in servo_labels) != 10:
        raise ValueError("merged source must contain exactly ten ZP10D servos")
    return result


if __name__ == "__main__":
    current_lower_in_original_tree()
