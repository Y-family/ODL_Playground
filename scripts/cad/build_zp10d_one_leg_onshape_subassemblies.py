#!/usr/bin/env python3
"""Export the six changed root subassemblies for audited Onshape replacement."""

from __future__ import annotations

from cadgen import build123d as bd
from cadgen import step

from build_zp10d_one_leg_original_tree import (
    open_duck_mini_one_leg_zp10d_original_tree,
)


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


def _robot() -> bd.Shape:
    linked = open_duck_mini_one_leg_zp10d_original_tree()
    linked.bounding_box()
    matches = [
        item
        for item in _walk(linked)
        if len(_children(item)) == 27
        and any(child.label == "trunk_assembly <1>" for child in _children(item))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one 27-item robot root, found {len(matches)}")
    return matches[0]


def _root_child(label: str) -> bd.Shape:
    matches = [child for child in _children(_robot()) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one root child {label!r}, found {len(matches)}")
    return matches[0]


@step(out="../../cad/onshape_staging/trunk_assembly_1_one_zp10d.step")
def trunk_assembly_1_one_zp10d():
    return _root_child("trunk_assembly <1>")


@step(out="../../cad/onshape_staging/hip_roll_assembly_1_zp10d.step")
def hip_roll_assembly_1_zp10d():
    return _root_child("hip_roll_assembly <1>")


@step(out="../../cad/onshape_staging/right_roll_to_pitch_assembly_1_zp10d.step")
def right_roll_to_pitch_assembly_1_zp10d():
    return _root_child("right_roll_to_pitch_assembly <1>")


@step(out="../../cad/onshape_staging/knee_and_ankle_assembly_1_zp10d.step")
def knee_and_ankle_assembly_1_zp10d():
    return _root_child("knee_and_ankle_assembly <1>")


@step(out="../../cad/onshape_staging/knee_and_ankle_assembly_3_zp10d.step")
def knee_and_ankle_assembly_3_zp10d():
    return _root_child("knee_and_ankle_assembly <3>")


@step(out="../../cad/onshape_staging/foot_assembly_1_zp10d.step")
def foot_assembly_1_zp10d():
    return _root_child("foot_assembly <1>")


if __name__ == "__main__":
    trunk_assembly_1_one_zp10d()
    hip_roll_assembly_1_zp10d()
    right_roll_to_pitch_assembly_1_zp10d()
    knee_and_ankle_assembly_1_zp10d()
    knee_and_ankle_assembly_3_zp10d()
    foot_assembly_1_zp10d()
