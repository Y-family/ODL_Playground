#!/usr/bin/env python3
"""Fuse three ZP10D locating sleeves into the existing trunk_top part.

The source assembly already carries three shaft-aligned ZP10D subassemblies.
This model clears their case envelopes from trunk_top, fuses the three open-end
sleeves into that same original solid, and exports no additional occurrence.
"""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ASSEMBLY = PROJECT_ROOT / "cad/source_exports/trunk_internal_zp10d_axis_aligned.step"
VENDOR_STEP = Path("/root/ZP10D舵机/金属舵机最小..双轴.STEP")

TRUNK_TOP_LABEL = "trunk_top"
SERVO_LOCAL_CENTER = (6.30656134575607, -0.0801681526999323, 51.0)
SLEEVE_LENGTH_X = 39.5
SLEEVE_OUTER_Y = 47.0
SLEEVE_OUTER_Z = 27.0
SLEEVE_INNER_LENGTH_X = 42.5
SLEEVE_INNER_Y = 40.5
SLEEVE_INNER_Z = 20.5
CASE_KEEPOUT_LENGTH_X = 53.5
MIN_FUSION_VOLUME_MM3 = 100.0
MAX_INTERFERENCE_MM3 = 0.01


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _intersection_volume(left: bd.Shape, right: bd.Shape) -> float:
    result = left.intersect(right)
    if result is None:
        return 0.0
    if isinstance(result, list):
        return sum(item.volume for item in result)
    return result.volume


def _single_child_by_label(assembly: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(assembly) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r} child, found {len(matches)}")
    return matches[0]


def _largest_child(assembly: bd.Shape) -> bd.Shape:
    return max(_children(assembly), key=lambda child: child.volume)


def _zp10d_subassemblies(assembly: bd.Shape) -> list[bd.Shape]:
    matches = [child for child in _children(assembly) if len(_children(child)) == 3]
    if len(matches) != 3:
        raise ValueError(f"expected three ZP10D subassemblies, found {len(matches)}")
    return matches


def _servo_locations(trunk: bd.Shape) -> list[bd.Location]:
    vendor_body = _largest_child(read_step(VENDOR_STEP))
    return [
        _largest_child(servo).location * vendor_body.location.inverse()
        for servo in _zp10d_subassemblies(trunk)
    ]


def _sleeve() -> bd.Solid:
    center = bd.Pos(*SERVO_LOCAL_CENTER)
    outer = bd.Box(
        SLEEVE_LENGTH_X,
        SLEEVE_OUTER_Y,
        SLEEVE_OUTER_Z,
        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER),
    ).moved(center)
    inner = bd.Box(
        SLEEVE_INNER_LENGTH_X,
        SLEEVE_INNER_Y,
        SLEEVE_INNER_Z,
        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER),
    ).moved(center)
    result = outer - inner
    if len(result.solids()) != 1 or not result.is_valid:
        raise ValueError("trunk sleeve is not one valid solid")
    return result


def _case_keepout() -> bd.Solid:
    return bd.Box(
        CASE_KEEPOUT_LENGTH_X,
        SLEEVE_INNER_Y,
        SLEEVE_INNER_Z,
        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER),
    ).moved(bd.Pos(*SERVO_LOCAL_CENTER))


@step(out="../../cad/remaining/trunk_internal_zp10d_no_new_parts.step")
def trunk_internal_zp10d_no_new_parts():
    trunk = read_step(SOURCE_ASSEMBLY)
    original_top = _single_child_by_label(trunk, TRUNK_TOP_LABEL)
    servos = _zp10d_subassemblies(trunk)
    locations = _servo_locations(trunk)
    sleeves = [location * _sleeve() for location in locations]
    keepouts = [location * _case_keepout() for location in locations]

    cleared_top = original_top.cut(*keepouts)
    if len(cleared_top.solids()) != 1 or not cleared_top.is_valid:
        raise ValueError("case clearances split or invalidate trunk_top")

    for index, sleeve in enumerate(sleeves, start=1):
        fusion = _intersection_volume(cleared_top, sleeve)
        if fusion < MIN_FUSION_VOLUME_MM3:
            raise ValueError(f"sleeve {index} fusion is only {fusion:.3f} mm3")

    revised_top = cleared_top.fuse(*sleeves).cut(*keepouts)
    if len(revised_top.solids()) != 1 or not revised_top.is_valid:
        raise ValueError("revised trunk_top is not one valid solid")
    revised_top.label = TRUNK_TOP_LABEL

    for servo_index, servo in enumerate(servos, start=1):
        for component in _children(servo):
            overlap = _intersection_volume(revised_top, component)
            if overlap > MAX_INTERFERENCE_MM3:
                raise ValueError(
                    f"trunk_top intersects servo {servo_index} component "
                    f"{component.label}: {overlap:.3f} mm3"
                )

    children = [
        revised_top if child.label == TRUNK_TOP_LABEL else child
        for child in _children(trunk)
    ]
    result = bd.Compound(children=children)
    result.label = "trunk_assembly_ZP10D_no_new_parts"
    return result


if __name__ == "__main__":
    trunk_internal_zp10d_no_new_parts()
