#!/usr/bin/env python3
"""Fuse a ZP10D case sleeve into the existing head_roll_mount.

The output retains the source assembly's component count.  Only the existing
head_roll_mount solid is replaced by its fused revision; the sleeve never
exists as a separate exported occurrence.
"""

from __future__ import annotations

from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step, step

import build_zp10d_left_leg_complete as m


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ASSEMBLY = PROJECT_ROOT / "cad/source_exports/head_internal_zp10d_axis_aligned.step"
VENDOR_STEP = Path("/root/ZP10D舵机/金属舵机最小..双轴.STEP")

MOUNT_LABEL = "head_roll_mount"
SERVO_LOCAL_CENTER = (6.30656134575607, -0.0801681526999323, 51.0)
SLEEVE_LENGTH_X = 40.8
SLEEVE_OUTER_Y = 47.0
SLEEVE_OUTER_Z = 27.0
SLEEVE_INNER_LENGTH_X = 43.0
SLEEVE_INNER_Y = 40.7
SLEEVE_INNER_Z = 20.7
MIN_FUSION_VOLUME_MM3 = 10.0
MAX_INTERFERENCE_MM3 = 0.01


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _single_child_by_label(assembly: bd.Shape, label: str) -> bd.Shape:
    matches = [child for child in _children(assembly) if child.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r} child, found {len(matches)}")
    return matches[0]


def _zp10d_subassembly(assembly: bd.Shape) -> bd.Shape:
    matches = [child for child in _children(assembly) if len(_children(child)) == 3]
    if len(matches) != 1:
        raise ValueError(f"expected one three-part ZP10D subassembly, found {len(matches)}")
    return matches[0]


def _largest_child(assembly: bd.Shape) -> bd.Shape:
    return max(_children(assembly), key=lambda child: child.volume)


def _intersection_volume(left: bd.Shape, right: bd.Shape) -> float:
    result = left.intersect(right)
    if result is None:
        return 0.0
    if isinstance(result, list):
        return sum(item.volume for item in result)
    return result.volume


def _servo_assembly_location(head: bd.Shape) -> bd.Location:
    placed_body = _largest_child(_zp10d_subassembly(head))
    vendor_body = _largest_child(read_step(VENDOR_STEP))
    return placed_body.location * vendor_body.location.inverse()


def _servo_placement(head: bd.Shape) -> m.ServoPlacement:
    servo = _zp10d_subassembly(head)
    parts = _children(servo)
    body = max(parts, key=lambda part: part.volume)
    passive, active = sorted(
        (part for part in parts if part is not body),
        key=lambda part: part.volume,
    )
    passive_center = passive.bounding_box().center()
    active_center = active.bounding_box().center()
    return m.ServoPlacement(
        pose=_servo_assembly_location(head),
        axis=(active_center - passive_center).normalized(),
        center=(passive_center + active_center) * 0.5,
        body=body,
        passive_horn=passive,
        active_horn=active,
    )


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
        raise ValueError("head sleeve is not one valid solid")
    return result


@step(out="../../cad/remaining/head_internal_zp10d_no_new_parts.step")
def head_internal_zp10d_no_new_parts():
    head = read_step(SOURCE_ASSEMBLY)
    original_mount = _single_child_by_label(head, MOUNT_LABEL)
    servo = _zp10d_subassembly(head)
    placement = _servo_placement(head)
    sleeve_world = _servo_assembly_location(head) * _sleeve()

    fusion_volume = _intersection_volume(original_mount, sleeve_world)
    if fusion_volume < MIN_FUSION_VOLUME_MM3:
        raise ValueError(f"sleeve does not fuse robustly: {fusion_volume:.3f} mm3")

    revised_mount = original_mount.fuse(sleeve_world)
    case_pads, case_holes = m._case_mount_tools(placement)
    for index, pad in enumerate(case_pads, start=1):
        if revised_mount.distance_to(pad) > 0.001:
            raise ValueError(f"head_roll_mount case boss {index} is disconnected")
        revised_mount = revised_mount.fuse(pad).clean()
    revised_mount = m._cut_sequential(
        revised_mount,
        [
            placement.body,
            placement.passive_horn,
            placement.active_horn,
            *case_holes,
        ],
    ).clean()
    if len(revised_mount.solids()) != 1 or not revised_mount.is_valid:
        raise ValueError("revised head_roll_mount is not one valid solid")
    revised_mount.label = MOUNT_LABEL

    for obstacle in _children(servo):
        overlap = _intersection_volume(revised_mount, obstacle)
        if overlap > MAX_INTERFERENCE_MM3:
            raise ValueError(
                f"revised mount intersects {obstacle.label}: {overlap:.3f} mm3"
            )

    children = [
        revised_mount if child.label == MOUNT_LABEL else child
        for child in _children(head)
    ]
    result = bd.Compound(children=children)
    result.label = "head_assembly_ZP10D_no_new_parts"
    return result


if __name__ == "__main__":
    head_internal_zp10d_no_new_parts()
