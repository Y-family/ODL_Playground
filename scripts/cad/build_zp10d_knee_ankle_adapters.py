#!/usr/bin/env python3
"""Build the reusable ZP10D knee/ankle adapter pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import cadquery as cq
import numpy as np
from build_zp10d_joint_fixture import (
    FixtureDimensions,
    lower_clamp,
    upper_clamp,
    vendor_parts,
)

OLD_SERVO = np.array(
    [
        [0, 0, -1, -0.0686744842091],
        [0, 1, 0, -0.0146771213994],
        [1, 0, 0, -0.00156509741007],
        [0, 0, 0, 1],
    ]
)
SHEET = np.array(
    [
        [1, 0, 0, -0.0573028571033],
        [0, 1, 0, -0.117192315936],
        [0, 0, 1, 0.206061910058],
        [0, 0, 0, 1],
    ]
)
OLD_SHAFT_POINT = np.array([-0.0255, 0.0044, 0.0, 1.0])
ZP10D_SHAFT_POINT = np.array([0.006306561, 0.010419832, 0.051, 1.0])
ZP10D_TO_OLD_SERVO = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def transformed(shape: cq.Shape, transform: np.ndarray) -> cq.Shape:
    transform = transform.copy()
    transform[:3, 3] *= 1000
    return shape.transformGeometry(cq.Matrix(transform.tolist()))


def load_single_solid(path: Path) -> cq.Shape:
    solids = list(cq.importers.importStep(str(path)).val().Solids())
    if len(solids) != 1:
        raise ValueError(f"expected one solid in {path}, found {len(solids)}")
    return solids[0]


def servo_transform() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = OLD_SERVO[:3, :3] @ ZP10D_TO_OLD_SERVO
    old_axis = (OLD_SERVO @ OLD_SHAFT_POINT)[:3]
    transform[:3, 3] = old_axis - transform[:3, :3] @ ZP10D_SHAFT_POINT[:3]
    return transform


def body_keepout(body: cq.Shape, clearance: float) -> cq.Shape:
    box = body.BoundingBox()
    return (
        cq.Workplane("XY")
        .box(
            box.xlen + 2 * clearance,
            box.ylen + 2 * clearance,
            box.zlen + 2 * clearance,
            centered=False,
        )
        .translate((box.xmin - clearance, box.ymin - clearance, box.zmin - clearance))
        .val()
    )


def assert_clear(name: str, part: cq.Shape, obstacles: dict[str, cq.Shape]) -> None:
    if not part.isValid() or len(part.Solids()) != 1:
        raise ValueError(f"{name} is not one valid solid")
    for obstacle_name, obstacle in obstacles.items():
        if part.intersect(obstacle).Volume() > 1e-6:
            raise ValueError(f"{name} intersects {obstacle_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left_sheet_step", type=Path)
    parser.add_argument("right_sheet_step", type=Path)
    parser.add_argument("zp10d_step", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    body, horn_a, horn_b = vendor_parts(args.zp10d_step)
    box = body.BoundingBox()
    origin = ((box.xmin + box.xmax) / 2, (box.ymin + box.ymax) / 2, box.zmin)
    dimensions = FixtureDimensions()
    lower = lower_clamp(dimensions).translate(origin).val().Solids()[0]
    upper = upper_clamp(dimensions).translate(origin).val().Solids()[0]
    transform = servo_transform()
    body, horn_a, horn_b, lower, upper = (
        transformed(part, transform) for part in (body, horn_a, horn_b, lower, upper)
    )
    keepout = body_keepout(body, dimensions.radial_clearance)

    left_sheet = transformed(load_single_solid(args.left_sheet_step), SHEET)
    left = (
        left_sheet.fuse(lower).cut(keepout).cut(horn_a).cut(horn_b).cut(upper).clean()
    )

    right_sheet = transformed(load_single_solid(args.right_sheet_step), SHEET)
    bridge = (
        cq.Workplane("XY").box(16, 23, 7, centered=False).translate((-82, -46, 3)).val()
    )
    right = (
        right_sheet.fuse(upper)
        .fuse(bridge)
        .cut(keepout)
        .cut(horn_a)
        .cut(horn_b)
        .cut(lower)
        .clean()
    )

    assert_clear(
        "left adapter",
        left,
        {
            "servo body": body,
            "main horn": horn_a,
            "passive horn": horn_b,
            "closure": upper,
        },
    )
    assert_clear(
        "right adapter",
        right,
        {
            "servo body": body,
            "main horn": horn_a,
            "passive horn": horn_b,
            "closure": lower,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for side, part in (("left", left), ("right", right)):
        path = args.output_dir / f"ZP10D_knee_ankle_{side}_R1.step"
        cq.exporters.export(part.Solids()[0], str(path))
        bounds = part.BoundingBox()
        print(
            f"output={path} volume_mm3={part.Volume():.3f} "
            f"bounds_mm={bounds.xlen:.3f}x{bounds.ylen:.3f}x{bounds.zlen:.3f}"
        )


if __name__ == "__main__":
    main()
