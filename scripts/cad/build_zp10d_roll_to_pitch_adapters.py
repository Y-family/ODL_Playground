#!/usr/bin/env python3
"""Build the mirrored ZP10D roll-to-pitch adapter pairs."""

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

OLD_SHAFT_POINT = np.array([-0.0255, 0.0044, 0.0, 1.0])
ZP10D_SHAFT_POINT = np.array([0.006306561, 0.010419832, 0.051, 1.0])
ZP10D_TO_OLD_SERVO = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)

LAYOUTS = {
    "left": {
        "servo": [
            1,
            0,
            0,
            -0.003148657774268959,
            0,
            1,
            0,
            0.003748140659605905,
            0,
            0,
            1,
            -0.00547809501165246,
            0,
            0,
            0,
            1,
        ],
        "link": [
            1,
            0,
            0,
            -0.0017616503060872701,
            0,
            1,
            0,
            -0.09886705387708833,
            0,
            0,
            1,
            0.06421027788264208,
            0,
            0,
            0,
            1,
        ],
        "main_half": "upper",
    },
    "right": {
        "servo": [
            1,
            0,
            0,
            0.015393682425092469,
            0,
            -1,
            0,
            0.005098838975742927,
            0,
            0,
            -1,
            -0.01752170425228127,
            0,
            0,
            0,
            1,
        ],
        "link": [
            1,
            0,
            0,
            0.016780689893274164,
            0,
            1,
            0,
            0.10771403351243715,
            0,
            0,
            1,
            0.04278992285342419,
            0,
            0,
            0,
            1,
        ],
        "main_half": "lower",
    },
}


def matrix(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(4, 4)


def transformed(shape: cq.Shape, transform: np.ndarray) -> cq.Shape:
    transform = transform.copy()
    transform[:3, 3] *= 1000
    return shape.transformGeometry(cq.Matrix(transform.tolist()))


def load_single_solid(path: Path) -> cq.Shape:
    solids = list(cq.importers.importStep(str(path)).val().Solids())
    if len(solids) != 1:
        raise ValueError(f"expected one solid in {path}, found {len(solids)}")
    return solids[0]


def zp10d_transform(old_servo: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = old_servo[:3, :3] @ ZP10D_TO_OLD_SERVO
    old_axis = (old_servo @ OLD_SHAFT_POINT)[:3]
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
            centered=(False, False, False),
        )
        .translate((box.xmin - clearance, box.ymin - clearance, box.zmin - clearance))
        .val()
    )


def build_side(
    side: str,
    link_path: Path,
    body: cq.Shape,
    horn_a: cq.Shape,
    horn_b: cq.Shape,
    lower: cq.Shape,
    upper: cq.Shape,
) -> tuple[cq.Shape, cq.Shape]:
    layout = LAYOUTS[side]
    servo_transform = zp10d_transform(matrix(layout["servo"]))
    link = transformed(load_single_solid(link_path), matrix(layout["link"]))
    parts = {
        name: transformed(shape, servo_transform)
        for name, shape in (
            ("body", body),
            ("horn_a", horn_a),
            ("horn_b", horn_b),
            ("lower", lower),
            ("upper", upper),
        )
    }
    main_name = str(layout["main_half"])
    closure_name = "lower" if main_name == "upper" else "upper"
    closure = parts[closure_name]
    main = (
        link.fuse(parts[main_name])
        .cut(body_keepout(parts["body"], FixtureDimensions().radial_clearance))
        .cut(parts["horn_a"])
        .cut(parts["horn_b"])
        .cut(closure)
        .clean()
    )
    for name, shape in (("main", main), ("closure", closure)):
        if not shape.isValid() or len(shape.Solids()) != 1:
            raise ValueError(f"{side} {name} adapter is not one valid solid")
    for name in ("body", "horn_a", "horn_b"):
        if main.intersect(parts[name]).Volume() > 1e-6:
            raise ValueError(f"{side} main adapter intersects {name}")
    if main.intersect(closure).Volume() > 1e-6:
        raise ValueError(f"{side} adapter halves intersect")
    return main.Solids()[0], closure.Solids()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left_link_step", type=Path)
    parser.add_argument("right_link_step", type=Path)
    parser.add_argument("zp10d_step", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    body, horn_a, horn_b = vendor_parts(args.zp10d_step)
    box = body.BoundingBox()
    origin = ((box.xmin + box.xmax) / 2, (box.ymin + box.ymax) / 2, box.zmin)
    lower = lower_clamp(FixtureDimensions()).translate(origin).val().Solids()[0]
    upper = upper_clamp(FixtureDimensions()).translate(origin).val().Solids()[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for side, link_path in (
        ("left", args.left_link_step),
        ("right", args.right_link_step),
    ):
        main_part, closure = build_side(
            side, link_path, body, horn_a, horn_b, lower, upper
        )
        for kind, shape in (("main", main_part), ("closure", closure)):
            path = args.output_dir / f"ZP10D_{side}_roll_to_pitch_{kind}_R1.step"
            cq.exporters.export(shape, str(path))
            print(f"output={path} volume_mm3={shape.Volume():.3f}")


if __name__ == "__main__":
    main()
