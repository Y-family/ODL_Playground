#!/usr/bin/env python3
"""Build ZP10D hip-roll brackets while retaining the original frame interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import cadquery as cq
from build_zp10d_joint_fixture import (
    FixtureDimensions,
    lower_clamp,
    upper_clamp,
    vendor_parts,
)

OLD_BRACKET_TRANSFORM = (
    1,
    0,
    0,
    -0.023031750164286,
    0,
    1,
    0,
    -0.047786496252609,
    0,
    0,
    1,
    0.010756611535078,
)
ZP10D_TRANSFORM = (
    1,
    0,
    0,
    -0.012423116627592,
    0,
    0,
    1,
    -0.068474869146903,
    0,
    -1,
    0,
    -0.054715563933104,
)


def matrix(values: tuple[float | int, ...]) -> cq.Matrix:
    return cq.Matrix(
        [
            [values[0], values[1], values[2], values[3] * 1000],
            [values[4], values[5], values[6], values[7] * 1000],
            [values[8], values[9], values[10], values[11] * 1000],
            [0, 0, 0, 1],
        ]
    )


def transformed(shape: cq.Shape, values: tuple[float | int, ...]) -> cq.Shape:
    return shape.transformGeometry(matrix(values))


def load_single_solid(path: Path) -> cq.Shape:
    solids = list(cq.importers.importStep(str(path)).val().Solids())
    if len(solids) != 1:
        raise ValueError(f"expected one solid in {path}, found {len(solids)}")
    return solids[0]


def box_from_bounds(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> cq.Shape:
    return (
        cq.Workplane("XY")
        .box(xmax - xmin, ymax - ymin, zmax - zmin, centered=(False, False, False))
        .translate((xmin, ymin, zmin))
        .val()
    )


def assert_printable(name: str, shape: cq.Shape) -> None:
    if not shape.isValid() or len(shape.Solids()) != 1:
        raise ValueError(f"{name} is not one valid solid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_bottom_step", type=Path)
    parser.add_argument("old_top_step", type=Path)
    parser.add_argument("zp10d_step", type=Path)
    parser.add_argument("output_step", type=Path)
    args = parser.parse_args()
    revision = args.output_step.stem.rsplit("_", 1)[-1]

    old_bottom = transformed(
        load_single_solid(args.old_bottom_step), OLD_BRACKET_TRANSFORM
    )
    old_top = transformed(load_single_solid(args.old_top_step), OLD_BRACKET_TRANSFORM)

    body, horn_a, horn_b = vendor_parts(args.zp10d_step)
    body_box = body.BoundingBox()
    fixture_origin = (
        (body_box.xmin + body_box.xmax) / 2,
        (body_box.ymin + body_box.ymax) / 2,
        body_box.zmin,
    )
    lower = transformed(
        lower_clamp(FixtureDimensions()).translate(fixture_origin).val().Solids()[0],
        ZP10D_TRANSFORM,
    )
    upper = transformed(
        upper_clamp(FixtureDimensions()).translate(fixture_origin).val().Solids()[0],
        ZP10D_TRANSFORM,
    )
    body, horn_a, horn_b = (
        transformed(part, ZP10D_TRANSFORM) for part in (body, horn_a, horn_b)
    )

    # A conservative rectangular keepout provides 0.25 mm clearance around the
    # complete vendor body, including both shaft-side protrusions.
    body_box = body.BoundingBox()
    clearance = FixtureDimensions().radial_clearance
    body_keepout = box_from_bounds(
        body_box.xmin - clearance,
        body_box.xmax + clearance,
        body_box.ymin - clearance,
        body_box.ymax + clearance,
        body_box.zmin - clearance,
        body_box.zmax + clearance,
    )
    lower_adapter = (
        old_bottom.fuse(lower).cut(body_keepout).cut(horn_a).cut(horn_b).clean()
    )

    # The old upper interface sits 2.9 mm above the new clamp. This plate joins
    # both exact solids while staying clear of the ZP10D body envelope.
    bridge = box_from_bounds(-24, -3, -20, -3, -33, -27)
    if bridge.intersect(body_keepout).Volume() > 1e-6:
        raise ValueError("upper bridge intersects the ZP10D body keepout")
    upper_adapter = old_top.fuse(upper).fuse(bridge).clean()

    for name, adapter in (
        ("lower adapter", lower_adapter),
        ("upper adapter", upper_adapter),
    ):
        assert_printable(name, adapter)
        if adapter.intersect(body).Volume() > 1e-6:
            raise ValueError(f"{name} intersects the ZP10D body")
        if adapter.intersect(horn_a).Volume() > 1e-6:
            raise ValueError(f"{name} intersects the ZP10D main horn")
        if adapter.intersect(horn_b).Volume() > 1e-6:
            raise ValueError(f"{name} intersects the ZP10D passive horn")

    assembly = cq.Assembly(name=f"ZP10D_hip_roll_adapter_{revision}")
    assembly.add(
        lower_adapter.Solids()[0],
        name=f"ZP10D_hip_roll_lower_{revision}",
        color=cq.Color(0.95, 0.55, 0.10),
    )
    assembly.add(
        upper_adapter.Solids()[0],
        name=f"ZP10D_hip_roll_upper_{revision}",
        color=cq.Color(0.95, 0.70, 0.10),
    )
    args.output_step.parent.mkdir(parents=True, exist_ok=True)
    assembly.save(str(args.output_step), exportType="STEP", mode="default")
    lower_path = args.output_step.with_name(args.output_step.stem + "_lower.step")
    upper_path = args.output_step.with_name(args.output_step.stem + "_upper.step")
    cq.exporters.export(lower_adapter.Solids()[0], str(lower_path))
    cq.exporters.export(upper_adapter.Solids()[0], str(upper_path))

    print(f"output={args.output_step}")
    print(f"lower_output={lower_path}")
    print(f"upper_output={upper_path}")
    print(f"lower_volume_mm3={lower_adapter.Volume():.3f}")
    print(f"upper_volume_mm3={upper_adapter.Volume():.3f}")
    print("body_interference_mm3=0")


if __name__ == "__main__":
    main()
