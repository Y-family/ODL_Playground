#!/usr/bin/env python3
"""Build ZP10D adapters for the neck pitch and yaw links."""

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
from build_zp10d_roll_to_pitch_adapters import (
    body_keepout,
    load_single_solid,
    transformed,
    zp10d_transform,
)

NECK_SERVO = np.array(
    [
        [0, 0, 1, -0.031184110571682577],
        [0, 1, 0, -0.017462393119792682],
        [-1, 0, 0, 0.05486114310196379],
        [0, 0, 0, 1],
    ]
)
NECK_SHEET = np.array(
    [
        [1, 0, 0, -0.04649573767738805],
        [0, 1, 0, -0.02997758765648688],
        [0, 0, 1, -0.021575864366217917],
        [0, 0, 0, 1],
    ]
)
YAW_SERVO = np.array(
    [
        [1, 0, 0, -0.033749323459106584],
        [0, 0, -1, -0.0319931301631853],
        [0, 1, 0, 0.0027146095558487713],
        [0, 0, 0, 1],
    ]
)
YAW_LINK = np.array(
    [
        [1, 0, 0, -0.06842231599092483],
        [0, 1, 0, -0.03668150305747986],
        [0, 0, 1, -0.13896058498084543],
        [0, 0, 0, 1],
    ]
)


def clear_part(
    part: cq.Shape, body: cq.Shape, horns: tuple[cq.Shape, cq.Shape], closure: cq.Shape
) -> cq.Shape:
    result = part.cut(body_keepout(body, FixtureDimensions().radial_clearance))
    for obstacle in (*horns, closure):
        result = result.cut(obstacle)
    result = result.clean()
    if not result.isValid() or len(result.Solids()) != 1:
        raise ValueError("adapter is not one valid solid")
    for obstacle in (body, *horns, closure):
        if result.intersect(obstacle).Volume() > 1e-6:
            raise ValueError("adapter has interference")
    return result.Solids()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("neck_left_step", type=Path)
    parser.add_argument("neck_right_step", type=Path)
    parser.add_argument("yaw_link_step", type=Path)
    parser.add_argument("zp10d_step", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    body, horn_a, horn_b = vendor_parts(args.zp10d_step)
    box = body.BoundingBox()
    origin = ((box.xmin + box.xmax) / 2, (box.ymin + box.ymax) / 2, box.zmin)
    lower = lower_clamp(FixtureDimensions()).translate(origin).val().Solids()[0]
    upper = upper_clamp(FixtureDimensions()).translate(origin).val().Solids()[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    neck_transform = zp10d_transform(NECK_SERVO)
    neck_body, neck_a, neck_b, neck_lower, neck_upper = (
        transformed(part, neck_transform)
        for part in (body, horn_a, horn_b, lower, upper)
    )
    left = transformed(load_single_solid(args.neck_left_step), NECK_SHEET).fuse(
        neck_lower
    )
    right_sheet = transformed(load_single_solid(args.neck_right_step), NECK_SHEET)
    bridge = (
        cq.Workplane("XY")
        .box(14, 20, 6, centered=False)
        .translate((-31, -48, 43))
        .val()
    )
    right = right_sheet.fuse(neck_upper).fuse(bridge)
    outputs = {
        "neck_pitch_left": clear_part(left, neck_body, (neck_a, neck_b), neck_upper),
        "neck_pitch_right": clear_part(right, neck_body, (neck_a, neck_b), neck_lower),
    }

    yaw_transform = zp10d_transform(YAW_SERVO)
    yaw_body, yaw_a, yaw_b, yaw_lower, yaw_upper = (
        transformed(part, yaw_transform)
        for part in (body, horn_a, horn_b, lower, upper)
    )
    yaw_link = transformed(load_single_solid(args.yaw_link_step), YAW_LINK).fuse(
        yaw_upper
    )
    outputs["neck_yaw_main"] = clear_part(yaw_link, yaw_body, (yaw_a, yaw_b), yaw_lower)
    outputs["neck_yaw_closure"] = yaw_lower

    for name, part in outputs.items():
        path = args.output_dir / f"ZP10D_{name}_R1.step"
        cq.exporters.export(part, str(path))
        print(f"output={path} volume_mm3={part.Volume():.3f}")


if __name__ == "__main__":
    main()
