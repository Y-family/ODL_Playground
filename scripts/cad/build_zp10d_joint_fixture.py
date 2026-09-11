#!/usr/bin/env python3
"""Build a STEP assembly for validating the ZP10D body and horn interfaces.

The fixture deliberately clamps the servo case instead of using the case-cover
screws. Dimensions are millimetres. CadQuery is the only non-stdlib dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cadquery as cq


@dataclass(frozen=True)
class FixtureDimensions:
    body_x: float = 40.0
    body_y: float = 40.0
    body_z: float = 20.0
    radial_clearance: float = 0.25
    clamp_x: float = 36.0
    wall: float = 3.2
    side_wall: float = 7.0
    base_flange_y: float = 56.0
    clamp_bolt_x: float = 13.0
    clamp_bolt_y: float = 23.75
    m3_clearance: float = 3.2
    m3_insert_diameter: float = 4.6
    m3_insert_depth: float = 5.5
    mount_hole_x: float = 13.0
    mount_hole_y: float = 26.0
    horn_hole_radius: float = 7.0
    horn_hole_diameter: float = 3.2
    horn_boss_clearance: float = 6.4


def cylinder_z(
    diameter: float, z0: float, height: float, x: float, y: float
) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .moveTo(x, y)
        .circle(diameter / 2)
        .extrude(height)
        .translate((0, 0, z0))
    )


def lower_clamp(d: FixtureDimensions) -> cq.Workplane:
    inner_half_y = d.body_y / 2 + d.radial_clearance
    outer_half_y = inner_half_y + d.side_wall

    base = (
        cq.Workplane("XY")
        .box(d.clamp_x, d.base_flange_y, d.wall, centered=(True, True, False))
        .translate((0, 0, -d.wall))
    )
    rail = cq.Workplane("XY").box(
        d.clamp_x, d.side_wall, d.body_z / 2, centered=(True, True, False)
    )
    part = base
    for y in (
        -outer_half_y + d.side_wall / 2,
        outer_half_y - d.side_wall / 2,
    ):
        part = part.union(rail.translate((0, y, 0)))

    # Blind pockets for M3 heat-set inserts, entered from the split plane.
    insert_z0 = d.body_z / 2 - d.m3_insert_depth
    for x in (-d.clamp_bolt_x, d.clamp_bolt_x):
        for y in (-d.clamp_bolt_y, d.clamp_bolt_y):
            part = part.cut(
                cylinder_z(
                    d.m3_insert_diameter, insert_z0, d.m3_insert_depth + 0.2, x, y
                )
            )

    # Independent frame holes in the base flange.
    for x in (-d.mount_hole_x, d.mount_hole_x):
        for y in (-d.mount_hole_y, d.mount_hole_y):
            part = part.cut(
                cylinder_z(d.m3_clearance, -d.wall - 0.1, d.wall + 0.2, x, y)
            )
    return part.clean()


def upper_clamp(d: FixtureDimensions) -> cq.Workplane:
    outer_y = d.body_y + 2 * (d.radial_clearance + d.side_wall)
    inner_half_y = d.body_y / 2 + d.radial_clearance
    outer_half_y = outer_y / 2
    split_z = d.body_z / 2
    top_z = d.body_z + d.radial_clearance
    profile = [
        (-outer_half_y, split_z),
        (-inner_half_y, split_z),
        (-inner_half_y, top_z),
        (inner_half_y, top_z),
        (inner_half_y, split_z),
        (outer_half_y, split_z),
        (outer_half_y, top_z + d.wall),
        (-outer_half_y, top_z + d.wall),
    ]
    part = cq.Workplane("YZ").polyline(profile).close().extrude(d.clamp_x / 2, both=True)

    for x in (-d.clamp_bolt_x, d.clamp_bolt_x):
        for y in (-d.clamp_bolt_y, d.clamp_bolt_y):
            part = part.cut(
                cylinder_z(
                    d.m3_clearance,
                    split_z - 0.1,
                    top_z + d.wall - split_z + 0.2,
                    x,
                    y,
                )
            )
    return part.clean()


def horn_fit_coupon(d: FixtureDimensions) -> cq.Workplane:
    """Make a separate coupon for the supplied R7 four-hole metal horn."""
    coupon = cq.Workplane("YZ").circle(12).extrude(3.0, both=True)
    coupon = coupon.cut(
        cq.Workplane("YZ").circle(d.horn_boss_clearance / 2).extrude(4, both=True)
    )
    for y, z in (
        (-d.horn_hole_radius, 0),
        (d.horn_hole_radius, 0),
        (0, -d.horn_hole_radius),
        (0, d.horn_hole_radius),
    ):
        hole = (
            cq.Workplane("YZ")
            .moveTo(y, z)
            .circle(d.horn_hole_diameter / 2)
            .extrude(4, both=True)
        )
        coupon = coupon.cut(hole)
    return coupon.translate((0, -38, 12)).clean()


def vendor_parts(step_path: Path) -> list[cq.Shape]:
    imported = cq.importers.importStep(str(step_path))
    solids = list(imported.val().Solids())
    if len(solids) != 3:
        raise ValueError(f"expected 3 solids in ZP10D STEP, found {len(solids)}")
    return sorted(solids, key=lambda shape: shape.Volume(), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_step", type=Path)
    parser.add_argument("output_step", type=Path)
    args = parser.parse_args()
    revision = args.output_step.stem.rsplit("_", 1)[-1]

    d = FixtureDimensions()
    body, horn_large, horn_small = vendor_parts(args.source_step)
    body_box = body.BoundingBox()
    fixture_origin = (
        (body_box.xmin + body_box.xmax) / 2,
        (body_box.ymin + body_box.ymax) / 2,
        body_box.zmin,
    )
    lower = lower_clamp(d).translate(fixture_origin)
    upper = upper_clamp(d).translate(fixture_origin)
    coupon = horn_fit_coupon(d)
    for name, part in (("lower", lower), ("upper", upper), ("coupon", coupon)):
        if not part.val().isValid() or len(part.val().Solids()) != 1:
            raise ValueError(f"{name} fixture part is not one valid solid")
    for name, part in (("lower", lower), ("upper", upper)):
        overlap = body.intersect(part.val().Solids()[0]).Volume()
        if overlap > 1e-6:
            raise ValueError(f"{name} clamp intersects servo body by {overlap} mm3")

    assembly = cq.Assembly(name=f"ZP10D_joint_fixture_{revision}")
    assembly.add(
        body, name="ZP10D_vendor_body_REFERENCE", color=cq.Color(0.35, 0.35, 0.38, 0.65)
    )
    assembly.add(
        horn_large,
        name="ZP10D_vendor_horn_A_REFERENCE",
        color=cq.Color(0.65, 0.65, 0.68, 0.65),
    )
    assembly.add(
        horn_small,
        name="ZP10D_vendor_horn_B_REFERENCE",
        color=cq.Color(0.65, 0.65, 0.68, 0.65),
    )
    assembly.add(
        lower.val().Solids()[0],
        name=f"ZP10D_case_clamp_lower_{revision}",
        color=cq.Color(0.95, 0.55, 0.10),
    )
    assembly.add(
        upper.val().Solids()[0],
        name=f"ZP10D_case_clamp_upper_{revision}",
        color=cq.Color(0.95, 0.70, 0.10),
    )
    assembly.add(
        coupon.val().Solids()[0],
        name=f"ZP10D_R7_horn_fit_coupon_{revision}",
        color=cq.Color(0.20, 0.55, 0.90),
    )

    args.output_step.parent.mkdir(parents=True, exist_ok=True)
    assembly.save(str(args.output_step), exportType="STEP", mode="default")

    pla_density_g_per_mm3 = 1.24e-3
    print(f"output={args.output_step}")
    print(f"lower_volume_mm3={lower.val().Volume():.3f}")
    print(f"upper_volume_mm3={upper.val().Volume():.3f}")
    print(f"coupon_volume_mm3={coupon.val().Volume():.3f}")
    print(
        "solid_pla_mass_g="
        f"{(lower.val().Volume() + upper.val().Volume() + coupon.val().Volume()) * pla_density_g_per_mm3:.3f}"
    )
    print(f"clamp_inner_y_mm={d.body_y + 2 * d.radial_clearance:.3f}")
    print(f"horn_pattern=4xD{d.horn_hole_diameter:.1f}_on_R{d.horn_hole_radius:.1f}")
    print("fixture_origin_mm=" + ",".join(f"{value:.3f}" for value in fixture_origin))


if __name__ == "__main__":
    main()
