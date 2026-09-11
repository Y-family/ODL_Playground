#!/usr/bin/env python3
"""Compare original and ZP10D right hip-pitch collision envelopes.

This is a read-only diagnostic.  It reconstructs the zero-angle geometry used
by the complete-leg builder, rotates the original upper-leg parts through the
MJCF range, and reports triangle-surface contacts with the fixed hip parts.
No STEP file or Onshape document is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

from cadgen import read_step
import vtk

import build_zp10d_left_leg_complete as m


@dataclass(frozen=True)
class SurfaceContact:
    variant: str
    angle_deg: float
    moving: str
    fixed: str
    inside_samples: int
    bounds: tuple[float, float, float, float, float, float]


def _mesh(shape: m.bd.Shape):
    return shape.tessellate(0.15, 0.10)


def _polydata(
    mesh,
    placement: m.ServoPlacement | None = None,
    angle_deg: float = 0.0,
) -> vtk.vtkPolyData:
    vertices, triangles = mesh
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(vertices))
    if placement is None or abs(angle_deg) < 0.001:
        for index, point in enumerate(vertices):
            points.SetPoint(index, point.X, point.Y, point.Z)
    else:
        theta = radians(angle_deg)
        cosine = cos(theta)
        sine = sin(theta)
        axis = placement.axis
        center = placement.center
        for index, point in enumerate(vertices):
            offset = point - center
            rotated = (
                offset * cosine
                + axis.cross(offset) * sine
                + axis * offset.dot(axis) * (1.0 - cosine)
            )
            transformed = center + rotated
            points.SetPoint(index, transformed.X, transformed.Y, transformed.Z)

    cells = vtk.vtkCellArray()
    for triangle in triangles:
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, triangle[0])
        cell.GetPointIds().SetId(1, triangle[1])
        cell.GetPointIds().SetId(2, triangle[2])
        cells.InsertNextCell(cell)
    result = vtk.vtkPolyData()
    result.SetPoints(points)
    result.SetPolys(cells)
    return result


def _surface_contact(
    variant: str,
    angle_deg: float,
    moving: m.bd.Shape,
    moving_mesh,
    fixed: m.bd.Shape,
    fixed_polydata: vtk.vtkPolyData,
) -> SurfaceContact | None:
    moved_polydata = _polydata(
        moving_mesh, _PLACEMENTS[variant], angle_deg
    )

    inside_points: list[tuple[float, float, float]] = []
    for samples, surface in (
        (moved_polydata, fixed_polydata),
        (fixed_polydata, moved_polydata),
    ):
        enclosed = vtk.vtkSelectEnclosedPoints()
        enclosed.SetInputData(samples)
        enclosed.SetSurfaceData(surface)
        enclosed.SetTolerance(1e-6)
        enclosed.SetCheckSurface(False)
        enclosed.Update()
        flags = enclosed.GetOutput().GetPointData().GetArray("SelectedPoints")
        points = samples.GetPoints()
        inside_points.extend(
            points.GetPoint(index)
            for index in range(points.GetNumberOfPoints())
            if flags.GetTuple1(index) > 0.5
        )

    if not inside_points:
        return None
    xs, ys, zs = zip(*inside_points, strict=True)
    bounds = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
    return SurfaceContact(
        variant=variant,
        angle_deg=angle_deg,
        moving=moving.label,
        fixed=fixed.label,
        inside_samples=len(inside_points),
        bounds=tuple(float(value) for value in bounds),
    )


def _old_case_parts(old_servo: m.bd.Shape) -> list[m.bd.Shape]:
    return [
        child
        for child in m._children(old_servo)
        if child.label.startswith("WJ-WK00-012")
    ]


def _original_geometry(robot: m.bd.Shape):
    roll_source = m._one_child(robot, "right_roll_to_pitch_assembly <1>")
    old_servo = m._one_prefix_child(roll_source, "STS32155")
    placement = m._servo_placement(old_servo, "hip_pitch")
    roll_link = m._one_child(roll_source, "right_roll_to_pitch")

    upper_source = m._one_child(robot, "knee_and_ankle_assembly <1>")
    moving = [
        child
        for child in m._children(upper_source)
        if child.label.startswith("left_knee_to_ankle_")
    ]
    fixed = [roll_link, *_old_case_parts(old_servo)]
    return placement, moving, fixed


def _zp10d_geometry(robot: m.bd.Shape):
    roll_source = m._one_child(robot, "right_roll_to_pitch_assembly <1>")
    old_pitch_servo = m._one_prefix_child(roll_source, "STS32155")
    pitch_servo = m._servo_placement(old_pitch_servo, "hip_pitch")

    interface = read_step(
        m.PROJECT_ROOT
        / "cad/hip_pitch_interface/zp10d_right_hip_pitch_interface.step"
    )
    roll_link = m._one_child(interface, "right_roll_to_pitch")
    servo = m._one_child(interface, "ZP10D S08 hip_pitch <1>")
    body = m._one_child(servo, "ZP10D_vendor_body")
    moving = [
        child
        for child in m._children(interface)
        if child.label.startswith("left_knee_to_ankle_")
    ]
    fixed = [body, roll_link]
    return pitch_servo, moving, fixed


def _print_header(
    variant: str,
    placement: m.ServoPlacement,
    moving: list[m.bd.Shape],
    fixed: list[m.bd.Shape],
) -> None:
    print(
        f"VARIANT {variant} axis="
        f"({placement.axis.X:.6f},{placement.axis.Y:.6f},{placement.axis.Z:.6f}) "
        f"center=({placement.center.X:.6f},{placement.center.Y:.6f},"
        f"{placement.center.Z:.6f})"
    )
    for role, parts in (("MOVING", moving), ("FIXED", fixed)):
        for part in parts:
            box = part.bounding_box()
            print(
                f"  {role} {part.label}: "
                f"[{box.min.X:.3f},{box.min.Y:.3f},{box.min.Z:.3f}].."
                f"[{box.max.X:.3f},{box.max.Y:.3f},{box.max.Z:.3f}]",
                flush=True,
            )


def main() -> None:
    robot = read_step(m.SOURCE_ROBOT)
    variants = {
        "original": _original_geometry(robot),
        "zp10d": _zp10d_geometry(robot),
    }
    _PLACEMENTS.update(
        {variant: geometry[0] for variant, geometry in variants.items()}
    )

    contacts: list[SurfaceContact] = []
    for variant, (placement, moving, fixed) in variants.items():
        _print_header(variant, placement, moving, fixed)
        moving_meshes = {id(part): _mesh(part) for part in moving}
        fixed_polydata = {
            id(part): _polydata(_mesh(part)) for part in fixed
        }
        for angle in m._rotation_angles(*m.PITCH_RANGE_DEG):
            print(f"ANGLE {variant} {angle:.1f}", flush=True)
            for moving_part in moving:
                for fixed_part in fixed:
                    item = _surface_contact(
                        variant,
                        angle,
                        moving_part,
                        moving_meshes[id(moving_part)],
                        fixed_part,
                        fixed_polydata[id(fixed_part)],
                    )
                    if item is not None:
                        contacts.append(item)
                        bounds = ",".join(f"{value:.3f}" for value in item.bounds)
                        print(
                            f"CONTACT {variant} angle={angle:.1f} "
                            f"moving={item.moving!r} fixed={item.fixed!r} "
                            f"inside_samples={item.inside_samples} bounds=[{bounds}]",
                            flush=True,
                        )

    for variant in variants:
        selected = [item for item in contacts if item.variant == variant]
        angles = sorted({item.angle_deg for item in selected})
        print(
            f"SUMMARY {variant} contacts={len(selected)} "
            f"angles={','.join(f'{angle:.1f}' for angle in angles) or 'none'}"
        )


_PLACEMENTS: dict[str, m.ServoPlacement] = {}


if __name__ == "__main__":
    main()
