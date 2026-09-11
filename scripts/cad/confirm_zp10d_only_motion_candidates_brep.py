#!/usr/bin/env python3
"""Confirm ZP10D-only broad-phase motion candidates with exact BRep solids."""

from __future__ import annotations

import json
from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOVING_REFS = {
    "right_hip_yaw": {13, 17, 20, 14, 18},
    "right_hip_roll": {17, 20, 14, 18},
    "right_hip_pitch": {20, 14, 18},
    "right_knee": {14, 18},
    "right_ankle": {18},
    "left_hip_yaw": {24, 26, 27, 21, 23},
    "left_hip_roll": {26, 27, 21, 23},
    "left_hip_pitch": {27, 21, 23},
    "left_knee": {21, 23},
    "left_ankle": {23},
    "neck_pitch": {9, 7, 16, 22, 2, 5, 10, 11},
    "head_pitch": {7, 16, 22, 2, 5, 10, 11},
    "head_yaw": {16, 22, 2, 5, 10, 11},
    "head_roll": {22, 2, 5, 10, 11},
}

ROBOT = PROJECT_ROOT / "cad/local_fix/open_duck_mini_14_zp10d_numbered.step"
ZP10D_REPORT = (
    PROJECT_ROOT
    / "cad/local_fix/open_duck_mini_14_zp10d_numbered.motion_scan.json"
)
ORIGINAL_REPORT = (
    PROJECT_ROOT
    / "cad/local_fix/open_duck_mini_original_14_sts3215.motion_scan.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "cad/local_fix/open_duck_mini_14_zp10d_numbered.motion_brep.json"
)
INTERFERENCE_TOLERANCE_MM3 = 0.05


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _leaves(shape: bd.Shape) -> list[bd.Shape]:
    children = _children(shape)
    if not children:
        return [shape]
    result = []
    for child in children:
        result.extend(_leaves(child))
    return result


def _boxes_overlap(left: bd.Shape, right: bd.Shape) -> bool:
    a = left.bounding_box()
    b = right.bounding_box()
    return not (
        a.max.X < b.min.X
        or b.max.X < a.min.X
        or a.max.Y < b.min.Y
        or b.max.Y < a.min.Y
        or a.max.Z < b.min.Z
        or b.max.Z < a.min.Z
    )


def _intersection_facts(
    left: bd.Shape,
    right: bd.Shape,
) -> tuple[float, dict[str, list[float]] | None]:
    result = left.intersect(right)
    if result is None:
        return 0.0, None
    pieces = (
        list(result)
        if isinstance(result, (list, bd.ShapeList))
        else list(result.solids())
    )
    pieces = [piece for piece in pieces if piece.volume > 0.0]
    if not pieces:
        return 0.0, None
    box = bd.Compound(children=pieces).bounding_box()
    return sum(piece.volume for piece in pieces), {
        "min": [box.min.X, box.min.Y, box.min.Z],
        "max": [box.max.X, box.max.Y, box.max.Z],
    }


def _ref_index(ref: str) -> int:
    prefix = "o1."
    if not ref.startswith(prefix):
        raise ValueError(f"unsupported top occurrence ref {ref!r}")
    return int(ref[len(prefix) :])


def main() -> None:
    new_joints = json.loads(ZP10D_REPORT.read_text())["joints"]
    original_joints = json.loads(ORIGINAL_REPORT.read_text())["joints"]
    kinematics = json.loads(Path(f"{ROBOT}.json").read_text())["kinematics"]
    mates = {
        mate["name"]: mate
        for mate in kinematics["mates"]
        if mate["kind"] == "revolute"
    }
    robot = read_step(ROBOT)
    top = _children(robot)

    checks = []
    for joint_name, joint in new_joints.items():
        original_pairs = {
            tuple(item["refs"])
            for item in original_joints[joint_name]["collisions"]
        }
        for candidate in joint["collisions"]:
            pair = tuple(candidate["refs"])
            if pair in original_pairs:
                continue

            indices = tuple(_ref_index(ref) for ref in pair)
            moving_flags = tuple(index in MOVING_REFS[joint_name] for index in indices)
            if sum(moving_flags) != 1:
                raise ValueError(
                    f"{joint_name} pair {pair!r} does not have one moving occurrence"
                )
            mate = mates[joint_name]
            axis = bd.Axis(mate["axis"]["origin"], mate["axis"]["dir"])

            for interval in candidate["intervals"]:
                angle = interval["max_depth_angle_deg"]
                leaf_sets = []
                for index, is_moving in zip(indices, moving_flags, strict=True):
                    leaves = _leaves(top[index - 1])
                    leaf_sets.append(
                        [leaf.rotate(axis, angle) for leaf in leaves]
                        if is_moving
                        else leaves
                    )

                volume = 0.0
                leaf_overlaps = []
                for left in leaf_sets[0]:
                    for right in leaf_sets[1]:
                        if not _boxes_overlap(left, right):
                            continue
                        overlap, bounds = _intersection_facts(left, right)
                        if overlap <= 0.0:
                            continue
                        volume += overlap
                        leaf_overlaps.append(
                            {
                                "left": left.label,
                                "right": right.label,
                                "volume_mm3": overlap,
                                "bounds_mm": bounds,
                            }
                        )

                check = {
                    "joint": joint_name,
                    "refs": list(pair),
                    "labels": candidate["labels"],
                    "angle_deg": angle,
                    "intersection_volume_mm3": volume,
                    "leaf_overlaps": leaf_overlaps,
                    "ok": volume <= INTERFERENCE_TOLERANCE_MM3,
                }
                checks.append(check)
                print(
                    f"{joint_name} {pair[0]}:{pair[1]} at {angle:+.1f} deg "
                    f"BRep={volume:.6f} mm3",
                    flush=True,
                )

    report = {
        "ok": all(check["ok"] for check in checks),
        "robot": str(ROBOT),
        "broad_phase_report": str(ZP10D_REPORT),
        "original_comparison_report": str(ORIGINAL_REPORT),
        "method": "exact BRep leaf intersections at every ZP10D-only interval maximum",
        "interference_tolerance_mm3": INTERFERENCE_TOLERANCE_MM3,
        "checks": checks,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report={OUTPUT}")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
