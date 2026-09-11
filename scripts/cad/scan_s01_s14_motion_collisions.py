#!/usr/bin/env python3
"""Scan every numbered ZP10D joint through its commanded range.

This is the fast broad-phase gate. It uses exact tessellations of the final
STEP occurrences with FCL at one-degree increments. Any reported interval
must be confirmed with BRep intersection before geometry is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from cadgen import read_step


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT = PROJECT_ROOT / "cad/local_fix/open_duck_mini_14_zp10d_numbered.step"
DEFAULT_REPORT = PROJECT_ROOT / "cad/local_fix/open_duck_mini_14_zp10d_numbered.motion_scan.json"
FRAME_REFS = {3, 4, 8, 12, 19}
PENETRATION_TOLERANCE_MM = 0.05


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


def _children(shape):
    return list(getattr(shape, "children", None) or [])


def _leaves(shape):
    children = _children(shape)
    if not children:
        return [shape]
    result = []
    for child in children:
        result.extend(_leaves(child))
    return result


def _mesh(shape, tolerance: float, angular_tolerance: float) -> trimesh.Trimesh:
    meshes = []
    for leaf in _leaves(shape):
        vertices, faces = leaf.tessellate(tolerance, angular_tolerance)
        meshes.append(
            trimesh.Trimesh(
                vertices=np.asarray([[v.X, v.Y, v.Z] for v in vertices]),
                faces=np.asarray(faces),
                process=False,
            )
        )
    return trimesh.util.concatenate(meshes)


def _angles(lower: float, upper: float, step: float) -> list[float]:
    result = list(np.arange(lower, upper + 0.5 * step, step))
    result.extend((lower, 0.0, upper))
    return sorted({round(max(lower, min(upper, value)), 9) for value in result})


def _pair_depths(data) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for contact in data:
        pair = tuple(sorted(int(name) for name in contact.names))
        result[pair] = max(result.get(pair, 0.0), max(0.0, float(contact.depth)))
    return result


def _intervals(samples: list[tuple[float, float]], step: float) -> list[dict]:
    if not samples:
        return []
    intervals = []
    start = samples[0][0]
    previous = samples[0][0]
    maximum = samples[0]
    for angle, depth in samples[1:]:
        if angle - previous > step + 1e-6:
            intervals.append(
                {
                    "start_deg": start,
                    "end_deg": previous,
                    "max_depth_mm": maximum[1],
                    "max_depth_angle_deg": maximum[0],
                }
            )
            start = angle
            maximum = (angle, depth)
        elif depth > maximum[1]:
            maximum = (angle, depth)
        previous = angle
    intervals.append(
        {
            "start_deg": start,
            "end_deg": previous,
            "max_depth_mm": maximum[1],
            "max_depth_angle_deg": maximum[0],
        }
    )
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--kinematics", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--step-deg", type=float, default=1.0)
    parser.add_argument("--mesh-tolerance", type=float, default=0.30)
    parser.add_argument("--mesh-angular-tolerance", type=float, default=0.15)
    parser.add_argument(
        "--joints",
        help="comma-separated joint names; defaults to every modeled joint",
    )
    args = parser.parse_args()
    if args.step_deg <= 0:
        raise ValueError("--step-deg must be positive")

    sidecar = args.kinematics or Path(f"{args.robot}.json")
    kinematics = json.loads(sidecar.read_text())["kinematics"]
    revolute_mates = {
        mate["name"]: mate
        for mate in kinematics["mates"]
        if mate["kind"] == "revolute"
    }
    if set(revolute_mates) != set(MOVING_REFS):
        raise ValueError("motion-scan joints do not match the STEP kinematics")
    selected_joints = (
        args.joints.split(",") if args.joints else list(MOVING_REFS)
    )
    unknown_joints = set(selected_joints) - set(MOVING_REFS)
    if unknown_joints:
        raise ValueError(f"unknown joints: {sorted(unknown_joints)!r}")

    robot = read_step(args.robot)
    top = _children(robot)
    physical_refs = [
        index for index in range(1, len(top) + 1) if index not in FRAME_REFS
    ]
    meshes = {
        index: _mesh(
            top[index - 1],
            args.mesh_tolerance,
            args.mesh_angular_tolerance,
        )
        for index in physical_refs
    }

    joints = {}
    for name in selected_joints:
        mate = revolute_mates[name]
        moving = MOVING_REFS[name]
        fixed = set(physical_refs) - moving
        moving_manager = trimesh.collision.CollisionManager()
        fixed_manager = trimesh.collision.CollisionManager()
        for index in moving:
            moving_manager.add_object(str(index), meshes[index])
        for index in fixed:
            fixed_manager.add_object(str(index), meshes[index])
        axis = mate["axis"]
        lower, upper = mate["limits"]["value"]
        angles = _angles(lower, upper, args.step_deg)
        pair_samples: dict[tuple[int, int], list[tuple[float, float]]] = {}

        _, _, neutral_contacts = moving_manager.in_collision_other(
            fixed_manager,
            return_names=True,
            return_data=True,
        )
        neutral_depths = _pair_depths(neutral_contacts)

        for angle in angles:
            transform = trimesh.transformations.rotation_matrix(
                np.radians(angle),
                axis["dir"],
                axis["origin"],
            )
            for index in moving:
                moving_manager.set_transform(str(index), transform)

            _, _, contacts = moving_manager.in_collision_other(
                fixed_manager,
                return_names=True,
                return_data=True,
            )
            for pair, depth in _pair_depths(contacts).items():
                increase = depth - neutral_depths.get(pair, 0.0)
                if increase <= PENETRATION_TOLERANCE_MM:
                    continue
                pair_samples.setdefault(pair, []).append((angle, increase))

        collisions = []
        for pair, samples in sorted(pair_samples.items()):
            collisions.append(
                {
                    "refs": [f"o1.{pair[0]}", f"o1.{pair[1]}"],
                    "labels": [top[pair[0] - 1].label, top[pair[1] - 1].label],
                    "intervals": _intervals(samples, args.step_deg),
                }
            )
        joints[name] = {
            "range_deg": [lower, upper],
            "sample_count": len(angles),
            "neutral_collision_pair_count": len(neutral_depths),
            "collisions": collisions,
        }
        print(
            f"{name}: samples={len(angles)} collision_pairs={len(collisions)}",
            flush=True,
        )

    report = {
        "ok": not any(item["collisions"] for item in joints.values()),
        "robot": str(args.robot),
        "method": "FCL triangle collision broad-phase; BRep confirmation required",
        "step_deg": args.step_deg,
        "penetration_tolerance_mm": PENETRATION_TOLERANCE_MM,
        "joints": joints,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report={args.out}")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
