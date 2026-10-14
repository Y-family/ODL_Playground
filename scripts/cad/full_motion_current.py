#!/usr/bin/env python3
"""Broad-phase collision scan over the limits declared by the MJCF."""

from __future__ import annotations

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from cadgen import read_step

from validate_s01_s10_interfaces_local import _children, _axis
from scan_s01_s14_motion_collisions import _angles

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(os.environ.get("OPEN_DUCK_MOTION_TARGET", ROOT / "cad/local_fix/open_duck_mini_upper_refit.step"))
REPORT = Path(os.environ.get("OPEN_DUCK_MOTION_REPORT", ROOT / "cad/local_fix/FULL_MOTION_CURRENT_REPORT.json"))
STEP_DEG = 5
PENETRATION_TOLERANCE_MM = 0.05

MJCF = ROOT / 'playground/open_duck_mini_zp10d/xmls/open_duck_mini_zp10d.xml'
JOINT_NAMES = dict(zip([f'S{index:02d}' for index in range(1, 15)],
    ['right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
     'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
     'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll']))


def joint_specs():
    root = ET.parse(MJCF).getroot()
    assert root.find('compiler').get('angle') == 'radian'
    result = {}
    def walk(body, rotation):
        local = trimesh.transformations.quaternion_matrix(
            [float(value) for value in body.get('quat', '1 0 0 0').split()])[:3, :3]
        world = rotation @ local
        for joint in body.findall('joint'):
            name = joint.get('name')
            if name not in JOINT_NAMES.values():
                continue
            direction = world @ np.array([float(value) for value in joint.get('axis', '0 0 1').split()])
            result[name] = dict(limits_deg=[round(math.degrees(float(value)), 8) for value in joint.get('range').split()],
                                direction=direction.tolist())
        for child in body.findall('body'):
            walk(child, world)
    for body in root.find('worldbody').findall('body'):
        walk(body, np.eye(3))
    assert set(result) == set(JOINT_NAMES.values())
    return {servo: result[name] for servo, name in JOINT_NAMES.items()}


SPECS = joint_specs()
LIMITS = {servo: tuple(round(value) for value in spec['limits_deg']) for servo, spec in SPECS.items()}


def measured_axes(robot):
    result = {}
    def walk(node):
        servo = node.label[:3]
        if servo in SPECS and node.label.startswith(servo + '_'):
            direction, origin = _axis(node)
            expected = np.array(SPECS[servo]['direction'])
            assert abs(np.dot(tuple(direction), expected)) > .99999, node.label
            result[servo] = (list(origin), expected.tolist())
        for child in _children(node):
            walk(child)
    walk(robot)
    assert set(result) == set(SPECS)
    return result


def is_moving(item, servo):
    if item['servo'] == servo:
        return ('horn' not in item['label']) if servo in ('S13', 'S14') else ('horn' in item['label'])
    return item['root'] in MOVING_ROOTS[servo]

AXES = {
    "S01": ((-19, -30.25, 64.365938812334), (0, 0, -1)),
    "S02": ((-18.375, -30.25, -0.074061187627), (1, 0, 0)),
    "S03": ((-35.06, -93.925, -0.074061187627), (0, -1, 0)),
    "S04": ((-35.06, -93.8, -78.7240611876), (0, 1, 0)),
    "S05": ((-35.06, -93.9, -157.3740611876), (0, 1, 0)),
    "S06": ((-19, 30.25, 64.365938812334), (0, 0, -1)),
    "S07": ((-18.375, 30.25, -0.074061187627), (1, 0, 0)),
    "S08": ((-35.06, 93.925, -0.074061187627), (0, 1, 0)),
    "S09": ((-35.06, 93.8, -78.724061187595), (0, -1, 0)),
    "S10": ((-35.06, 93.9, -157.37406118755), (0, -1, 0)),
    "S11": ((1, 0.725, 90.000938812376), (0, 1, 0)),
    "S12": ((1, 0.725, 156.00093881237), (0, 1, 0)),
    "S13": ((9, 0.1, 197.57593881241), (0, 0, 1)),
    "S14": ((75.95, 0.1, 196.85093881236), (-1, 0, 0)),
}

# Root occurrences are flattened in the STEP but nested in the MJCF. These
# sets describe the rigid subtree that travels for each revolute joint.
RIGHT = {"right_cache"}
LEFT = {"left_cache"}
HEAD = {"right_antenna_holder", "left_antenna_holder", "antenna"}
MOVING_ROOTS = {
    "S01": {"hip_roll_assembly <1>", "right_roll_to_pitch_assembly <1>", "knee_and_ankle_assembly <1>", "knee_and_ankle_assembly <3>", "foot_assembly <1>"} | RIGHT,
    "S02": {"right_roll_to_pitch_assembly <1>", "knee_and_ankle_assembly <1>", "knee_and_ankle_assembly <3>", "foot_assembly <1>"} | RIGHT,
    "S03": {"knee_and_ankle_assembly <1>", "knee_and_ankle_assembly <3>", "foot_assembly <1>"} | RIGHT,
    "S04": {"knee_and_ankle_assembly <3>", "foot_assembly <1>"},
    "S05": {"foot_assembly <1>"},
    "S06": {"hip_roll_assembly <2>", "left_roll_to_pitch_assembly <1>", "knee_and_ankle_assembly <2>", "knee_and_ankle_assembly <4>", "foot_assembly <2>"} | LEFT,
    "S07": {"left_roll_to_pitch_assembly <1>", "knee_and_ankle_assembly <2>", "knee_and_ankle_assembly <4>", "foot_assembly <2>"} | LEFT,
    "S08": {"knee_and_ankle_assembly <2>", "knee_and_ankle_assembly <4>", "foot_assembly <2>"} | LEFT,
    "S09": {"knee_and_ankle_assembly <4>", "foot_assembly <2>"},
    "S10": {"foot_assembly <2>"},
    "S11": {"neck_pitch_assembly <1>", "head_pitch_to_yaw", "neck_yaw_assembly <1>", "head_assembly <1>"} | HEAD,
    "S12": {"head_pitch_to_yaw", "neck_yaw_assembly <1>", "head_assembly <1>"} | HEAD,
    "S13": {"neck_yaw_assembly <1>", "head_assembly <1>"} | HEAD,
    "S14": {"head_assembly <1>"} | HEAD,
}


def _mesh(shape):
    vertices, faces = shape.tessellate(0.5, 0.3)
    return trimesh.Trimesh(
        vertices=np.asarray([[point.X, point.Y, point.Z] for point in vertices]),
        faces=np.asarray(faces), process=False,
    )


def _depths(manager, mesh):
    collided, names, contacts = manager.in_collision_single(
        mesh, return_names=True, return_data=True
    )
    if not collided:
        return {}
    result = {}
    for contact in contacts:
        depth = max(0.0, float(contact.depth))
        for name in contact.names:
            if name in names:
                result[name] = max(result.get(name, 0.0), depth)
    return result


def _leaves(robot):
    result = []

    def walk(node, root=None, servo=None, path=""):
        children = _children(node)
        if not path:
            for index, child in enumerate(children):
                walk(child, child.label, None, f'/{child.label}[{index}]')
            return
        current_servo = servo or (node.label[:3] if node.label.startswith("S") else None)
        if children:
            for index, child in enumerate(children):
                walk(child, root, current_servo, path + f'/{child.label}[{index}]')
        else:
            if not str(root).startswith("Frame"):
                result.append({"path": path, "shape": node, "root": root, "servo": current_servo, "label": node.label})

    walk(robot)
    return result


def main():
    robot = read_step(TARGET)
    leaves = _leaves(robot)
    assert len({item['path'] for item in leaves}) == len(leaves)
    axes = measured_axes(robot)
    meshes = {item["path"]: _mesh(item["shape"]) for item in leaves}
    report = {
        "target_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        "scanner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mjcf_sha256": hashlib.sha256(MJCF.read_bytes()).hexdigest(),
        "axes": axes,
        "angle_convention": "MJCF joint angle; current STEP neutral pose and measured servo axes",
        "limit_source": "playground/open_duck_mini_zp10d/xmls/open_duck_mini_zp10d.xml",
        "excluded_reference_roots": ["Frame <1>", "Frame <2>", "Frame <3>", "Frame <4>", "Frame <5>"],
        "step_deg": STEP_DEG,
        "penetration_tolerance_mm": PENETRATION_TOLERANCE_MM,
        "joints": [],
    }
    by_path = {item['path']: item for item in leaves}
    for servo_id in LIMITS:
        origin, direction = axes[servo_id]
        moving = [
            item for item in leaves
            if is_moving(item, servo_id)
        ]
        moving_paths = {item["path"] for item in moving}
        fixed = [item for item in leaves if item["path"] not in moving_paths]
        manager = trimesh.collision.CollisionManager()
        for item in fixed:
            manager.add_object(item["path"], meshes[item["path"]])
        baseline = {}
        for item in moving:
            for fixed_path, depth in _depths(manager, meshes[item["path"]]).items():
                baseline[(item["path"], fixed_path)] = depth
        pair_angles = {}
        angles = _angles(*LIMITS[servo_id], STEP_DEG)
        for angle in angles:
            transform = trimesh.transformations.rotation_matrix(math.radians(angle), direction, origin)
            for item in moving:
                posed = meshes[item["path"]].copy()
                posed.apply_transform(transform)
                for fixed_path, depth in _depths(manager, posed).items():
                    if item['servo'] and item['servo'] == by_path[fixed_path]['servo']:
                        continue
                    if depth > PENETRATION_TOLERANCE_MM:
                        pair_angles.setdefault((item["path"], fixed_path), []).append(angle)
        candidates = [
            {"moving": moving_path, "fixed": fixed_path, "angles_deg": sorted(set(angles))}
            for (moving_path, fixed_path), angles in sorted(pair_angles.items())
        ]
        report["joints"].append({"id": servo_id, "name": JOINT_NAMES[servo_id], "limits": list(LIMITS[servo_id]), "sampled_angles_deg": angles, "moving_count": len(moving), "candidate_collisions": candidates})
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        print(servo_id, "moving", len(moving), "candidate pairs", len(candidates), flush=True)


if __name__ == "__main__":
    main()
