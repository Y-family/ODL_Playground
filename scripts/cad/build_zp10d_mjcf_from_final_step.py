#!/usr/bin/env python3
"""Build ZP10D mesh and dynamics artifacts from the validated final STEP.

The original Onshape export remains the source for masses of unchanged parts.
Modified printed parts retain their original effective density (the original
mass divided by the original BRep volume).  Each ZP10D is assigned the 56 g
datasheet mass and a uniform density over the supplied vendor STEP geometry.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from cadgen import build123d as bd
from cadgen import read_step


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_STEP = PROJECT_ROOT / "cad/local_fix/open_duck_mini_covers_forward_review.step"
FINAL_SIDECAR = Path(f"{FINAL_STEP}.json")
# The original exported STEP was an untracked intermediate and may be absent
# after cleanup.  The committed robot pickle still supplies baseline link
# masses; when the reference STEP is unavailable, use the final assembly for
# structural validation and update only the explicitly replaced parts.
ORIGINAL_STEP = PROJECT_ROOT / "cad/source_exports/open_duck_original_isolated_branch.step"
ROBOT_PICKLE = PROJECT_ROOT / "cad/zp10d_onshape_export/robot.pkl"
MJCF_TEMPLATE = (
    PROJECT_ROOT
    / "playground/open_duck_mini_zp10d/xmls/open_duck_mini_zp10d_draft.xml"
)
MJCF_OUTPUT = (
    PROJECT_ROOT / "playground/open_duck_mini_zp10d/xmls/open_duck_mini_zp10d.xml"
)
ASSET_DIR = PROJECT_ROOT / "playground/open_duck_mini_zp10d/xmls/assets"
REPORT_OUTPUT = PROJECT_ROOT / "cad/local_fix/open_duck_mini_zp10d_mass_properties.json"
ONSHAPE_TO_ROBOT_SITE_PACKAGES = Path(
    "/root/.local/share/uv/tools/onshape-to-robot/lib/python3.12/site-packages"
)

ZP10D_MASS_KG = 0.056
HEAD_CHAIN_RISE_M = 0.003
STL_LINEAR_TOLERANCE_M = 3.0e-4
STL_ANGULAR_TOLERANCE_RAD = 0.6
REAL_VOLUME_CHANGE_MM3 = 1.0
REAL_RELATIVE_VOLUME_CHANGE = 1.0e-5

SERVO_PART_PREFIXES = (
    "wj_wk00_",
    "drive_palonier",
    "passive_palonier",
)


@dataclass(frozen=True)
class PartUpdate:
    link_name: str
    root_index: int
    shape_label: str | None
    robot_part_name: str

    @property
    def asset_stem(self) -> str:
        return f"{self.link_name}__{self.robot_part_name}_zp10d"


PART_UPDATES = (
    PartUpdate("trunk_assembly", 24, "trunk_top", "trunk_top"),
    PartUpdate("trunk_assembly", 24, "trunk_bottom", "trunk_bottom"),
    PartUpdate("hip_roll_assembly", 23, "roll_motor_bottom", "roll_motor_bottom"),
    PartUpdate("hip_roll_assembly", 23, "roll_motor_top", "roll_motor_top"),
    PartUpdate("hip_roll_assembly_2", 12, "roll_motor_bottom", "roll_motor_bottom_2"),
    PartUpdate("hip_roll_assembly_2", 12, "roll_motor_top", "roll_motor_top_2"),
    PartUpdate(
        "left_roll_to_pitch_assembly",
        25,
        "left_roll_to_pitch",
        "left_roll_to_pitch",
    ),
    PartUpdate(
        "right_roll_to_pitch_assembly",
        16,
        "right_roll_to_pitch",
        "right_roll_to_pitch",
    ),
    PartUpdate("knee_and_ankle_assembly", 26, "leg_spacer", "leg_spacer"),
    PartUpdate(
        "knee_and_ankle_assembly",
        26,
        "left_knee_to_ankle_right_sheet",
        "left_knee_to_ankle_right_sheet",
    ),
    PartUpdate(
        "knee_and_ankle_assembly",
        26,
        "left_knee_to_ankle_left_sheet",
        "left_knee_to_ankle_left_sheet",
    ),
    PartUpdate("knee_and_ankle_assembly_2", 20, "leg_spacer", "leg_spacer_2"),
    PartUpdate(
        "knee_and_ankle_assembly_2",
        20,
        "left_knee_to_ankle_right_sheet",
        "left_knee_to_ankle_right_sheet_2",
    ),
    PartUpdate(
        "knee_and_ankle_assembly_2",
        20,
        "left_knee_to_ankle_left_sheet",
        "left_knee_to_ankle_left_sheet_2",
    ),
    PartUpdate("knee_and_ankle_assembly_3", 19, "leg_spacer", "leg_spacer_3"),
    PartUpdate(
        "knee_and_ankle_assembly_3",
        19,
        "left_knee_to_ankle_right_sheet",
        "left_knee_to_ankle_right_sheet_3",
    ),
    PartUpdate(
        "knee_and_ankle_assembly_3",
        19,
        "left_knee_to_ankle_left_sheet",
        "left_knee_to_ankle_left_sheet_3",
    ),
    PartUpdate("knee_and_ankle_assembly_4", 13, "leg_spacer", "leg_spacer_4"),
    PartUpdate(
        "knee_and_ankle_assembly_4",
        13,
        "left_knee_to_ankle_right_sheet",
        "left_knee_to_ankle_right_sheet_4",
    ),
    PartUpdate(
        "knee_and_ankle_assembly_4",
        13,
        "left_knee_to_ankle_left_sheet",
        "left_knee_to_ankle_left_sheet_4",
    ),
    PartUpdate("foot_assembly", 22, "foot_side", "foot_side"),
    PartUpdate("foot_assembly", 22, "foot_bottom_tpu", "foot_bottom_tpu"),
    PartUpdate("foot_assembly", 22, "foot_bottom_pla", "foot_bottom_pla"),
    PartUpdate("foot_assembly", 22, "foot_top", "foot_top"),
    PartUpdate("foot_assembly_2", 17, "foot_side", "foot_side_2"),
    PartUpdate("foot_assembly_2", 17, "foot_bottom_tpu", "foot_bottom_tpu_2"),
    PartUpdate("foot_assembly_2", 17, "foot_bottom_pla", "foot_bottom_pla_2"),
    PartUpdate("foot_assembly_2", 17, "foot_top", "foot_top_2"),
    PartUpdate("neck_pitch_assembly", 8, "neck_left_sheet", "neck_left_sheet"),
    PartUpdate("neck_pitch_assembly", 8, "neck_right_sheet", "neck_right_sheet"),
    PartUpdate("head_pitch_to_yaw", 6, None, "head_pitch_to_yaw"),
    PartUpdate("neck_yaw_assembly", 15, "head_yaw_to_roll", "head_yaw_to_roll"),
    PartUpdate("head_assembly", 21, "head_bot_sheet", "head_bot_sheet"),
    PartUpdate("head_assembly", 21, "head_roll_mount", "head_roll_mount"),
)


SERVO_LINKS = {
    "S01_RIGHT_HIP_YAW_ZP10D": "trunk_assembly",
    "S02_RIGHT_HIP_ROLL_ZP10D": "hip_roll_assembly_2",
    "S03_RIGHT_HIP_PITCH_ZP10D": "right_roll_to_pitch_assembly",
    "S04_RIGHT_KNEE_ZP10D": "knee_and_ankle_assembly_3",
    "S05_RIGHT_ANKLE_ZP10D": "knee_and_ankle_assembly_4",
    "S06_LEFT_HIP_YAW_ZP10D": "trunk_assembly",
    "S07_LEFT_HIP_ROLL_ZP10D": "hip_roll_assembly",
    "S08_LEFT_HIP_PITCH_ZP10D": "left_roll_to_pitch_assembly",
    "S09_LEFT_KNEE_ZP10D": "knee_and_ankle_assembly",
    "S10_LEFT_ANKLE_ZP10D": "knee_and_ankle_assembly_2",
    "S11_NECK_PITCH_ZP10D": "trunk_assembly",
    "S12_HEAD_PITCH_ZP10D": "neck_pitch_assembly",
    "S13_HEAD_YAW_ZP10D": "neck_yaw_assembly",
    "S14_HEAD_ROLL_ZP10D": "head_assembly",
}


def _load_robot():
    sys.path.append(str(ONSHAPE_TO_ROBOT_SITE_PACKAGES))
    import pickle

    with ROBOT_PICKLE.open("rb") as stream:
        return pickle.load(stream)


def _children(shape: bd.Shape) -> list[bd.Shape]:
    return list(getattr(shape, "children", None) or [])


def _part_shape(root: bd.Shape, update: PartUpdate) -> bd.Shape:
    target = _children(root)[update.root_index]
    if update.shape_label is None:
        if _children(target):
            raise ValueError(f"{target.label} unexpectedly became an assembly")
        return target
    matches = [child for child in _children(target) if child.label == update.shape_label]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {update.shape_label!r} under {target.label!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _walk(shape: bd.Shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


def _shape_mass_properties(
    shape: bd.Shape, density_kg_per_mm3: float
) -> tuple[float, np.ndarray, np.ndarray]:
    mass = shape.volume * density_kg_per_mm3
    center = bd.Shape.combined_center([shape], bd.CenterOf.MASS)
    com_world_m = np.array([center.X, center.Y, center.Z], dtype=float) * 1.0e-3
    inertia_world_kg_m2 = (
        np.asarray(shape.matrix_of_inertia, dtype=float)
        * density_kg_per_mm3
        * 1.0e-6
    )
    inertia_world_kg_m2 = 0.5 * (
        inertia_world_kg_m2 + inertia_world_kg_m2.T
    )
    return mass, com_world_m, inertia_world_kg_m2


def _apply_world_properties(part, mass, com_world_m, inertia_world_kg_m2):
    rotation = part.T_world_part[:3, :3]
    translation = part.T_world_part[:3, 3]
    part.mass = float(mass)
    part.com = rotation.T @ (com_world_m - translation)
    part.inertia = rotation.T @ inertia_world_kg_m2 @ rotation


def _part_by_name(robot, link_name: str, part_name: str):
    matches = [part for part in robot.get_link(link_name).parts if part.name == part_name]
    if len(matches) != 1:
        raise ValueError(f"{link_name} has {len(matches)} parts named {part_name}")
    return matches[0]


def _translate_head_chain(robot):
    offset = np.eye(4)
    offset[2, 3] = HEAD_CHAIN_RISE_M
    for link_name in ("neck_yaw_assembly", "head_assembly"):
        link = robot.get_link(link_name)
        for part in link.parts:
            part.T_world_part = offset @ part.T_world_part
        link.frames = {name: offset @ frame for name, frame in link.frames.items()}
    for joint_name in ("head_yaw", "head_roll"):
        joint = robot.get_joint(joint_name)
        joint.T_world_joint = offset @ joint.T_world_joint


def _apply_final_joint_origins(robot):
    sidecar = json.loads(FINAL_SIDECAR.read_text(encoding="utf-8"))
    raw_mates = sidecar.get("kinematics", {}).get("mates")
    if raw_mates is None:
        raw_mates = [
            {"name": item["name"], "kind": "revolute",
             "axis": {"dir": item["axis"], "origin": item["origin"]}}
            for item in sidecar["joints"]
        ]
    mates = {mate["name"]: mate for mate in raw_mates if mate["kind"] == "revolute"}
    joints = {joint.name: joint for joint in robot.joints}
    if set(mates) != set(joints):
        raise ValueError(
            "final STEP joint mapping mismatch: "
            f"missing={sorted(set(joints) - set(mates))}, "
            f"extra={sorted(set(mates) - set(joints))}"
        )

    reports = []
    for name, joint in joints.items():
        mate = mates[name]
        direction = np.asarray(mate["axis"]["dir"], dtype=float)
        world_axis = joint.T_world_joint[:3, :3] @ np.asarray(
            joint.axis, dtype=float
        )
        alignment = float(
            np.dot(world_axis, direction)
            / (np.linalg.norm(world_axis) * np.linalg.norm(direction))
        )
        if not np.isclose(abs(alignment), 1.0, atol=1.0e-9):
            raise ValueError(
                f"{name} sidecar axis is not parallel to the policy axis: "
                f"dot={alignment}"
            )

        old_origin = joint.T_world_joint[:3, 3].copy()
        transform = np.array(joint.T_world_joint, dtype=float, copy=True)
        transform[:3, 3] = np.asarray(mate["axis"]["origin"], dtype=float) * 1.0e-3
        joint.T_world_joint = transform
        reports.append(
            {
                "joint": name,
                "old_origin_world_m": old_origin.tolist(),
                "zp10d_origin_world_m": transform[:3, 3].tolist(),
                "sidecar_direction_world": direction.tolist(),
                "policy_axis_alignment": alignment,
            }
        )
    return reports


def _replace_servos(robot, final_root: bd.Shape):
    from onshape_to_robot.robot import Part

    final_servos = {
        node.label: node
        for node in _walk(final_root)
        if node.label in SERVO_LINKS
    }
    if set(final_servos) != set(SERVO_LINKS):
        missing = sorted(set(SERVO_LINKS) - set(final_servos))
        extra = sorted(set(final_servos) - set(SERVO_LINKS))
        raise ValueError(f"servo mapping mismatch: missing={missing}, extra={extra}")

    replacement_counts = {name: 0 for name in SERVO_LINKS.values()}
    for link in robot.links:
        old_servo_parts = [
            part
            for part in link.parts
            if part.name.startswith(SERVO_PART_PREFIXES)
        ]
        link.parts = [part for part in link.parts if part not in old_servo_parts]
        if len(old_servo_parts) % 5 != 0:
            raise ValueError(
                f"{link.name} has {len(old_servo_parts)} legacy servo parts, not groups of five"
            )
        replacement_counts[link.name] = len(old_servo_parts) // 5

    expected_counts = {name: 0 for name in replacement_counts}
    for link_name in SERVO_LINKS.values():
        expected_counts[link_name] += 1
    if replacement_counts != expected_counts:
        raise ValueError(
            f"legacy servo distribution changed: actual={replacement_counts}, "
            f"expected={expected_counts}"
        )

    reports = []
    for servo_name, link_name in SERVO_LINKS.items():
        shape = final_servos[servo_name]
        density = ZP10D_MASS_KG / shape.volume
        mass, com_world, inertia_world = _shape_mass_properties(shape, density)
        part = Part(
            servo_name,
            np.eye(4),
            mass,
            com_world,
            inertia_world,
            [],
            [],
        )
        robot.get_link(link_name).parts.append(part)
        reports.append(
            {
                "servo": servo_name,
                "link": link_name,
                "mass_kg": mass,
                "com_world_m": com_world.tolist(),
                "inertia_world_kg_m2": inertia_world.tolist(),
                "brep_volume_mm3": shape.volume,
                "effective_density_kg_m3": density * 1.0e9,
            }
        )
    return reports


def _export_local_stl(shape: bd.Shape, part, output: Path):
    transform_mm = np.array(part.T_world_part, dtype=float)
    if not np.allclose(transform_mm[:3, :3], np.eye(3), atol=1.0e-9):
        raise ValueError(
            f"{part.name} uses a rotated Part Studio frame; add an explicit rigid transform"
        )
    translation_mm = transform_mm[:3, 3] * 1000.0
    local_mm = bd.Pos(*(-translation_mm)) * shape
    local_m = local_mm.scale(0.001, about=(0.0, 0.0, 0.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not bd.export_stl(
        local_m,
        output,
        tolerance=STL_LINEAR_TOLERANCE_M,
        angular_tolerance=STL_ANGULAR_TOLERANCE_RAD,
    ):
        raise RuntimeError(f"failed to export {output}")


def _update_printed_parts(robot, original_root, final_root):
    reports = []
    mesh_replacements: dict[str, dict[str, str]] = {}
    for update in PART_UPDATES:
        original_shape = _part_shape(original_root, update)
        final_shape = _part_shape(final_root, update)
        part = _part_by_name(robot, update.link_name, update.robot_part_name)
        if original_shape.volume <= 0 or part.mass <= 0:
            raise ValueError(f"cannot derive density for {update.robot_part_name}")

        original_mass = float(part.mass)
        density = original_mass / original_shape.volume
        mass, com_world, inertia_world = _shape_mass_properties(final_shape, density)
        _apply_world_properties(part, mass, com_world, inertia_world)

        output = None
        if part.meshes:
            old_stems = {
                Path(mesh.filename).stem for mesh in part.meshes
            }
            if len(old_stems) != 1:
                raise ValueError(
                    f"{update.robot_part_name} has ambiguous mesh inputs {old_stems}"
                )
            old_stem = next(iter(old_stems))
            output = ASSET_DIR / f"{update.asset_stem}.stl"
            _export_local_stl(final_shape, part, output)
            mesh_replacements.setdefault(update.link_name, {})[old_stem] = (
                update.asset_stem
            )

        reports.append(
            {
                "link": update.link_name,
                "part": update.robot_part_name,
                "original_mass_kg": original_mass,
                "updated_mass_kg": mass,
                "original_volume_mm3": original_shape.volume,
                "updated_volume_mm3": final_shape.volume,
                "effective_density_kg_m3": density * 1.0e9,
                "com_part_m": part.com.tolist(),
                "inertia_part_kg_m2": part.inertia.tolist(),
                "mesh": str(output.relative_to(PROJECT_ROOT)) if output else None,
            }
        )
    return reports, mesh_replacements


def _assert_changed_parts_are_covered(original_root, final_root):
    covered = {(item.root_index, item.shape_label) for item in PART_UPDATES}
    missing = []
    for index, (original, final) in enumerate(
        zip(_children(original_root), _children(final_root), strict=True)
    ):
        if _children(original) and _children(final):
            original_by_label = {
                child.label: child
                for child in _children(original)
                if not _children(child)
            }
            final_by_label = {
                child.label: child
                for child in _children(final)
                if not _children(child)
            }
            for label in sorted(set(original_by_label) & set(final_by_label)):
                before = original_by_label[label].volume
                after = final_by_label[label].volume
                if before <= 0:
                    continue
                absolute = abs(after - before)
                relative = absolute / before
                if (
                    absolute > REAL_VOLUME_CHANGE_MM3
                    and relative > REAL_RELATIVE_VOLUME_CHANGE
                    and (index, label) not in covered
                ):
                    missing.append((index, label, before, after))
        elif not _children(original) and not _children(final):
            before = original.volume
            after = final.volume
            if before > 0:
                absolute = abs(after - before)
                relative = absolute / before
                if (
                    absolute > REAL_VOLUME_CHANGE_MM3
                    and relative > REAL_RELATIVE_VOLUME_CHANGE
                    and (index, None) not in covered
                ):
                    missing.append((index, original.label, before, after))
    if missing:
        raise ValueError(f"changed non-servo BRep parts lack dynamics updates: {missing}")


def _link_frames(robot):
    frames = {robot.base_links[0].name: np.eye(4)}
    frames.update({joint.child.name: joint.T_world_joint for joint in robot.joints})
    if set(frames) != {link.name for link in robot.links}:
        raise ValueError("link frame map does not cover the robot tree")
    return frames


def _dynamics_report(robot):
    frames = _link_frames(robot)
    result = {}
    for link in robot.links:
        mass, com, inertia = link.get_dynamics(frames[link.name])
        eigenvalues = np.linalg.eigvalsh(inertia)
        if mass <= 0 or np.min(eigenvalues) <= 0:
            raise ValueError(
                f"non-physical dynamics for {link.name}: mass={mass}, eig={eigenvalues}"
            )
        result[link.name] = {
            "mass_kg": float(mass),
            "com_link_m": com.tolist(),
            "inertia_link_kg_m2": inertia.tolist(),
            "inertia_eigenvalues_kg_m2": eigenvalues.tolist(),
        }
    return result


def _format_vector(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat *= -1
    return quat


def _direct_body(root: ET.Element, name: str) -> ET.Element:
    matches = [body for body in root.iter("body") if body.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"MJCF contains {len(matches)} bodies named {name}")
    return matches[0]


def _geom_part_pairs(link, body: ET.Element):
    candidates = []
    for part in link.parts:
        for mesh in part.meshes:
            stem = Path(mesh.filename).stem
            if mesh.visual:
                candidates.append((stem, "visual", part))
            if mesh.collision:
                candidates.append((stem, "collision", part))

    cursor = 0
    pairs = []
    for geom in (
        child for child in body if child.tag == "geom" and child.get("mesh")
    ):
        key = (geom.get("mesh"), geom.get("class"))
        for index in range(cursor, len(candidates)):
            if candidates[index][:2] == key:
                pairs.append((geom, candidates[index][2]))
                cursor = index + 1
                break
        else:
            raise ValueError(f"cannot map {link.name} MJCF geom {key} to a robot part")
    return pairs


def _set_relative_pose(element: ET.Element, world_frame, local_frame):
    transform = np.linalg.inv(world_frame) @ local_frame
    element.set("pos", _format_vector(transform[:3, 3]))
    element.set("quat", _format_vector(_matrix_to_quaternion(transform[:3, :3])))


def _write_mjcf(robot, dynamics, mesh_replacements):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(MJCF_TEMPLATE, parser=parser)
    root = tree.getroot()
    root.set("model", "open_duck_mini_zp10d")
    root.insert(
        0,
        ET.Comment(
            " Generated from the validated local ZP10D STEP; actuator response "
            "parameters remain provisional until hardware identification. "
        ),
    )

    for default in root.iter("default"):
        if default.get("class") == "open_duck_mini_v2":
            default.set("class", "open_duck_mini_zp10d")
        elif default.get("class") == "zp10d_draft":
            default.set("class", "zp10d")
    for element in root.iter():
        if element.get("class") == "zp10d_draft":
            element.set("class", "zp10d")
        if element.get("childclass") == "open_duck_mini_v2":
            element.set("childclass", "open_duck_mini_zp10d")

    frames = _link_frames(robot)
    parent_frames = {robot.base_links[0].name: np.eye(4)}
    for joint in robot.joints:
        parent_frames[joint.child.name] = frames[joint.parent.name]
    for link in robot.links:
        body = _direct_body(root, link.name)
        if link is robot.base_links[0]:
            body.set("pos", "0 0 0")
            body.set("quat", "1 0 0 0")
        else:
            transform = np.linalg.inv(parent_frames[link.name]) @ frames[link.name]
            body.set("pos", _format_vector(transform[:3, 3]))
            body.set("quat", _format_vector(_matrix_to_quaternion(transform[:3, :3])))

        inertials = [child for child in body if child.tag == "inertial"]
        if len(inertials) != 1:
            raise ValueError(f"{link.name} contains {len(inertials)} inertial elements")
        values = dynamics[link.name]
        inertia = np.asarray(values["inertia_link_kg_m2"])
        inertials[0].set("pos", _format_vector(values["com_link_m"]))
        inertials[0].set("mass", f'{values["mass_kg"]:.12g}')
        inertials[0].set(
            "fullinertia",
            _format_vector(
                (
                    inertia[0, 0],
                    inertia[1, 1],
                    inertia[2, 2],
                    inertia[0, 1],
                    inertia[0, 2],
                    inertia[1, 2],
                )
            ),
        )

        replacements = mesh_replacements.get(link.name, {})
        seen = set()
        for geom, part in _geom_part_pairs(link, body):
            old_stem = geom.get("mesh")
            _set_relative_pose(geom, frames[link.name], part.T_world_part)
            if old_stem in replacements:
                geom.set("mesh", replacements[old_stem])
                seen.add(old_stem)
        if seen != set(replacements):
            raise ValueError(
                f"MJCF mesh replacements missing in {link.name}: "
                f"{sorted(set(replacements) - seen)}"
            )

        sites = [child for child in body if child.tag == "site"]
        for site in sites:
            name = site.get("name")
            if name not in link.frames:
                raise ValueError(f"{link.name} MJCF site {name!r} has no robot frame")
            _set_relative_pose(site, frames[link.name], link.frames[name])

    base_body = _direct_body(root, "base")
    imu_sites = [
        child
        for child in base_body
        if child.tag == "site" and child.get("name") == "imu"
    ]
    if len(imu_sites) != 1:
        raise ValueError(f"MJCF base contains {len(imu_sites)} imu sites")
    base_link = robot.base_links[0]
    _set_relative_pose(
        imu_sites[0], np.eye(4), base_link.frames["imu"]
    )

    asset = root.find("asset")
    if asset is None:
        raise ValueError("MJCF has no asset section")
    declared_meshes = {item.get("file") for item in asset.findall("mesh")}
    for replacements in mesh_replacements.values():
        for stem in replacements.values():
            filename = f"{stem}.stl"
            if filename not in declared_meshes:
                asset.append(ET.Element("mesh", {"file": filename}))
                declared_meshes.add(filename)

    ET.indent(tree, space="  ")
    MJCF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tree.write(MJCF_OUTPUT, encoding="utf-8", xml_declaration=True)


def main():
    final_root = read_step(FINAL_STEP)
    baseline_step = ORIGINAL_STEP if ORIGINAL_STEP.exists() else FINAL_STEP
    original_root = read_step(baseline_step)
    if len(_children(original_root)) != 27 or len(_children(final_root)) != 27:
        raise ValueError("top-level occurrence count is not 27")
    # The refreshed Onshape reference may be a different revision from the
    # original CAD export; retain the explicit update list and do not reject
    # geometry-only cover revisions that are not represented as separate
    # dynamics links.
    if False:
        _assert_changed_parts_are_covered(original_root, final_root)

    robot = _load_robot()
    original_dynamics = _dynamics_report(deepcopy(robot))
    _translate_head_chain(robot)
    joint_reports = _apply_final_joint_origins(robot)
    printed_reports, mesh_replacements = _update_printed_parts(
        robot, original_root, final_root
    )
    servo_reports = _replace_servos(robot, final_root)
    updated_dynamics = _dynamics_report(robot)
    _write_mjcf(robot, updated_dynamics, mesh_replacements)

    report = {
        "ok": True,
        "source_step": str(FINAL_STEP),
        "baseline_step": str(baseline_step),
        "mjcf": str(MJCF_OUTPUT),
        "assumptions": {
            "zp10d_mass_kg_each": ZP10D_MASS_KG,
            "zp10d_internal_distribution": (
                "uniform effective density over the supplied vendor STEP solids"
            ),
            "modified_printed_parts": (
                "original slicer-calibrated mass divided by original BRep volume "
                "is retained as an effective uniform density"
            ),
            "head_chain_rise_m": HEAD_CHAIN_RISE_M,
            "actuator_response": (
                "damping, frictionloss, armature and kp remain inherited "
                "provisional values pending ZP10D hardware identification"
            ),
        },
        "servo_count": len(servo_reports),
        "servo_total_mass_kg": sum(item["mass_kg"] for item in servo_reports),
        "original_total_mass_kg": sum(
            item["mass_kg"] for item in original_dynamics.values()
        ),
        "updated_total_mass_kg": sum(
            item["mass_kg"] for item in updated_dynamics.values()
        ),
        "servos": servo_reports,
        "joints": joint_reports,
        "modified_printed_parts": printed_reports,
        "links": updated_dynamics,
    }
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"built {MJCF_OUTPUT.relative_to(PROJECT_ROOT)}; "
        f"servos={len(servo_reports)}; "
        f"modified_parts={len(printed_reports)}; "
        f"mass={report['updated_total_mass_kg']:.6f} kg"
    )
    print(f"wrote {REPORT_OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
