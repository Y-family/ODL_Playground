#!/usr/bin/env python3
"""Synchronize the validated S01-S05 ZP10D right leg into the native assembly.

The operation is deliberately narrow: seven existing target occurrences are
updated one-for-one, no mechanical occurrence is added, and only the five
right-leg revolute mates plus one right-cache rigid relation are rebuilt.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from onshape_import_step import signed_request
from onshape_manage_rigid_group import rigid_group_feature


DOCUMENT_ID = "2b9c1a0eb9b7da6426df90e5"
WORKSPACE_ID = "126224a19cf9e463c24ad750"
TARGET_ASSEMBLY_ID = "e492fe1531afa9374de3c1b8"
PART_STUDIO_ID = "70e5ee2039a1914df4c6313f"

IDENTITY = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


@dataclass(frozen=True)
class Replacement:
    label: str
    target_instance_id: str
    source_element_id: str
    source_type: str = "ASSEMBLY"
    source_part_id: str | None = None
    desired_transform: tuple[float, ...] = tuple(IDENTITY)


REPLACEMENTS = (
    Replacement("right trunk / S01", "MiEvZbyJuPaig3d8P", "4a3fd171b8a727140d73295a"),
    Replacement("right hip roll / S02", "Mv7mjtoLHrxJfpJOr", "016cd4aa51eb08770b50f8e9"),
    Replacement("right hip pitch / S03", "MdpQNTQVWIQVA2XPv", "25cd49741d32942e6259fd03"),
    Replacement("right upper leg / S04", "MoB647zxmlVieawCK", "b3e6337a6434aebfbce77aaf"),
    Replacement("right lower leg / S05", "MqymZLKrI8WnAicE+", "904c00187191f120449c6bcd"),
    Replacement("right foot", "MC1dJZUdK6hFtGmjb", "fd9b1cb0a66f0823fd1a5457"),
    Replacement(
        "right cache",
        "MBtFsLan/52XkruX7",
        PART_STUDIO_ID,
        "PARTSTUDIO",
        "JFL",
        (
            1.0, 0.0, 0.0, -0.019,
            0.0, 1.0, 0.0, -0.0075,
            0.0, 0.0, 1.0, 0.064925938812371,
            0.0, 0.0, 0.0, 1.0,
        ),
    ),
)

# Root instance IDs inside the imported native subassemblies.
RIGID_GROUPS = (
    ("f74bee119a319ac80b3ae5d3", "S01_ZP10D_INTERNAL_RIGID", ("MKOE/WWepTnGFElQ8", "M2Ohb6WNzh5RKfDo3", "M+b8jd2hZ1UmPast4")),
    ("bc7f753c8f324229804d965d", "S02_ZP10D_INTERNAL_RIGID", ("MH0MN4KsVSC6wu5Vp", "MGlUyuCYmWLKqTTII", "MPquQx6E7Q3iRM2+b")),
    ("e5885c0cebc4e25877cf8508", "S03_ZP10D_INTERNAL_RIGID", ("My+WIrh8qqjaZIpER", "MdgtX4hJhnZwRH9c/", "MFWzGtl9zio3w4iTg")),
    ("d7189a1ee1f19aed64e1e481", "S04_ZP10D_INTERNAL_RIGID", ("MRg5n/fMVLmc9Ghos", "MZbowue9OWWG3XPy0", "MZEjrThSgqiD6a/E0")),
    ("a6dc23a8370db326cd4c01f8", "S05_ZP10D_INTERNAL_RIGID", ("M0/3jaZ5I8WWuzT0s", "Moqwg5CtO0NmC20mC", "Mbzjqdbh9hWyYeM80")),
    ("4a3fd171b8a727140d73295a", "RIGHT_TRUNK_S01_RIGID", ("MxC/SnrgXMG5JXCRj", "M1UahUgZYq8kKCxA4", "Mf01mfYAKgsuqsDBX", "MZNGM2nUb4lWq2M/P", "M5HNOSh3XjrHElGd3", "MlKywk7HtTSni6WTH", "MHH0PP72Dc61GlCOO", "MkYXebJMOK+4E0pz1", "MgDN8IAAIx9DX8Lo3")),
    ("016cd4aa51eb08770b50f8e9", "RIGHT_HIP_ROLL_S02_RIGID", ("MxuRGA02wQTFnXvSD", "MCPBgvxJ+ympbeCj9", "MJKbLtvnYZ6uZfc07")),
    ("25cd49741d32942e6259fd03", "RIGHT_HIP_PITCH_S03_RIGID", ("MuYf2soAsx6HyowUg", "MB0c1EN7JJ8zzqVjv")),
    ("b3e6337a6434aebfbce77aaf", "RIGHT_UPPER_LEG_S04_RIGID", ("M73q7yu1aVCOIZlPU", "McXsutjWuzsRObFzJ", "Mjrdb/bDpiWye4X63", "M+MRUnTc4aruu9Hn5")),
    ("904c00187191f120449c6bcd", "RIGHT_LOWER_LEG_S05_RIGID", ("MxnHQOWalqowwfOpj", "ME47pekUT0n2WdYDY", "MTJVyGJczrQa7daeE", "MmpRtX1CZdCGCMmZT")),
    ("fd9b1cb0a66f0823fd1a5457", "RIGHT_FOOT_RIGID", ("MinhM3et+AjM4zck1", "MxePRs4VBMqZHpdHy", "MHJ7/hi1KMwdU3xUA", "MQuuL8M5RAZ/0YNno")),
)

OLD_RIGHT_MATE_IDS = {
    "dof_right_hip_yaw": "MpZOgq8JC9FVI1mg5",
    "dof_right_hip_roll": "MtQitW3SF/Vzi8bNq",
    "dof_right_hip_pitch": "MJe5WF9WwQMjp7NOi",
    "dof_right_knee": "MO1TNmc/UnyNMfpMA",
    "dof_right_ankle": "MtzHX4bGzERDECzlv",
}

ACTIVE_HORN_CONNECTOR = "FJfTUbkQZTWkaBI_133"
LINK_CONNECTORS = {
    "dof_right_hip_yaw": "FWVIPKHWHMceN53_133",
    "dof_right_hip_roll": "Fp2AZRXrAjwORTc_133",
    "dof_right_hip_pitch": "FfY6ivu4V7kdhaq_133",
    "dof_right_knee": "FGyvwdaWOiLWMFq_133",
    "dof_right_ankle": "FM2zmZyxg5G3sMA_133",
}

# Full paths after the seven target occurrence references are updated.
MATE_PATHS = {
    "dof_right_hip_yaw": (
        ("Mv7mjtoLHrxJfpJOr", "MJKbLtvnYZ6uZfc07"),
        ("MiEvZbyJuPaig3d8P", "MxC/SnrgXMG5JXCRj", "MKOE/WWepTnGFElQ8"),
    ),
    "dof_right_hip_roll": (
        ("MdpQNTQVWIQVA2XPv", "MB0c1EN7JJ8zzqVjv"),
        ("Mv7mjtoLHrxJfpJOr", "MxuRGA02wQTFnXvSD", "MH0MN4KsVSC6wu5Vp"),
    ),
    "dof_right_hip_pitch": (
        ("MoB647zxmlVieawCK", "Mjrdb/bDpiWye4X63"),
        ("MdpQNTQVWIQVA2XPv", "MuYf2soAsx6HyowUg", "My+WIrh8qqjaZIpER"),
    ),
    "dof_right_knee": (
        ("MqymZLKrI8WnAicE+", "MTJVyGJczrQa7daeE"),
        ("MoB647zxmlVieawCK", "McXsutjWuzsRObFzJ", "MRg5n/fMVLmc9Ghos"),
    ),
    "dof_right_ankle": (
        ("MC1dJZUdK6hFtGmjb", "MxePRs4VBMqZHpdHy"),
        ("MqymZLKrI8WnAicE+", "ME47pekUT0n2WdYDY", "M0/3jaZ5I8WWuzT0s"),
    ),
}


def node_id() -> str:
    return "M" + base64.b64encode(secrets.token_bytes(12)).decode("ascii")


def assembly(element_id: str) -> dict:
    return signed_request(
        "GET",
        f"/api/v12/assemblies/d/{DOCUMENT_ID}/w/{WORKSPACE_ID}/e/{element_id}",
        params={"includeMateFeatures": "true", "includeNonSolids": "true"},
    ).json()


def features(element_id: str) -> dict:
    return signed_request(
        "GET",
        f"/api/v12/assemblies/d/{DOCUMENT_ID}/w/{WORKSPACE_ID}/e/{element_id}/features",
    ).json()


def add_feature(element_id: str, feature: dict, feature_list: dict) -> dict:
    return signed_request(
        "POST",
        f"/api/v12/assemblies/d/{DOCUMENT_ID}/w/{WORKSPACE_ID}/e/{element_id}/features",
        json={
            "btType": "BTFeatureDefinitionCall-1406",
            "feature": feature,
            "libraryVersion": feature_list["libraryVersion"],
            "serializationVersion": feature_list["serializationVersion"],
            "sourceMicroversion": feature_list["sourceMicroversion"],
            "rejectMicroversionSkew": False,
        },
    ).json()


def ensure_rigid_group(element_id: str, name: str, instance_ids: tuple[str, ...]) -> str:
    current = features(element_id)
    existing = next(
        (feature for feature in current.get("features", []) if feature.get("name") == name),
        None,
    )
    if existing:
        return existing["featureId"]
    available = {instance["id"] for instance in assembly(element_id)["rootAssembly"]["instances"]}
    missing = sorted(set(instance_ids) - available)
    if missing:
        raise RuntimeError(f"{element_id} is missing rigid-group instances: {missing}")
    proposal = rigid_group_feature(name, list(instance_ids))
    response = add_feature(element_id, proposal, current)
    state = response.get("featureState", {}).get("featureStatus")
    if state not in (None, "OK"):
        raise RuntimeError(f"{name} returned {state}")
    return response.get("feature", {}).get("featureId") or proposal["featureId"]


def inverse_rigid(transform: list[float]) -> list[float]:
    rotation = [[transform[r * 4 + c] for c in range(3)] for r in range(3)]
    translation = [transform[3], transform[7], transform[11]]
    inverse_rotation = [[rotation[c][r] for c in range(3)] for r in range(3)]
    inverse_translation = [
        -sum(inverse_rotation[r][c] * translation[c] for c in range(3))
        for r in range(3)
    ]
    return [
        inverse_rotation[0][0], inverse_rotation[0][1], inverse_rotation[0][2], inverse_translation[0],
        inverse_rotation[1][0], inverse_rotation[1][1], inverse_rotation[1][2], inverse_translation[1],
        inverse_rotation[2][0], inverse_rotation[2][1], inverse_rotation[2][2], inverse_translation[2],
        0.0, 0.0, 0.0, 1.0,
    ]


def multiply(left: tuple[float, ...] | list[float], right: list[float]) -> list[float]:
    return [
        sum(left[row * 4 + k] * right[k * 4 + column] for k in range(4))
        for row in range(4)
        for column in range(4)
    ]


def update_references() -> None:
    updates = []
    for item in REPLACEMENTS:
        target = {
            "documentId": DOCUMENT_ID,
            "workspaceId": WORKSPACE_ID,
            "elementId": item.source_element_id,
            "elementType": item.source_type,
        }
        if item.source_part_id:
            target["partId"] = item.source_part_id
        updates.append(
            {
                "toReference": target,
                "idsToUpdate": [item.target_instance_id],
                "ignoreChildren": False,
            }
        )
    signed_request(
        "POST",
        f"/api/v12/elements/d/{DOCUMENT_ID}/w/{WORKSPACE_ID}/e/{TARGET_ASSEMBLY_ID}/updatereferences",
        json={"editDescription": "Replace right leg S01-S05", "referenceUpdates": updates},
    )


def reset_updated_occurrences(before: dict) -> None:
    transforms = {
        tuple(occurrence["path"]): occurrence["transform"]
        for occurrence in before["rootAssembly"].get("occurrences", [])
        if len(occurrence.get("path", [])) == 1
    }
    path = f"/api/v12/assemblies/d/{DOCUMENT_ID}/w/{WORKSPACE_ID}/e/{TARGET_ASSEMBLY_ID}/occurrencetransforms"
    for item in REPLACEMENTS:
        old = transforms[(item.target_instance_id,)]
        signed_request(
            "POST",
            path,
            json={
                "isRelative": True,
                "occurrences": [{"path": [item.target_instance_id]}],
                "transform": multiply(item.desired_transform, inverse_rigid(old)),
            },
        )


def feature_query(path: tuple[str, ...], feature_id: str) -> dict:
    return {
        "btType": "BTMFeatureQueryWithOccurrence-157",
        "path": list(path),
        "featureId": feature_id,
        "nodeId": node_id(),
        "queryData": "",
    }


def rebuilt_mate(template: dict) -> dict:
    result = copy.deepcopy(template)
    name = result["name"]
    result["featureId"] = node_id()
    result["nodeId"] = result["featureId"]
    for parameter in result.get("parameters", []):
        parameter["nodeId"] = node_id()
        if parameter.get("parameterId") != "mateConnectorsQuery":
            continue
        link_path, horn_path = MATE_PATHS[name]
        parameter["queries"] = [
            feature_query(link_path, LINK_CONNECTORS[name]),
            feature_query(horn_path, ACTIVE_HORN_CONNECTOR),
        ]
    return result


def delete_feature(feature_id: str) -> None:
    signed_request(
        "DELETE",
        f"/api/v12/assemblies/d/{DOCUMENT_ID}/w/{WORKSPACE_ID}/e/{TARGET_ASSEMBLY_ID}/features/featureid/{quote(feature_id, safe='')}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()

    before = assembly(TARGET_ASSEMBLY_ID)
    before_features = features(TARGET_ASSEMBLY_ID)
    root_instances = before["rootAssembly"]["instances"]
    by_id = {instance["id"]: instance for instance in root_instances}
    missing = sorted(
        item.target_instance_id for item in REPLACEMENTS if item.target_instance_id not in by_id
    )
    if missing:
        raise SystemExit(f"target occurrence IDs changed: {missing}")
    old_mates = {
        feature["name"]: feature
        for feature in before_features.get("features", [])
        if feature.get("name") in OLD_RIGHT_MATE_IDS
    }
    if set(old_mates) != set(OLD_RIGHT_MATE_IDS):
        raise SystemExit("the five original right-leg mate templates are not all present")
    proposal = {
        "targetAssemblyId": TARGET_ASSEMBLY_ID,
        "rootInstanceCount": len(root_instances),
        "replacementCount": len(REPLACEMENTS),
        "replacements": [item.__dict__ for item in REPLACEMENTS],
        "rigidGroupCount": len(RIGID_GROUPS),
        "mates": list(OLD_RIGHT_MATE_IDS),
        "apply": args.apply,
    }
    if args.backup:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(
            json.dumps({"assembly": before, "features": before_features, "proposal": proposal}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if not args.apply:
        print(json.dumps(proposal, ensure_ascii=False, indent=2))
        return

    group_ids = {}
    for element_id, name, instance_ids in RIGID_GROUPS:
        group_ids[name] = ensure_rigid_group(element_id, name, instance_ids)

    for feature_id in OLD_RIGHT_MATE_IDS.values():
        delete_feature(feature_id)
    update_references()
    reset_updated_occurrences(before)

    current = features(TARGET_ASSEMBLY_ID)
    new_mate_ids = {}
    for name in OLD_RIGHT_MATE_IDS:
        mate = rebuilt_mate(old_mates[name])
        response = add_feature(TARGET_ASSEMBLY_ID, mate, current)
        state = response.get("featureState", {}).get("featureStatus")
        if state not in (None, "OK"):
            raise RuntimeError(f"{name} returned {state}")
        new_mate_ids[name] = response.get("feature", {}).get("featureId") or mate["featureId"]
        current = features(TARGET_ASSEMBLY_ID)

    cache_group = rigid_group_feature(
        "RIGHT_CACHE_TO_UPPER_LEG_RIGID",
        ["MBtFsLan/52XkruX7", "MoB647zxmlVieawCK"],
    )
    cache_group_response = add_feature(TARGET_ASSEMBLY_ID, cache_group, current)
    cache_group_id = cache_group_response.get("feature", {}).get("featureId") or cache_group["featureId"]

    after = assembly(TARGET_ASSEMBLY_ID)
    after_features = features(TARGET_ASSEMBLY_ID)
    states = {
        feature_id: state.get("featureStatus")
        for feature_id, state in after_features.get("featureStates", {}).items()
    }
    result = {
        **proposal,
        "groupFeatureIds": group_ids,
        "newMateIds": new_mate_ids,
        "newMateStates": {name: states.get(fid) for name, fid in new_mate_ids.items()},
        "rightCacheRigidGroupId": cache_group_id,
        "rightCacheRigidGroupState": states.get(cache_group_id),
        "rootInstanceCountAfter": len(after["rootAssembly"]["instances"]),
        "referencesAfter": {
            item.label: by_id_after.get(item.target_instance_id, {}).get("elementId")
            for item in REPLACEMENTS
            for by_id_after in ({instance["id"]: instance for instance in after["rootAssembly"]["instances"]},)
        },
    }
    expected_refs = {item.label: item.source_element_id for item in REPLACEMENTS}
    if result["rootInstanceCountAfter"] != len(root_instances):
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    if result["referencesAfter"] != expected_refs:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    if any(state != "OK" for state in result["newMateStates"].values()):
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    if result["rightCacheRigidGroupState"] != "OK":
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
