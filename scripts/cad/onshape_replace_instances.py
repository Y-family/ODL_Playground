#!/usr/bin/env python3
"""Perform audited one-for-one replacements in an Onshape assembly.

The JSON spec supplies old root instance IDs, replacement instance definitions,
and explicit 4x4 transforms. No Part Studio, part, or extra occurrence is
created: a successful run requires the final root instance count to equal the
initial count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

from onshape_import_step import signed_request
from onshape_manage_rigid_group import assembly, feature_list, rigid_group_feature


def delete_instance(
    document_id: str, workspace_id: str, element_id: str, instance_id: str
) -> None:
    encoded = quote(instance_id, safe="")
    signed_request(
        "DELETE",
        f"/api/v12/assemblies/d/{document_id}/w/{workspace_id}"
        f"/e/{element_id}/instance/nodeid/{encoded}",
    )


def add_rigid_group(
    document_id: str,
    workspace_id: str,
    element_id: str,
    name: str,
    instance_ids: list[str],
) -> str:
    features = feature_list(document_id, workspace_id, element_id)
    proposal = rigid_group_feature(name, instance_ids)
    payload = {
        "btType": "BTFeatureDefinitionCall-1406",
        "feature": proposal,
        "libraryVersion": features["libraryVersion"],
        "serializationVersion": features["serializationVersion"],
        "sourceMicroversion": features["sourceMicroversion"],
        "rejectMicroversionSkew": False,
    }
    response = signed_request(
        "POST",
        f"/api/v12/assemblies/d/{document_id}/w/{workspace_id}"
        f"/e/{element_id}/features",
        json=payload,
    ).json()
    return response.get("feature", {}).get("featureId") or proposal["featureId"]


def create_instance(
    document_id: str,
    workspace_id: str,
    element_id: str,
    definition: dict,
) -> str:
    before_ids = {
        instance["id"]
        for instance in assembly(document_id, workspace_id, element_id)["rootAssembly"][
            "instances"
        ]
    }
    signed_request(
        "POST",
        f"/api/v12/assemblies/d/{document_id}/w/{workspace_id}"
        f"/e/{element_id}/instances",
        json=definition,
    )
    after_ids = {
        instance["id"]
        for instance in assembly(document_id, workspace_id, element_id)["rootAssembly"][
            "instances"
        ]
    }
    inserted_ids = sorted(after_ids - before_ids)
    if len(inserted_ids) != 1:
        raise RuntimeError(f"expected one inserted root instance, got {inserted_ids}")
    return inserted_ids[0]


def transform_instance(
    document_id: str,
    workspace_id: str,
    element_id: str,
    instance_id: str,
    transform: list[float],
) -> None:
    signed_request(
        "POST",
        f"/api/v12/assemblies/d/{document_id}/w/{workspace_id}"
        f"/e/{element_id}/occurrencetransforms",
        json={
            # A relative rigid motion preserves the internal transforms of a
            # nested vendor assembly. Absolute mode rewrites descendants.
            "isRelative": True,
            "occurrences": [{"path": [instance_id]}],
            "transform": transform,
        },
    )


def resolve_part_microversion(
    document_id: str, workspace_id: str, definition: dict
) -> None:
    if definition.get("isAssembly"):
        return
    element_id = definition["elementId"]
    part_id = definition["partId"]
    parts = signed_request(
        "GET",
        f"/api/v12/parts/d/{document_id}/w/{workspace_id}/e/{element_id}",
    ).json()
    matches = [part for part in parts if part.get("partId") == part_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one target part {part_id} in {element_id}, got {len(matches)}"
        )
    definition["microversionId"] = matches[0]["microversionId"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("element_id")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    replacements = spec["replacements"]
    delete_feature_ids = spec.get("deleteFeatureIds", [])
    before = assembly(args.document_id, args.workspace_id, args.element_id)
    before_features = feature_list(
        args.document_id, args.workspace_id, args.element_id
    )
    root_instances = before["rootAssembly"]["instances"]
    root_by_id = {instance["id"]: instance for instance in root_instances}
    old_ids = [replacement["oldInstanceId"] for replacement in replacements]
    missing = sorted(set(old_ids) - set(root_by_id))
    if missing:
        parser.error("old root instance IDs not found: " + ", ".join(missing))
    if len(set(old_ids)) != len(old_ids):
        parser.error("old root instance IDs must be unique")
    for replacement in replacements:
        transform = replacement["transform"]
        if len(transform) != 16:
            parser.error(
                f"replacement {replacement['oldInstanceId']} transform must have 16 values"
            )

    before_feature_ids = {
        feature["featureId"] for feature in before_features.get("features", [])
    }
    missing_features = sorted(set(delete_feature_ids) - before_feature_ids)
    if missing_features:
        parser.error("feature IDs not found: " + ", ".join(missing_features))

    audit = {
        "elementId": args.element_id,
        "rootInstanceCountBefore": len(root_instances),
        "oldInstanceIds": old_ids,
        "replacementCount": len(replacements),
        "deleteFeatureIds": delete_feature_ids,
        "rigidGroupName": spec["rigidGroupName"],
        "apply": args.apply,
    }
    if args.backup:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(
            json.dumps(
                {"assembly": before, "features": before_features, "spec": spec},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if not args.apply:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    instance_definitions = []
    for replacement in replacements:
        definition = dict(replacement["definition"])
        definition.setdefault("documentId", args.document_id)
        definition.setdefault("configuration", "default")
        definition.setdefault("isWholePartStudio", False)
        resolve_part_microversion(args.document_id, args.workspace_id, definition)
        instance_definitions.append(definition)

    inserted_ids = []
    current_old_id = "unknown"
    try:
        for definition, replacement in zip(
            instance_definitions, replacements, strict=True
        ):
            current_old_id = replacement["oldInstanceId"]
            instance_id = create_instance(
                args.document_id,
                args.workspace_id,
                args.element_id,
                definition,
            )
            inserted_ids.append(instance_id)
            transform_instance(
                args.document_id,
                args.workspace_id,
                args.element_id,
                instance_id,
                replacement["transform"],
            )
    except Exception as error:
        for instance_id in inserted_ids:
            delete_instance(
                args.document_id, args.workspace_id, args.element_id, instance_id
            )
        response = getattr(error, "response", None)
        detail = response.text if response is not None else str(error)
        raise SystemExit(
            f"Onshape rejected replacement for {current_old_id}: {detail}"
        ) from error

    unchanged_ids = [instance_id for instance_id in root_by_id if instance_id not in old_ids]
    final_ids = unchanged_ids + inserted_ids
    try:
        group_id = add_rigid_group(
            args.document_id,
            args.workspace_id,
            args.element_id,
            spec["rigidGroupName"],
            final_ids,
        )
    except Exception:
        for instance_id in inserted_ids:
            delete_instance(
                args.document_id, args.workspace_id, args.element_id, instance_id
            )
        raise

    for feature_id in delete_feature_ids:
        encoded_feature_id = quote(feature_id, safe="")
        signed_request(
            "DELETE",
            f"/api/v12/assemblies/d/{args.document_id}/w/{args.workspace_id}"
            f"/e/{args.element_id}/features/featureid/{encoded_feature_id}",
        )
    for instance_id in old_ids:
        delete_instance(args.document_id, args.workspace_id, args.element_id, instance_id)

    after = assembly(args.document_id, args.workspace_id, args.element_id)
    after_features = feature_list(args.document_id, args.workspace_id, args.element_id)
    after_instances = after["rootAssembly"]["instances"]
    states = {
        feature_id: state.get("featureStatus")
        for feature_id, state in after_features.get("featureStates", {}).items()
    }
    audit.update(
        {
            "insertedInstanceIds": inserted_ids,
            "replacementMap": dict(zip(old_ids, inserted_ids, strict=True)),
            "rigidGroupFeatureId": group_id,
            "rigidGroupFeatureState": states.get(group_id),
            "featureStatesAfter": states,
            "rootInstanceCountAfter": len(after_instances),
            "rootInstancesAfter": [
                {
                    key: instance.get(key)
                    for key in ("id", "name", "type", "elementId", "partId")
                }
                for instance in after_instances
            ],
        }
    )
    if len(after_instances) != len(root_instances):
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    if states.get(group_id) != "OK":
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
