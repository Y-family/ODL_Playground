#!/usr/bin/env python3
"""Replace failed assembly mates with one explicit rigid group.

This utility only edits assembly constraints. It does not create parts or
instances. By default it prints the proposed change; ``--apply`` is required
for any Onshape mutation.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

from onshape_import_step import signed_request


def node_id() -> str:
    return "M" + base64.b64encode(secrets.token_bytes(12)).decode("ascii")


def feature_list(document_id: str, workspace_id: str, element_id: str) -> dict:
    return signed_request(
        "GET",
        f"/api/v12/assemblies/d/{document_id}/w/{workspace_id}"
        f"/e/{element_id}/features",
    ).json()


def assembly(document_id: str, workspace_id: str, element_id: str) -> dict:
    return signed_request(
        "GET",
        f"/api/v12/assemblies/d/{document_id}/w/{workspace_id}"
        f"/e/{element_id}",
        params={"includeMateFeatures": "true", "includeNonSolids": "true"},
    ).json()


def rigid_group_feature(name: str, instance_ids: list[str]) -> dict:
    queries = [
        {
            "btType": "BTMIndividualOccurrenceQuery-626",
            "path": [instance_id],
            "nodeId": node_id(),
        }
        for instance_id in instance_ids
    ]
    feature_id = node_id()
    return {
        "btType": "BTMMateGroup-65",
        "suppressionState": None,
        "parameterLibraries": [],
        "returnAfterSubfeatures": False,
        "subFeatures": [],
        "namespace": "",
        "version": 2,
        "name": name,
        "parameters": [
            {
                "btType": "BTMParameterQueryWithOccurrenceList-67",
                "libraryRelationType": "DEFAULT",
                "queries": queries,
                "nodeId": node_id(),
                "parameterId": "occurrencesQuery",
                "parameterName": "",
            }
        ],
        "nodeId": feature_id,
        "featureId": feature_id,
        "suppressed": False,
        "featureType": "mateGroup",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("element_id")
    parser.add_argument("name")
    parser.add_argument(
        "--instances",
        nargs="+",
        help="Root instance IDs to group. Defaults to every root instance.",
    )
    parser.add_argument(
        "--delete-feature",
        action="append",
        default=[],
        help="Feature ID to delete after the new group is accepted.",
    )
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    before_assembly = assembly(args.document_id, args.workspace_id, args.element_id)
    before_features = feature_list(
        args.document_id, args.workspace_id, args.element_id
    )
    root_instances = before_assembly["rootAssembly"]["instances"]
    available = {instance["id"] for instance in root_instances}
    instance_ids = args.instances or [instance["id"] for instance in root_instances]
    missing_instances = sorted(set(instance_ids) - available)
    if missing_instances:
        parser.error("root instance IDs not found: " + ", ".join(missing_instances))

    feature_by_id = {
        feature["featureId"]: feature for feature in before_features.get("features", [])
    }
    missing_features = sorted(set(args.delete_feature) - set(feature_by_id))
    if missing_features:
        parser.error("feature IDs not found: " + ", ".join(missing_features))

    proposal = rigid_group_feature(args.name, instance_ids)
    result = {
        "elementId": args.element_id,
        "rootInstanceCountBefore": len(root_instances),
        "groupedInstanceIds": instance_ids,
        "deleteFeatureIds": args.delete_feature,
        "apply": args.apply,
    }
    if args.backup:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(
            json.dumps(
                {"assembly": before_assembly, "features": before_features},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if not args.apply:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    payload = {
        "btType": "BTFeatureDefinitionCall-1406",
        "feature": proposal,
        "libraryVersion": before_features["libraryVersion"],
        "serializationVersion": before_features["serializationVersion"],
        "sourceMicroversion": before_features["sourceMicroversion"],
        "rejectMicroversionSkew": False,
    }
    add_path = (
        f"/api/v12/assemblies/d/{args.document_id}/w/{args.workspace_id}"
        f"/e/{args.element_id}/features"
    )
    try:
        added = signed_request("POST", add_path, json=payload).json()
    except Exception as error:
        response = getattr(error, "response", None)
        detail = response.text if response is not None else str(error)
        raise SystemExit(f"Onshape rejected rigid-group feature: {detail}") from error
    added_feature = added.get("feature", {})
    added_id = added_feature.get("featureId") or proposal["featureId"]

    for feature_id in args.delete_feature:
        encoded_feature_id = quote(feature_id, safe="")
        signed_request(
            "DELETE",
            f"/api/v12/assemblies/d/{args.document_id}/w/{args.workspace_id}"
            f"/e/{args.element_id}/features/featureid/{encoded_feature_id}",
        )

    after_assembly = assembly(args.document_id, args.workspace_id, args.element_id)
    after_features = feature_list(args.document_id, args.workspace_id, args.element_id)
    states = {
        feature_id: state.get("featureStatus")
        for feature_id, state in after_features.get("featureStates", {}).items()
    }
    result.update(
        {
            "addedFeatureId": added_id,
            "addedFeatureState": states.get(added_id),
            "featureStatesAfter": states,
            "rootInstanceCountAfter": len(
                after_assembly["rootAssembly"]["instances"]
            ),
        }
    )
    if result["rootInstanceCountAfter"] != result["rootInstanceCountBefore"]:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    if result["addedFeatureState"] != "OK":
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
