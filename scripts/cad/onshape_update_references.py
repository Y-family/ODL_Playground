#!/usr/bin/env python3
"""Replace selected Onshape element references without changing instance IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from onshape_import_step import signed_request


def assembly(document_id: str, workspace_id: str, element_id: str) -> dict:
    return signed_request(
        "GET",
        f"/api/v12/assemblies/d/{document_id}/w/{workspace_id}/e/{element_id}",
        params={"includeMateFeatures": "true", "includeNonSolids": "true"},
    ).json()


def feature_list(document_id: str, workspace_id: str, element_id: str) -> dict:
    return signed_request(
        "GET",
        f"/api/v12/assemblies/d/{document_id}/w/{workspace_id}"
        f"/e/{element_id}/features",
    ).json()


def reference(
    document_id: str,
    workspace_id: str,
    element_id: str,
    element_type: str,
) -> dict:
    result = {
        "documentId": document_id,
        "workspaceId": workspace_id,
        "elementId": element_id,
        "elementType": element_type,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("target_element_id")
    parser.add_argument("from_element_id")
    parser.add_argument("to_element_id")
    parser.add_argument("instance_ids", nargs="+")
    parser.add_argument(
        "--element-type",
        choices=("ASSEMBLY", "PARTSTUDIO"),
        default="ASSEMBLY",
    )
    parser.add_argument("--description", default="Replace assembly reference")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    before = assembly(
        args.document_id, args.workspace_id, args.target_element_id
    )
    before_features = feature_list(
        args.document_id, args.workspace_id, args.target_element_id
    )
    instances = before["rootAssembly"]["instances"]
    by_id = {instance["id"]: instance for instance in instances}
    missing = sorted(set(args.instance_ids) - set(by_id))
    if missing:
        parser.error("root instance IDs not found: " + ", ".join(missing))
    wrong_source = [
        instance_id
        for instance_id in args.instance_ids
        if by_id[instance_id].get("elementId") != args.from_element_id
    ]
    if wrong_source:
        parser.error("instances do not use from_element_id: " + ", ".join(wrong_source))

    if args.backup:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(
            json.dumps(
                {"assembly": before, "features": before_features},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    result = {
        "targetElementId": args.target_element_id,
        "fromElementId": args.from_element_id,
        "toElementId": args.to_element_id,
        "instanceIds": args.instance_ids,
        "rootInstanceCountBefore": len(instances),
        "apply": args.apply,
    }
    if not args.apply:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    path = (
        f"/api/v12/elements/d/{args.document_id}/w/{args.workspace_id}"
        f"/e/{args.target_element_id}/updatereferences"
    )
    payload = {
        "editDescription": args.description,
        "referenceUpdates": [
            {
                "toReference": reference(
                    args.document_id,
                    args.workspace_id,
                    args.to_element_id,
                    args.element_type,
                ),
                "idsToUpdate": args.instance_ids,
                "ignoreChildren": False,
            }
        ],
    }
    try:
        signed_request("POST", path, json=payload)
    except Exception as error:
        response = getattr(error, "response", None)
        detail = response.text if response is not None else str(error)
        raise SystemExit(f"Onshape rejected reference update: {detail}") from error

    after = assembly(args.document_id, args.workspace_id, args.target_element_id)
    after_features = feature_list(
        args.document_id, args.workspace_id, args.target_element_id
    )
    after_instances = after["rootAssembly"]["instances"]
    after_by_id = {instance["id"]: instance for instance in after_instances}
    preserved = all(
        instance_id in after_by_id
        and after_by_id[instance_id].get("elementId") == args.to_element_id
        for instance_id in args.instance_ids
    )
    states = {
        feature_id: state.get("featureStatus")
        for feature_id, state in after_features.get("featureStates", {}).items()
    }
    result.update(
        {
            "rootInstanceCountAfter": len(after_instances),
            "instanceIdsPreserved": preserved,
            "featureStatesAfter": states,
        }
    )
    if len(after_instances) != len(instances) or not preserved:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
