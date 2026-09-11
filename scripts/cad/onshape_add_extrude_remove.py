#!/usr/bin/env python3
"""Add one audited blind extrude-remove feature to an Onshape Part Studio.

The selected existing planar face supplies the removal profile. The operation
only subtracts from the explicitly selected existing part and creates no part
or assembly instance. By default this script prints the proposal; ``--apply``
is required for an Onshape mutation.
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


def unwrap(value: dict) -> dict:
    return value.get("message", value)


def parameter(feature: dict, parameter_id: str) -> dict:
    for item in feature.get("parameters", []):
        message = unwrap(item)
        if message.get("parameterId") == parameter_id:
            return message
    raise KeyError(parameter_id)


def individual_query(deterministic_id: str) -> dict:
    return {
        "btType": "BTMIndividualQuery-138",
        "queryStatement": None,
        "queryString": "",
        "nodeId": node_id(),
        "deterministicIds": [deterministic_id],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("element_id")
    parser.add_argument("name")
    parser.add_argument("face_id")
    parser.add_argument("part_id")
    parser.add_argument("depth", help="Quantity expression, for example 8.6_mm")
    parser.add_argument(
        "--opposite-direction",
        action="store_true",
        help="Reverse the selected planar face's default extrusion direction.",
    )
    parser.add_argument("--template-feature", default="FvGzZfrRvP0Emyi_3")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    features_path = (
        f"/api/v12/partstudios/d/{args.document_id}/w/{args.workspace_id}"
        f"/e/{args.element_id}/features"
    )
    feature_list = signed_request(
        "GET",
        features_path,
        params=[("featureId", args.template_feature), ("noSketchGeometry", "true")],
    ).json()
    templates = feature_list.get("features", [])
    if len(templates) != 1:
        parser.error(f"expected one template feature, received {len(templates)}")
    feature = unwrap(templates[0])
    if feature.get("featureType") != "extrude":
        parser.error("template feature is not an extrude feature")

    if args.backup:
        complete = signed_request(
            "GET", features_path, params=[("noSketchGeometry", "true")]
        ).json()
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(
            json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    feature_id = node_id()
    feature["featureId"] = feature_id
    feature["nodeId"] = node_id()
    feature["name"] = args.name
    feature["suppressed"] = False

    parameter(feature, "operationType")["value"] = "REMOVE"
    parameter(feature, "entities")["queries"] = [individual_query(args.face_id)]
    parameter(feature, "endBound")["value"] = "BLIND"
    parameter(feature, "depth")["expression"] = args.depth.replace("_", " ")
    parameter(feature, "oppositeDirection")["value"] = args.opposite_direction
    parameter(feature, "symmetric")["value"] = False
    parameter(feature, "hasSecondDirection")["value"] = False
    parameter(feature, "defaultScope")["value"] = False
    parameter(feature, "booleanScope")["queries"] = [
        individual_query(args.part_id)
    ]

    proposal = {
        "featureId": feature_id,
        "name": args.name,
        "faceId": args.face_id,
        "partId": args.part_id,
        "depth": parameter(feature, "depth")["expression"],
        "oppositeDirection": parameter(feature, "oppositeDirection")["value"],
        "apply": args.apply,
    }
    if not args.apply:
        print(json.dumps(proposal, ensure_ascii=False, indent=2))
        return

    payload = {
        "btType": "BTFeatureDefinitionCall-1406",
        "feature": feature,
        "libraryVersion": feature_list["libraryVersion"],
        "serializationVersion": feature_list["serializationVersion"],
        "sourceMicroversion": feature_list["sourceMicroversion"],
        "rejectMicroversionSkew": False,
    }
    try:
        response = signed_request("POST", features_path, json=payload).json()
        added = response.get("feature", {})
        added_id = added.get("featureId") or feature_id
        after = signed_request(
            "GET",
            features_path,
            params=[("featureId", added_id), ("noSketchGeometry", "true")],
        ).json()
        state = after.get("featureStates", {}).get(added_id, {}).get("featureStatus")
        proposal.update({"addedFeatureId": added_id, "featureState": state})
        if state != "OK":
            encoded = quote(added_id, safe="")
            signed_request("DELETE", f"{features_path}/featureid/{encoded}")
            raise SystemExit(json.dumps(proposal, ensure_ascii=False, indent=2))
    except Exception as error:
        response = getattr(error, "response", None)
        detail = response.text if response is not None else str(error)
        raise SystemExit(f"Onshape rejected extrude-remove feature: {detail}") from error
    print(json.dumps(proposal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
