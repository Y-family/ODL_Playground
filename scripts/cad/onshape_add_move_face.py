#!/usr/bin/env python3
"""Add one audited offset-face feature to an existing Onshape Part Studio.

The operation changes an existing part in place. It does not create a body,
part, or assembly instance. By default the script only prints the proposal;
``--apply`` is required for an Onshape mutation.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("element_id")
    parser.add_argument("name")
    parser.add_argument(
        "face_ids",
        nargs="+",
        help="One or more deterministic Onshape face IDs moved as one feature.",
    )
    parser.add_argument("distance", help="Quantity expression, for example 8.1_mm")
    parser.add_argument(
        "--same-direction",
        action="store_true",
        help="Move along the selected face direction instead of opposite to it.",
    )
    parser.add_argument("--template-feature", default="F1G6w9OxKyzEIY6_8")
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
    if feature.get("featureType") != "moveFace":
        parser.error("template feature is not a moveFace feature")

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

    parameter(feature, "moveFaceType")["value"] = "OFFSET"
    parameter(feature, "offsetDistance")["expression"] = args.distance.replace(
        "_", " "
    )
    parameter(feature, "oppositeDirection")["value"] = not args.same_direction
    move_faces = parameter(feature, "moveFaces")
    move_faces["queries"] = [
        {
            "btType": "BTMIndividualQuery-138",
            "queryStatement": None,
            "queryString": "",
            "nodeId": node_id(),
            "deterministicIds": [face_id],
        }
        for face_id in args.face_ids
    ]

    proposal = {
        "featureId": feature_id,
        "name": args.name,
        "faceIds": args.face_ids,
        "distance": parameter(feature, "offsetDistance")["expression"],
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
            "GET", features_path, params=[("featureId", added_id), ("noSketchGeometry", "true")]
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
        raise SystemExit(f"Onshape rejected move-face feature: {detail}") from error
    print(json.dumps(proposal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
