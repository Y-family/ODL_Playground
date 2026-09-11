#!/usr/bin/env python3
"""Add audited, parameterless Feature Studio features to one Part Studio."""

from __future__ import annotations

import argparse
import base64
import json
import secrets
from urllib.parse import quote

from dotenv import load_dotenv

from onshape_import_step import signed_request


def node_id() -> str:
    return "M" + base64.b64encode(secrets.token_bytes(12)).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("part_studio_id")
    parser.add_argument("feature_studio_id")
    parser.add_argument(
        "features",
        nargs="+",
        help="Feature definitions as featureType=name.",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    requested = []
    for value in args.features:
        if "=" not in value:
            parser.error(f"feature must be featureType=name: {value}")
        feature_type, name = value.split("=", 1)
        requested.append((feature_type, name))

    specs_path = (
        f"/api/v12/featurestudios/d/{args.document_id}/w/{args.workspace_id}"
        f"/e/{args.feature_studio_id}/featurespecs"
    )
    specs_response = signed_request("GET", specs_path).json()
    specs = {
        spec.get("message", spec)["featureType"]: spec.get("message", spec)
        for spec in specs_response.get("featureSpecs", [])
    }
    missing = sorted({feature_type for feature_type, _ in requested} - set(specs))
    if missing:
        parser.error("Feature Studio lacks feature types: " + ", ".join(missing))

    features_path = (
        f"/api/v12/partstudios/d/{args.document_id}/w/{args.workspace_id}"
        f"/e/{args.part_studio_id}/features"
    )
    before = signed_request(
        "GET", features_path, params={"noSketchGeometry": "true"}
    ).json()
    proposal = {
        "partStudioId": args.part_studio_id,
        "featureStudioId": args.feature_studio_id,
        "features": [
            {"featureType": feature_type, "name": name}
            for feature_type, name in requested
        ],
        "apply": args.apply,
    }
    if not args.apply:
        print(json.dumps(proposal, ensure_ascii=False, indent=2))
        return

    added_ids: list[str] = []
    try:
        for feature_type, name in requested:
            feature_id = node_id()
            feature = {
                "btType": "BTMFeature-134",
                "namespace": specs[feature_type]["namespace"],
                "name": name,
                "suppressed": False,
                "parameters": [],
                "featureId": feature_id,
                "nodeId": feature_id,
                "featureType": feature_type,
                "returnAfterSubfeatures": False,
                "subFeatures": [],
                "parameterLibraries": [],
                "suppressionState": None,
            }
            payload = {
                "btType": "BTFeatureDefinitionCall-1406",
                "feature": feature,
                "libraryVersion": before["libraryVersion"],
                "serializationVersion": before["serializationVersion"],
                "sourceMicroversion": before["sourceMicroversion"],
                "rejectMicroversionSkew": False,
            }
            response = signed_request("POST", features_path, json=payload).json()
            added_id = (
                response.get("feature", {}).get("featureId") or feature_id
            )
            added_ids.append(added_id)
            before = signed_request(
                "GET", features_path, params={"noSketchGeometry": "true"}
            ).json()

        states = {
            feature_id: before.get("featureStates", {})
            .get(feature_id, {})
            .get("featureStatus")
            for feature_id in added_ids
        }
        if any(state != "OK" for state in states.values()):
            raise RuntimeError(f"custom feature states are not OK: {states}")
    except Exception:
        for feature_id in reversed(added_ids):
            encoded = quote(feature_id, safe="")
            signed_request("DELETE", f"{features_path}/featureid/{encoded}")
        raise

    proposal.update({"addedFeatureIds": added_ids, "featureStates": states})
    print(json.dumps(proposal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
