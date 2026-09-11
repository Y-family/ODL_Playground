#!/usr/bin/env python3
"""Update selected scalar parameters on an existing Onshape feature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from onshape_import_step import signed_request


def unwrap(value: dict) -> dict:
    return value.get("message", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("element_id")
    parser.add_argument("feature_id")
    parser.add_argument(
        "updates",
        nargs="+",
        metavar="PARAMETER_ID=VALUE",
        help="Quantity values use underscores for spaces, for example 3.2_mm",
    )
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    requested = {}
    for update in args.updates:
        parameter_id, separator, value = update.partition("=")
        if not separator:
            parser.error(f"invalid update: {update}")
        requested[parameter_id] = value.replace("_", " ")

    feature_list_path = (
        f"/api/v12/partstudios/d/{args.document_id}/w/{args.workspace_id}"
        f"/e/{args.element_id}/features"
    )
    feature_list = signed_request(
        "GET",
        feature_list_path,
        params=[("featureId", args.feature_id), ("noSketchGeometry", "true")],
    ).json()
    features = feature_list.get("features", [])
    if len(features) != 1:
        parser.error(f"expected one feature, received {len(features)}")
    feature = unwrap(features[0])

    if args.backup:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(
            json.dumps(feature_list, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    found = set()
    changes = []
    for parameter in feature.get("parameters", []):
        message = unwrap(parameter)
        parameter_id = message.get("parameterId")
        if parameter_id not in requested:
            continue
        if "expression" in message:
            field = "expression"
            after: object = requested[parameter_id]
        elif "value" in message and isinstance(message["value"], bool):
            field = "value"
            normalized = requested[parameter_id].lower()
            if normalized not in {"true", "false"}:
                parser.error(f"{parameter_id} expects true or false")
            after = normalized == "true"
        elif "value" in message:
            field = "value"
            after = requested[parameter_id]
        else:
            parser.error(f"unsupported parameter type for {parameter_id}")
        before = message.get(field)
        message[field] = after
        changes.append(
            {
                "parameter": parameter_id,
                "field": field,
                "before": before,
                "after": after,
            }
        )
        found.add(parameter_id)

    missing = sorted(set(requested) - found)
    if missing:
        parser.error("parameter IDs not found: " + ", ".join(missing))

    result = {
        "featureId": args.feature_id,
        "featureName": feature.get("name"),
        "changes": changes,
        "applied": args.apply,
    }
    if args.apply:
        response = signed_request(
            "POST",
            (
                f"/api/v12/partstudios/d/{args.document_id}/w/{args.workspace_id}"
                f"/e/{args.element_id}/features/featureid/{args.feature_id}"
            ),
            json={
                "feature": feature,
                "serializationVersion": feature_list["serializationVersion"],
                "sourceMicroversion": feature_list["sourceMicroversion"],
                "libraryVersion": feature_list["libraryVersion"],
            },
        ).json()
        result["response"] = {
            key: response.get(key)
            for key in ("featureState", "sourceMicroversion", "serializationVersion")
            if key in response
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
