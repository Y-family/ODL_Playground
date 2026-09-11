#!/usr/bin/env python3
"""Update named dimensional constraints in an existing Onshape sketch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from onshape_import_step import signed_request


def unwrap(value: dict) -> dict:
    return value.get("message", value)


def dimension_parameter(constraint: dict) -> dict:
    for parameter in unwrap(constraint).get("parameters", []):
        message = unwrap(parameter)
        if message.get("parameterId") in {"length", "angle"}:
            return message
    raise ValueError(
        f"constraint {unwrap(constraint).get('entityId')} has no dimension parameter"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("element_id")
    parser.add_argument("feature_id")
    parser.add_argument(
        "updates",
        nargs="+",
        metavar="CONSTRAINT_ID=EXPRESSION",
        help="For example: EA71vwCibh8K=40.5_mm",
    )
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    requested = {}
    for update in args.updates:
        constraint_id, separator, expression = update.partition("=")
        if not separator:
            parser.error(f"invalid update: {update}")
        requested[constraint_id] = expression.replace("_", " ")

    feature_list_path = (
        f"/api/v12/partstudios/d/{args.document_id}/w/{args.workspace_id}"
        f"/e/{args.element_id}/features"
    )
    feature_list = signed_request(
        "GET",
        feature_list_path,
        params=[("featureId", args.feature_id), ("noSketchGeometry", "false")],
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
    for constraint in feature.get("constraints", []):
        message = unwrap(constraint)
        constraint_id = message.get("entityId")
        if constraint_id not in requested:
            continue
        parameter = dimension_parameter(constraint)
        before = parameter.get("expression")
        after = requested[constraint_id]
        parameter["expression"] = after
        changes.append({"constraint": constraint_id, "before": before, "after": after})
        found.add(constraint_id)

    missing = sorted(set(requested) - found)
    if missing:
        parser.error("constraint IDs not found: " + ", ".join(missing))

    result = {
        "featureId": args.feature_id,
        "featureName": feature.get("name"),
        "changes": changes,
        "applied": args.apply,
    }
    if args.apply:
        payload = {
            "feature": feature,
            "serializationVersion": feature_list["serializationVersion"],
            "sourceMicroversion": feature_list["sourceMicroversion"],
            "libraryVersion": feature_list["libraryVersion"],
        }
        response = signed_request(
            "POST",
            (
                f"/api/v12/partstudios/d/{args.document_id}/w/{args.workspace_id}"
                f"/e/{args.element_id}/features/featureid/{args.feature_id}"
            ),
            json=payload,
        ).json()
        result["response"] = {
            key: response.get(key)
            for key in (
                "featureState",
                "featureStatus",
                "sourceMicroversion",
                "serializationVersion",
            )
            if key in response
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
