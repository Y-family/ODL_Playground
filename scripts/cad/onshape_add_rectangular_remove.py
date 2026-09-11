#!/usr/bin/env python3
"""Add an audited rectangular sketch and blind remove to an Onshape part."""

from __future__ import annotations

import argparse
import base64
import json
import math
import secrets
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

from onshape_add_extrude_remove import parameter, unwrap
from onshape_import_step import signed_request


def node_id() -> str:
    return "M" + base64.b64encode(secrets.token_bytes(12)).decode("ascii")


def line(entity_id: str, start: tuple[float, float], end: tuple[float, float], index: int) -> dict:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    return {
        "btType": "BTMSketchCurveSegment-155",
        "geometry": {
            "btType": "BTCurveGeometryLine-117",
            "pntX": (start[0] + end[0]) / 2,
            "pntY": (start[1] + end[1]) / 2,
            "dirX": dx / length,
            "dirY": dy / length,
        },
        "isConstruction": False,
        "isFromSplineControlPolygon": False,
        "startPointId": f"{entity_id}.start",
        "endPointId": f"{entity_id}.end",
        "startParam": -length / 2,
        "endParam": length / 2,
        "offsetCurveExtensions": [],
        "centerId": "",
        "isFromSplineHandle": False,
        "isFromEndpointSplineHandle": False,
        "internalIds": [],
        "curvedTextIds": [],
        "namespace": "",
        "name": "",
        "index": index,
        "parameters": [],
        "nodeId": node_id(),
        "entityId": entity_id,
    }


def individual_query(deterministic_id: str) -> dict:
    return {
        "btType": "BTMIndividualQuery-138",
        "queryStatement": None,
        "queryString": "",
        "nodeId": node_id(),
        "deterministicIds": [deterministic_id],
    }


def delete_feature(path: str, feature_id: str) -> None:
    signed_request("DELETE", f"{path}/featureid/{quote(feature_id, safe='')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("element_id")
    parser.add_argument("name")
    parser.add_argument("plane_face_id")
    parser.add_argument("part_id")
    parser.add_argument("x_min_mm", type=float)
    parser.add_argument("x_max_mm", type=float)
    parser.add_argument("y_min_mm", type=float)
    parser.add_argument("y_max_mm", type=float)
    parser.add_argument("depth")
    parser.add_argument("--opposite-direction", action="store_true")
    parser.add_argument("--template-sketch", default="FG2w5FVJzcSOLPY_25")
    parser.add_argument("--template-extrude", default="FvGzZfrRvP0Emyi_3")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.x_min_mm >= args.x_max_mm or args.y_min_mm >= args.y_max_mm:
        parser.error("rectangle minimums must be smaller than maximums")

    load_dotenv()
    features_path = (
        f"/api/v12/partstudios/d/{args.document_id}/w/{args.workspace_id}"
        f"/e/{args.element_id}/features"
    )
    template_list = signed_request(
        "GET",
        features_path,
        params=[
            ("featureId", args.template_sketch),
            ("featureId", args.template_extrude),
            ("noSketchGeometry", "false"),
        ],
    ).json()
    templates = {unwrap(f)["featureId"]: unwrap(f) for f in template_list["features"]}
    if set(templates) != {args.template_sketch, args.template_extrude}:
        parser.error("could not load sketch and extrude templates")

    if args.backup:
        complete = signed_request(
            "GET", features_path, params=[("noSketchGeometry", "true")]
        ).json()
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(
            json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    sketch_id = node_id()
    sketch = templates[args.template_sketch]
    sketch["featureId"] = sketch_id
    sketch["nodeId"] = node_id()
    sketch["name"] = f"{args.name} profile"
    sketch["suppressed"] = False
    sketch["constraints"] = []
    x0, x1 = args.x_min_mm / 1000, args.x_max_mm / 1000
    y0, y1 = args.y_min_mm / 1000, args.y_max_mm / 1000
    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    sketch["entities"] = [
        line(node_id(), points[index], points[(index + 1) % 4], index + 1)
        for index in range(4)
    ]
    parameter(sketch, "sketchPlane")["queries"] = [individual_query(args.plane_face_id)]

    proposal = {
        "name": args.name,
        "planeFaceId": args.plane_face_id,
        "partId": args.part_id,
        "rectangleMm": [args.x_min_mm, args.x_max_mm, args.y_min_mm, args.y_max_mm],
        "depth": args.depth.replace("_", " "),
        "oppositeDirection": args.opposite_direction,
        "sketchFeatureId": sketch_id,
        "apply": args.apply,
    }
    if not args.apply:
        print(json.dumps(proposal, ensure_ascii=False, indent=2))
        return

    base_payload = {
        "libraryVersion": template_list["libraryVersion"],
        "serializationVersion": template_list["serializationVersion"],
        "sourceMicroversion": template_list["sourceMicroversion"],
        "rejectMicroversionSkew": False,
    }
    created: list[str] = []
    try:
        payload = {"btType": "BTFeatureDefinitionCall-1406", "feature": sketch, **base_payload}
        added = signed_request("POST", features_path, json=payload).json()
        sketch_id = added.get("feature", {}).get("featureId") or sketch_id
        created.append(sketch_id)

        extrude_id = node_id()
        extrude = templates[args.template_extrude]
        extrude["featureId"] = extrude_id
        extrude["nodeId"] = node_id()
        extrude["name"] = args.name
        extrude["suppressed"] = False
        parameter(extrude, "operationType")["value"] = "REMOVE"
        parameter(extrude, "entities")["queries"] = [
            {
                "btType": "BTMIndividualSketchRegionQuery-140",
                "queryStatement": None,
                "filterInnerLoops": True,
                "queryString": f'query = qSketchRegion(id + "{sketch_id}", true);',
                "featureId": sketch_id,
                "nodeId": node_id(),
                "deterministicIds": [],
            }
        ]
        parameter(extrude, "endBound")["value"] = "BLIND"
        parameter(extrude, "depth")["expression"] = args.depth.replace("_", " ")
        parameter(extrude, "oppositeDirection")["value"] = args.opposite_direction
        parameter(extrude, "symmetric")["value"] = False
        parameter(extrude, "hasSecondDirection")["value"] = False
        parameter(extrude, "defaultScope")["value"] = False
        parameter(extrude, "booleanScope")["queries"] = [individual_query(args.part_id)]
        payload = {"btType": "BTFeatureDefinitionCall-1406", "feature": extrude, **base_payload}
        added = signed_request("POST", features_path, json=payload).json()
        extrude_id = added.get("feature", {}).get("featureId") or extrude_id
        created.append(extrude_id)

        states = signed_request(
            "GET", features_path, params=[("featureId", extrude_id), ("noSketchGeometry", "true")]
        ).json().get("featureStates", {})
        state = states.get(extrude_id, {}).get("featureStatus")
        proposal.update({"sketchFeatureId": sketch_id, "extrudeFeatureId": extrude_id, "featureState": state})
        if state != "OK":
            raise RuntimeError(json.dumps(proposal, ensure_ascii=False))
    except Exception as error:
        for feature_id in reversed(created):
            try:
                delete_feature(features_path, feature_id)
            except Exception:
                pass
        response = getattr(error, "response", None)
        detail = response.text if response is not None else str(error)
        raise SystemExit(f"Onshape rejected rectangular remove: {detail}") from error

    print(json.dumps(proposal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
