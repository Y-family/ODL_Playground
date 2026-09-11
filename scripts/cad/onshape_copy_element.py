#!/usr/bin/env python3
"""Copy one Onshape element into a target workspace with an explicit audit."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from onshape_import_step import signed_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_document_id")
    parser.add_argument("target_workspace_id")
    parser.add_argument("source_document_id")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-workspace-id")
    source.add_argument("--source-version-id")
    parser.add_argument("source_element_id")
    parser.add_argument("--anchor-element-id")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    payload = {
        "documentIdSource": args.source_document_id,
        "elementIdSource": args.source_element_id,
        "isGroupAnchor": False,
    }
    payload["workspaceIdSource"] = (
        args.source_workspace_id or args.target_workspace_id
    )
    if args.source_version_id:
        payload["versionIdSource"] = args.source_version_id
    if args.anchor_element_id:
        payload["anchorElementId"] = args.anchor_element_id

    audit = {
        "targetDocumentId": args.target_document_id,
        "targetWorkspaceId": args.target_workspace_id,
        "sourceDocumentId": args.source_document_id,
        "sourceWorkspaceId": args.source_workspace_id,
        "sourceVersionId": args.source_version_id,
        "sourceElementId": args.source_element_id,
        "apply": args.apply,
    }
    if not args.apply:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    path = (
        f"/api/v12/elements/copyelement/{args.target_document_id}"
        f"/workspace/{args.target_workspace_id}"
    )
    try:
        copied = signed_request("POST", path, json=payload).json()
    except Exception as error:
        response = getattr(error, "response", None)
        detail = response.text if response is not None else str(error)
        raise SystemExit(f"Onshape rejected element copy: {detail}") from error

    audit.update(
        {
            "copiedElementId": copied.get("id") or copied.get("elementId"),
            "copiedElementName": copied.get("name"),
            "copiedElementType": copied.get("elementType"),
        }
    )
    if not audit["copiedElementId"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
