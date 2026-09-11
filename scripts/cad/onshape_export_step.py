#!/usr/bin/env python3
"""Export an Onshape assembly as STEP and download the result."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

from onshape_import_step import signed_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("element_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    load_dotenv()
    path = (
        f"/api/v12/assemblies/d/{args.document_id}/w/{args.workspace_id}"
        f"/e/{args.element_id}/translations"
    )
    translation = signed_request(
        "POST",
        path,
        json={
            "formatName": "STEP",
            "flatten": False,
            "includeExportIds": True,
            "storeInDocument": False,
            "triggerAutoDownload": False,
            "unit": "millimeter",
        },
    ).json()

    for _ in range(120):
        translation = signed_request(
            "GET", f"/api/v12/translations/{translation['id']}"
        ).json()
        state = translation["requestState"]
        if state == "DONE":
            break
        if state == "FAILED":
            parser.error(str(translation.get("failureReason")))
        time.sleep(1)
    else:
        parser.error("translation did not finish within 120 seconds")

    external_ids = translation.get("resultExternalDataIds") or []
    if len(external_ids) != 1:
        parser.error(f"expected one exported file, received {len(external_ids)}")
    response = signed_request(
        "GET",
        f"/api/v12/documents/d/{args.document_id}/externaldata/{external_ids[0]}",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(response.content)
    print(f"output={args.output} bytes={len(response.content)}")


if __name__ == "__main__":
    main()
