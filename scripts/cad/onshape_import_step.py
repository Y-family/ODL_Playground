#!/usr/bin/env python3
"""Import a STEP file into an existing Onshape workspace and wait for it."""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv


def signed_request(
    method: str,
    path: str,
    *,
    api_base: str | None = None,
    follow_redirects: bool = True,
    **kwargs: object,
) -> requests.Response:
    base_url = (api_base or os.environ["ONSHAPE_API"]).rstrip("/")
    access_key = os.environ["ONSHAPE_ACCESS_KEY"]
    secret_key = os.environ["ONSHAPE_SECRET_KEY"].encode()
    request = requests.Request(method, base_url + path, **kwargs).prepare()
    content_type = request.headers.get("Content-Type", "application/json")
    query = urlsplit(request.url).query
    date = datetime.datetime.now(datetime.UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
    nonce = secrets.token_urlsafe(20)
    signature_input = (
        (f"{method}\n{nonce}\n{date}\n{content_type}\n{path}\n{query}\n")
        .lower()
        .encode()
    )
    signature = base64.b64encode(
        hmac.new(secret_key, signature_input, hashlib.sha256).digest()
    ).decode()
    request.headers.update(
        {
            "Accept": "application/json",
            "Authorization": f"On {access_key}:HmacSHA256:{signature}",
            "Content-Type": content_type,
            "Date": date,
            "On-Nonce": nonce,
            "User-Agent": "Open-Duck-Mini-ZP10D-conversion",
        }
    )
    response = requests.Session().send(
        request, timeout=120, allow_redirects=follow_redirects
    )
    response.raise_for_status()
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("workspace_id")
    parser.add_argument("step", type=Path)
    args = parser.parse_args()

    load_dotenv()
    missing = [
        key
        for key in ("ONSHAPE_API", "ONSHAPE_ACCESS_KEY", "ONSHAPE_SECRET_KEY")
        if not os.getenv(key)
    ]
    if missing:
        parser.error("missing environment variables: " + ", ".join(missing))
    if not args.step.is_file():
        parser.error(f"STEP file does not exist: {args.step}")

    path = f"/api/v6/translations/d/{args.document_id}/w/{args.workspace_id}"
    with args.step.open("rb") as step_file:
        response = signed_request(
            "POST",
            path,
            data={
                "allowFaultyParts": "false",
                "flattenAssemblies": "false",
                "storeInDocument": "true",
                "translate": "true",
                "unit": "millimeter",
            },
            files={"file": (args.step.name, step_file, "application/step")},
        )
    translation = response.json()
    translation_id = translation["id"]
    print(f"translation_id={translation_id}")

    for _ in range(120):
        translation = signed_request(
            "GET", f"/api/v6/translations/{translation_id}"
        ).json()
        state = translation["requestState"]
        print(f"state={state}")
        if state == "DONE":
            print(
                "result_element_ids=" + ",".join(translation["resultElementIds"] or [])
            )
            return
        if state == "FAILED":
            raise SystemExit(
                "translation failed: " + str(translation.get("failureReason"))
            )
        # Translation normally takes several seconds; avoid spending API quota
        # on rapid status polling.
        time.sleep(5)
    raise SystemExit("translation did not finish within 120 seconds")


if __name__ == "__main__":
    main()
