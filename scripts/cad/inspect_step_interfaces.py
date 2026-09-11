#!/usr/bin/env python3
"""Report useful envelopes and cylindrical interfaces from an AP203 STEP file.

This intentionally uses only the Python standard library. It is a dimensional
audit helper, not a replacement for a CAD kernel or tolerance drawing.
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

ENTITY_RE = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*?)\)\s*;", re.DOTALL)
REFERENCE_RE = re.compile(r"#(\d+)")
TUPLE_RE = re.compile(r"\(\s*([-+0-9.Ee, ]+)\s*\)\s*$")
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:E[-+]?\d+)?")


def references(arguments: str) -> list[int]:
    return [int(value) for value in REFERENCE_RE.findall(arguments)]


def numeric_tuple(arguments: str) -> tuple[float, ...] | None:
    match = TUPLE_RE.search(arguments)
    if not match:
        return None
    return tuple(float(value) for value in match.group(1).split(","))


def descendants(root: int, entities: dict[int, tuple[str, str]]) -> set[int]:
    found: set[int] = set()
    pending = [root]
    while pending:
        entity_id = pending.pop()
        if entity_id in found or entity_id not in entities:
            continue
        found.add(entity_id)
        pending.extend(references(entities[entity_id][1]))
    return found


def surface_radius(arguments: str) -> float:
    tail = re.sub(r"^.*#\d+\s*,", "", arguments, flags=re.DOTALL)
    return float(NUMBER_RE.findall(tail)[0])


def placement(
    entity_id: int, entities: dict[int, tuple[str, str]]
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    refs = references(entities[entity_id][1])
    if len(refs) < 2:
        return None
    origin = numeric_tuple(entities[refs[0]][1])
    axis = numeric_tuple(entities[refs[1]][1])
    if origin is None or axis is None or len(origin) < 3 or len(axis) < 3:
        return None
    return origin, axis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=Path)
    args = parser.parse_args()

    text = args.step.read_text(encoding="latin1")
    entities = {
        int(entity_id): (kind, arguments)
        for entity_id, kind, arguments in ENTITY_RE.findall(text)
    }

    for root, (kind, arguments) in entities.items():
        if kind != "MANIFOLD_SOLID_BREP":
            continue

        name_match = re.search(r"'([^']*)'", arguments)
        raw_name = name_match.group(1) if name_match else f"solid_{root}"
        try:
            name = raw_name.encode("latin1").decode("gbk")
        except UnicodeError:
            name = raw_name

        owned = descendants(root, entities)
        points = [
            numeric_tuple(entities[item][1])
            for item in owned
            if entities[item][0] == "CARTESIAN_POINT"
        ]
        points = [point for point in points if point is not None and len(point) >= 3]
        if not points:
            continue
        bounds = [
            (min(point[axis] for point in points), max(point[axis] for point in points))
            for axis in range(3)
        ]

        cylinders: collections.Counter[
            tuple[float, tuple[float, ...], tuple[float, ...]]
        ] = collections.Counter()
        for item in owned:
            surface_kind, surface_args = entities[item]
            if surface_kind != "CYLINDRICAL_SURFACE":
                continue
            surface_refs = references(surface_args)
            if not surface_refs:
                continue
            located = placement(surface_refs[0], entities)
            if located is None:
                continue
            origin, axis = located
            key = (
                round(surface_radius(surface_args), 6),
                tuple(round(value, 6) for value in origin),
                tuple(round(value, 6) for value in axis),
            )
            cylinders[key] += 1

        spans = [upper - lower for lower, upper in bounds]
        print(f"#{root} {name}")
        print("  envelope mm: " + " x ".join(f"{span:.6g}" for span in spans))
        print("  cylindrical surfaces (radius, origin, axis):")
        for (radius, origin, axis), count in sorted(cylinders.items()):
            print(f"    r={radius:g}  origin={origin}  axis={axis}  faces={count}")


if __name__ == "__main__":
    main()
