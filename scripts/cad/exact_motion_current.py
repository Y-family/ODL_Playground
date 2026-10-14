#!/usr/bin/env python3
"""Confirm broad-phase candidates with exact B-Rep intersections."""

from __future__ import annotations

import json
import gc
import sys
import hashlib
import os
from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step

from full_motion_current import LIMITS, REPORT as BROAD_REPORT, TARGET, MJCF, _leaves
from validate_s01_s10_interfaces_local import _children

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(os.environ.get("OPEN_DUCK_EXACT_REPORT", ROOT / "cad/local_fix/FULL_MOTION_EXACT_REPORT.json"))
VOLUME_TOLERANCE_MM3 = 0.05


def _boxes_overlap(left, right):
    a = left.bounding_box()
    b = right.bounding_box()
    return not (
        a.max.X < b.min.X or b.max.X < a.min.X
        or a.max.Y < b.min.Y or b.max.Y < a.min.Y
        or a.max.Z < b.min.Z or b.max.Z < a.min.Z
    )


def _overlap(left, right):
    if not _boxes_overlap(left, right):
        return 0.0
    intersection = left.intersect(right)
    if intersection is None:
        return 0.0
    return sum(piece.volume for piece in intersection.solids())


def main():
    selected = set(sys.argv[1:]) or set(LIMITS)
    unknown = selected - set(LIMITS)
    if unknown:
        raise ValueError(f"unknown joints: {sorted(unknown)}")

    robot = read_step(TARGET)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    broad = json.loads(BROAD_REPORT.read_text())
    assert broad['target_sha256'] == hashlib.sha256(TARGET.read_bytes()).hexdigest()
    assert broad['mjcf_sha256'] == hashlib.sha256(MJCF.read_bytes()).hexdigest()
    assert broad['scanner_sha256'] == hashlib.sha256((Path(__file__).parent / 'full_motion_current.py').read_bytes()).hexdigest()
    broad_digest = hashlib.sha256(BROAD_REPORT.read_bytes()).hexdigest()
    checks = []
    completed = []

    def save_progress():
        progress = {"target_sha256": broad["target_sha256"], "checks": []}
        if REPORT.exists():
            previous = json.loads(REPORT.read_text())
            if isinstance(previous, dict) and previous.get('broad_sha256') == broad_digest:
                progress = previous
        retained = [row for row in progress["checks"] if row["joint"] not in selected]
        progress.update({
            "limit_source": broad["limit_source"],
            "step_deg": broad["step_deg"],
            "broad_sha256": broad_digest,
            "axes": broad['axes'],
            "completed_joints": sorted((set(progress.get('completed_joints', [])) - selected) | set(completed)),
            "volume_tolerance_mm3": VOLUME_TOLERANCE_MM3,
            "checks": sorted(retained + checks, key=lambda row: (row["joint"], row["moving"], row["fixed"])),
        })
        REPORT.write_text(json.dumps(progress, indent=2) + "\n")

    for joint in broad["joints"]:
        servo_id = joint["id"]
        if servo_id not in selected:
            continue
        axis = bd.Axis(*broad['axes'][servo_id])
        for candidate in joint["candidate_collisions"]:
            moving_path = candidate["moving"]
            fixed_path = candidate["fixed"]
            mover = shapes[moving_path]
            fixed = shapes[fixed_path]
            baseline = _overlap(mover, fixed)
            exact = []
            for angle in candidate["angles_deg"]:
                posed = mover.rotate(axis, angle)
                volume = _overlap(posed, fixed)
                del posed
                gc.collect()
                exact.append({
                    "angle_deg": angle,
                    "overlap_mm3": volume,
                    "increase_mm3": volume - baseline,
                })
            for sign in (-1, 1):
                hits = [abs(row['angle_deg']) for row in exact if row['angle_deg']*sign > 0 and row['overlap_mm3'] > VOLUME_TOLERANCE_MM3]
                if not hits:
                    continue
                first = int(min(hits))
                for magnitude in range(max(1, first-4), first):
                    angle = sign*magnitude
                    posed = mover.rotate(axis, angle)
                    overlap = _overlap(posed, fixed)
                    exact.append(dict(angle_deg=angle, overlap_mm3=overlap, increase_mm3=overlap-baseline))
                    del posed
            exact.sort(key=lambda row: row['angle_deg'])
            collisions = [
                row for row in exact
                if row["overlap_mm3"] > VOLUME_TOLERANCE_MM3
            ]
            checks.append({
                "joint": servo_id,
                "moving": moving_path,
                "fixed": fixed_path,
                "baseline_overlap_mm3": baseline,
                "exact": exact,
                "new_collision_samples": collisions,
                "collision_samples": collisions,
            })
            save_progress()
            print(
                servo_id,
                moving_path.rsplit("/", 1)[-1],
                "<->",
                fixed_path.rsplit("/", 1)[-1],
                "interference",
                [(row["angle_deg"], round(row["overlap_mm3"], 3)) for row in collisions],
                flush=True,
            )
            # Keep each exact B-Rep candidate independent. A full robot STEP
            # can otherwise retain several gigabytes of transient OCC copies.
            gc.collect()
        completed.append(servo_id)
        save_progress()

    save_progress()


if __name__ == "__main__":
    main()
