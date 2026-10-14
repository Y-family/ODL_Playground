"""Rebuild the original speaker hanger as one tangent, smoothly bent plate.

Preserve the original roof mounting, integral speaker screw seats and all
four hole axes. Never install without STEP round-trip, motion and visual checks.
Run serially through run_memory_limited.py.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import math
import os
import tempfile
from pathlib import Path

from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder

import refit_upper_servo_interfaces_local as refit
from repair_speaker_layout_local import by_label, children, leaves, rebuild, TARGET
from full_motion_current import measured_axes
from remove_s01_s06_upper_horns import compare

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "cad/local_fix/open_duck_mini_upper_refit.step"
WORK = Path("/tmp/head-smooth-fix")
CANDIDATE = WORK / "review.step"
REPORT = ROOT / "cad/local_fix/SPEAKER_STAND_CONTINUITY_REPORT.json"
BASE_SHA = "fe9839fec9dd21712f7f1d403f47d2f0a46e3ca15c0bac3491fcd13f834bfbef"


def fixing_cylinders(stand):
    return [face for face in stand.faces()
            if face.geom_type == bd.GeomType.CYLINDER
            and abs(BRepAdaptor_Surface(face.wrapped, True).Cylinder().Radius()-1.025) < 1e-6]


def verify_bearing_material(old, new):
    checks = []
    for face in fixing_cylinders(old):
        cy = BRepAdaptor_Surface(face.wrapped, True).Cylinder()
        origin, direction = bd.Vector(cy.Location().Coord()), bd.Vector(cy.Axis().Direction().Coord())
        guard = refit._rod(origin, direction, 1.50001, 5.99999, 1.575)
        material = refit._shape(old.intersect(guard))
        lost = sum(x.volume for x in refit._items(material.cut(new)))
        assert lost < 1e-6, lost
        checks.append(dict(axis_origin=list(origin), guard_radius_mm=1.575,
                           original_bearing_volume_mm3=material.volume, lost_bearing_volume_mm3=lost))
    assert len(checks) == 2
    return checks


def verify_tangent_bend(stand):
    candidates = [f for f in stand.faces() if f.geom_type == bd.GeomType.CYLINDER
                  and abs(BRepAdaptor_Surface(f.wrapped, True).Cylinder().Radius()-6) < 1e-6]
    expected = 6*math.radians(70)*(93.641783-38.5)
    assert len(candidates) == 1 and abs(candidates[0].area-expected) < 1e-4
    cylinder = BRepAdaptor_Surface(candidates[0].wrapped, True).Cylinder()
    origin = bd.Vector(cylinder.Location().Coord())
    direction = bd.Vector(cylinder.Axis().Direction().Coord())
    errors = []
    for edge in candidates[0].edges():
        box = edge.bounding_box()
        if box.max.X-box.min.X < 50: continue
        point = edge.center()
        normal = (point-origin-direction*(point-origin).dot(direction)).normalized()
        expected_normal = bd.Vector(0, 0, 1) if abs(point.Z-228.90093881236) < 1e-5 else bd.Vector(0, -math.cos(math.radians(20)), math.sin(math.radians(20)))
        errors.append((normal-expected_normal).length)
    assert len(errors) == 2 and max(errors) < 1e-7
    return dict(one_full_width_analytic_outer_cylinder=True, radius_mm=6,
                angular_span_deg=70, width_mm=55.141783,
                tangent_normal_errors=errors, passed=True)


def repair_stand(stand, interface):
    # A single analytic YZ profile: horizontal plate, tangent R6/R3 bend,
    # then the original speaker mounting plane. No piecewise loft or patch.
    z, k = 228.90093881236, 128.20942525997708
    ny, nz = -math.cos(math.radians(20)), math.sin(math.radians(20))
    zc = z-6
    yc = (k-6-nz*zc)/ny
    x0, x1 = 38.5, 93.641783
    p = lambda y, height: (x0, y, height)
    a, b = p(-41.7, z), p(yc, z)
    c, d = p(yc+ny*6, zc+nz*6), p((k-nz*210)/ny, 210)
    e, f = p((k-3-nz*210)/ny, 210), p(yc+ny*3, zc+nz*3)
    g, h = p(yc, zc+3), p(-41.7, z-3)
    edges = [bd.Edge.make_line(a, b),
             bd.Edge.make_three_point_arc(b, p(yc-6*math.sin(math.radians(35)), zc+6*math.cos(math.radians(35))), c),
             bd.Edge.make_line(c, d), bd.Edge.make_line(d, e), bd.Edge.make_line(e, f),
             bd.Edge.make_three_point_arc(f, p(yc-3*math.sin(math.radians(35)), zc+3*math.cos(math.radians(35))), g),
             bd.Edge.make_line(g, h), bd.Edge.make_line(h, a)]
    body = bd.extrude(bd.Face(bd.Wire(edges)), amount=x1-x0, dir=(1, 0, 0))
    body = refit._cut(body, [refit._world_box(30, 48.9341, -100, 0, 200, 223.5)])
    bottom = bd.Plane(origin=(70.90405789115124, -55.65539616793591, 215.89057894724132),
                      z_dir=(-.07762277503612615, -.10931032590796642, -.9909722283926957))
    body = refit._cut(body, [bottom*refit._world_box(-300, 300, -300, 300, 0, 300)])
    roof = refit._shape(stand.intersect(refit._world_box(30, 100, -41.8, 0, 225.90093881234, 240)))
    inside = bd.Plane(origin=(0, k/ny, 0), z_dir=(0, ny, nz))
    seats = refit._shape(stand.intersect(inside*refit._world_box(-300, 300, -300, 300, -10, -2.999)))
    # Keep only the two existing integral screw seats, excluding the obsolete
    # central rib and the rejected added bridge from the previous iteration.
    guards = []
    for face in fixing_cylinders(stand):
        cy = BRepAdaptor_Surface(face.wrapped, True).Cylinder()
        guards.append(refit._rod(bd.Vector(cy.Location().Coord()), bd.Vector(cy.Axis().Direction().Coord()), 1.49, 6.0001, 3.05))
    seats = refit._shape(seats.intersect(bd.Compound(guards)))
    result = refit._fuse(body, seats).clean().fix()
    result = refit._fuse(result, roof).clean().fix()
    for face in fixing_cylinders(stand):
        cy = BRepAdaptor_Surface(face.wrapped, True).Cylinder()
        result = refit._cut(result, [refit._rod(bd.Vector(cy.Location().Coord()), bd.Vector(cy.Axis().Direction().Coord()), -15, 15, 1.025)])
    result = refit._cut(result, [interface]).clean().fix()
    result.label, result.color = stand.label, stand.color
    return result


def local_rows(robot):
    rows = list(leaves(robot))
    return {row["label"]: row["shape"] for row in rows}, rows


def build():
    WORK.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(FORMAL.read_bytes()).hexdigest() != BASE_SHA:
        raise SystemExit("formal STEP changed; refusing unexpected input")
    robot = read_step(FORMAL)
    old_by, rows = local_rows(robot)
    stand = repair_stand(old_by["speaker_stand"], old_by["speaker_interface"])
    validity = check_occurrence_shape(stand.wrapped)
    if validity["solidCount"] != 1 or validity["reasons"]:
        raise ValueError({"speaker_stand": validity})
    candidate_robot = rebuild(robot, {"speaker_stand": stand})
    bd.export_step(refit._encode_colours(candidate_robot), CANDIDATE)
    previous = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report = {
        "history": {"superseded_sha256": BASE_SHA, "reason": "User rejected segmented patch appearance", "viewer_cache_fix": previous.get("viewer_cache_fix")},
        "status": "candidate built; verification pending",
        "source_sha256": BASE_SHA,
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "geometry": "One planar top, tangent cylindrical bend and planar speaker wall",
        "bend_radii_mm": [6, 3],
        "plate_thickness_mm": 3,
        "stand_before_volume_mm3": old_by["speaker_stand"].volume,
        "stand_after_volume_mm3": stand.volume,
        "stand_validity": validity,
        "candidate": str(CANDIDATE),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("built", CANDIDATE, report["candidate_sha256"], flush=True)


def verify():
    report = json.loads(REPORT.read_text())
    if hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() != report["candidate_sha256"]:
        raise ValueError("candidate changed")
    old_robot, new_robot = read_step(FORMAL), read_step(CANDIDATE)
    old_by, old_rows = local_rows(old_robot)
    new_by, new_rows = local_rows(new_robot)
    if len(old_rows) != 135 or len(new_rows) != 135:
        raise ValueError((len(old_rows), len(new_rows)))
    if [r["path"] for r in old_rows] != [r["path"] for r in new_rows]:
        raise ValueError("leaf path order changed")
    for old, new in zip(old_rows, new_rows):
        if old["label"] != "speaker_stand":
            compare(old["shape"], new["shape"])
        if old["shape"].color is not None:
            assert max(abs(a - b) for a, b in zip(old["shape"].color, new["shape"].color)) < 1e-6
    assert len(measured_axes(old_robot)) == len(measured_axes(new_robot)) == 14
    for joint, axes in measured_axes(old_robot).items():
        for a, b in zip(axes, measured_axes(new_robot)[joint]):
            assert max(abs(x - y) for x, y in zip(a, b)) < 1e-5, joint

    stand = new_by["speaker_stand"]
    validity = check_occurrence_shape(stand.wrapped)
    assert validity["solidCount"] == 1 and not validity["reasons"], validity
    # Check original hole axes, open bores and their original bearing material.
    from repair_speaker_layout_local import roof_fixing_axes
    assert roof_fixing_axes(old_by["speaker_stand"]) == roof_fixing_axes(stand)
    fixings = []
    for face in fixing_cylinders(old_by["speaker_stand"]):
        cy = BRepAdaptor_Surface(face.wrapped, True).Cylinder()
        axis_origin = bd.Vector(cy.Location().Coord())
        direction = bd.Vector(cy.Axis().Direction().Coord()).normalized()
        point = face.center()
        axial = axis_origin + direction*(point-axis_origin).dot(direction)
        radial = (point-axial).normalized()
        tangent = direction.cross(radial).normalized()
        samples = [axial + 1.575*(radial*math.cos(i*math.pi/16)+tangent*math.sin(i*math.pi/16)) for i in range(32)]
        for sample in samples:
            assert old_by["speaker_stand"].is_inside(sample, tolerance=1e-6)
            assert stand.is_inside(sample, tolerance=1e-6), list(sample)
        bore = refit._rod(axis_origin, direction, -15, 15, 1.025-1e-5)
        blocked = sum(x.volume for x in refit._items(stand.intersect(bore)))
        assert blocked < 1e-6, blocked
        fixings.append(dict(axis_origin=list(axis_origin), direction=list(direction), radius_mm=1.025,
                            retained_bearing_samples=32, bore_blockage_mm3=blocked))
    assert len(fixings) == 2
    report["bearing_material_checks"] = verify_bearing_material(old_by["speaker_stand"], stand)
    report["smooth_surface"] = verify_tangent_bend(stand)
    report["speaker_fixing_checks"] = fixings
    report["roof_fixing_axes"] = roof_fixing_axes(stand)
    checks = []
    for label in ("head", "speaker_interface", "speaker_bottom", "speaker_top",
                  "head_roll_mount", "head_yaw_to_roll", "head_pitch_to_yaw"):
        other = new_by[label]
        gap = stand.distance_to(other)
        overlap = 0.0
        if gap < 1e-7:
            overlap = sum(item.volume for item in refit._items(stand.intersect(other)))
        checks.append({"part": label, "gap_mm": gap, "overlap_mm3": overlap,
                       "passed": overlap < 1e-6 and (label in ("head", "speaker_interface") or gap >= .2 - 1e-7)})
        assert checks[-1]["passed"], checks[-1]
    report.update(status="verified", leaf_count=len(new_rows), unchanged_leaf_count=134,
                  main_axes_preserved=14, speaker_stand_validity=validity,
                  local_clearance=checks)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("verified stand continuity; local pairs", len(checks), flush=True)


def install():
    report = json.loads(REPORT.read_text())
    if report.get("status") != "verified":
        raise SystemExit("run build and verify first")
    motion = json.loads((WORK / "motion.json").read_text())
    assert motion.get("passed") and motion["source_sha256"] == report["candidate_sha256"]
    assert report.get("visual_review_sha256") == report["candidate_sha256"]
    assert report.get("standard_validation_passed")
    assert hashlib.sha256(FORMAL.read_bytes()).hexdigest() == BASE_SHA
    assert hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() == report["candidate_sha256"]
    fd, temporary = tempfile.mkstemp(prefix=".stand-smooth-", suffix=".tmp", dir=FORMAL.parent)
    os.close(fd)
    try:
        shutil.copyfile(CANDIDATE, temporary)
        os.replace(temporary, FORMAL)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    report["motion_checks"] = motion
    installed = hashlib.sha256(FORMAL.read_bytes()).hexdigest()
    report.update(status="installed and reread", installed_sha256=installed)
    reread = read_step(FORMAL)
    by, rows = local_rows(reread)
    assert len(rows) == 135 and len(by["speaker_stand"].solids()) == 1
    final_validity = check_occurrence_shape(by["speaker_stand"].wrapped)
    assert final_validity["solidCount"] == 1 and not final_validity["reasons"], final_validity
    report["installed_reread_leaf_count"] = len(rows)
    report["installed_stand_validity"] = final_validity
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("installed", installed, flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in {"build", "verify", "install"}:
        raise SystemExit("usage: repair_speaker_stand_continuity.py build|verify|install")
    {"build": build, "verify": verify, "install": install}[sys.argv[1]]()
