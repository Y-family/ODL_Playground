"""Audit U08 case screw outlines without changing accepted geometry."""

import hashlib
import json
from pathlib import Path

from cadgen import build123d as bd, read_step
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Circle

import refit_upper_servo_interfaces_local as refit
import refit_case_bracket_width_local as brackets
import validate_s01_s10_interfaces_local as interfaces


TARGET = refit.ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
REPORT = refit.ROOT / 'cad/local_fix/U08_OUTLINE_REPORT.json'
HOSTS = {
    'S01': (112, 120), 'S02': (28, 29), 'S03': (54,),
    'S04': (66, 67), 'S05': (34, 38), 'S06': (112, 120),
    'S07': (108, 107), 'S08': (127,), 'S09': (131, 132),
    'S10': (72, 73), 'S11': (112, 120), 'S12': (17, 18),
    'S13': (53,), 'S14': (90,),
}


def audit():
    fingerprint = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    robot = read_step(TARGET)
    vendor = read_step(interfaces.VENDOR)
    vendor_body = max(interfaces._children(vendor), key=lambda child: child.volume)
    leaves = []

    def collect(node, path=()):
        children = interfaces._children(node)
        if children:
            for child in children:
                collect(child, path + (child.label,))
        else:
            leaves.append(('/'.join(path), node))

    collect(robot)
    assert len(leaves) == 137
    results = []
    for servo_id, indices in HOSTS.items():
        body = next(node for path, node in leaves
                    if f'/{servo_id}_' in '/' + path and node.label == 'ZP10D_vendor_body')
        lines = interfaces._case_mount_lines(body, vendor_body)
        findings = []
        for point, direction in lines:
            matches = []
            for index in indices:
                path, host = leaves[index]
                for face in host.faces():
                    if face.geom_type != bd.GeomType.CYLINDER:
                        continue
                    cylinder = BRepAdaptor_Surface(face.wrapped).Cylinder()
                    axis = bd.Vector(cylinder.Axis().Direction().Coord())
                    origin = bd.Vector(cylinder.Location().Coord())
                    if abs(axis.dot(direction)) < .999:
                        continue
                    if interfaces._line_distance(point, direction, origin, axis) > .01:
                        continue
                    surface_point = face.position_at(.5, .5)
                    radial = surface_point - origin
                    radial -= axis * radial.dot(axis)
                    outward = face.normal_at(surface_point).dot(radial.normalized()) > .5
                    match = dict(host=path, radius_mm=round(cylinder.Radius(), 5),
                                 external=outward, area_mm2=round(face.area, 5))
                    if not outward and cylinder.Radius() <= 1.3:
                        centre = origin + axis * (face.center() - origin).dot(axis)
                        probe = refit._rod(centre, axis, -.02, .02, cylinder.Radius() - .02)
                        outer = refit._rod(centre, axis, -.02, .02, cylinder.Radius() + .5)
                        inner = refit._rod(centre, axis, -.03, .03, cylinder.Radius() + .02)
                        ring = refit._shape(outer.cut(inner))
                        blocked = brackets.volume(host.intersect(probe))
                        unsupported = brackets.export_difference(ring, host)
                        match.update(bore_midsection_blockage_mm3=blocked,
                                     unsupported_half_mm_ring_mm3=unsupported)
                    matches.append(match)
            findings.append(dict(point=list(point), direction=list(direction), cylinders=matches))
        external = [match for finding in findings for match in finding['cylinders'] if match['external']]
        missing = [position for position, finding in enumerate(findings)
                   if not any(not match['external'] and match['radius_mm'] <= 1.3
                              for match in finding['cylinders'])]
        shared_faces = []
        for index in indices:
            path, host = leaves[index]
            for face in host.faces():
                if face.geom_type != bd.GeomType.PLANE:
                    continue
                if abs(face.normal_at().dot(lines[0][1])) < .999:
                    continue
                axes = set()
                for edge in face.edges():
                    curve = BRepAdaptor_Curve(edge.wrapped)
                    if curve.GetType() != GeomAbs_Circle:
                        continue
                    circle = curve.Circle()
                    if circle.Radius() > 2.8:
                        continue
                    origin = bd.Vector(circle.Location().Coord())
                    axis = bd.Vector(circle.Axis().Direction().Coord())
                    for position, (point, direction) in enumerate(lines):
                        if abs(axis.dot(direction)) > .999 and interfaces._line_distance(point, direction, origin, axis) < .01:
                            axes.add(position)
                if len(axes) >= 2:
                    shared_faces.append(dict(host=path, axes=sorted(axes), area_mm2=round(face.area,4),
                                             outer_edges=len(face.outer_wire().edges())))
        covered = set(position for face in shared_faces for position in face['axes'])
        result = dict(servo=servo_id, hosts=[leaves[index][0] for index in indices],
                      case_axes=findings, external_coaxial_bosses=external, missing_holes=missing,
                      shared_planar_mount_faces=shared_faces, axes_without_shared_face=sorted(set(range(4)) - covered))
        results.append(result)
        print(servo_id, 'external bosses', len(external), 'missing holes', missing,
              'shared plane axes', sorted(covered), flush=True)
    assert fingerprint == hashlib.sha256(TARGET.read_bytes()).hexdigest()
    defects = []
    support_findings = []
    for result in results:
        if result['external_coaxial_bosses'] or result['missing_holes'] or result['axes_without_shared_face']:
            defects.append(result['servo'])
        for finding in result['case_axes']:
            for match in finding['cylinders']:
                if match.get('bore_midsection_blockage_mm3', 0) > .001:
                    defects.append(result['servo'])
                if match.get('unsupported_half_mm_ring_mm3', 0) > .001:
                    support_findings.append(dict(servo=result['servo'], point=finding['point'], **match))
    REPORT.write_text(json.dumps(dict(target_sha256=fingerprint, geometry_changed=False,
                                     status='passed' if not defects else 'needs review',
                                     defects=sorted(set(defects)),
                                     support_findings=support_findings,
                                     support_note='0.5 mm radial ring is a diagnostic sample, not a specified strength criterion; partial rings require separate mounting-seat review.',
                                     scope='U08 case mounting outlines and hole midsection support; no joint motion certification',
                                     joints=results), indent=2))
    if defects:
        raise ValueError(f'U08 checks failed for {sorted(set(defects))}')


if __name__ == '__main__':
    refit._start_memory_watchdog()
    audit()
