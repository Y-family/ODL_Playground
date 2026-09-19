"""Measure existing screw seats against the supplied 4 mm servo screws."""

import hashlib
import json

from cadgen import build123d as bd, read_step
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Circle
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

import audit_case_mount_outlines_local as outlines
import validate_s01_s10_interfaces_local as interfaces


REPORT = outlines.TARGET.parent / 'SERVO_SCREW_LENGTH_REPORT.json'
SCREW_LENGTH_MM = 4.0


def audit(target=outlines.TARGET, report_path=REPORT):
    fingerprint = hashlib.sha256(target.read_bytes()).hexdigest()
    robot = read_step(target)
    vendor = read_step(interfaces.VENDOR)
    vendor_body = max(interfaces._children(vendor), key=lambda child: child.volume)
    leaves = []
    servos = {}

    def collect(node, path=''):
        path = path + '/' + node.label
        children = interfaces._children(node)
        if node.label.startswith('S') and '_ZP10D' in node.label:
            servos[node.label.split('_')[0]] = node
        if children:
            for child in children:
                collect(child, path)
        else:
            leaves.append((path, node))

    collect(robot)
    assert len(leaves) == 137 and len(servos) == 14
    seats = []
    for index, (path, host) in enumerate(leaves):
        if 'ZP10D_vendor' in path:
            continue
        for face in host.faces():
            if face.geom_type != bd.GeomType.PLANE:
                continue
            for edge in face.edges():
                curve = BRepAdaptor_Curve(edge.wrapped)
                if curve.GetType() != GeomAbs_Circle:
                    continue
                circle = curve.Circle()
                if not .9 <= circle.Radius() <= 1.8:
                    continue
                seats.append((index, path, bd.Vector(circle.Location().Coord()),
                              bd.Vector(circle.Axis().Direction().Coord()),
                              face.normal_at(), circle.Radius()))

    results = []
    for servo_id, servo in sorted(servos.items()):
        body, passive, active = interfaces._servo_parts(servo)
        bore_ray = IntCurvesFace_ShapeIntersector()
        bore_ray.Load(bd.Compound([body, passive, active]).wrapped, 1e-7)
        joint_axis, joint_center = interfaces._axis(servo)
        patterns = [('case', None, interfaces._case_mount_lines(body, vendor_body))]
        for role, horn in [('active', active), ('passive', passive)]:
            if role == 'passive' and servo_id in ('S01', 'S06', 'S14'):
                continue
            patterns.append((role, horn, interfaces._horn_pattern_lines(horn, joint_axis, joint_center)))
        for role, horn, lines in patterns:
            for hole, (point, direction) in enumerate(lines, 1):
                outward = direction if (point - body.bounding_box().center()).dot(direction) > 0 else -direction
                entry = point
                if horn is not None:
                    endpoints = []
                    for face in horn.faces():
                        if face.geom_type != bd.GeomType.CYLINDER:
                            continue
                        surface = BRepAdaptor_Surface(face.wrapped)
                        cylinder = surface.Cylinder()
                        axis = bd.Vector(cylinder.Axis().Direction().Coord())
                        origin = bd.Vector(cylinder.Location().Coord())
                        if abs(cylinder.Radius() - 1.5) > .001:
                            continue
                        if interfaces._line_distance(point, direction, origin, axis) > .01:
                            continue
                        endpoints.extend((origin + axis * value - point).dot(outward)
                                         for value in (surface.FirstVParameter(), surface.LastVParameter()))
                    assert endpoints
                    entry = point + outward * max(endpoints)
                candidates = []
                for index, path, seat, axis, normal, radius in seats:
                    if role == 'case' and index not in outlines.HOSTS[servo_id]:
                        continue
                    if normal.dot(outward) < .999:
                        continue
                    if interfaces._line_distance(entry, outward, seat, axis) > .01:
                        continue
                    distance = (seat - entry).dot(outward)
                    if not .01 < distance < 15:
                        continue
                    candidate = dict(host=path, seat_distance_mm=round(distance, 5),
                                     geometric_insertion_mm=round(SCREW_LENGTH_MM - distance, 5),
                                     bore_radius_mm=round(radius, 5))
                    if candidate not in candidates:
                        candidates.append(candidate)
                candidates.sort(key=lambda item: item['seat_distance_mm'])
                status = 'seat_unresolved'
                if len(candidates) == 1:
                    insertion = candidates[0]['geometric_insertion_mm']
                    status = 'no_thread_reach' if insertion <= .001 else 'positive_reach_not_tightening_verified'
                elif candidates:
                    status = 'multiple_seats_review'
                bottom_depths = []
                helper = bd.Vector(0, 0, 1) if abs(outward.Z) < .9 else bd.Vector(0, 1, 0)
                radial = outward.cross(helper).normalized()
                tangent = outward.cross(radial)
                probe_radius = .5 if role == 'case' else 1.0
                for offset in (bd.Vector(0, 0, 0), radial * probe_radius, -radial * probe_radius,
                               tangent * probe_radius, -tangent * probe_radius):
                    start = entry + offset
                    bore_ray.Perform(gp_Lin(gp_Pnt(*start), gp_Dir(*(-outward))), .01, 4.0)
                    bottom_depths.extend(bore_ray.WParameter(position)
                                         for position in range(1, bore_ray.NbPnt() + 1))
                bottom = min(bottom_depths) if bottom_depths else None
                clearance = bottom - candidates[0]['geometric_insertion_mm'] if bottom is not None and len(candidates) == 1 else None
                results.append(dict(servo=servo_id, interface=role, hole=hole,
                                    entry_point_mm=list(entry), outward=list(outward),
                                    status=status, candidates=candidates,
                                    sampled_servo_obstruction_depth_mm=bottom,
                                    sampled_tip_clearance_mm=clearance))
        print(servo_id, 'seats measured', flush=True)
    assert fingerprint == hashlib.sha256(target.read_bytes()).hexdigest()
    report = dict(
        target_sha256=fingerprint, geometry_changed=False, screw_length_mm=SCREW_LENGTH_MM,
        confirmed_length='4 mm under-head length of supplier servo screws; thread size not inferred',
        scope='servo case and attached horn screws only; passive supports S01/S06/S14 excluded',
        limitations=['Positive reach does not establish effective full-thread engagement or tightening strength.',
                     'Five axial rays screen modeled tip clearance; they do not measure full-thread engagement.',
                     'Actual tapped depth, head fit, complete bearing annulus and tool access are not verified.',
                     'Existing planar bore rims identify candidate seats; unresolved or multiple seats require review.'],
        holes=results)
    report_path.write_text(json.dumps(report, indent=2))
    for servo_id in sorted(servos):
        summary = {}
        for role in ('case', 'active', 'passive'):
            rows = [row for row in results if row['servo'] == servo_id and row['interface'] == role]
            summary[role] = dict(distances=sorted({candidate['seat_distance_mm'] for row in rows
                                                 for candidate in row['candidates']}),
                                 unresolved=sum(len(row['candidates']) != 1 for row in rows),
                                 no_reach=sum(row['status'] == 'no_thread_reach' for row in rows))
        print(servo_id, summary, flush=True)
    return report


if __name__ == '__main__':
    audit()
