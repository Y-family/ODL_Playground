"""Adapt existing servo fixing holes to 4 mm under-head screws, in millimeters.

Keep servo transforms and the accepted external outlines. Bring existing
recessed mating faces flush before deepening screw counterbores, targeting
2 mm from head seat to servo entry. Existing shorter seats stay unchanged.
No new parts are introduced. Thread strength is not inferred from STEP holes.
"""

import gc
import hashlib
import json
import resource
import shutil
import sys
from pathlib import Path

from cadgen import build123d as bd, read_step

import refit_case_bracket_width_local as brackets
import refit_upper_servo_interfaces_local as refit


WORK = Path('/tmp/servo4')
TARGET = brackets.TARGET
AUDIT = TARGET.parent / 'SERVO_SCREW_LENGTH_REPORT.json'
REPORT = TARGET.parent / 'SERVO_SCREW_REFIT_REPORT.json'
SEAT_DISTANCE_MM = 2.0
MAX_RSS_KIB = 4_800_000


def check_memory():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < MAX_RSS_KIB


def prepare():
    WORK.mkdir(exist_ok=True)
    audit = json.loads(AUDIT.read_text())
    fingerprint = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    assert fingerprint == audit['target_sha256']
    shutil.copyfile(TARGET, WORK / 'baseline.step')
    (WORK / 'audit.json').write_text(json.dumps(audit, indent=2))
    robot = read_step(TARGET)
    rows = []

    def visit(node, path=()):
        children = refit._children(node)
        if children:
            for child in children:
                visit(child, path + (child.label,))
            return
        filename = WORK / f'leaf_{len(rows)}.step'
        bd.export_step(bd.Compound(node.solids()), filename)
        bounds = node.bounding_box()
        rows.append(dict(path='/'.join(path), file=str(filename), label=node.label,
                         volume=node.volume, bounds=list(bounds.min) + list(bounds.max),
                         color=list(node.color) if node.color else None))

    visit(robot)
    assert len(rows) == 137
    (WORK / 'leaves.json').write_text(json.dumps(rows, indent=2))
    check_memory()
    print('Captured current 137-leaf assembly and screw-seat baseline', flush=True)


def ring(point, direction, start, end, inner, outer):
    return refit._shape(refit._rod(point, direction, start, end, outer).cut(
        refit._rod(point, direction, start - .01, end + .01, inner)))


def signed_bounds(part, point, direction):
    bounds = part.bounding_box()
    corners = [bd.Vector(horizontal, lateral, vertical)
               for horizontal in (bounds.min.X, bounds.max.X)
               for lateral in (bounds.min.Y, bounds.max.Y)
               for vertical in (bounds.min.Z, bounds.max.Z)]
    distances = [(corner - point).dot(direction) for corner in corners]
    return min(distances), max(distances)


def close_old_recess(part, hole):
    point, outward = bd.Vector(hole['entry_point_mm']), bd.Vector(hole['outward'])
    if hole['servo'] in ('S03', 'S08'):
        floor = .225 if hole['interface'] == 'passive' else 0.0
    elif hole['servo'] in ('S04', 'S09'):
        floor = .2 if hole['interface'] == 'passive' else 0.0
    else:
        floor = 0.0
    faces = []
    for face in part.faces():
        if face.geom_type != bd.GeomType.PLANE or face.normal_at().dot(outward) > -.999:
            continue
        distance = (face.center() - point).dot(outward)
        if 1.3 < distance < 1.7 and face.area > .001:
            faces.append((face, distance))
    assert faces, (hole['servo'], hole['interface'])
    for face, distance in faces:
        extension = bd.Solid.extrude(face, -outward * (distance - floor))
        part = refit._fuse(part, extension).clean()
    return part, dict(
        servo=hole['servo'], interface=hole['interface'], old_recess_mm=max(distance for face, distance in faces),
        new_mating_face_mm=floor, extended_original_face_area_mm2=sum(face.area for face, distance in faces))


def build(indices=None):
    rows = json.loads((WORK / 'leaves.json').read_text())
    audit = json.loads((WORK / 'audit.json').read_text())
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == audit['target_sha256']
    groups = {}
    for hole in audit['holes']:
        assert len(hole['candidates']) == 1
        path = hole['candidates'][0]['host'].split('/', 2)[2]
        groups.setdefault(path, []).append(hole)
    report = dict(target_sha256=audit['target_sha256'], seat_target_mm=SEAT_DISTANCE_MM,
                  counterbore_diameter_mm=dict(case=5.0, horn=6.5), parts=[])
    if indices is not None:
        report = json.loads(REPORT.read_text())
        report['parts'] = [part for part in report['parts'] if part['index'] not in indices]
    report['status'] = 'building candidates'
    REPORT.write_text(json.dumps(report, indent=2))
    for index, row in enumerate(rows):
        if indices is not None and index not in indices:
            continue
        holes = groups.get(row['path'], [])
        changed = [hole for hole in holes if hole['candidates'][0]['seat_distance_mm'] > SEAT_DISTANCE_MM + .001]
        if not changed:
            continue
        original = read_step(row['file'])
        part = original
        face_changes = []
        seen = set()
        for hole in changed:
            key = hole['servo'], hole['interface']
            if hole['servo'] in ('S02', 'S03', 'S04', 'S05', 'S07', 'S08', 'S09', 'S10') and hole['interface'] != 'case' and key not in seen:
                part, face_change = close_old_recess(part, hole)
                face_changes.append(face_change)
                seen.add(key)
        cap_hole = next((hole for hole in changed if hole['servo'] in ('S01', 'S06')
                         and hole['interface'] == 'active'), None)
        if cap_hole:
            point, outward = bd.Vector(cap_hole['entry_point_mm']), bd.Vector(cap_hole['outward'])
            assert outward.Z < -.999
            cap_height = point.Z + outward.Z * SEAT_DISTANCE_MM
            bounds = part.bounding_box()
            part = refit._cut(part, [refit._world_box(-37.875, 2.125,
                                                     bounds.min.Y - .1, bounds.max.Y + .1,
                                                     35.201, cap_height)]).clean()
            face_changes.append(dict(servo=cap_hole['servo'], interface='active',
                                     method='thin underside between existing end walls; retain top outline and wall connections',
                                     before_mm=cap_hole['candidates'][0]['seat_distance_mm'], after_mm=SEAT_DISTANCE_MM))
        hole_changes = []
        for hole in changed:
            point, outward = bd.Vector(hole['entry_point_mm']), bd.Vector(hole['outward'])
            old = hole['candidates'][0]
            head_radius = 2.5 if hole['interface'] == 'case' else 3.25
            end = signed_bounds(part, point, outward)[1] + .1
            part = refit._cut(part, [
                refit._rod(point, outward, -.01, end, old['bore_radius_mm']),
                refit._rod(point, outward, SEAT_DISTANCE_MM, end, head_radius),
            ]).clean()
            hole_changes.append(dict(servo=hole['servo'], interface=hole['interface'], hole=hole['hole'],
                                     before_seat_mm=old['seat_distance_mm'], after_seat_mm=SEAT_DISTANCE_MM,
                                     head_radius_mm=head_radius))
        assert part.is_valid and len(part.solids()) == 1, row['path']
        if cap_hole:
            cap_faces = [face for face in part.faces() if face.geom_type == bd.GeomType.PLANE
                         and abs(face.center().Z - cap_height) < .0001 and face.area > 100]
            assert len(cap_faces) == 1
            assert len(cap_faces[0].outer_wire().edges()) == 4
            assert len(cap_faces[0].inner_wires()) == 5
            assert abs(part.bounding_box().size.Y - 20) < .001
        added = refit._items(part.cut(original))
        added = [item for item in added if item.volume > .0001]
        added_volume = sum(item.volume for item in added)
        neighbour_checks = []
        if added_volume > .001:
            additions = bd.Compound(added)
            bounds = additions.bounding_box()
            for other_index, other in enumerate(rows):
                if other_index == index:
                    continue
                box = other['bounds']
                if any(box[axis] > list(bounds.max)[axis] + .001 or
                       box[axis + 3] < list(bounds.min)[axis] - .001 for axis in range(3)):
                    continue
                neighbour = read_step(other['file'])
                overlap = brackets.volume(additions.intersect(neighbour))
                assert overlap < .001, (row['path'], other['path'], overlap)
                neighbour_checks.append(dict(part=other['path'], added_overlap_mm3=overlap))
        support_checks = []
        for hole in changed:
            point, outward = bd.Vector(hole['entry_point_mm']), bd.Vector(hole['outward'])
            bore = hole['candidates'][0]['bore_radius_mm']
            bearing = ring(point, outward, .8, SEAT_DISTANCE_MM - .01, bore + .03, bore + .55)
            missing = brackets.export_difference(bearing, part)
            assert missing < .001, (hole['servo'], hole['interface'], hole['hole'], 'bearing', missing)
            head_radius = 2.5 if hole['interface'] == 'case' else 3.25
            end = signed_bounds(part, point, outward)[1] + .1
            passage = refit._rod(point, outward, SEAT_DISTANCE_MM + .01, end, head_radius - .02)
            blockage = brackets.volume(part.intersect(passage))
            assert blockage < .001
            support_checks.append(dict(servo=hole['servo'], interface=hole['interface'], hole=hole['hole'],
                                       bearing_test_axial_thickness_mm=1.19,
                                       bearing_missing_mm3=missing, head_passage_blockage_mm3=blockage))
        filename = WORK / f'candidate_{index}.step'
        bd.export_step(bd.Compound(part.solids()), filename)
        report['parts'].append(dict(index=index, path=row['path'], file=str(filename),
                                    volume_before_mm3=original.volume, volume_after_mm3=part.volume,
                                    added_volume_mm3=added_volume, original_face_changes=face_changes,
                                    holes=hole_changes, bearing_checks=support_checks,
                                    added_neighbour_checks=neighbour_checks))
        REPORT.write_text(json.dumps(report, indent=2))
        print(row['path'], len(changed), 'seats, added', round(added_volume, 4), 'mm3', flush=True)
        del original, part
        gc.collect()
        check_memory()
    report['status'] = 'candidates built and locally checked'
    report['max_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    REPORT.write_text(json.dumps(report, indent=2))


def geometry_signature(part):
    def coordinates(vector):
        return tuple(round(value, 5) for value in vector)

    faces = sorted((str(face.geom_type), round(face.area, 5), coordinates(face.center()),
                    coordinates(face.bounding_box().min), coordinates(face.bounding_box().max),
                    coordinates(face.normal_at()), len(face.wires()), len(face.edges()),
                    tuple(sorted(coordinates(face.position_at(horizontal, vertical))
                                 for horizontal in (.2, .5, .8) for vertical in (.2, .5, .8))))
                   for face in part.faces())
    edges = sorted((str(edge.geom_type), round(edge.length, 5),
                    tuple(sorted(coordinates(edge.position_at(position)) for position in (0, .25, .5, .75, 1))))
                   for edge in part.edges())
    vertices = sorted(coordinates(vertex.center()) for vertex in part.vertices())
    return faces, edges, vertices


def check_added_motion():
    from audit_case_mount_outlines_local import HOSTS
    rows = json.loads((WORK / 'leaves.json').read_text())
    audit = json.loads((WORK / 'audit.json').read_text())
    report = json.loads(REPORT.read_text())
    replacements = {part['index']: part['file'] for part in report['parts']}
    checks = []
    for part in report['parts']:
        if part['added_volume_mm3'] < .001:
            continue
        candidate = read_step(part['file'])
        original = read_step(rows[part['index']]['file'])
        additions = refit._shape(candidate.cut(original))
        for change in part['original_face_changes']:
            if 'old_recess_mm' not in change:
                continue
            servo, role = change['servo'], change['interface']
            holes = [hole for hole in audit['holes'] if hole['servo'] == servo and hole['interface'] == role]
            center = sum((bd.Vector(hole['entry_point_mm']) for hole in holes), bd.Vector(0, 0, 0)) / len(holes)
            axis = bd.Vector(holes[0]['outward'])
            envelope = refit._rod(center, axis, -.001, change['old_recess_mm'] + .001, 16)
            moving = refit._shape(additions.intersect(envelope))
            body_index = next(index for index, row in enumerate(rows)
                              if ('/' + servo + '_') in ('/' + row['path']) and row['label'] == 'ZP10D_vendor_body')
            for other_index in (body_index, *HOSTS[servo]):
                other = read_step(replacements.get(other_index, rows[other_index]['file']))
                swept_overlap = brackets.volume(envelope.intersect(other))
                if swept_overlap < .001:
                    checks.append(dict(servo=servo, interface=role, neighbour=rows[other_index]['path'],
                                       method='continuous 360 degree conservative cylindrical envelope',
                                       overlap_mm3=swept_overlap))
                    continue
                if servo in ('S02', 'S07'):
                    angles = range(-25, 26, 5)
                elif servo == 'S03':
                    angles = range(-30, 71, 5)
                elif servo == 'S08':
                    angles = range(-70, 31, 5)
                else:
                    angles = range(-180, 181, 5)
                maximum = 0.0
                for angle in angles:
                    posed = moving.rotate(bd.Axis(center, axis), angle)
                    overlap = brackets.volume(posed.intersect(other))
                    maximum = max(maximum, overlap)
                    assert overlap < .001, (servo, role, rows[other_index]['path'], angle, overlap)
                checks.append(dict(servo=servo, interface=role, neighbour=rows[other_index]['path'],
                                   method='added material only; 5 degree samples',
                                   range_deg=[angles.start, angles.stop - 1], maximum_overlap_mm3=maximum))
            print('Added material motion checked:', servo, role, flush=True)
        check_memory()
    report['added_material_motion_checks'] = checks
    REPORT.write_text(json.dumps(report, indent=2))


def install(reuse=False):
    report = json.loads(REPORT.read_text())
    assert report['status'] == 'candidates built and locally checked'
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == report['target_sha256']
    replacements = {row['path']: row for row in report['parts']}
    robot = None if reuse else read_step(TARGET)
    baseline = []

    def rebuild(node, path=()):
        children = refit._children(node)
        if children:
            fresh = bd.Compound(children=[rebuild(child, path + (child.label,)) for child in children], label=node.label)
            fresh.location, fresh.color = node.location, node.color
            return fresh
        bounds = node.bounding_box()
        baseline.append(('/'.join(path), node.volume, list(bounds.min) + list(bounds.max),
                         list(node.color) if node.color else None))
        if '/'.join(path) not in replacements:
            return node
        candidate = read_step(replacements['/'.join(path)]['file'])
        fresh = bd.Compound(candidate.solids(), label=node.label)
        fresh.color = node.color
        return fresh

    temporary = WORK / 'installed.step'
    if reuse:
        rows = json.loads((WORK / 'leaves.json').read_text())
        baseline.extend((row['path'], row['volume'], row['bounds'], row['color']) for row in rows)
    else:
        result = refit._encode_colours(rebuild(robot))
        bd.export_step(result, temporary)
        del result, robot
    gc.collect()
    exported = read_step(temporary)
    leaves = []

    def collect(node, path=()):
        children = refit._children(node)
        if children:
            for child in children:
                collect(child, path + (child.label,))
        else:
            leaves.append(('/'.join(path), node))

    collect(exported)
    assert len(leaves) == len(baseline) == 137
    checks = []
    for (path, node), (old_path, volume, bounds, colour) in zip(leaves, baseline):
        assert path == old_path
        actual_colour = list(node.color) if node.color else None
        assert (colour is None) == (actual_colour is None)
        assert colour is None or max(abs(left - right) for left, right in zip(colour, actual_colour)) < 1e-6
        if path in replacements:
            candidate = read_step(replacements[path]['file'])
            assert node.is_valid and len(node.solids()) == 1
            if geometry_signature(node) == geometry_signature(candidate):
                checks.append(dict(path=path, method='matching face/edge/vertex geometry signatures',
                                   coordinate_decimals=5, surface_samples_per_face=9, samples_per_edge=5))
            else:
                extra = brackets.export_difference(node, candidate)
                missing = brackets.export_difference(candidate, node)
                assert max(extra, missing) < .001, (path, extra, missing)
                checks.append(dict(path=path, method='bidirectional tolerant boolean difference',
                                   extra_mm3=extra, missing_mm3=missing))
            print('Export verified:', path, flush=True)
        else:
            actual = node.bounding_box()
            assert abs(node.volume - volume) < .001
            assert max(abs(left - right) for left, right in zip(bounds, list(actual.min) + list(actual.max))) < 1e-5
    del exported, leaves, candidate, node
    gc.collect()
    from audit_servo_screw_length_local import audit
    final_audit = audit(target=temporary, report_path=WORK / 'final_audit.json')
    for hole in final_audit['holes']:
        assert len(hole['candidates']) == 1, hole
        assert hole['candidates'][0]['seat_distance_mm'] <= SEAT_DISTANCE_MM + .001, hole
        assert hole['sampled_tip_clearance_mm'] is None or hole['sampled_tip_clearance_mm'] >= .1, hole
    pending = TARGET.with_suffix('.pending.step')
    shutil.copyfile(temporary, pending)
    pending.replace(TARGET)
    shutil.copyfile(WORK / 'final_audit.json', AUDIT)
    report.update(status='installed and export checked', export_checks=checks,
                  final_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                  unchanged_leaves=137 - len(replacements), all_colours_retained=True,
                  max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    REPORT.write_text(json.dumps(report, indent=2))
    check_memory()
    print('Installed', len(replacements), 'original parts; preserved remaining leaves and colours', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'build': build, 'tops': lambda: build((28, 108)),
     'install': install, 'verify-export': lambda: install(reuse=True), 'motion': check_added_motion}[sys.argv[1]]()
