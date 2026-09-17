"""Match S03/S08 inner sheet thickness to the existing outer sheet."""

import gc
import hashlib
import json
import math
import sys
from pathlib import Path

from cadgen import build123d as bd, read_step
import trimesh

import refit_case_bracket_width_local as brackets
import refit_upper_servo_interfaces_local as refit
from scan_s01_s14_motion_collisions import _mesh


WORK = Path('/tmp/u09')
TARGET = brackets.TARGET
REPORT = refit.ROOT / 'cad/local_fix/U09_THICKNESS_REPORT.json'
EXTRA_MM = 1.45
INDICES = (66, 131)


def source(row):
    candidate = brackets.candidate_file(row)
    return candidate if candidate.exists() else Path(row['file'])


def build():
    WORK.mkdir(exist_ok=True)
    rows = json.loads((brackets.WORK / 'leaves.json').read_text())
    reports = []
    for index in INDICES:
        row = rows[index]
        original = read_step(source(row))
        sign = -1 if original.bounding_box().center().Y < 0 else 1
        faces = [face for face in original.faces() if face.geom_type == bd.GeomType.PLANE
                 and face.bounding_box().size.Y < 1e-5
                 and abs(face.center().Y - sign * 65.3) < 1e-4]
        candidate = original
        for face in faces:
            candidate = refit._fuse(candidate, bd.Solid.extrude(face, (0, -sign * EXTRA_MM, 0)))
        candidate = candidate.clean()
        if not candidate.is_valid or len(candidate.solids()) != 1:
            raise ValueError('Thickened sheet is not a valid single solid')
        if brackets.export_difference(original, candidate) > .001:
            raise ValueError('Existing sheet material was removed')
        added = refit._shape(candidate.cut(original))
        bounds = added.bounding_box()
        neighbours = []
        for other_index, other in enumerate(rows):
            if other_index == index or any(other['bounds'][axis] > list(bounds.max)[axis] or
                    other['bounds'][axis + 3] < list(bounds.min)[axis] for axis in range(3)):
                continue
            neighbour = read_step(source(other))
            overlap = brackets.volume(added.intersect(neighbour))
            if overlap > .001:
                raise ValueError(f'Added material collides with {other["path"]}: {overlap}')
            neighbours.append(dict(part=other['path'], added_clearance_mm=added.distance_to(neighbour)))
        bores = []
        for face in original.faces():
            if face.geom_type != bd.GeomType.CYLINDER:
                continue
            cylinder = refit.BRepAdaptor_Surface(face.wrapped).Cylinder()
            direction = bd.Vector(cylinder.Axis().Direction().Coord())
            if abs(direction.Y) < .999 or cylinder.Radius() > 3.3:
                continue
            centre = bd.Vector(cylinder.Location().Coord())
            face_bounds = face.bounding_box()
            low, high = face_bounds.min.Y + .001, face_bounds.max.Y - .001
            if sign < 0 and abs(face_bounds.max.Y + 65.3) < 1e-4:
                high += EXTRA_MM
            if sign > 0 and abs(face_bounds.min.Y - 65.3) < 1e-4:
                low -= EXTRA_MM
            bore = refit._rod(bd.Vector(centre.X, 0, centre.Z), bd.Vector(0, 1, 0),
                              low, high, cylinder.Radius() - .005)
            if brackets.volume(original.intersect(bore)) < .001:
                if brackets.volume(candidate.intersect(bore)) > .001:
                    raise ValueError('Existing screw passage obstructed')
                bores.append(dict(x=centre.X, z=centre.Z, radius=cylinder.Radius()))
        output = WORK / f'leaf_{index}.step'
        bd.export_step(candidate, output)
        reports.append(dict(index=index, path=row['path'], file=str(output),
                            before_mm=2.4, after_mm=3.85, outward_extension_mm=EXTRA_MM,
                            valid_single_solid=True, bores=bores, neighbours=neighbours))
        print(f'{row["path"]}: 2.40 -> 3.85 mm; valid, static clearances and bores passed', flush=True)
    REPORT.write_text(json.dumps(dict(status='candidate', target_sha256=hashlib.sha256(
        TARGET.read_bytes()).hexdigest(), parts=reports, full_joint_motion='not checked'), indent=2))


def install():
    report = json.loads(REPORT.read_text())
    if hashlib.sha256(TARGET.read_bytes()).hexdigest() != report['target_sha256']:
        raise ValueError('Target changed since preparation')
    replacements = {tuple(row['path']): row for row in report['parts']}
    robot = read_step(TARGET)
    baseline = []

    def rebuild(node, path=()):
        children = refit._children(node)
        if children:
            fresh = bd.Compound(children=[rebuild(child, path + (child.label,)) for child in children], label=node.label)
            fresh.location, fresh.color = node.location, node.color
            return fresh
        bounds = node.bounding_box()
        baseline.append((path, node.volume, list(bounds.min) + list(bounds.max), list(node.color) if node.color else None))
        if path not in replacements:
            return node
        row = replacements[path]
        original = read_step(brackets.WORK / f'leaf_{row["index"]}_candidate.step')
        if max(brackets.export_difference(node, original), brackets.export_difference(original, node)) > .001:
            raise ValueError('Cached source differs from installed part')
        candidate = read_step(row['file'])
        fresh = bd.Compound(candidate.solids(), label=node.label)
        fresh.color = node.color
        return fresh

    result = refit._encode_colours(rebuild(robot))
    temporary = TARGET.with_suffix('.pending.step')
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
            leaves.append((path, node))

    collect(exported)
    assert len(leaves) == len(baseline)
    for (path, node), (old_path, volume, bounds, colour) in zip(leaves, baseline):
        assert path == old_path
        actual_colour = list(node.color) if node.color else None
        assert (colour is None) == (actual_colour is None)
        assert colour is None or max(abs(first - second) for first, second in zip(colour, actual_colour)) < 1e-6
        if path in replacements:
            candidate = read_step(replacements[path]['file'])
            assert node.is_valid and len(node.solids()) == 1
            assert abs(node.volume - candidate.volume) < .001
            assert max(brackets.export_difference(node, candidate), brackets.export_difference(candidate, node)) < .001
        else:
            actual = node.bounding_box()
            assert abs(node.volume - volume) < .001
            assert max(abs(first - second) for first, second in zip(bounds, list(actual.min) + list(actual.max))) < 1e-5
    temporary.replace(TARGET)
    report.update(status='installed; awaiting visual acceptance', unchanged_leaves=len(leaves)-2,
                  all_colours_retained=True, export_checks='passed')
    REPORT.write_text(json.dumps(report, indent=2))
    print('Installed two sheets; 135 other leaves and all colours retained', flush=True)


def check_motion():
    report = json.loads(REPORT.read_text())
    rows = json.loads((brackets.WORK / 'leaves.json').read_text())
    for part_report, fixed_indices, limits in zip(report['parts'], ((54, 55), (127, 128)), ((-30, 70), (-70, 30))):
        body_index = fixed_indices[1]
        passive = read_step(source(rows[body_index + 1])).bounding_box().center()
        active = read_step(source(rows[body_index + 2])).bounding_box().center()
        axis = bd.Axis((passive + active) * .5, (active - passive).normalized())
        candidate = read_step(part_report['file'])
        original = read_step(source(rows[part_report['index']]))
        candidate_manager = trimesh.collision.CollisionManager()
        original_manager = trimesh.collision.CollisionManager()
        candidate_manager.add_object('sheet', _mesh(candidate, .03, .1))
        original_manager.add_object('sheet', _mesh(original, .03, .1))
        checked = []
        for fixed_index in fixed_indices:
            fixed = read_step(source(rows[fixed_index]))
            fixed_manager = trimesh.collision.CollisionManager()
            fixed_manager.add_object('fixed', _mesh(fixed, .03, .1))
            minimum = float('inf')
            closest_angle = None
            existing_collisions = []
            clear_angles = []
            for angle in range(limits[0], limits[1] + 1):
                transform = trimesh.transformations.rotation_matrix(math.radians(angle), list(axis.direction), list(axis.position))
                candidate_manager.set_transform('sheet', transform)
                original_manager.set_transform('sheet', transform)
                collision = candidate_manager.in_collision_other(fixed_manager)
                old_collision = original_manager.in_collision_other(fixed_manager)
                if collision:
                    if not old_collision:
                        overlap = brackets.volume(candidate.rotate(axis, angle).intersect(fixed))
                        if overlap > .001:
                            raise ValueError(f'New motion collision at {angle}: {overlap}')
                    existing_collisions.append(angle)
                else:
                    clear_angles.append(angle)
                    gap = candidate_manager.min_distance_other(fixed_manager)
                    if gap < minimum:
                        minimum, closest_angle = gap, angle
            posed = candidate.rotate(axis, closest_angle)
            assert brackets.volume(posed.intersect(fixed)) < .001
            checked.append(dict(part=rows[fixed_index]['path'], min_clearance_at_clear_samples_mm=minimum,
                                closest_angle_deg=closest_angle, exact_closest_clearance_mm=posed.distance_to(fixed),
                                clear_angles_deg=clear_angles, existing_collision_angles_deg=existing_collisions))
        part_report['motion'] = dict(range_deg=limits, step_deg=1, fixed_parts=checked)
        print(f'{part_report["path"]}: previously clear hip samples remain clear; existing collisions recorded', flush=True)
        REPORT.write_text(json.dumps(report, indent=2))
    REPORT.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    refit._start_memory_watchdog()
    if '--install' in sys.argv:
        install()
    elif '--motion' in sys.argv:
        check_motion()
    else:
        build()
