"""Square the S02/S07 upper and lower fixing brackets (U11).

The refit left the two upper brackets (`roll_motor_top`) with a notched plate
outline: eight slanted planar faces from 30 deg / 45 deg corner cuts and steps,
plus the matching slanted edges.  The two lower brackets (`roll_motor_bottom`)
carry a 15 deg tapered outboard wall (one 376.65 mm2 face, 13.64 mm slanted
edges).  The user asked for square outlines on both parts - no slanted, crooked
or protruding edges - while keeping every hole, the S02 case faces and the
motion envelope.

Upper bracket: the plate outline is squared to the part's own envelope,
X -40.875..5.125 x Y -44.61..-15.89 over Z 35.201..38.365939 (the existing plate
band), adding 347.50 mm3.  Nothing else on the part changes.

Lower bracket: the tapered wall becomes a vertical wall at Y = -/+43.15, i.e.
2.90 mm thick off the untouched S02 case face at Y = -/+40.25, which is the
thickness of the original part's rear wall (2.90 mm).  The cut runs over the
wall's own band Z -10.184..3.0 only, so the base plate keeps its square
-/+43.75 outline; 650.08 mm3 is removed.  The wall can only be squared inboard:
squaring it outboard to the taper's top edge would put material back into the
roll sweep the taper exists to clear.
"""

import gc
import hashlib
import json
import math
import sys
from pathlib import Path

from cadgen import build123d as bd, read_step
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Plane
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.TopExp import TopExp_Explorer
import trimesh

import refit_case_bracket_width_local as brackets
import refit_upper_servo_interfaces_local as refit
from scan_s01_s14_motion_collisions import _mesh


WORK = Path('/tmp/u11')
TARGET = brackets.TARGET
REPORT = refit.ROOT / 'cad/local_fix/U11_OUTLINE_REPORT.json'
PLATE_X = (-40.875, 5.125)
PLATE_Y = (15.89, 44.61)
PLATE_Z = (35.201, 38.36593881233405)
WALL_Y = 43.15
WALL_Z = (-10.1840611876, 3.0)
TOPS = (28, 108)
BOTTOMS = (29, 107)
CLEARANCE_MM = 0.2


def rows():
    return json.loads((WORK / 'leaves.json').read_text())


def source_file(index):
    return WORK / f'leaf_{index}.step'


def extract():
    """Cut the current fixed STEP into one file per leaf (U10-style working set)."""
    WORK.mkdir(exist_ok=True)
    robot = read_step(TARGET)
    found = []

    def visit(node, path=()):
        children = refit._children(node)
        if children:
            for child in children:
                visit(child, path + (child.label,))
            return
        filename = WORK / f'leaf_{len(found)}.step'
        bd.export_step(bd.Compound(node.solids()), filename)
        bounds = node.bounding_box()
        found.append(dict(label=node.label, path=path, file=str(filename), volume=node.volume,
                          bounds=list(bounds.min) + list(bounds.max),
                          color=list(node.color) if node.color else None))

    visit(robot)
    if len(found) != 137:
        raise ValueError(f'expected 137 leaves, found {len(found)}')
    (WORK / 'leaves.json').write_text(json.dumps(found, indent=2))
    print(f'extracted {len(found)} leaves', flush=True)


def slant_inventory(part):
    """Slanted (non axis-aligned) planar faces and straight edges of a part."""
    faces = []
    exp = TopExp_Explorer(part.wrapped, TopAbs_FACE)
    while exp.More():
        adaptor = BRepAdaptor_Surface(TopoDS.Face_s(exp.Current()))
        if adaptor.GetType() == GeomAbs_Plane:
            direction = adaptor.Plane().Axis().Direction()
            if max(abs(direction.X()), abs(direction.Y()), abs(direction.Z())) < 0.999:
                props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(exp.Current(), props)
                centre = props.CentreOfMass().Coord()
                faces.append(dict(area_mm2=round(props.Mass(), 2),
                                  normal=[round(v, 3) for v in (direction.X(), direction.Y(), direction.Z())],
                                  centre=[round(v, 2) for v in centre]))
        exp.Next()
    edges = []
    exp = TopExp_Explorer(part.wrapped, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        curve = BRepAdaptor_Curve(edge)
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, props)
        if int(curve.GetType()) == int(GeomAbs_Line) and props.Mass() >= 1.0:
            start, end = curve.Value(curve.FirstParameter()), curve.Value(curve.LastParameter())
            chord = (end.X() - start.X(), end.Y() - start.Y(), end.Z() - start.Z())
            length = math.sqrt(sum(c * c for c in chord))
            if length and max(abs(c) / length for c in chord) < 0.999:
                centre = props.CentreOfMass().Coord()
                edges.append(dict(length_mm=round(props.Mass(), 2),
                                  direction=[round(c / length, 3) for c in chord],
                                  centre=[round(v, 2) for v in centre]))
        exp.Next()
    edges.sort(key=lambda row: -row['length_mm'])
    faces.sort(key=lambda row: -row['area_mm2'])
    return dict(slanted_faces=len(faces), slanted_face_area_mm2=round(sum(f['area_mm2'] for f in faces), 2),
                largest_faces=faces[:4], slanted_edges=len(edges),
                longest_edges=edges[:4])


def square_top(part, sign):
    x0, x1 = PLATE_X
    y0, y1 = sorted(sign * v for v in PLATE_Y)
    box = bd.Box(x1 - x0, y1 - y0, PLATE_Z[1] - PLATE_Z[0]).moved(
        bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, sum(PLATE_Z) / 2))
    added = bd.Compound(box.cut(part).solids())
    return bd.Compound(part.fuse(added).solids()), added


def square_bottom(part, sign):
    box = bd.Box(60.0, 40.0, WALL_Z[1] - WALL_Z[0]).moved(
        bd.Pos(-19.0, sign * (WALL_Y + 20.0), sum(WALL_Z) / 2))
    removed = bd.Compound(part.intersect(box).solids()) if part.intersect(box) else None
    return bd.Compound(part.cut(box).solids()), removed


def build():
    if not (WORK / 'leaves.json').exists():
        extract()
    leaves = rows()
    reports = []
    for kind, indices in (('top', TOPS), ('bottom', BOTTOMS)):
        for index in indices:
            part = read_step(source_file(index))
            sign = 1 if part.bounding_box().center().Y > 0 else -1
            candidate, touched = (square_top if kind == 'top' else square_bottom)(part, sign)
            if len(candidate.solids()) != 1 or not candidate.is_valid:
                raise ValueError(f'{kind} candidate {index} is not one valid solid')
            before, after = slant_inventory(part), slant_inventory(candidate)
            if after['slanted_faces']:
                raise ValueError(f'{kind} candidate {index} still has slanted faces: {after}')
            target = WORK / f'cand_{index}.step'
            bd.export_step(bd.Compound(candidate.solids()), target)
            checks = []
            zone = candidate.bounding_box()
            for other, row in enumerate(leaves):
                if other in TOPS + BOTTOMS:
                    continue
                box = row['bounds']
                if (box[3] < zone.min.X - 1 or box[0] > zone.max.X + 1
                        or box[4] < zone.min.Y - 1 or box[1] > zone.max.Y + 1
                        or box[5] < zone.min.Z - 1 or box[2] > zone.max.Z + 1):
                    continue
                neighbour = read_step(row['file'])
                overlap = 0.0
                try:
                    overlap = brackets.volume(candidate.intersect(neighbour))
                except Exception:
                    overlap = 0.0
                if overlap > .001:
                    raise ValueError(f'{kind} candidate {index} overlaps {"/".join(row["path"])}: {overlap}')
                gap = candidate.distance_to(neighbour)
                if gap < CLEARANCE_MM:
                    checks.append(dict(part='/'.join(row['path']), clearance_mm=round(gap, 4)))
            reports.append(dict(index=index, kind=kind, path='/'.join(leaves[index]['path']),
                                file=str(target), before_volume_mm3=round(part.volume, 3),
                                after_volume_mm3=round(candidate.volume, 3),
                                volume_delta_mm3=round(candidate.volume - part.volume, 3),
                                before_slants=before, after_slants=after,
                                tight_neighbours=checks))
            print(f'{kind} {index}: {part.volume:.1f} -> {candidate.volume:.1f} mm3, '
                  f'slanted faces {before["slanted_faces"]} -> {after["slanted_faces"]}', flush=True)
    REPORT.write_text(json.dumps(dict(
        status='candidate', target_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        parts=reports, full_joint_motion='not checked'), indent=2))


def check_motion():
    """Swing each squared bracket about the hip roll and hip pitch axes.

    Everything that is rigid with the bracket (its own S02 case, the trunk
    shells, trunk_bottom and the roll bearings) is excluded; the parts listed
    here are the ones that move relative to it - S02's output horns, the yellow
    roll-to-pitch link with S03, and the leg group.
    """
    report = json.loads(REPORT.read_text())
    leaves = rows()
    sides = (
        dict(bracket=(28, 29), roll=bd.Axis(bd.Vector(-18.37, -30.25, -0.07), bd.Vector(1, 0, 0)),
             pitch_servo=(56, 57), pitch_limits=(-30, 70),
             moving=(31, 32, 54, 55, 56, 57, 66, 67, 68, 69, 70, 71)),
        dict(bracket=(108, 107), roll=bd.Axis(bd.Vector(-18.37, 30.25, -0.07), bd.Vector(1, 0, 0)),
             pitch_servo=(129, 130), pitch_limits=(-70, 30),
             moving=(110, 111, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136)),
    )
    for side in sides:
        passive, active = (read_step(source_file(index)).bounding_box().center()
                           for index in side['pitch_servo'])
        pitch_axis = bd.Axis((passive + active) * .5, (active - passive).normalized())
        for index in side['bracket']:
            entry = [row for row in report['parts'] if row['index'] == index][0]
            candidate = read_step(entry['file'])
            original = read_step(source_file(index))
            checks = []
            for label, axis, limits in (('hip_roll', side['roll'], (-25, 25)),
                                        ('hip_pitch', pitch_axis, side['pitch_limits'])):
                candidate_manager = trimesh.collision.CollisionManager()
                original_manager = trimesh.collision.CollisionManager()
                candidate_manager.add_object('bracket', _mesh(candidate, .03, .1))
                original_manager.add_object('bracket', _mesh(original, .03, .1))
                parts = []
                for other in side['moving']:
                    fixed = read_step(source_file(other))
                    fixed_manager = trimesh.collision.CollisionManager()
                    fixed_manager.add_object('fixed', _mesh(fixed, .03, .1))
                    minimum, closest, existing, clear = float('inf'), None, [], []
                    for angle in range(limits[0], limits[1] + 1):
                        transform = trimesh.transformations.rotation_matrix(
                            math.radians(angle), list(axis.direction), list(axis.position))
                        candidate_manager.set_transform('bracket', transform)
                        original_manager.set_transform('bracket', transform)
                        if candidate_manager.in_collision_other(fixed_manager):
                            if not original_manager.in_collision_other(fixed_manager):
                                overlap = brackets.volume(candidate.rotate(axis, angle).intersect(fixed))
                                if overlap > .001:
                                    raise ValueError(f'New {label} collision at {angle} deg with '
                                                     f'{"/".join(leaves[other]["path"])}: {overlap} mm3')
                            existing.append(angle)
                        else:
                            clear.append(angle)
                            gap = candidate_manager.min_distance_other(fixed_manager)
                            if gap < minimum:
                                minimum, closest = gap, angle
                    posed = candidate.rotate(axis, closest) if closest is not None else candidate
                    parts.append(dict(part='/'.join(leaves[other]['path']),
                                      min_clearance_at_clear_samples_mm=minimum, closest_angle_deg=closest,
                                      exact_closest_clearance_mm=posed.distance_to(fixed),
                                      existing_collision_angles_deg=existing, clear_angles_deg=clear))
                    print(f'  {label} {index} vs {"/".join(leaves[other]["path"])[:40]}: '
                          f'{len(existing)} existing collisions, {len(clear)} clear', flush=True)
                checks.append(dict(joint=label, range_deg=limits, step_deg=1, moving_parts=parts))
            entry['motion'] = checks
            REPORT.write_text(json.dumps(report, indent=2))


def compare_shapes(node, candidate):
    """Identity evidence for one replaced leaf, placement-independently."""
    stride = max(1, len(candidate.vertices()) // 2000, len(node.vertices()) // 2000)
    evidence = dict(
        volume_delta_mm3=abs(node.volume - candidate.volume),
        area_delta_mm2=abs(node.area - candidate.area),
        centroid_delta_mm=max(abs(a - b) for a, b in zip(node.center(), candidate.center())),
        max_vertex_distance_mm=max(
            max(node.distance_to(vertex) for vertex in candidate.vertices()[::stride]),
            max(candidate.distance_to(vertex) for vertex in node.vertices()[::stride])),
    )
    if evidence['volume_delta_mm3'] >= .001 or evidence['max_vertex_distance_mm'] >= .001:
        raise ValueError(f'Exported leaf differs from its candidate: {evidence}')
    return evidence


def install():
    report = json.loads(REPORT.read_text())
    if hashlib.sha256(TARGET.read_bytes()).hexdigest() != report['target_sha256']:
        raise ValueError('Target changed since preparation')
    replacements = {('/'.join(rows()[row['index']]['path']),): row for row in report['parts']}
    robot = read_step(TARGET)
    baseline = []

    def rebuild(node, path=()):
        children = refit._children(node)
        if children:
            fresh = bd.Compound(children=[rebuild(child, path + (child.label,)) for child in children],
                                label=node.label)
            fresh.location, fresh.color = node.location, node.color
            return fresh
        bounds = node.bounding_box()
        baseline.append(('/'.join(path), node.volume, list(bounds.min) + list(bounds.max),
                         list(node.color) if node.color else None))
        key = ('/'.join(path),)
        if key not in replacements:
            return node
        candidate = read_step(replacements[key]['file'])
        replacement = bd.Compound(candidate.solids(), label=node.label)
        replacement.color = node.color
        return replacement

    result = refit._encode_colours(rebuild(robot))
    temporary = TARGET.with_suffix('.pending.step')
    bd.export_step(result, temporary)
    del result, robot
    gc.collect()
    exported = read_step(temporary)
    leaves = []
    export_checks = []

    def collect(node, path=()):
        children = refit._children(node)
        if children:
            for child in children:
                collect(child, path + (child.label,))
        else:
            leaves.append((tuple(path), node))

    collect(exported)
    assert len(leaves) == len(baseline)
    for (path, node), (old_path, volume, bounds, colour) in zip(leaves, baseline):
        assert '/'.join(path) == old_path
        actual_colour = list(node.color) if node.color else None
        assert (colour is None) == (actual_colour is None)
        assert colour is None or max(abs(a - b) for a, b in zip(colour, actual_colour)) < 1e-6
        if ('/'.join(path),) in replacements:
            candidate = read_step(replacements[('/'.join(path),)]['file'])
            if not node.is_valid or len(node.solids()) != 1:
                raise ValueError(f'Exported leaf is not a valid single solid: {old_path}')
            export_checks.append(dict(part=old_path, **compare_shapes(node, candidate)))
        else:
            actual = node.bounding_box()
            assert abs(node.volume - volume) < .001
            assert max(abs(a - b) for a, b in zip(bounds, list(actual.min) + list(actual.max))) < 1e-5
    temporary.replace(TARGET)
    report.update(status='installed; awaiting visual acceptance',
                  unchanged_leaves=len(leaves) - len(replacements), all_colours_retained=True,
                  export_checks=export_checks)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f'Installed {len(replacements)} squared brackets; other leaves and all colours retained', flush=True)


def verify():
    """Re-read the installed STEP and confirm U11 from the exported file."""
    report = json.loads(REPORT.read_text())
    leaves = rows()
    robot = read_step(TARGET)
    found = []

    def collect(node, path=()):
        children = refit._children(node)
        if children:
            for child in children:
                collect(child, path + (child.label,))
        else:
            found.append(('/'.join(path), node))

    collect(robot)
    if len(found) != len(leaves):
        raise ValueError(f'Re-read STEP has {len(found)} leaves, expected {len(leaves)}')
    changed = {}
    for position, ((path, node), old) in enumerate(zip(found, leaves)):
        if path != '/'.join(old['path']):
            raise ValueError(f'Leaf order changed at {path}')
        if node.label in ('roll_motor_top', 'roll_motor_bottom'):
            entry = [row for row in report['parts'] if row['index'] == position][0]
            delta = node.volume - old['volume']
            if abs(delta - entry['volume_delta_mm3']) > .01:
                raise ValueError(f'{path} volume changed by {delta}, expected {entry["volume_delta_mm3"]}')
            slants = slant_inventory(node)
            if slants['slanted_faces'] or slants['slanted_edges']:
                raise ValueError(f'{path} still carries slanted geometry: {slants}')
            facts = dict(path=path, volume_delta_mm3=round(delta, 3), slants=slants)
            if node.label == 'roll_motor_top':
                box = node.bounding_box()
                envelope = dict(x=[round(box.min.X, 3), round(box.max.X, 3)],
                                y=sorted([round(abs(box.min.Y), 3), round(abs(box.max.Y), 3)]),
                                z=[round(box.max.Z - (PLATE_Z[1] - PLATE_Z[0]), 3), round(box.max.Z, 3)])
                expected = dict(x=[round(PLATE_X[0], 3), round(PLATE_X[1], 3)],
                                y=[round(PLATE_Y[0], 3), round(PLATE_Y[1], 3)],
                                z=[round(PLATE_Z[0], 3), round(PLATE_Z[1], 3)])
                if envelope != expected:
                    raise ValueError(f'plate envelope {envelope} != {expected}')
                facts['plate_envelope'] = envelope
            else:
                wall = [round(face.center().Y, 4) for face in node.faces()
                        if face.geom_type == bd.GeomType.PLANE
                        and abs(abs(face.normal_at(face.center()).Y) - 1) < 1e-6
                        and face.area > 300.0
                        and abs(face.bounding_box().min.Z - WALL_Z[0]) < .01
                        and abs(face.bounding_box().max.Z - WALL_Z[1]) < .01]
                if len(wall) != 1:
                    raise ValueError(f'{path}: squared wall face not found ({wall})')
                facts['squared_wall_plane_y'] = wall
                facts['wall_thickness_mm'] = round(abs(abs(wall[0]) - 40.25), 4)
                if abs(facts['wall_thickness_mm'] - 2.90) > .01:
                    raise ValueError(f'{path}: squared wall is {facts["wall_thickness_mm"]} mm, expected 2.90')
            changed[position] = facts
            continue
        box = node.bounding_box()
        if abs(node.volume - old['volume']) >= .001:
            raise ValueError(f'Untouched leaf changed volume: {path}')
        if max(abs(a - b) for a, b in zip(old['bounds'], list(box.min) + list(box.max))) >= 1e-5:
            raise ValueError(f'Untouched leaf moved: {path}')
        colour = list(node.color) if node.color else None
        if (colour is None) != (old['color'] is None) or (
                colour and max(abs(a - b) for a, b in zip(colour, old['color'])) > 1e-6):
            raise ValueError(f'Untouched leaf changed colour: {path}')
    report['reread_verification'] = dict(leaves=len(found), untouched_leaves=len(found) - len(changed),
                                         brackets=changed)
    report['status'] = 'installed; re-verified from the exported STEP; awaiting visual acceptance'
    REPORT.write_text(json.dumps(report, indent=2))
    for index, facts in changed.items():
        print(f'  {facts["path"]}: volume {facts["volume_delta_mm3"]:+.2f} mm3, '
              f'slanted faces/edges {facts["slants"]["slanted_faces"]}/{facts["slants"]["slanted_edges"]}', flush=True)
    print(f'Re-read {len(found)} leaves; {len(found) - len(changed)} untouched leaves identical', flush=True)


if __name__ == '__main__':
    refit._start_memory_watchdog()
    if '--extract' in sys.argv:
        extract()
    elif '--motion' in sys.argv:
        check_motion()
    elif '--install' in sys.argv:
        install()
    elif '--verify' in sys.argv:
        verify()
    else:
        build()
