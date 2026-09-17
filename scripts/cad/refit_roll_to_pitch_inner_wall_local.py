"""Thicken the inboard wall of right/left_roll_to_pitch to the original 5.0 mm.

The ZP10D refit left the inner end wall of both yellow roll-to-pitch brackets
3.526 mm thick where the original part is 5.000 mm.  This model keeps the
leg-facing surface (the clearance side towards the thigh) exactly where it is
and grows the wall 0.974 mm further inboard, as one fused solid, so no new
part, pad or spacer is introduced.  Growing it the full 1.474 mm to the
original thickness would close the last 1.246 mm of hip-roll clearance to
trunk_bottom at +25 deg (no collision is allowed and 0.2 mm must remain).
"""

import gc
import hashlib
import json
import math
import sys
from pathlib import Path

from cadgen import build123d as bd, read_step
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
import trimesh

import refit_case_bracket_width_local as brackets
import refit_upper_servo_interfaces_local as refit
from scan_s01_s14_motion_collisions import _mesh


WORK = Path('/tmp/u10')
TARGET = brackets.TARGET
REPORT = refit.ROOT / 'cad/local_fix/U10_THICKNESS_REPORT.json'
SOURCE_THICKNESS_MM = 3.526
TARGET_THICKNESS_MM = 4.5
EXTRA_MM = round(TARGET_THICKNESS_MM - SOURCE_THICKNESS_MM, 3)
CAP_Y_MM = 58.774
INDICES = (54, 127)


def rows():
    return json.loads((WORK / 'leaves.json').read_text())


def source_file(index):
    return WORK / f'leaf_{index}.step'


def inboard_faces(part):
    """Every face of the inner wall that points inboard, cap plane included.

    The wall's inboard end is a flat cap plus two short slanted steps outboard
    of it; moving only the flat cap would leave a 1.5 mm step on the plate, so
    the whole inboard end surface travels together and the existing profile is
    kept.
    """
    sign = -1 if part.bounding_box().center().Y < 0 else 1
    found = []
    for face in part.faces():
        box = face.bounding_box()
        if sign * box.max.Y < 49.0 or sign * box.min.Y > CAP_Y_MM + 0.4:
            continue
        if face.area < 5.0:
            continue
        normal = face.normal_at(face.center())
        if sign * normal.Y > -0.5:
            continue
        found.append(face)
    return sign, found


def build():
    WORK.mkdir(exist_ok=True)
    leaves = rows()
    reports = []
    for index in INDICES:
        original = read_step(source_file(index))
        sign, faces = inboard_faces(original)
        if not faces:
            raise ValueError(f'No inboard cap face found on leaf {index}')
        candidate = original
        for face in faces:
            candidate = refit._fuse(candidate, bd.Solid.extrude(face, (0, -sign * EXTRA_MM, 0)))
        candidate = candidate.clean()
        if not candidate.is_valid or len(candidate.solids()) != 1:
            raise ValueError('Thickened bracket is not a valid single solid')
        if brackets.export_difference(original, candidate) > .001:
            raise ValueError('Existing bracket material was removed')
        added = refit._shape(candidate.cut(original))
        bounds = added.bounding_box()
        neighbours = []
        for other_index, other in enumerate(leaves):
            if other_index == index:
                continue
            if any(other['bounds'][axis] > list(bounds.max)[axis] or
                   other['bounds'][axis + 3] < list(bounds.min)[axis] for axis in range(3)):
                continue
            neighbour = read_step(source_file(other_index))
            overlap = brackets.volume(added.intersect(neighbour))
            if overlap > .001:
                raise ValueError(f'Added material collides with {other["path"]}: {overlap}')
            neighbours.append(dict(part='/'.join(other['path']),
                                   added_clearance_mm=added.distance_to(neighbour)))
        bores = []
        for face in original.faces():
            if face.geom_type != bd.GeomType.CYLINDER:
                continue
            cylinder = refit.BRepAdaptor_Surface(face.wrapped).Cylinder()
            direction = bd.Vector(cylinder.Axis().Direction().Coord())
            if abs(abs(direction.Y) - 1) > .999 or cylinder.Radius() > 3.3:
                continue
            box = face.bounding_box()
            if sign * box.max.Y < CAP_Y_MM or sign * box.min.Y > CAP_Y_MM + SOURCE_THICKNESS_MM + EXTRA_MM:
                continue
            centre = bd.Vector(cylinder.Location().Coord())
            low, high = box.min.Y - 0.001, box.max.Y + 0.001
            if low == high:
                continue
            bore = refit._rod(bd.Vector(centre.X, 0, centre.Z), bd.Vector(0, 1, 0),
                              low, high, cylinder.Radius() - .005)
            if brackets.volume(added.intersect(bore)) > .001:
                raise ValueError('Added material obstructs an existing passage')
            bores.append(dict(x=centre.X, z=centre.Z, radius=cylinder.Radius(),
                              bore_span_mm=[round(low, 3), round(high, 3)]))
        output = WORK / f'wall_{index}.step'
        bd.export_step(candidate, output)
        reports.append(dict(index=index, path='/'.join(leaves[index]['path']), file=str(output),
                            before_mm=SOURCE_THICKNESS_MM, after_mm=TARGET_THICKNESS_MM,
                            inboard_extension_mm=EXTRA_MM, cap_faces=len(faces),
                            valid_single_solid=True, bores=bores, neighbours=neighbours))
        print(f'{reports[-1]["path"]}: {SOURCE_THICKNESS_MM} -> {TARGET_THICKNESS_MM} mm; '
              f'valid single solid, neighbours clear', flush=True)
        gc.collect()
    REPORT.write_text(json.dumps(dict(status='candidate', target_sha256=hashlib.sha256(
        TARGET.read_bytes()).hexdigest(), parts=reports, full_joint_motion='not checked'), indent=2))


def compare_shapes(node, candidate):
    """Identity evidence for one replaced leaf, placement-independently.

    The OCC boolean returns an empty intersection for two identical solids on
    the positive-Y (left) side, so identity is established from volume,
    surface area, centroid and the bidirectional vertex-to-surface distance,
    and whatever the boolean says is recorded rather than trusted blindly.
    """
    evidence = dict(
        volume_delta_mm3=abs(node.volume - candidate.volume),
        area_delta_mm2=abs(node.area - candidate.area),
        centroid_delta_mm=max(abs(a - b) for a, b in zip(node.center(), candidate.center())),
    )
    stride = max(1, len(candidate.vertices()) // 2000, len(node.vertices()) // 2000)
    evidence['max_vertex_distance_mm'] = max(
        max(node.distance_to(vertex) for vertex in candidate.vertices()[::stride]),
        max(candidate.distance_to(vertex) for vertex in node.vertices()[::stride]),
    )
    try:
        difference = [brackets.export_difference(node, candidate),
                      brackets.export_difference(candidate, node)]
    except Exception:
        difference = None
    evidence['boolean_difference_mm3'] = difference
    evidence['boolean_agrees'] = difference is not None and max(difference) < .001
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
    print(f'Installed {len(replacements)} walls; other leaves and all colours retained', flush=True)


def check_motion():
    """Rotate the thickened bracket about the hip roll and hip pitch axes."""
    report = json.loads(REPORT.read_text())
    leaves = rows()
    servo = ((54, 55, 56, 57), (127, 128, 129, 130))
    roll_axes = (bd.Axis(bd.Vector(-18.37, -30.25, -0.07), bd.Vector(1, 0, 0)),
                 bd.Axis(bd.Vector(-18.37, 30.25, -0.07), bd.Vector(1, 0, 0)))
    # The two S02 horns are the bolted output interface of this bracket (the
    # bracket is screwed to them and turns with them), so they are not motion
    # neighbours; everything the bracket moves against is listed here.
    roll_fixed = (28, 29, 30, 40, 112, 115, 116, 117, 120, 124)
    pitch_fixed = ((66, 67, 68, 69, 70, 71), (131, 132, 134, 135, 136, 133))
    for part_report, (bracket_index, body, passive, active) in zip(report['parts'], servo):
        candidate = read_step(part_report['file'])
        original = read_step(source_file(part_report['index']))
        centre_passive = read_step(source_file(passive)).bounding_box().center()
        centre_active = read_step(source_file(active)).bounding_box().center()
        pitch_axis = bd.Axis((centre_passive + centre_active) * .5, (centre_active - centre_passive).normalized())
        checks = []
        for label, axis, limits, fixed_indices in (
                ('hip_roll', roll_axes[0 if bracket_index == 54 else 1], (-25, 25), roll_fixed),
                ('hip_pitch', pitch_axis, (-30, 70) if bracket_index == 54 else (-70, 30),
                 pitch_fixed[0] if bracket_index == 54 else pitch_fixed[1])):
            candidate_manager = trimesh.collision.CollisionManager()
            original_manager = trimesh.collision.CollisionManager()
            candidate_manager.add_object('bracket', _mesh(candidate, .03, .1))
            original_manager.add_object('bracket', _mesh(original, .03, .1))
            parts = []
            for fixed_index in fixed_indices:
                fixed = read_step(source_file(fixed_index))
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
                                raise ValueError(f'New {label} collision at {angle} deg: {overlap} mm3')
                        existing.append(angle)
                    else:
                        clear.append(angle)
                        gap = candidate_manager.min_distance_other(fixed_manager)
                        if gap < minimum:
                            minimum, closest = gap, angle
                posed = candidate.rotate(axis, closest) if closest is not None else candidate
                parts.append(dict(part='/'.join(leaves[fixed_index]['path']),
                                  min_clearance_at_clear_samples_mm=minimum, closest_angle_deg=closest,
                                  exact_closest_clearance_mm=posed.distance_to(fixed),
                                  existing_collision_angles_deg=existing, clear_angles_deg=clear))
                print(f'  {label} vs {"/".join(leaves[fixed_index]["path"])}: '
                      f'{len(existing)} existing collisions, {len(clear)} clear', flush=True)
            checks.append(dict(joint=label, range_deg=limits, step_deg=1, fixed_parts=parts))
        part_report['motion'] = checks
        REPORT.write_text(json.dumps(report, indent=2))


def _ray_hits(part, direction):
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(part.wrapped, 1e-6)
    samples = []
    for x in (-25.0, -20.0, -10.0):
        for z in (-10.0, 0.0, 10.0):
            inter.Perform(gp_Lin(gp_Pnt(x, direction * 400.0, z), gp_Dir(0, -direction, 0)), -400, 400)
            values = sorted(float(inter.Pnt(i).Coord()[1]) for i in range(1, inter.NbPnt() + 1))
            merged = []
            for value in values:
                if not merged or abs(value - merged[-1]) > 1e-4:
                    merged.append(value)
            pairs = [(abs((merged[i] + merged[i + 1]) / 2), merged[i], merged[i + 1])
                     for i in range(0, len(merged) - 1, 2)]
            if not pairs:
                continue
            _, start, end = min(pairs, key=lambda pair: abs(pair[0] - 60.0))
            samples.append(dict(x=x, z=z, inboard_y=round(min(start, end), 4),
                                outboard_y=round(max(start, end), 4),
                                thickness_mm=round(abs(end - start), 4)))
    return samples


def wall_thickness(part, direction):
    """Measure the inner wall at nine X/Z stations by casting rays along Y.

    `direction` is the geometric side the material sits on (+1 = positive Y).
    The ray starts on that side and runs inboard, because the -400..400
    parameter range only reaches y <= 0 when the ray starts at y = -400; probing
    the right-hand bracket from the left end therefore reports no hit at all.
    The other direction is tried as a fallback and whichever produced hits is
    recorded.
    """
    for candidate in (direction, -direction):
        samples = _ray_hits(part, candidate)
        if samples:
            thicknesses = [sample["thickness_mm"] for sample in samples]
            return dict(ray_from=f'{"+" if candidate > 0 else "-"}Y', samples=samples,
                        samples_hit=len(samples), thickness_min_mm=min(thicknesses),
                        thickness_max_mm=max(thicknesses))
    return dict(ray_from=None, samples=[], samples_hit=0, thickness_min_mm=None,
                thickness_max_mm=None)


def verify():
    """Re-read the installed STEP and confirm U10 from the exported file."""
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
    thickness = {}
    for position, ((path, node), old) in enumerate(zip(found, leaves)):
        if path != '/'.join(old['path']):
            raise ValueError(f'Leaf order changed at {path}')
        if node.label in ('right_roll_to_pitch', 'left_roll_to_pitch'):
            direction = 1 if node.bounding_box().center().Y > 0 else -1
            before = wall_thickness(read_step(old['file']), direction)
            after = wall_thickness(node, direction)
            if not after['samples_hit']:
                raise ValueError(f'No ray hit on the installed {path}')
            if abs(after['thickness_min_mm'] - TARGET_THICKNESS_MM) > .01:
                raise ValueError(f'{path} wall is {after["thickness_min_mm"]} mm, expected {TARGET_THICKNESS_MM}')
            if after['thickness_min_mm'] <= before['thickness_min_mm']:
                raise ValueError(f'{path} wall was not thickened: {before} -> {after}')
            thickness[position] = dict(path=path, volume_delta_mm3=round(node.volume - old['volume'], 3),
                                    before=before, after=after)
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
    report['reread_verification'] = dict(
        leaves=len(found), untouched_leaves=len(found) - len(thickness), wall_thickness=thickness,
        note='Thickness is the inboard/outboard Y hit pair whose midpoint is nearest Y=60; the wall '
             'face itself is a B-spline, not a plane. Rays are cast from the side the bracket sits '
             'on, because the -400..400 parameter range only covers that half-space.')
    report['status'] = 'installed; re-verified from the exported STEP; awaiting visual acceptance'
    REPORT.write_text(json.dumps(report, indent=2))
    for index, entry in thickness.items():
        print(f'  {entry["path"]}: {entry["before"]["thickness_min_mm"]} -> '
              f'{entry["after"]["thickness_min_mm"]} mm, volume +{entry["volume_delta_mm3"]} mm3', flush=True)
    print(f'Re-read {len(found)} leaves; {len(found) - len(thickness)} untouched leaves identical', flush=True)


if __name__ == '__main__':
    refit._start_memory_watchdog()
    if '--install' in sys.argv:
        install()
    elif '--motion' in sys.argv:
        check_motion()
    elif '--verify' in sys.argv:
        verify()
    else:
        build()
