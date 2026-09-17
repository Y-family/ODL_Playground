"""Give eight leg case brackets straight sides and flush screw mounting faces."""

import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

from cadgen import build123d as bd, read_step
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.TopTools import TopTools_ListOfShape

import refit_upper_servo_interfaces_local as refit


WORK = Path('/tmp/u07')
CENTRE_X_MM = -35.06
CASE_END_WIDTH_MM = 23.4
TARGET = refit.ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
SPACER_HOLE_HALF_PITCH_MM = 6.5


def volume(shape):
    return sum(item.volume for item in refit._items(shape))


def export_difference(left, right):
    arguments = TopTools_ListOfShape()
    arguments.Append(left.wrapped)
    cutters = TopTools_ListOfShape()
    cutters.Append(right.wrapped)
    operation = BRepAlgoAPI_Cut()
    operation.SetArguments(arguments)
    operation.SetTools(cutters)
    operation.SetFuzzyValue(1e-5)
    operation.Build()
    if not operation.IsDone():
        raise ValueError('Tolerant export comparison failed')
    return bd.Compound(operation.Shape()).volume


def bracket_rows():
    rows = json.loads((WORK / 'leaves.json').read_text())
    return [row for row in rows if row['label'] in (
        'left_knee_to_ankle_left_sheet', 'left_knee_to_ankle_right_sheet')]


def candidate_file(row):
    return WORK / (Path(row['file']).stem + '_candidate.step')


def affected_rows():
    rows = json.loads((WORK / 'leaves.json').read_text())
    return bracket_rows() + [row for row in rows if row['label'] == 'leg_spacer']


def capture_baseline():
    fingerprint = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    manifest = WORK / 'installed_baseline.json'
    if manifest.exists() and json.loads(manifest.read_text())['target'] == fingerprint:
        return
    for row in affected_rows():
        source = candidate_file(row)
        if not source.exists():
            source = Path(row['file'])
        shutil.copyfile(source, WORK / (Path(row['file']).stem + '_installed.step'))
    manifest.write_text(json.dumps({'target': fingerprint}))


def width_cut(part):
    return refit._cut(part, [
        refit._world_box(-150, CENTRE_X_MM - CASE_END_WIDTH_MM / 2, -200, 200, -250, 100),
        refit._world_box(CENTRE_X_MM + CASE_END_WIDTH_MM / 2, 150, -200, 200, -250, 100),
    ]).clean()


def spacer_hole_faces(part, height):
    matches = []
    for face in part.faces():
        if face.geom_type != bd.GeomType.CYLINDER:
            continue
        cylinder = refit.BRepAdaptor_Surface(face.wrapped).Cylinder()
        position = bd.Vector(cylinder.Location().Coord())
        direction = bd.Vector(cylinder.Axis().Direction().Coord())
        if (abs(abs(position.X - CENTRE_X_MM) - 8) < .001 and
                abs(position.Z - height) < .001 and abs(direction.Y) > .999 and
                cylinder.Radius() < 3.3):
            matches.append((face, cylinder.Radius(), position))
    if not matches:
        raise ValueError('Missing original spacer fastening holes')
    return matches


def relocate_spacer_holes(part, height, reference=None):
    holes = spacer_hole_faces(reference if reference is not None else part, height)
    for face, radius, position in holes:
        bounds = face.bounding_box()
        part = refit._fuse(part, refit._rod(bd.Vector(position.X, 0, height), bd.Vector(0, 1, 0),
                                          bounds.min.Y, bounds.max.Y, radius))
    for face, radius, position in holes:
        bounds = face.bounding_box()
        sign = 1 if position.X > CENTRE_X_MM else -1
        axis = bd.Vector(CENTRE_X_MM + sign * SPACER_HOLE_HALF_PITCH_MM, 0, height)
        part = refit._cut(part, [refit._rod(axis, bd.Vector(0, 1, 0),
                                          bounds.min.Y - 1e-5, bounds.max.Y + 1e-5, radius)])
    return part.clean(), holes


def validate_moved_holes(candidate, holes):
    verified = []
    for face, radius, position in holes:
        bounds = face.bounding_box()
        sign = 1 if position.X > CENTRE_X_MM else -1
        axis = bd.Vector(CENTRE_X_MM + sign * SPACER_HOLE_HALF_PITCH_MM, 0, position.Z)
        bore = refit._rod(axis, bd.Vector(0, 1, 0), bounds.min.Y + .001,
                          bounds.max.Y - .001, radius - .005)
        if volume(candidate.intersect(bore)) > .001:
            raise ValueError('Relocated spacer bore blocked')
        outer = refit._rod(axis, bd.Vector(0, 1, 0), bounds.min.Y + .001,
                           bounds.max.Y - .001, radius + 1.25)
        inner = refit._rod(axis, bd.Vector(0, 1, 0), bounds.min.Y,
                           bounds.max.Y, radius + .005)
        ring = refit._shape(outer.cut(inner))
        if volume(ring.cut(candidate)) > .001:
            raise ValueError(f'Relocated hole lacks 1.25 mm surrounding material: {position}, R{radius}')
        verified.append(dict(centre_x=axis.X, centre_z=axis.Z, radius=radius,
                             axial_min=bounds.min.Y, axial_max=bounds.max.Y))
    return verified


def build_spacer(row, all_rows):
    original = read_step(row['file'])
    sheet = next(item for item in all_rows if item['path'][0] == row['path'][0] and
                 item['label'] == 'left_knee_to_ankle_left_sheet')
    height = sheet['bounds'][5] - 47.825
    candidate, holes = relocate_spacer_holes(original, height)
    candidate = width_cut(candidate)
    if not candidate.is_valid or len(candidate.solids()) != 1:
        raise ValueError('Spacer must remain one valid solid')
    hole_report = validate_moved_holes(candidate, holes)
    bd.export_step(candidate, candidate_file(row))
    print(f'{row["path"]}: spacer narrowed, paired holes moved and supported', flush=True)
    return dict(part=row['path'], old_width_mm=original.bounding_box().size.X,
                new_width_mm=candidate.bounding_box().size.X,
                relocated_spacer_holes=hole_report, spacer_hole_pitch_mm=13)


def flush_mount(part, original):
    bosses = []
    for face in original.faces():
        if face.geom_type == bd.GeomType.CYLINDER:
            cylinder = refit.BRepAdaptor_Surface(face.wrapped).Cylinder()
            if abs(cylinder.Radius() - 4) < .001 and face.bounding_box().max.Z < -40:
                bosses.append(face)
    if len(bosses) != 2:
        raise ValueError(f'Expected two case mounting bosses, got {len(bosses)}')
    bounds = original.bounding_box()
    bottom = min(face.bounding_box().min.Z for face in bosses)
    top = bounds.max.Z - 54.54
    front = min(face.bounding_box().min.Y for face in bosses)
    back = max(face.bounding_box().max.Y for face in bosses)
    part = refit._cut(part, [refit._world_box(-100, 100, -200, 200, -250, top)])
    extension = refit._world_box(CENTRE_X_MM - CASE_END_WIDTH_MM / 2,
                                CENTRE_X_MM + CASE_END_WIDTH_MM / 2,
                                bounds.min.Y, bounds.max.Y, min(bottom, bounds.min.Z), top)
    part = refit._fuse(part, extension)
    case_holes = []
    for face in original.faces():
        if face.geom_type != bd.GeomType.CYLINDER:
            continue
        cylinder = refit.BRepAdaptor_Surface(face.wrapped).Cylinder()
        position = bd.Vector(cylinder.Location().Coord())
        if abs(cylinder.Radius() - 1) < .001 and bottom < position.Z < top:
            case_holes.append(position)
            part = refit._cut(part, [refit._rod(bd.Vector(position.X, 0, position.Z),
                                              bd.Vector(0, 1, 0), bounds.min.Y - 1,
                                              bounds.max.Y + 1, 1)])
    if len(case_holes) != 2:
        raise ValueError('Case fastening holes not found')
    return part.clean(), dict(bottom=min(bottom, bounds.min.Z), top=top, front=front, back=back)


def backing_face(part, height):
    bounds = part.bounding_box()
    faces = [face for face in part.faces() if face.geom_type == bd.GeomType.PLANE and
             face.bounding_box().size.Y < 1e-5 and face.bounding_box().min.Z < height and
             face.bounding_box().max.Z > bounds.max.Z - 2]
    if len(faces) < 2:
        raise ValueError('Main plate faces not found')
    faces = sorted(faces, key=lambda face: face.area, reverse=True)[:2]
    lower = min(face.center().Y for face in faces)
    upper = max(face.center().Y for face in faces)
    return refit._world_box(CENTRE_X_MM - CASE_END_WIDTH_MM / 2,
                            CENTRE_X_MM + CASE_END_WIDTH_MM / 2,
                            lower, upper, bounds.min.Z, height + 5)


def clear_leg_cover(candidate, row, all_rows):
    if row['label'] != 'left_knee_to_ankle_right_sheet' or row['path'][0] not in (
            'knee_and_ankle_assembly <1>', 'knee_and_ankle_assembly <2>'):
        return candidate, None
    name = 'right_cache' if row['path'][0].endswith('<1>') else 'left_cache'
    cache = read_step(next(item['file'] for item in all_rows if item['label'] == name))
    bounds = candidate.bounding_box()
    size = 3.5
    points = [(bounds.min.X, -200, bounds.min.Z),
              (bounds.min.X, -200, bounds.min.Z + size),
              (bounds.min.X + size, -200, bounds.min.Z)]
    triangle = bd.Face(bd.Wire.make_polygon(points, close=True))
    candidate = refit._cut(candidate, [bd.Solid.extrude(triangle, (0, 400, 0))]).clean()
    original = read_step(row['file'])
    old_gap = original.distance_to(cache)
    if volume(candidate.intersect(cache)) > .001 or candidate.distance_to(cache) < min(.2, old_gap) - 1e-5:
        raise ValueError('Leg cover interference or reduced existing clearance')
    return candidate, dict(part=name, lower_rear_chamfer_mm=size,
                           clearance_mm=candidate.distance_to(cache), old_clearance_mm=old_gap)


def install():
    report = json.loads((WORK / 'width_report.json').read_text())
    for filename, fingerprint in report['candidate_fingerprints'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != fingerprint:
            raise ValueError('Candidate changed after validation')
    robot = read_step(TARGET)
    baseline = []
    replaced = []
    rows = {tuple(row['path']): row for row in affected_rows()}
    if len(rows) != 12:
        raise ValueError('Expected eight brackets and four spacers')
    manifest = json.loads((WORK / 'installed_baseline.json').read_text())
    if hashlib.sha256(TARGET.read_bytes()).hexdigest() != manifest['target']:
        raise ValueError('Assembly changed since candidate preparation')

    def rebuild(node, path=()):
        children = refit._children(node)
        if children:
            fresh = bd.Compound(children=[rebuild(child, path + (child.label,))
                                          for child in children], label=node.label)
            fresh.location, fresh.color = node.location, node.color
            return fresh
        bounds = node.bounding_box()
        baseline.append((path, node.volume, list(bounds.min) + list(bounds.max),
                         list(node.color) if node.color else None))
        if path not in rows:
            return node
        original = read_step(WORK / (Path(rows[path]['file']).stem + '_installed.step'))
        if sum(item.volume for item in refit._items(node.cut(original))) > .001 or sum(
                item.volume for item in refit._items(original.cut(node))) > .001:
            raise ValueError('Extracted bracket no longer matches current assembly')
        candidate = read_step(candidate_file(rows[path]))
        fresh = bd.Compound(candidate.solids(), label=node.label)
        fresh.color = node.color
        replaced.append(path)
        return fresh

    result = refit._encode_colours(rebuild(robot))
    if set(replaced) != set(rows):
        raise ValueError(f'Unexpected replacement list: {replaced}')
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
    if len(leaves) != len(baseline):
        raise ValueError('Leaf count changed')
    export_checks = []
    for (path, node), (old_path, volume, bounds, colour) in zip(leaves, baseline):
        if path != old_path:
            raise ValueError('Assembly hierarchy changed')
        actual_colour = list(node.color) if node.color else None
        if (colour is None) != (actual_colour is None) or (colour is not None and any(
                abs(first - second) > 1e-6 for first, second in zip(colour, actual_colour))):
            raise ValueError(f'Colour changed: {path}')
        if path in rows:
            candidate = read_step(candidate_file(rows[path]))
            difference = (export_difference(node, candidate), export_difference(candidate, node))
            if (not node.is_valid or len(node.solids()) != 1 or
                    abs(node.volume - candidate.volume) > .001 or max(difference) > .001):
                raise ValueError(f'Exported candidate differs: {path}, differences={difference}')
            export_checks.append(dict(part=path, valid_single_solid=True,
                                      volume_delta_mm3=node.volume - candidate.volume,
                                      bidirectional_difference_mm3=difference))
        else:
            actual = node.bounding_box()
            if abs(node.volume - volume) > .001 or any(abs(first - second) > 1e-5 for
                    first, second in zip(bounds, list(actual.min) + list(actual.max))):
                raise ValueError(f'Unmodified leaf changed: {path}')
    temporary.replace(TARGET)
    report = json.loads((WORK / 'width_report.json').read_text())
    report.update(status='installed; U07 straight sides and U08 flush faces; awaiting visual acceptance',
                  unchanged_leaves=len(leaves) - len(rows), all_colours_retained=True,
                  export_checks=export_checks, export_comparison_tolerance_mm=1e-5)
    (refit.ROOT / 'cad/local_fix/U07_WIDTH_REPORT.json').write_text(json.dumps(report, indent=2))
    print(f'Updated {TARGET}; {len(leaves) - len(rows)} other leaves and all colours retained', flush=True)


def build_bracket(row, all_rows):
    original = read_step(row['file'])
    assembly = row['path'][0]
    servo_row = next(item for item in all_rows if item['path'][0] == assembly and
                     item['label'] == 'ZP10D_vendor_body')
    top = original.bounding_box().max.Z - 44.54
    spacer_height = top - 3.285
    candidate = refit._fuse(original, backing_face(original, spacer_height))
    candidate, moved_holes = relocate_spacer_holes(candidate, spacer_height, original)
    candidate = width_cut(candidate)
    candidate, flush_region = flush_mount(candidate, original)
    candidate, cover_clearance = clear_leg_cover(candidate, row, all_rows)
    candidate = bd.Compound(refit._shape_fix(candidate.wrapped))
    if not candidate.is_valid or len(candidate.solids()) != 1:
        raise ValueError('Width reduction must preserve one valid solid')
    hole_report = validate_moved_holes(candidate, moved_holes)
    for face in candidate.faces():
        if face.geom_type == bd.GeomType.CYLINDER:
            cylinder = refit.BRepAdaptor_Surface(face.wrapped).Cylinder()
            if abs(cylinder.Radius() - 4) < .001 and face.bounding_box().max.Z < -40:
                raise ValueError('Protruding round case mount remains')
    checked = 0
    open_bores = 0
    for face in original.faces():
        if face.geom_type != bd.GeomType.CYLINDER:
            continue
        cylinder = refit.BRepAdaptor_Surface(face.wrapped).Cylinder()
        radius = cylinder.Radius()
        position = bd.Vector(cylinder.Location().Coord())
        if abs(abs(position.X - CENTRE_X_MM) - 8) < .001 and abs(position.Z - spacer_height) < .001:
            continue
        if radius > 3.25 or abs(radius - 3) < .001:
            continue
        retained = sum(item.area for item in refit._items(face.intersect(candidate)))
        if face.area - retained > .001:
            raise ValueError(f'Width reduction damages cylindrical feature R{radius}')
        checked += 1
        direction = bd.Vector(cylinder.Axis().Direction().Coord())
        if abs(direction.Y) > .999:
            position = bd.Vector(cylinder.Location().Coord())
            face_bounds = face.bounding_box()
            bore = refit._rod(bd.Vector(position.X, 0, position.Z), bd.Vector(0, 1, 0),
                              face_bounds.min.Y + .001, face_bounds.max.Y - .001,
                              radius - .005)
            if volume(original.intersect(bore)) < .001:
                if volume(candidate.intersect(bore)) > .001:
                    raise ValueError(f'Blocked bore R{radius}')
                open_bores += 1
    added = volume(candidate.cut(original))
    servo = read_step(servo_row['file'])
    before_overlap = sum(item.volume for item in refit._items(original.intersect(servo)))
    after_overlap = sum(item.volume for item in refit._items(candidate.intersect(servo)))
    if after_overlap > .001:
        raise ValueError(f'Bracket intersects servo body: {after_overlap} mm3')
    added_shape = refit._shape(candidate.cut(original))
    added_bounds = added_shape.bounding_box()
    neighbours = []
    for neighbour_row in all_rows:
        if neighbour_row['file'] == row['file']:
            continue
        if any(neighbour_row['bounds'][axis] > list(added_bounds.max)[axis] or
               neighbour_row['bounds'][axis + 3] < list(added_bounds.min)[axis]
               for axis in range(3)):
            continue
        neighbour = read_step(neighbour_row['file'])
        overlap = sum(item.volume for item in refit._items(added_shape.intersect(neighbour)))
        if overlap > .001:
            raise ValueError(f'Inward wall thickening interferes with {neighbour_row["path"]}: {overlap}')
        neighbours.append(neighbour_row['path'])
    upper_region = refit._rod(
        bd.Vector(CENTRE_X_MM, 0, original.bounding_box().max.Z - 11),
        bd.Vector(0, 1, 0), original.bounding_box().min.Y - 1,
        original.bounding_box().max.Y + 1, 11)
    if volume(original.cut(candidate).intersect(upper_region)) > .001 or volume(
            candidate.cut(original).intersect(upper_region)) > .001:
        raise ValueError('Upper horn interface changed')
    report = dict(
        part=row['path'],
        servo=servo_row['path'][1],
        old_width_mm=original.bounding_box().size.X,
        new_width_mm=candidate.bounding_box().size.X,
        servo_width_mm=servo.bounding_box().size.X,
        case_end_width_mm=CASE_END_WIDTH_MM,
        case_mount_face='continuous planar face with two screw holes',
        side_profile='straight and parallel',
        spacer_hole_pitch_mm=2 * SPACER_HOLE_HALF_PITCH_MM,
        flush_region=flush_region,
        leg_cover_clearance=cover_clearance,
        relocated_spacer_holes=hole_report,
        retained_cylindrical_faces=checked,
        open_bores_verified=open_bores,
        added_volume_mm3=added,
        old_servo_overlap_mm3=before_overlap,
        new_servo_overlap_mm3=after_overlap,
        removed_volume_mm3=original.volume - candidate.volume,
        checked_added_material_neighbours=neighbours,
        status='local candidate only; not installed',
    )
    bd.export_step(candidate, candidate_file(row))
    print(f'{servo_row["path"][1]} / {row["label"]}: valid, holes retained, no new static interference', flush=True)
    return report


def main():
    refit._start_memory_watchdog()
    capture_baseline()
    all_rows = json.loads((WORK / 'leaves.json').read_text())
    rows = bracket_rows()
    if len(rows) != 8:
        raise ValueError('Expected four servos and eight brackets')
    reports = [build_bracket(row, all_rows) for row in rows]
    reports.extend(build_spacer(row, all_rows) for row in all_rows if row['label'] == 'leg_spacer')
    (WORK / 'width_report.json').write_text(json.dumps(dict(parts=reports), indent=2))
    validate_set()


def validate_set():
    rows = affected_rows()
    candidates = {tuple(row['path']): read_step(candidate_file(row)) for row in rows}
    for row in rows:
        part = candidates[tuple(row['path'])]
        original = read_step(row['file'])
        added = refit._shape(part.cut(original))
        bounds = added.bounding_box()
        for other_path, other in candidates.items():
            if other_path == tuple(row['path']):
                continue
            other_bounds = other.bounding_box()
            if any(list(bounds.min)[axis] > list(other_bounds.max)[axis] or
                   list(bounds.max)[axis] < list(other_bounds.min)[axis] for axis in range(3)):
                continue
            if volume(added.intersect(other)) > .001:
                raise ValueError(f'New interference between modified parts: {row["path"]}, {other_path}')
    report = json.loads((WORK / 'width_report.json').read_text())
    paired_axes = []
    for assembly in sorted({row['path'][0] for row in rows}):
        group = [part for path, part in candidates.items() if path[0] == assembly]
        sheet_row = next(row for row in rows if row['path'][0] == assembly and
                         row['label'] == 'left_knee_to_ankle_left_sheet')
        height = sheet_row['bounds'][5] - 47.825
        for sign in (-1, 1):
            axis = bd.Vector(CENTRE_X_MM + sign * SPACER_HOLE_HALF_PITCH_MM, 0, height)
            screw_path = refit._rod(axis, bd.Vector(0, 1, 0), -150, 150, .995)
            if any(volume(part.intersect(screw_path)) > .001 for part in group):
                raise ValueError(f'Paired screw passage obstructed: {assembly}')
            paired_axes.append(dict(assembly=assembly, centre_x=axis.X, centre_z=axis.Z))
    for part_report in report['parts']:
        part_report.pop('case_end_inner_width_mm', None)
        part_report.pop('retained_side_wall_mm', None)
        if 'flush_region' not in part_report:
            continue
        part_report['case_mount_face'] = 'continuous planar face with two screw holes'
        part = candidates[tuple(part_report['part'])]
        region = part_report['flush_region']
        outer = region['front'] if part_report['part'][-1].endswith('left_sheet') else region['back']
        if part.bounding_box().center().Y > 0:
            outer = region['back'] if part_report['part'][-1].endswith('left_sheet') else region['front']
        faces = [face for face in part.faces() if face.geom_type == bd.GeomType.PLANE and
                 face.bounding_box().size.Y < 1e-5 and abs(face.center().Y - outer) < 1e-4 and
                 face.bounding_box().size.X > CASE_END_WIDTH_MM - .01 and
                 face.bounding_box().min.Z <= region['bottom'] + 1e-4 and
                 face.bounding_box().max.Z >= region['top'] - 1e-4]
        if len(faces) != 1:
            raise ValueError(f'Expected one flush case face: {part_report["part"]}, got {len(faces)}')
        part_report['flush_face_area_mm2'] = faces[0].area
    report.update(paired_spacer_axes=paired_axes, modified_parts_interference='passed',
                  full_joint_motion_sweep='not performed',
                  candidate_fingerprints={str(candidate_file(row)): hashlib.sha256(
                      candidate_file(row).read_bytes()).hexdigest() for row in rows})
    (WORK / 'width_report.json').write_text(json.dumps(report, indent=2))
    print('Validated twelve modified parts, eight continuous mounting faces and eight paired spacer axes', flush=True)


if __name__ == '__main__':
    if '--validate' in sys.argv:
        refit._start_memory_watchdog()
        validate_set()
    elif '--install' in sys.argv:
        refit._start_memory_watchdog()
        install()
    else:
        main()
