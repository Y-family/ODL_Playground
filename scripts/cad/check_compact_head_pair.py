"""Confirm one local motion pair with exact STEP intersections, serially."""

import gc
import json
import sys

import compact_head_local as compact
from cadgen import build123d as bd, read_step
import refit_upper_servo_interfaces_local as refit


def static_all():
    import trimesh
    from scan_s01_s14_motion_collisions import _mesh

    compact.rows()
    manager = trimesh.collision.CollisionManager()
    indices = compact.HEAD_LEAVES | {12, 17, 18, 19, 20, 21}
    for index in sorted(indices):
        print('Meshing', index, flush=True)
        prefix = 'candidate' if index in compact.HEAD_LEAVES | {12} else 'leaf'
        shape = read_step(compact.WORK / f'{prefix}_{index}.step')
        manager.add_object(str(index), _mesh(shape, .25, .3))
        del shape
        gc.collect()
    _, _, contacts = manager.in_collision_internal(return_names=True, return_data=True)
    pairs = sorted({tuple(sorted(map(int, contact.names))) for contact in contacts if contact.depth > .05})
    del manager, contacts
    gc.collect()
    results = []
    for first, second in pairs:
        volumes = {}
        for mode in ('leaf', 'candidate'):
            shapes = [read_step(compact.WORK / f'{mode if mode == "candidate" and index in compact.HEAD_LEAVES | {12} else "leaf"}_{index}.step')
                      for index in (first, second)]
            overlap = shapes[0].intersect(shapes[1])
            volumes[mode] = sum(piece.volume for piece in refit._items(overlap))
            del shapes, overlap
            gc.collect()
        results.append(dict(pair=[first, second], baseline_mm3=volumes['leaf'], candidate_mm3=volumes['candidate']))
        print(results[-1], flush=True)
    output = compact.TARGET.parent / 'HEAD_COMPACT_STATIC_REPORT.json'
    output.write_text(json.dumps(dict(status='coarse diagnostic; not a clearance certificate',
                                     mesh_tolerance_mm=.25, mesh_angular_tolerance=.3, pairs=results), indent=2))


def inspect_part():
    prefix = sys.argv[3] if len(sys.argv) > 3 else 'candidate'
    shape = read_step(compact.WORK / f'{prefix}_{int(sys.argv[2])}.step')
    print('bounds', shape.bounding_box(), flush=True)
    for face in shape.faces():
        if face.geom_type == bd.GeomType.CYLINDER:
            cylinder = compact.BRepAdaptor_Surface(face.wrapped).Cylinder()
            if cylinder.Radius() > (1 if int(sys.argv[2]) in (85, 91, 102) else 15):
                print('cylinder', cylinder.Radius(), cylinder.Location().Coord(),
                      cylinder.Axis().Direction().Coord(), face.bounding_box(), flush=True)
        elif int(sys.argv[2]) == 85 and face.geom_type == bd.GeomType.PLANE and abs(face.normal_at().Z) > .99:
            print('horizontal face', face.area, face.bounding_box(), flush=True)


def antenna_seats():
    shell = read_step(compact.WORK / 'leaf_82.step')
    motor = read_step(compact.WORK / 'leaf_91.step')
    print('original motor/shell distance', motor.distance_to(shell), flush=True)
    for face in motor.faces():
        if face.distance_to(shell) < .001:
            print('motor contact', str(face.geom_type), face.area, face.bounding_box(), flush=True)
    bounds = motor.bounding_box()
    for face in shell.faces():
        box = face.bounding_box()
        if any(lower > upper + 2 for lower, upper in zip(box.min, bounds.max)) or any(upper < lower - 2 for upper, lower in zip(box.max, bounds.min)):
            continue
        if face.distance_to(motor) < .001:
            print('shell contact', str(face.geom_type), face.area, face.bounding_box(), flush=True)


def speaker_checks():
    original = read_step(compact.WORK / 'leaf_85.step').moved(bd.Pos(-8, compact.SHORTEN_Y_SIDE_MM, 0))
    stand = read_step(compact.WORK / 'candidate_85.step')
    for face in original.faces():
        if face.geom_type != bd.GeomType.CYLINDER:
            continue
        cylinder = compact.BRepAdaptor_Surface(face.wrapped).Cylinder()
        if not .7 <= cylinder.Radius() <= 2:
            continue
        point = bd.Vector(cylinder.Location().Coord())
        direction = bd.Vector(cylinder.Axis().Direction().Coord())
        lower, upper = refit._projection(face, point, direction)
        region = refit._rod(point, direction, lower, upper, cylinder.Radius() + 1)
        expected = original
        if abs(direction.Z) < .9:
            shift = bd.Pos(compact.SPEAKER_FORWARD_MM, compact.SPEAKER_INWARD_MM, 0)
            region = region.moved(shift)
            expected = original.moved(shift)
        before = sum(piece.volume for piece in refit._items(expected.intersect(region)))
        after = sum(piece.volume for piece in refit._items(stand.intersect(region)))
        assert abs(before - after) < .001, ('mount material', list(point), before, after)
    print('Speaker mount annuli retained: roof holes stationary, device holes translated', flush=True)
    ring = read_step(compact.WORK / 'candidate_53.step')
    axes = [(bd.Axis((75.95, .1, 196.85093881236), (-1, 0, 0)), range(-30, 31, 2))]
    findings = []
    for index in sorted(compact.SPEAKER):
        shape = read_step(compact.WORK / f'candidate_{index}.step')
        for fixed_index in (82, 90, 98, 78, 89, 91, 102):
            fixed = read_step(compact.WORK / f'candidate_{fixed_index}.step')
            overlap = sum(piece.volume for piece in refit._items(shape.intersect(fixed)))
            print('speaker static', index, fixed_index, overlap, flush=True)
            findings.append(dict(part=index, fixed=fixed_index, overlap_mm3=overlap))
        minimum, maximum = 1000, 0
        for axis, angles in axes:
            for angle in angles:
                posed = shape.rotate(axis, angle)
                minimum = min(minimum, posed.distance_to(ring))
                maximum = max(maximum, sum(piece.volume for piece in refit._items(posed.intersect(ring))))
        print('speaker roll', index, 'overlap', maximum, 'min_gap', minimum, flush=True)
        findings.append(dict(part=index, joint='S14', angles=[-30, 30, 2], overlap_mm3=maximum, minimum_gap_mm=minimum))
    (compact.WORK / 'speaker_checks.json').write_text(json.dumps(findings, indent=2))


def rear_layout():
    turn, tilt, horizontal, lateral, height = map(float, sys.argv[2:7])
    motor = read_step(compact.WORK / 'candidate_91.step')
    bounds = motor.bounding_box()
    pivot = (bounds.min + bounds.max) * .5
    shell = read_step(compact.WORK / 'candidate_82.step')
    ring = read_step(compact.WORK / 'candidate_53.step')
    axis = bd.Axis((65.95, .1, 196.85093881236), (-1, 0, 0))
    for index in (91, 22, 10):
        original = read_step(compact.WORK / f'candidate_{index}.step')
        candidate = original.moved(bd.Pos(*(-pivot))).rotate(bd.Axis.Z, turn).rotate(bd.Axis.X, tilt).moved(bd.Pos(horizontal, lateral, height))
        print(index, candidate.bounding_box(), flush=True)
        overlap = sum(piece.volume for piece in refit._items(candidate.intersect(shell)))
        maximum, closest = 0, 1000
        for angle in range(-30, 31, 5):
            posed = candidate.rotate(axis, angle)
            maximum = max(maximum, sum(piece.volume for piece in refit._items(posed.intersect(ring))))
            closest = min(closest, posed.distance_to(ring))
            del posed
            gc.collect()
        print('shell_overlap', overlap, 'ring_max_overlap', maximum, 'ring_min_distance', closest, flush=True)
        bd.export_step(candidate, compact.WORK / f'probe_rear_{index}.step')


def housing_layout():
    angle = float(sys.argv[2])
    offset = tuple(map(float, sys.argv[3:6])) if len(sys.argv) >= 6 else (0, 0, 0)
    motor = read_step(compact.WORK / 'candidate_91.step')
    shaft = bd.Axis((-51.63914929544, 60.762837687316, 197.55500826739),
                    (.26308374005901913, -.9354088535420682, -.23621435695501722))
    candidate = motor.rotate(shaft, angle).moved(bd.Pos(*offset))
    print('motor bounds', candidate.bounding_box(), flush=True)
    for index in (82, 89, 22):
        fixed = read_step(compact.WORK / f'candidate_{index}.step')
        if index == 22:
            fixed = fixed.moved(bd.Pos(*offset))
        overlap = sum(piece.volume for piece in refit._items(candidate.intersect(fixed)))
        print('static', index, overlap, 'distance', candidate.distance_to(fixed), flush=True)
    ring = read_step(compact.WORK / 'candidate_53.step')
    axis = bd.Axis((65.95, .1, 196.85093881236), (-1, 0, 0))
    maximum, closest = 0, 1000
    for pose in range(-30, 31, 5):
        rotated = candidate.rotate(axis, pose)
        maximum = max(maximum, sum(piece.volume for piece in refit._items(rotated.intersect(ring))))
        closest = min(closest, rotated.distance_to(ring))
    print('ring_max_overlap', maximum, 'ring_min_distance', closest, flush=True)
    bd.export_step(candidate, compact.WORK / 'probe_housing_91.step')


def speaker_layout():
    turn, horizontal, lateral = map(float, sys.argv[2:5])
    ring = read_step(compact.WORK / 'candidate_53.step')
    axis = bd.Axis((65.95, .1, 196.85093881236), (-1, 0, 0))
    for index in sorted(compact.SPEAKER):
        original = read_step(compact.WORK / f'candidate_{index}.step')
        candidate = original.moved(bd.Pos(-7, 55, -205)).rotate(bd.Axis.Z, turn).moved(bd.Pos(horizontal, lateral, 205))
        print(index, candidate.bounding_box(), flush=True)
        for fixed_index in (82, 90, 98):
            fixed = read_step(compact.WORK / f'candidate_{fixed_index}.step')
            overlap = sum(piece.volume for piece in refit._items(candidate.intersect(fixed)))
            print('static', fixed_index, overlap, flush=True)
        maximum, closest = 0, 1000
        for pose in range(-30, 31, 5):
            rotated = candidate.rotate(axis, pose)
            maximum = max(maximum, sum(piece.volume for piece in refit._items(rotated.intersect(ring))))
            closest = min(closest, rotated.distance_to(ring))
        print('ring_max_overlap', maximum, 'ring_min_distance', closest, flush=True)
        bd.export_step(candidate, compact.WORK / f'probe_speaker_{index}.step')


def oriented_rear():
    horizontal, lateral, height = map(float, sys.argv[2:5])
    pivot = bd.Vector(-51.63914929544, 60.762837687316, 197.55500826739)
    ring = read_step(compact.WORK / 'candidate_53.step')
    shell = read_step(compact.WORK / 'candidate_82.step')
    axis = bd.Axis((65.95, .1, 196.85093881236), (-1, 0, 0))
    for index in (91, 22, 10):
        original = read_step(compact.WORK / f'candidate_{index}.step')
        candidate = original.moved(bd.Pos(*(-pivot))).rotate(bd.Axis.Z, -15.71).rotate(bd.Axis.X, -13.66)
        if index == 91:
            candidate = candidate.rotate(bd.Axis.Y, -90)
        candidate = candidate.moved(bd.Pos(horizontal, lateral, height))
        print(index, candidate.bounding_box(), flush=True)
        overlap = sum(piece.volume for piece in refit._items(candidate.intersect(shell)))
        maximum, closest = 0, 1000
        for pose in range(-30, 31, 5):
            rotated = candidate.rotate(axis, pose)
            maximum = max(maximum, sum(piece.volume for piece in refit._items(rotated.intersect(ring))))
            closest = min(closest, rotated.distance_to(ring))
        print('shell_overlap', overlap, 'ring_max_overlap', maximum, 'ring_min_distance', closest, flush=True)
        bd.export_step(candidate, compact.WORK / f'probe_oriented_{index}.step')


def main():
    mode, joint, moving, fixed = sys.argv[1:5]
    moving, fixed = int(moving), int(fixed)
    axes = {'S12': ((9, .725, 156.00093881237), (0, 1, 0), 45),
            'S13': ((9, .1, 197.57593881241), (0, 0, 1), 90),
            'S14': ((75.95, .1, 196.85093881236), (-1, 0, 0), 30),
            'static': ((0, 0, 0), (0, 0, 1), 0)}
    centre, direction, limit = axes[joint]
    if mode == 'candidate' and joint in ('S13', 'S14'):
        centre = (centre[0] - compact.REARWARD_MM, *centre[1:])

    def filename(index):
        if mode == 'candidate' and index in compact.HEAD_LEAVES | {12}:
            return compact.WORK / f'candidate_{index}.step'
        return compact.WORK / f'leaf_{index}.step'

    compact.rows()
    mover, obstacle = read_step(filename(moving)), read_step(filename(fixed))
    offset = tuple(map(float, sys.argv[5:8])) if len(sys.argv) >= 8 else (0, 0, 0)
    mover = mover.moved(bd.Pos(*offset))
    results = []
    for angle in range(-limit, limit + 1, 1 if '--fine' in sys.argv else 5):
        posed = mover.rotate(bd.Axis(centre, direction), angle)
        intersection = posed.intersect(obstacle)
        overlap = sum(piece.volume for piece in refit._items(intersection))
        distance = posed.distance_to(obstacle)
        results.append(dict(angle=angle, overlap_mm3=overlap, distance_mm=distance))
        if overlap > .001:
            print(joint, moving, fixed, angle, round(overlap, 5), flush=True)
            if joint == 'static' or '--fine' in sys.argv:
                for piece in refit._items(intersection):
                    print('intersection bounds', piece.bounding_box(), flush=True)
        del posed, intersection
        gc.collect()
    suffix = '' if offset == (0, 0, 0) else '_offset_' + '_'.join(map(str, offset))
    destination = compact.WORK / f'{mode}_{joint}_{moving}_{fixed}{suffix}.json'
    destination.write_text(json.dumps(results, indent=2))
    print(destination, 'max overlap', max(row['overlap_mm3'] for row in results), flush=True)


if __name__ == '__main__':
    refit._start_memory_watchdog()
    if sys.argv[1] == 'static_all':
        static_all()
    elif sys.argv[1] == 'inspect':
        inspect_part()
    elif sys.argv[1] == 'antenna_seats':
        antenna_seats()
    elif sys.argv[1] == 'speaker_checks':
        speaker_checks()
    elif sys.argv[1] == 'rear_layout':
        rear_layout()
    elif sys.argv[1] == 'housing_layout':
        housing_layout()
    elif sys.argv[1] == 'speaker_layout':
        speaker_layout()
    elif sys.argv[1] == 'oriented_rear':
        oriented_rear()
    else:
        main()
