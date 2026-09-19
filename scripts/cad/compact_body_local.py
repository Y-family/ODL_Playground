"""Compact the existing body and place its rear board horizontally."""

import json
import sys
import hashlib
import shutil

from compact_head_local import WORK, read_step, rows, bd, clip, compress_axis, refit

CHANGED = {40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 112, 113, 114}


def horizontal(part):
    board = read_step(WORK / 'leaf_114.step')
    bounds = board.bounding_box()
    centre = (bounds.min + bounds.max) * .5
    return part.rotate(bd.Axis(centre, (0, 0, 1)), 90).moved(bd.Pos(-81-centre.X, -centre.Y, 0))


def inspect():
    measured = []
    for entry in rows():
        index = entry['index']
        if index not in set(range(39, 50)) | set(range(112, 127)):
            continue
        part = read_step(WORK / f'leaf_{index}.step')
        bounds = part.bounding_box()
        result = dict(index=index, name=entry['path'], minimum=list(bounds.min),
                      maximum=list(bounds.max), size=list(bounds.size))
        measured.append(result)
        print(json.dumps(result), flush=True)
    (WORK / 'body_measurements.json').write_text(json.dumps(measured, indent=2))


def interfaces():
    for index in (112, 113, 114):
        part = read_step(WORK / f'leaf_{index}.step')
        for face in part.faces():
            bounds = face.bounding_box()
            if bounds.max.X < -53 and face.area > 8:
                print(index, str(face.geom_type), round(face.area, 3), list(bounds.min), list(bounds.max), flush=True)


def build():
    original = read_step(WORK / 'leaf_112.step')
    rear = clip(original, 0, -200, -59.11)
    lower = clip(rear, 2, -100, 51.79)
    lower = compress_axis(lower, 0, -76, -59.11, 10, True)
    upper = clip(rear, 2, 51.79, 150)
    for old_x, new_x in ((-103.11, -103.00200116), (-65.99, -65.99)):
        for sign in (-1, 1):
            old_y = sign * 9.25
            new_y = sign * 14
            region = refit._world_box(old_x-4, old_x+4, old_y-4, old_y+4, 55.790938812376, 59)
            boss = bd.Compound(refit._items(upper.intersect(region)))
            upper = bd.Compound(refit._items(upper.cut(region)))
            plug = refit._rod(bd.Vector(old_x, old_y, 0), bd.Vector(0, 0, 1), 51.79, 55.790938812376, 1)
            upper = refit._fuse(upper, plug, boss.moved(bd.Pos(new_x-old_x, new_y-old_y, 0)))
            bore = refit._rod(bd.Vector(new_x, new_y, 0), bd.Vector(0, 0, 1), 51, 60, 1)
            upper = bd.Compound(refit._items(upper.cut(bore)))
    upper = clip(horizontal(upper), 0, -99.5, 200)
    front = clip(original, 0, -59.11, 200)
    candidate = refit._fuse(front, lower, upper).clean()
    for position_x in (-103.00200116, -65.99):
        for lateral in (-14, 14):
            bore = refit._rod(bd.Vector(position_x, lateral, 0), bd.Vector(0, 0, 1), 51, 60, 1)
            candidate = refit._shape(candidate.cut(horizontal(bore)))
    assert candidate.is_valid and len(candidate.solids()) == 1
    bd.export_step(candidate, WORK / 'body_candidate_112.step')
    if '--board-only' in sys.argv:
        bd.export_step(horizontal(read_step(WORK / 'leaf_114.step')), WORK / 'body_candidate_114.step')
        print('Horizontal board and integral carrier built', flush=True)
        return
    for index in (40, 44):
        part = read_step(WORK / f'leaf_{index}.step')
        part = compress_axis(part, 0, -90, -65, 10, True)
        assert part.is_valid and len(part.solids()) == 1
        bd.export_step(part, WORK / f'body_candidate_{index}.step')
    for index in (*range(41, 44), *range(45, 50), 113, 114):
        part = read_step(WORK / f'leaf_{index}.step')
        part = horizontal(part) if index == 114 else part.moved(bd.Pos(10, 0, 0))
        bd.export_step(part, WORK / f'body_candidate_{index}.step')
    print('Candidate built; awaiting interface and collision checks', flush=True)


def check():
    outcomes = []
    checked = {112, 114} if '--board-only' in sys.argv else CHANGED
    for moving in sorted(checked):
        candidate = read_step(WORK / f'body_candidate_{moving}.step')
        source = read_step(WORK / f'leaf_{moving}.step')
        for fixed in list(range(39, 50)) + list(range(112, 127)):
            if fixed == moving or fixed in checked and fixed < moving:
                continue
            neighbour = read_step(WORK / (f'body_candidate_{fixed}.step' if fixed in CHANGED else f'leaf_{fixed}.step'))
            first, second = candidate.bounding_box(), neighbour.bounding_box()
            if any(list(first.min)[axis] > list(second.max)[axis] + .2 or list(second.min)[axis] > list(first.max)[axis] + .2 for axis in range(3)):
                continue
            overlap = sum(part.volume for part in refit._items(candidate.intersect(neighbour)))
            old_neighbour = read_step(WORK / f'leaf_{fixed}.step')
            old_overlap = sum(part.volume for part in refit._items(source.intersect(old_neighbour)))
            gap = candidate.distance_to(neighbour)
            row = dict(parts=[moving, fixed], overlap_mm3=overlap, baseline_overlap_mm3=old_overlap, gap_mm=gap)
            outcomes.append(row)
            print(json.dumps(row), flush=True)
    (WORK / 'body_checks.json').write_text(json.dumps(outcomes, indent=2))
    assert all(row['overlap_mm3'] < max(.001, row['baseline_overlap_mm3'] + .001) for row in outcomes), 'New overlaps remain'


def mounts():
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from refit_servo_screw_length_local import geometry_signature
    for index in (40, 44):
        part = read_step(WORK / f'leaf_{index}.step')
        for face in part.faces():
            if face.geom_type == bd.GeomType.CYLINDER:
                cylinder = BRepAdaptor_Surface(face.wrapped).Cylinder()
                point = cylinder.Location()
                print('shell cylinder', index, cylinder.Radius(), point.X(), point.Y(), point.Z())
    original = read_step(WORK / 'leaf_112.step')
    candidate = read_step(WORK / 'body_candidate_112.step')
    unchanged = clip(original, 0, -54, 200)
    other = clip(candidate, 0, -54, 200)
    assert abs(unchanged.volume - other.volume) < .001
    assert geometry_signature(unchanged) == geometry_signature(other), 'Servo interface geometry signature changed'
    for position_x in (-103.00200116, -65.99):
        for lateral in (-14, 14):
            outer = refit._rod(bd.Vector(position_x, lateral, 0), bd.Vector(0, 0, 1), 58.29, 58.79, 1.8)
            inner = refit._rod(bd.Vector(position_x, lateral, 0), bd.Vector(0, 0, 1), 58, 59, 1.05)
            ring = horizontal(bd.Compound(refit._items(outer.cut(inner))))
            missing = sum(part.volume for part in refit._items(ring.cut(candidate)))
            assert missing < .001, ('board screw seat missing', position_x, lateral, missing)
            bore = refit._rod(bd.Vector(position_x, lateral, 0), bd.Vector(0, 0, 1), 51.7, 60.5, .95)
            bore = horizontal(bore)
            blockage = sum(part.volume for part in refit._items(bore.intersect(candidate)))
            assert blockage < .001, ('board screw blocked', blockage)
    print('Four board seats and bores, unchanged servo interface region passed', flush=True)


def picture():
    import os
    os.environ['MPLCONFIGDIR'] = '/tmp/mpl-config'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    figure = plt.figure(figsize=(13, 6))
    for column, prefix in enumerate(('leaf', 'body_candidate'), 1):
        plot = figure.add_subplot(1, 2, column, projection='3d')
        for index, colour in ((112, '#777777'), (113, '#008cba'), (114, '#2aaf55'), (41, '#aabbee'), (47, '#dcac45')):
            part = read_step(WORK / f'{prefix}_{index}.step')
            vertices, faces = part.tessellate(.3, .4)
            polygons = np.array([list(vertex) for vertex in vertices])[np.array(faces)]
            plot.add_collection3d(Poly3DCollection(polygons, facecolor=colour, alpha=.08 if index == 41 else .25 if index == 112 else 1, linewidth=0))
        plot.set(xlim=(-160, 30), ylim=(-60, 60), zlim=(-30, 115), xlabel='X mm', ylabel='Y mm', zlabel='Z mm', title='Original' if column == 1 else 'Compact body: board horizontal')
        plot.set_box_aspect((190, 120, 145))
        plot.set_proj_type('ortho')
        plot.view_init(elev=20, azim=-135)
    figure.savefig(WORK / 'body_layout.png', dpi=150)


def motion():
    results = []
    for joint, centre, limit in (('S11', (9, .725, 90.000938812376), 30), ('S12', (9, .725, 156.00093881237), 45)):
        for index in (12, 17, 18, 19, 82, 89):
            if joint == 'S12' and index in (17, 18, 19):
                continue
            filename = WORK / (f'candidate_{index}.step' if index in (12, 82, 89) else f'leaf_{index}.step')
            moving = read_step(filename)
            for fixed in ((112, 114) if '--board-only' in sys.argv else (40, 41, 44, 112, 113, 114)):
                candidate = read_step(WORK / f'body_candidate_{fixed}.step')
                original = read_step(WORK / f'leaf_{fixed}.step')
                maximum, old_maximum = 0, 0
                for angle in range(-limit, limit + 1, 5):
                    posed = moving.rotate(bd.Axis(centre, (0, 1, 0)), angle)
                    first, second = posed.bounding_box(), candidate.bounding_box()
                    if any(list(first.min)[axis] > list(second.max)[axis] + .2 or list(second.min)[axis] > list(first.max)[axis] + .2 for axis in range(3)):
                        continue
                    overlap = sum(part.volume for part in refit._items(posed.intersect(candidate)))
                    old_overlap = sum(part.volume for part in refit._items(posed.intersect(original)))
                    maximum, old_maximum = max(maximum, overlap), max(old_maximum, old_overlap)
                    assert overlap <= max(.001, old_overlap + .001), (joint, index, fixed, angle, overlap, old_overlap)
                row = dict(joint=joint, moving=index, fixed=fixed, overlap_mm3=maximum, baseline_overlap_mm3=old_maximum)
                results.append(row)
                print(json.dumps(row), flush=True)
    (WORK / 'body_motion.json').write_text(json.dumps(results, indent=2))


def review():
    baseline_path = WORK / 'body_horizontal_input.step'
    if not baseline_path.exists():
        shutil.copyfile(WORK / 'review.step', baseline_path)
    baseline = read_step(baseline_path)
    catalog = rows()
    position = 0
    expected = []

    def rebuild(node):
        nonlocal position
        children = refit._children(node)
        if children:
            fresh = bd.Compound(children=[rebuild(child) for child in children], label=node.label)
            fresh.location, fresh.color = node.location, node.color
            return fresh
        index = position
        position += 1
        assert node.label == catalog[index]['label']
        if index in {112, 114}:
            part = read_step(WORK / f'body_candidate_{index}.step')
            assert part.is_valid and len(part.solids()) == 1
            fresh = bd.Compound(part.solids(), label=node.label)
            fresh.color = node.color
        else:
            fresh = node
        bounds = fresh.bounding_box()
        expected.append((fresh.label, fresh.volume, list(bounds.min) + list(bounds.max), str(fresh.color)))
        return fresh

    result = refit._encode_colours(rebuild(baseline))
    assert position == 137
    pending = WORK / 'body_pending.step'
    bd.export_step(result, pending)
    imported = read_step(pending)
    observed = []

    def visit(node):
        children = refit._children(node)
        if children:
            for child in children:
                visit(child)
        else:
            bounds = node.bounding_box()
            observed.append((node.label, node.volume, list(bounds.min) + list(bounds.max), str(node.color)))

    visit(imported)
    assert len(observed) == len(expected)
    for index, (before, after) in enumerate(zip(expected, observed)):
        assert before[0] == after[0] and before[3] == after[3], (index, 'label or colour')
        assert abs(before[1] - after[1]) < .01, (index, 'volume')
        assert max(abs(first-second) for first, second in zip(before[2], after[2])) < .001, (index, 'bounds')
    report = dict(status='body review candidate; not installed as official STEP',
                  input_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(pending.read_bytes()).hexdigest(),
                  changed_leaves=[112, 114], body_x_before_mm=190, body_x_after_mm=190,
                  body_external_dimensions_mm=[190, 110, 143],
                  board_rotation_y_deg=0, board_rotation_z_deg=90, bno055_rotation_deg=0,
                  mount_checks='four board seats and bores passed; original servo-interface geometry signature retained',
                  static_checks=json.loads((WORK / 'body_checks.json').read_text()),
                  motion_checks=json.loads((WORK / 'body_motion.json').read_text()),
                  export_readback_leaves=137, unchanged_leaves=135,
                  limitations=['Board components and connectors absent in source model', 'No multi-joint combined-motion certification', 'Existing baseline overlaps retained'])
    pending.replace(WORK / 'review.step')
    from compact_head_local import TARGET
    (TARGET.parent / 'BODY_COMPACT_REPORT.json').write_text(json.dumps(report, indent=2))
    print('Body preview exported and read back:', WORK / 'review.step', flush=True)


if __name__ == '__main__':
    if 'build' in sys.argv:
        build()
    elif 'check' in sys.argv:
        check()
    elif 'mounts' in sys.argv:
        mounts()
    elif 'picture' in sys.argv:
        picture()
    elif 'motion' in sys.argv:
        motion()
    elif 'review' in sys.argv:
        review()
    elif 'interfaces' in sys.argv:
        interfaces()
    else:
        inspect()
