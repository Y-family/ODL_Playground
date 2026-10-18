"""First compact-head layout, millimeters, preserving purchased parts.

Target: original shell and floor below 175 mm X/Y, unchanged height.
Compress feature-free shell bands, retain the front optical interface and
rear antenna mounting region, retain the original head position and
the original S12-to-S13 cross-link. Validate candidates
against the current assembly before replacing the single official STEP.
"""

import gc
import hashlib
import json
import resource
import sys
from pathlib import Path

if resource.getrlimit(resource.RLIMIT_AS)[1] not in range(1, 4_000_000_001):
    raise SystemExit('Run through scripts/cad/run_memory_limited.py before importing CAD libraries')

from cadgen import build123d as bd, read_step
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.TopTools import TopTools_ListOfShape

import refit_upper_servo_interfaces_local as refit
import refit_case_bracket_width_local as brackets


WORK = Path('/tmp/head_compact')
TARGET = brackets.TARGET
REPORT = TARGET.parent / 'HEAD_COMPACT_REPORT.json'
REARWARD_MM = 0.0
SHORTEN_X_MM = 24.845704
SHORTEN_Y_SIDE_MM = 11.157980
X_BAND = (-40.0, -10.0)
Y_BAND = (66.0, 82.0)
HEAD_LEAVES = set(range(78, 103)) | {1, 10, 22, 23, 50, 51, 52, 53}
REAR_DEVICES = {1, 10, 22, 23, 91, 102}
SPEAKER = {80, 85, 92, 93}
SPEAKER_FORWARD_MM = 45.0
SPEAKER_INWARD_MM = 5.0
FLASH = {79, 86}


def rows():
    baseline = json.loads((WORK / 'baseline.json').read_text())
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == baseline['sha256'], 'Baseline changed; prepare again'
    return json.loads((WORK / 'leaves.json').read_text())


def prepare():
    WORK.mkdir(exist_ok=True)
    robot = read_step(TARGET)
    found = []

    def visit(node, path=()):
        children = refit._children(node)
        if children:
            for child in children:
                visit(child, path + (child.label,))
            return
        index = len(found)
        filename = WORK / f'leaf_{index}.step'
        bd.export_step(bd.Compound(node.solids()), filename)
        found.append(dict(index=index, path='/'.join(path), label=node.label, file=str(filename)))

    visit(robot)
    assert len(found) == 137
    assert found[82]['label'] == 'head' and found[89]['label'] == 'head_bot_sheet'
    (WORK / 'leaves.json').write_text(json.dumps(found, indent=2))
    (WORK / 'baseline.json').write_text(json.dumps(dict(sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest())))
    print('Prepared 137 baseline leaves; official STEP unchanged', flush=True)


def checkpoint():
    rows()
    report = json.loads(REPORT.read_text())
    report['status'] = 'not installed: static and motion interference remains'
    report['official_step_unchanged'] = hashlib.sha256(TARGET.read_bytes()).hexdigest() == report['input_sha256']
    report['motion_diagnostics'] = {path.stem: json.loads(path.read_text())
                                    for pattern in ('candidate_S*.json', 'baseline_S*.json', 'candidate_static*.json')
                                    for path in sorted(WORK.glob(pattern))}
    report['static_report'] = 'HEAD_COMPACT_STATIC_REPORT.json'
    report['snapshot'] = dict(browser='failed: required Playwright browser executable missing',
                              fallback='HEAD_COMPACT_LAYOUT.png', status='diagnostic only')
    report['resource_policy'] = dict(total_budget_bytes=8_000_000_000,
                                      process_address_space_limit_bytes=4_000_000_000,
                                      component_workers=1, validate_workers=1, daemon=False,
                                      aggregate_cgroup_enforcement=False)
    report['remaining'] = ['Resolve shell/speaker stand and speaker top static interference',
                           'Resolve antenna servo/ring and speaker/ring motion interference',
                           'Recheck yaw and neck motion plus moved mounting interfaces and 4 mm screw seats',
                           'Resolve fine shell tessellation failure before final viewer/export validation']
    REPORT.write_text(json.dumps(report, indent=2))
    print('Saved diagnostics; official STEP unchanged', flush=True)


def review():
    baseline = rows()
    report = json.loads(REPORT.read_text())
    assert report['rearward_mm'] == REARWARD_MM == 0
    for index in (12, 50, 51, 52, 53, 98, 99, 100):
        source = read_step(WORK / f'leaf_{index}.step')
        candidate = read_step(WORK / f'candidate_{index}.step')
        assert abs(source.volume - candidate.volume) < .001
        assert max(abs(left - right) for left, right in zip(
            list(source.bounding_box().min) + list(source.bounding_box().max),
            list(candidate.bounding_box().min) + list(candidate.bounding_box().max))) < 1e-5
        del source, candidate
    robot = read_step(TARGET)
    position = 0

    def rebuild(node):
        nonlocal position
        children = refit._children(node)
        if children:
            fresh = bd.Compound(children=[rebuild(child) for child in children], label=node.label)
            fresh.location, fresh.color = node.location, node.color
            return fresh
        index = position
        position += 1
        assert node.label == baseline[index]['label']
        if index not in HEAD_LEAVES | {12}:
            return node
        candidate = read_step(WORK / f'candidate_{index}.step')
        fresh = bd.Compound(candidate.solids(), label=node.label)
        fresh.color = node.color
        return fresh

    result = refit._encode_colours(rebuild(robot))
    assert position == 137
    destination = WORK / 'review.step'
    bd.export_step(result, destination)
    report.update(status='intermediate review: compact head at original position; collision issues unresolved',
                  restored_position_checks=[12, 50, 51, 52, 53, 98, 99, 100],
                  review_file=str(destination), motion_diagnostics={})
    REPORT.write_text(json.dumps(report, indent=2))
    print('UNVALIDATED intermediate assembly:', destination, flush=True)


def clip(part, axis, lower, upper):
    limits = [-400, 400, -400, 400, -400, 500]
    limits[axis * 2:axis * 2 + 2] = [lower, upper]
    result = part.intersect(refit._world_box(*limits))
    pieces = [piece for piece in refit._items(result) if piece.volume > .000001]
    return bd.Compound(pieces) if pieces else None


def affine(part, matrix):
    return part.transform_geometry(bd.Matrix(matrix))


def compress_axis(part, axis, lower, upper, removal, move_lower):
    low = clip(part, axis, -400, lower)
    middle = clip(part, axis, lower, upper)
    high = clip(part, axis, upper, 400)
    factor = (upper - lower - removal) / (upper - lower)
    assert factor > 0
    shift = [0.0, 0.0, 0.0]
    shift[axis] = removal if move_lower else -removal
    matrix = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    matrix[axis][axis] = factor
    matrix[axis][3] = (upper if move_lower else lower) * (1 - factor)
    pieces = []
    if low:
        pieces.append(low.moved(bd.Pos(*shift)) if move_lower else low)
    if middle:
        pieces.append(affine(middle, matrix))
    if high:
        pieces.append(high if move_lower else high.moved(bd.Pos(*shift)))
    return refit._fuse(*pieces).clean()


def compact_skin(part):
    part = compress_axis(part, 0, *X_BAND, SHORTEN_X_MM, True)
    part = compress_axis(part, 1, Y_BAND[0] + .1, Y_BAND[1] + .1, SHORTEN_Y_SIDE_MM, False)
    return compress_axis(part, 1, -Y_BAND[1] + .1, -Y_BAND[0] + .1, SHORTEN_Y_SIDE_MM, True)


def mapped_point(point):
    horizontal, lateral, height = point
    if horizontal < X_BAND[0]:
        horizontal += SHORTEN_X_MM
    elif horizontal < X_BAND[1]:
        horizontal = X_BAND[1] + (horizontal - X_BAND[1]) * (1 - SHORTEN_X_MM / (X_BAND[1] - X_BAND[0]))
    sign = 1 if lateral >= .1 else -1
    distance = abs(lateral - .1)
    if distance > Y_BAND[1]:
        distance -= SHORTEN_Y_SIDE_MM
    elif distance > Y_BAND[0]:
        distance = Y_BAND[0] + (distance - Y_BAND[0]) * (1 - SHORTEN_Y_SIDE_MM / (Y_BAND[1] - Y_BAND[0]))
    return bd.Vector(horizontal, .1 + sign * distance, height)


def floor_hole_axes():
    original = read_step(WORK / 'leaf_89.step')
    found = []
    for face in original.faces():
        if face.geom_type != bd.GeomType.CYLINDER:
            continue
        cylinder = BRepAdaptor_Surface(face.wrapped).Cylinder()
        if abs(cylinder.Radius() - 1.6) > .001:
            continue
        point = bd.Vector(cylinder.Location().Coord())
        point = bd.Vector(point.X, point.Y, 0)
        if not any((point - old).length < .01 for old in found):
            found.append(point)
    assert len(found) == 8
    return found


def restore_floor_screws(candidate, original, shell):
    for point in floor_hole_axes():
        mapped = mapped_point(point)
        distance = abs(point.Y - .1)
        if not (Y_BAND[0] - 4 < distance < Y_BAND[1] + 4 or
                X_BAND[0] - 4 < point.X < X_BAND[1] + 4):
            continue
        if shell:
            pillar_zone = refit._rod(point, bd.Vector(0, 0, 1), 174.876878, 196, 4)
            pillars = [solid for solid in refit._items(original.intersect(pillar_zone)) if solid.volume > .001]
            candidate = refit._fuse(candidate, *[pillar.moved(bd.Pos(*(mapped - point))) for pillar in pillars])
            candidate = refit._cut(candidate, [refit._rod(mapped, bd.Vector(0, 0, 1), 174.87, 190, 1.6)])
        else:
            cutters = [refit._rod(mapped, bd.Vector(0, 0, 1), 171.8, 175, 1.6),
                       refit._rod(mapped, bd.Vector(0, 0, 1), 171.8, 172.876878, 3.25)]
            arguments = TopTools_ListOfShape()
            arguments.Append(candidate.wrapped)
            tools = TopTools_ListOfShape()
            for cutter in cutters:
                tools.Append(cutter.wrapped)
            operation = BRepAlgoAPI_Cut()
            operation.SetArguments(arguments)
            operation.SetTools(tools)
            operation.SetFuzzyValue(1e-5)
            operation.SetRunParallel(False)
            operation.Build()
            assert operation.IsDone()
            candidate = bd.Compound(operation.Shape())
    return candidate.clean()


def head_shell(original):
    support_box = refit._world_box(-120, -20, -20, 20, 177, 250)
    support = refit._shape(original.intersect(support_box))
    skin = refit._cut(original, [support_box])
    candidate = compact_skin(skin)
    rear_wall = clip(support, 0, -120, -65)
    rear_wall = rear_wall.moved(bd.Pos(SHORTEN_X_MM, 0, 0))
    support = clip(support, 0, -90.84570345975 + SHORTEN_X_MM, 0)
    candidate = refit._fuse(candidate, support, rear_wall).clean()
    speaker_mount_box = refit._world_box(10, 62, -58, -27, 228.85, 233.900938812362)
    mounts = [solid for solid in refit._items(original.intersect(speaker_mount_box))
              if solid.volume > 100 and solid.center().Y < -35]
    assert len(mounts) == 2
    candidate = refit._cut(candidate, mounts)
    candidate = refit._fuse(candidate, *[mount.moved(bd.Pos(-8, SHORTEN_Y_SIDE_MM, 0)) for mount in mounts]).clean()
    flash_box = refit._world_box(74, 104, 80, 130, 180, 231.5)
    flash_wall = refit._shape(original.intersect(flash_box))
    shift = bd.Pos(0, -SHORTEN_Y_SIDE_MM, 0)
    candidate = refit._cut(candidate, [flash_box.moved(shift)])
    candidate = refit._fuse(candidate, flash_wall.moved(shift)).clean()
    return restore_floor_screws(candidate, original, True)


def head_floor(original):
    candidate = compact_skin(original)
    opening = refit._rod(bd.Vector(9, .1, 0), bd.Vector(0, 0, 1), 171, 176, 60)
    opening = opening.scale((.9, 1, 1), about=(9, .1, 0))
    return restore_floor_screws(refit._cut(candidate, [opening]).clean(), original, False)


def speaker_stand(original):
    positioned = original.moved(bd.Pos(-8, SHORTEN_Y_SIDE_MM, 0))
    roof = clip(positioned, 2, 223.5, 250)
    outer = clip(roof, 1, -150, -33)
    inner = clip(roof, 1, -33, 150)
    inner = clip(inner, 0, -150, 47.8)
    outer = compress_axis(outer, 0, 46, 47, -SPEAKER_FORWARD_MM, False)
    lower = clip(positioned, 2, 170, 223.5)
    lower = compress_axis(lower, 2, 221.8, 223.5, -2.402, False)
    lower = lower.moved(bd.Pos(SPEAKER_FORWARD_MM, SPEAKER_INWARD_MM, 0))
    lower = clip(lower, 0, 47, 150)
    return refit._fuse(inner, outer, lower).clean()


def check_rear_wall():
    shell = read_step(WORK / 'candidate_82.step')
    assert shell.is_valid and len(shell.solids()) == 1
    assert abs(shell.bounding_box().size.X - 175) < .001
    assert abs(shell.bounding_box().size.Y - 175) < .001
    checks = []
    for lateral in (-15, 0, 15):
        for height in (180, 190, 200, 210, 220):
            ray = refit._rod(bd.Vector(0, lateral, height), bd.Vector(1, 0, 0), -70, -42, .1)
            volume = sum(piece.volume for piece in refit._items(shell.intersect(ray)))
            checks.append(dict(y=lateral, z=height, wall_volume_mm3=volume))
            assert volume > .001, checks[-1]
    for index in (81, 89, 91, 102):
        other = read_step(WORK / f'candidate_{index}.step')
        overlap = sum(piece.volume for piece in refit._items(shell.intersect(other)))
        print('rear wall neighbour', index, overlap, flush=True)
        assert overlap < .001
    (WORK / 'rear_wall_check.json').write_text(json.dumps(checks, indent=2))
    print('Rear wall sampling, single solid, dimensions and local neighbours passed', flush=True)


def cross_link(original):
    if REARWARD_MM == 0:
        return original
    pieces = []
    for lower, upper, constant, slope in [(-100, -23.5, 0, 0), (-23.5, -12, -23.5 / 11.5 * REARWARD_MM, -REARWARD_MM / 11.5),
                                          (-12, 12.2, -REARWARD_MM, 0), (12.2, 23.7, -23.7 / 11.5 * REARWARD_MM, REARWARD_MM / 11.5),
                                          (23.7, 100, 0, 0)]:
        piece = clip(original, 1, lower, upper)
        if piece:
            if slope:
                piece = affine(piece, [[1, slope, 0, constant], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            elif constant:
                piece = piece.moved(bd.Pos(constant, 0, 0))
            pieces.append(piece)
    return refit._fuse(*pieces).clean()


def build(indices=None):
    fingerprint = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    report = dict(status='building first compact-head candidates', input_sha256=fingerprint,
                  units='mm', rearward_mm=REARWARD_MM, target_shell_xy_mm=175, parts=[])
    if indices and REPORT.exists():
        report = json.loads(REPORT.read_text())
        assert report['input_sha256'] == fingerprint
        report['parts'] = [part for part in report['parts']
                           if part['index'] not in indices and Path(part['file']).exists()]
    report['rearward_mm'] = REARWARD_MM
    report['motion_diagnostics'] = {}
    REPORT.write_text(json.dumps(report, indent=2))
    for row in rows():
        index = row['index']
        if indices and index not in indices:
            continue
        if index not in HEAD_LEAVES | {12}:
            continue
        original = read_step(row['file'])
        candidate = original
        offset = [-REARWARD_MM, 0., 0.]
        if index == 12:
            candidate = cross_link(original)
            offset = [0., 0., 0.]
        elif index == 82:
            candidate = head_shell(original)
        elif index == 89:
            candidate = head_floor(original)
        elif index == 85:
            candidate = speaker_stand(original)
            offset = [0., 0., 0.]
        elif index in REAR_DEVICES:
            offset[0] += SHORTEN_X_MM
        elif index in SPEAKER:
            offset[0] += SPEAKER_FORWARD_MM - 8
            offset[1] = SHORTEN_Y_SIDE_MM + SPEAKER_INWARD_MM
        elif index in FLASH:
            offset[1] = -SHORTEN_Y_SIDE_MM
        candidate = candidate.moved(bd.Pos(*offset))
        if len(candidate.solids()) != len(original.solids()):
            for solid in candidate.solids():
                print('fragment', solid.volume, solid.bounding_box(), flush=True)
            bd.export_step(candidate, WORK / f'failed_{index}.step')
        assert candidate.is_valid and len(candidate.solids()) == len(original.solids()), (index, candidate.is_valid, len(candidate.solids()))
        filename = WORK / f'candidate_{index}.step'
        bd.export_step(bd.Compound(candidate.solids()), filename)
        bounds = candidate.bounding_box()
        report['parts'].append(dict(index=index, path=row['path'], file=str(filename),
                                   offset_mm=offset, volume_mm3=candidate.volume,
                                   bounds=list(bounds.min) + list(bounds.max),
                                   size_mm=list(bounds.size)))
        REPORT.write_text(json.dumps(report, indent=2))
        print(index, row['label'], 'size', [round(value, 3) for value in bounds.size], flush=True)
        del original, candidate
        gc.collect()
    report['status'] = 'partial candidates built; checks pending'
    REPORT.write_text(json.dumps(report, indent=2))


def restore_position():
    rows()
    report = json.loads(REPORT.read_text())
    previous = report['rearward_mm']
    assert previous in (0, 10)
    for part in report['parts']:
        index = part['index']
        filename = Path(part['file'])
        original = read_step(WORK / f'leaf_{index}.step')
        if index == 12:
            candidate = read_step(WORK / 'leaf_12.step')
        else:
            assert index in HEAD_LEAVES
            cached = read_step(filename)
            shift = part['bounds'][0] + previous - REARWARD_MM - cached.bounding_box().min.X
            candidate = cached.moved(bd.Pos(shift, 0, 0))
            assert abs(candidate.volume - cached.volume) < .001
            assert max(abs(left - right) for left, right in zip(candidate.bounding_box().size, cached.bounding_box().size)) < 1e-5
            part['offset_mm'][0] += previous - REARWARD_MM
        assert candidate.is_valid
        bd.export_step(bd.Compound(candidate.solids()), filename)
        bounds = candidate.bounding_box()
        part.update(bounds=list(bounds.min) + list(bounds.max), size_mm=list(bounds.size), volume_mm3=candidate.volume)
        del original, candidate
        gc.collect()
    report.update(rearward_mm=REARWARD_MM, status='original head position restored; compact dimensions retained; review only',
                  motion_diagnostics={}, remaining=['Previously identified internal interference remains; neck motion requires recheck at restored position'])
    REPORT.write_text(json.dumps(report, indent=2))
    review()
    print('Restored original head position and original cross-link; compact part sizes unchanged', flush=True)


if __name__ == '__main__':
    refit._start_memory_watchdog()
    {'prepare': prepare, 'build': build, 'part': lambda: build({int(sys.argv[2])}),
     'parts': lambda: build(set(map(int, sys.argv[2:]))),
     'checkpoint': checkpoint,
     'review': review,
     'check-rear-wall': check_rear_wall,
     'restore-position': restore_position,
     'layout': lambda: build((HEAD_LEAVES | {12}) - {82, 89}),
     'shell': lambda: build({82, 89, 80, 85, 92, 93})}[sys.argv[1]]()
