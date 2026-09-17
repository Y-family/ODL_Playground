"""Enlarge S11 shell openings and relocate their existing fastening interfaces."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

from cadgen import build123d as bd, read_step

import refit_upper_servo_interfaces_local as refit


TARGET = refit.ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
CHANGED = ('body_front', 'body_middle_top', 'body_back',
           'neck_left_sheet', 'neck_right_sheet')
AXIS = bd.Axis((9, .725, 90.000938812376), (0, 1, 0))


def extract(work):
    work.mkdir(parents=True, exist_ok=True)
    robot = read_step(TARGET)
    rows = []
    local_names = set(CHANGED) | {'body_middle_bottom', 'head', 'head_bot_sheet', 'trunk_top'}

    def visit(node, path=()):
        children = refit._children(node)
        if children:
            for child in children:
                visit(child, path + (child.label,))
            return
        moving = (node.label in refit.S11_FOLLOW_LABELS or any(
            path[:len(prefix)] == prefix for prefix in refit.S11_FOLLOW_PATHS))
        if 'S11_NECK_PITCH_ZP10D' in path and node.label == 'ZP10D_vendor_body':
            moving = False
        filename = work / f'leaf_{len(rows)}.step'
        shape = bd.Compound(node.solids())
        bd.export_step(shape, filename)
        if node.label in local_names:
            bd.export_step(shape, work / f'{node.label}.step')
        bounds = node.bounding_box()
        rows.append(dict(label=node.label, path=path, moving=moving, file=str(filename),
                         volume=node.volume, bounds=list(bounds.min) + list(bounds.max),
                         color=list(node.color) if node.color else None))

    visit(robot)
    (work / 'leaves.json').write_text(json.dumps(rows, indent=2))
    print(f'Extracted {len(rows)} leaves; {sum(row["moving"] for row in rows)} move with S11', flush=True)


def drill(part, lateral, height, start, end, radius):
    cutter = refit._rod(bd.Vector(0, lateral, height), bd.Vector(1, 0, 0),
                        start, end, radius)
    return refit._cut(part, [cutter])


def revise_shell(part, name):
    if name == 'body_front':
        for sign in (-1, 1):
            centre = sign * 29.407864376269
            old_bore = refit._rod(bd.Vector(0, centre, 97.890938812376),
                                  bd.Vector(1, 0, 0), 36, 46, 3.25)
            part = refit._fuse(part, old_bore)
            part = drill(part, sign * 45, 65, 35.9, 46.1, 1.75)
            part = drill(part, sign * 45, 65, 39, 46.1, 3.25)
        part = refit._cut(part, [refit._world_box(35.9, 46.1, -26.9, 27.4, 107.6, 120)])
    elif name == 'body_middle_top':
        for sign in (-1, 1):
            lower, upper = (41, 52) if sign > 0 else (-52, -41)
            part = refit._fuse(part, refit._world_box(
                29, 36, lower, upper, 51.115938812376, 78.3))
            part = drill(part, sign * 45, 65, 28.9, 36.1, 1.5)
            part = drill(part, sign * 45, 65, 30.2, 36.1, 2)
    return part.clean()


def relieve_neck(part, trunk):
    overlap = part.rotate(AXIS, -45).intersect(trunk)
    if sum(item.volume for item in refit._items(overlap)) < 1e-5:
        return part
    bounds = refit._shape(overlap).bounding_box()
    clearance = .3
    cutter = refit._world_box(
        bounds.min.X - clearance, bounds.max.X + clearance,
        bounds.min.Y - clearance, bounds.max.Y + clearance,
        bounds.min.Z - clearance, bounds.max.Z + clearance).rotate(AXIS, 45)
    fresh = refit._cut(part, [cutter]).clean()
    for face in part.faces():
        if face.geom_type == bd.GeomType.CYLINDER:
            radius = refit.BRepAdaptor_Surface(face.wrapped).Cylinder().Radius()
            if radius < 4:
                retained = sum(item.area for item in refit._items(face.intersect(fresh)))
                if face.area - retained > .001:
                    raise ValueError('Neck relief would remove a screw bore')
    return fresh


def build(work):
    trunk = read_step(work / 'trunk_top.step')
    for name in CHANGED:
        old = read_step(work / f'{name}.step')
        fresh = (relieve_neck(old, trunk) if name.startswith('neck_')
                 else revise_shell(old, name))
        if not fresh.is_valid or len(fresh.solids()) != 1:
            raise ValueError(f'{name}: expected one valid solid')
        before, after = old.bounding_box(), fresh.bounding_box()
        if any(second < first - 1e-5 for first, second in zip(before.min, after.min)) or any(
                second > first + 1e-5 for first, second in zip(before.max, after.max)):
            raise ValueError(f'{name}: outer bounds enlarged')
        bd.export_step(fresh, work / f'{name}_candidate.step')
        print(f'{name}: valid single solid, no enlargement of outer bounds', flush=True)


def assemble(work):
    fingerprints = {name: hashlib.sha256((work / f'{name}_candidate.step').read_bytes()).hexdigest()
                    for name in CHANGED}
    for name in ('body_front', 'body_middle_top', 'body_middle_bottom', 'body_back', 'trunk_top'):
        report = json.loads((work / f'{name}_motion.json').read_text())
        if report['issues']:
            raise ValueError(f'{name}: motion verification failed')
        relevant = ('neck_left_sheet', 'neck_right_sheet') + ((name,) if name in CHANGED else ())
        if any(report['fingerprints'][part] != fingerprints[part] for part in relevant):
            raise ValueError(f'{name}: motion report is stale')
    robot = read_step(TARGET)
    replaced = []

    def replace(node):
        children = refit._children(node)
        if children:
            fresh = bd.Compound(children=[replace(child) for child in children], label=node.label)
            fresh.location = node.location
            fresh.color = node.color
            return fresh
        if node.label not in CHANGED:
            return node
        fresh = read_step(work / f'{node.label}_candidate.step')
        fresh = bd.Compound(fresh.solids(), label=node.label)
        fresh.color = node.color
        replaced.append(node.label)
        return fresh

    result = refit._encode_colours(replace(robot))
    if sorted(replaced) != sorted(CHANGED):
        raise ValueError(f'Unexpected replacement list: {replaced}')
    temporary = TARGET.with_suffix('.pending.step')
    bd.export_step(result, temporary)
    del result, robot
    gc.collect()
    exported = read_step(temporary)
    baseline = json.loads((work / 'leaves.json').read_text())
    leaves = []

    def collect(node):
        children = refit._children(node)
        if children:
            for child in children:
                collect(child)
        else:
            leaves.append(node)

    collect(exported)
    if len(leaves) != len(baseline):
        raise ValueError('Assembly leaf count changed')
    for node, before in zip(leaves, baseline):
        if node.label != before['label']:
            raise ValueError('Assembly labels or order changed')
        color = list(node.color) if node.color else None
        if (color is None) != (before['color'] is None) or (color is not None and any(
                abs(first - second) > 1e-6 for first, second in zip(color, before['color']))):
            raise ValueError(f'{node.label}: color changed')
        if node.label in CHANGED:
            candidate = read_step(work / f'{node.label}_candidate.step')
            if not node.is_valid or len(node.solids()) != 1 or abs(node.volume - candidate.volume) > .001:
                raise ValueError(f'{node.label}: exported geometry failed validation')
        else:
            bounds = node.bounding_box()
            if abs(node.volume - before['volume']) > .001 or any(
                    abs(first - second) > 1e-5 for first, second in zip(
                        list(bounds.min) + list(bounds.max), before['bounds'])):
                raise ValueError(f'{node.label}: unmodified part changed')
    temporary.replace(TARGET)
    print(f'{len(leaves) - len(CHANGED)} unmodified leaves and all colors retained', flush=True)
    print(f'Updated {TARGET}', flush=True)


def restore_rear(work):
    replacements = {name: revise_shell(read_step(work / f'{name}.step'), name)
                    for name in ('body_back', 'body_middle_top')}
    if any(not part.is_valid or len(part.solids()) != 1 for part in replacements.values()):
        raise ValueError('Restored shells must be valid single solids')
    robot = read_step(TARGET)
    baseline = []

    def rebuild(node):
        children = refit._children(node)
        if children:
            fresh = bd.Compound(children=[rebuild(child) for child in children], label=node.label)
            fresh.location, fresh.color = node.location, node.color
            return fresh
        bounds = node.bounding_box()
        baseline.append((node.label, node.volume, list(bounds.min) + list(bounds.max),
                         list(node.color) if node.color else None))
        if node.label not in replacements:
            return node
        fresh = bd.Compound(replacements[node.label].solids(), label=node.label)
        fresh.color = node.color
        return fresh

    temporary = TARGET.with_suffix('.pending.step')
    result = refit._encode_colours(rebuild(robot))
    bd.export_step(result, temporary)
    del result, robot
    gc.collect()
    exported = read_step(temporary)
    leaves = [node for node in refit._walk(exported) if not refit._children(node)]
    assert len(leaves) == len(baseline)
    for node, (label, volume, bounds, color) in zip(leaves, baseline):
        assert node.label == label
        actual = list(node.color) if node.color else None
        assert (actual is None) == (color is None)
        assert actual is None or max(abs(first-second) for first, second in zip(actual, color)) < 1e-6
        if label in replacements:
            assert node.is_valid and len(node.solids()) == 1
            assert abs(node.volume - replacements[label].volume) < .001
        else:
            actual_bounds = node.bounding_box()
            assert abs(node.volume - volume) < .001
            assert max(abs(first-second) for first, second in zip(
                list(actual_bounds.min) + list(actual_bounds.max), bounds)) < 1e-5
    temporary.replace(TARGET)
    print('Rear shells restored; 135 other leaves and all colors retained', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=Path('/tmp/s11_local'))
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--restore-rear', action='store_true')
    args = parser.parse_args()
    refit._start_memory_watchdog()
    if args.restore_rear:
        restore_rear(args.work)
    elif args.extract:
        extract(args.work)
    elif args.assemble:
        assemble(args.work)
    else:
        build(args.work)


if __name__ == '__main__':
    main()
