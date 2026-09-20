"""Extend existing rear hip openings while preserving fastening seats."""
import gc
import hashlib
import json
from pathlib import Path
from cadgen import build123d as bd, read_step
import refit_upper_servo_interfaces_local as refit
from full_motion_current import AXES, MOVING_ROOTS, TARGET, _leaves
from validate_s01_s10_interfaces_local import _axis

BASELINE = '7a98135bef434052fc506fd699d5d8b264d819e87876664c8a0c65b4f4b0e781'
WORK = Path('/tmp/open-duck-hip-repair')


def detached(node):
    return bd.Compound(node.solids())


def all_leaves(node):
    if node.children:
        return [leaf for child in node.children for leaf in all_leaves(child)]
    return [node]


def volume(shape):
    return sum(item.volume for item in refit._items(shape))


def face_facts(shape):
    rows = []
    for face in shape.faces():
        bounds = face.bounding_box()
        rows.append((str(face.geom_type), [face.area, *face.center(), *bounds.min, *bounds.max]))
    return sorted(rows, key=lambda row: (row[0], *[round(value, 3) for value in row[1]]))


def main():
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == BASELINE
    WORK.mkdir(exist_ok=True)
    robot = read_step(TARGET)
    items = _leaves(robot)
    original = detached(next(item['shape'] for item in items if item['label'] == 'body_middle_bottom'))
    fresh = original
    for lower, upper in ((40, 56), (-56, -40)):
        fresh = refit._cut(fresh, [refit._world_box(-64, -53, lower, upper, -36, 17)])
    assert fresh.is_valid and len(fresh.solids()) == 1
    removed = refit._shape(original.cut(fresh))
    for face in original.faces():
        if face.geom_type == bd.GeomType.CYLINDER:
            bounds = face.bounding_box()
            guard = refit._world_box(bounds.min.X-2, bounds.max.X+2,
                                     bounds.min.Y-2, bounds.max.Y+2,
                                     bounds.min.Z-2, bounds.max.Z+2)
            assert volume(removed.intersect(guard)) < 1e-6, 'Fastening seat changed'
    bd.export_step(fresh, WORK / 'shell.step')
    fresh = read_step(WORK / 'shell.step')
    report = dict(input_sha256=BASELINE, changed=['body_middle_bottom'],
                  removed_mm3=removed.volume, screw_guard_mm=2, checks=[])
    for joint, angles in (('S01', range(0, 31)), ('S06', range(-30, 1))):
        def servo_nodes(node):
            if node.label.startswith(joint + '_'):
                yield node
            for child in node.children:
                yield from servo_nodes(child)
        servo, = servo_nodes(robot)
        direction, origin = _axis(servo)
        assert (origin-bd.Vector(AXES[joint][0])).length < 1e-5
        assert direction.dot(bd.Vector(AXES[joint][1])) > .99999
        moving = [item for item in items if item['root'] in MOVING_ROOTS[joint]]
        minimum = float('inf')
        tested = 0
        for index, item in enumerate(moving):
            shape = detached(item['shape'])
            if shape.bounding_box().max.Z < -36.5:
                continue
            for angle in angles:
                posed = shape.rotate(bd.Axis(origin, direction), angle)
                distance = fresh.distance_to(posed)
                assert distance >= .199, (joint, item['path'], angle, distance)
                minimum = min(minimum, distance)
                tested += 1
                del posed
            print(joint, index, item['label'], 'checked', flush=True)
            gc.collect()
        report['checks'].append(dict(joint=joint, angles_deg=list(angles),
                                     minimum_distance_mm=minimum, pair_poses=tested))
    def replace(node):
        if node.children:
            result = bd.Compound(children=[replace(child) for child in node.children], label=node.label)
            result.location = node.location
        else:
            result = detached(fresh if node.label == 'body_middle_bottom' else node)
            result.label = node.label
        result.color = node.color
        return result
    output = WORK / 'review.step'
    bd.export_step(refit._encode_colours(replace(robot)), output)
    exported = read_step(output)
    before, after = all_leaves(robot), all_leaves(exported)
    assert len(before) == len(after)
    for old, new in zip(before, after):
        assert old.label == new.label
        assert new.is_valid
        expected = fresh if old.label == 'body_middle_bottom' else old
        assert abs(new.volume-expected.volume) < .01
        assert (new.bounding_box().min-expected.bounding_box().min).length < 1e-4
        assert (new.bounding_box().max-expected.bounding_box().max).length < 1e-4
        first, second = face_facts(expected), face_facts(new)
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert left[0] == right[0]
            assert max(abs(before-after) for before, after in zip(left[1], right[1])) < .0001
    report['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    report['limitations'] = 'Single-joint 1 degree samples only; opposite yaw direction and other joints remain unresolved.'
    (WORK / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
