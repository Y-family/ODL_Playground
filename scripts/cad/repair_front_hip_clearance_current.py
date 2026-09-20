"""Relocate original front lower shell fastenings to clear hip yaw."""
import gc
import hashlib
import json
from pathlib import Path
from cadgen import build123d as bd, read_step
import refit_upper_servo_interfaces_local as refit
from full_motion_current import TARGET, LIMITS, measured_axes, _leaves, is_moving
from repair_motion_clearance_current import detached, all_leaves, face_facts, volume

BASELINE = '993ac313d66fab58c424a1c54d9b6e69165a0e24e090927024f7b47b1b0a0af3'
WORK = Path('/tmp/open-duck-front-hip-repair')
OLD_HEIGHT = 2.857885438806
NEW_HEIGHT = 25


def bbox_gap(left, right):
    first, second = left.bounding_box(), right.bounding_box()
    return sum(max(0, lower-other_upper, other_lower-upper)**2
               for lower, upper, other_lower, other_upper in zip(first.min, first.max, second.min, second.max))**.5


def main():
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == BASELINE
    WORK.mkdir(exist_ok=True)
    robot = read_step(TARGET)
    items = _leaves(robot)
    axes = measured_axes(robot)
    old = {item['label']: detached(item['shape']) for item in items
           if item['label'] in ('body_middle_bottom', 'body_front')}
    fresh = dict(old)
    for sign in (-1, 1):
        lower, upper = (40, 56) if sign > 0 else (-56, -40)
        seat_box = refit._world_box(29, 36, lower, upper, OLD_HEIGHT-4, OLD_HEIGHT+4)
        seat = refit._shape(old['body_middle_bottom'].intersect(seat_box))
        shifted = seat.translate((0, 0, NEW_HEIGHT-OLD_HEIGHT))
        shell = refit._cut(fresh['body_middle_bottom'], [refit._world_box(15.9, 32.5, lower, upper, -36, 17)])
        shell = refit._fuse(shell, shifted).clean()
        centre = bd.Vector(0, sign*45, NEW_HEIGHT)
        fresh['body_middle_bottom'] = refit._cut(shell, [
            refit._rod(centre, bd.Vector(1, 0, 0), 15.8, 30.2, 1.5),
            refit._rod(centre, bd.Vector(1, 0, 0), 30.2, 36.1, 2)]).clean()
        centre = bd.Vector(0, sign*45, OLD_HEIGHT)
        front = refit._fuse(fresh['body_front'], refit._rod(centre, bd.Vector(1, 0, 0), 36, 46, 3.25))
        centre = bd.Vector(0, sign*45, NEW_HEIGHT)
        fresh['body_front'] = refit._cut(front, [
            refit._rod(centre, bd.Vector(1, 0, 0), 35.9, 46.1, 1.75),
            refit._rod(centre, bd.Vector(1, 0, 0), 39, 46.1, 3.25)]).clean()
    for name, shape in fresh.items():
        assert shape.is_valid and len(shape.solids()) == 1, name
        assert (shape.bounding_box().min-old[name].bounding_box().min).length < 1e-4
        assert (shape.bounding_box().max-old[name].bounding_box().max).length < 1e-4
        bd.export_step(shape, WORK / f'{name}.step')
        fresh[name] = read_step(WORK / f'{name}.step')
        print(name, 'valid single solid; bounds unchanged', flush=True)
    for sign in (-1, 1):
        centre = bd.Vector(0, sign*45, NEW_HEIGHT)
        ring = refit._cut(refit._rod(centre, bd.Vector(1, 0, 0), 29.05, 30.15, 2.1),
                         [refit._rod(centre, bd.Vector(1, 0, 0), 29, 30.2, 1.51)])
        assert volume(ring.cut(fresh['body_middle_bottom'])) < .001, 'Thread seat lost'
        bearing = refit._cut(refit._rod(centre, bd.Vector(1, 0, 0), 38.5, 38.95, 3),
                            [refit._rod(centre, bd.Vector(1, 0, 0), 38.4, 39, 1.76)])
        assert volume(bearing.cut(fresh['body_front'])) < .001, 'Head bearing lost'
        assert volume(fresh['body_front'].intersect(refit._rod(centre, bd.Vector(1, 0, 0), 36, 46, 1.74))) < .001
    print('Both relocated fastenings: bore and bearing checks passed', flush=True)
    added = {name: refit._shape(shape.cut(old[name])) for name, shape in fresh.items()}
    for name, material in added.items():
        for item in items:
            if item['label'] == name:
                continue
            neighbor = fresh.get(item['label'], detached(item['shape']))
            if bbox_gap(material, neighbor) < .001:
                assert volume(material.intersect(neighbor)) < .001, (name, item['path'], 'new static interference')
    report = dict(input_sha256=BASELINE, changed=list(fresh), old_fastening_z=OLD_HEIGHT,
                  new_fastening_z=NEW_HEIGHT, checks=[])
    for joint in LIMITS:
        moving = [item for item in items if is_moving(item, joint)]
        minimum = float('inf')
        tested = 0
        for item in moving:
            mover = detached(item['shape'])
            dense = joint in ('S01', 'S06') and item['label'] in ('right_roll_to_pitch', 'left_roll_to_pitch')
            increment = 1 if dense else 5
            for angle in range(LIMITS[joint][0], LIMITS[joint][1]+1, increment):
                posed = mover.rotate(bd.Axis(*axes[joint]), angle)
                for name, material in added.items():
                    gap = bbox_gap(posed, material)
                    if gap < .2:
                        gap = posed.distance_to(material)
                    assert gap >= .199, (joint, angle, item['path'], name, 'new material', gap)
                if joint in ('S01', 'S06'):
                    gap = bbox_gap(posed, fresh['body_middle_bottom'])
                    if gap < .2:
                        gap = posed.distance_to(fresh['body_middle_bottom'])
                    assert gap >= .199, (joint, angle, item['path'], gap)
                    minimum = min(minimum, gap)
                    tested += 1
                del posed
            gc.collect()
        report['checks'].append(dict(joint=joint, new_material_clearance_pass=True,
                                     hip_pose_checks=tested, clearance_lower_bound_mm=minimum if tested else None))
        (WORK / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(joint, 'new material clear', tested, minimum if tested else '', flush=True)
    def replace(node):
        if node.children:
            result = bd.Compound(children=[replace(child) for child in node.children], label=node.label)
            result.location = node.location
        else:
            result = detached(fresh.get(node.label, node))
            result.label = node.label
        result.color = node.color
        return result
    output = WORK / 'review.step'
    bd.export_step(refit._encode_colours(replace(robot)), output)
    exported = read_step(output)
    before, after = all_leaves(robot), all_leaves(exported)
    assert len(before) == len(after)
    for original, candidate in zip(before, after):
        expected = fresh.get(original.label, original)
        assert original.label == candidate.label
        assert candidate.is_valid and abs(candidate.volume-expected.volume) < .01
        first, second = face_facts(expected), face_facts(candidate)
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert left[0] == right[0]
            assert max(abs(before-after) for before, after in zip(left[1], right[1])) < .0001
    report['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    report['export_leaf_count'] = len(after)
    (WORK / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Export verified', report['output_sha256'], flush=True)


if __name__ == '__main__':
    main()
