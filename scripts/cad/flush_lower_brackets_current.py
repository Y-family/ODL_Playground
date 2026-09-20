"""Make both edges of the lower hip bracket bases flush with their walls."""
import hashlib
import json
from pathlib import Path
from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape
import refit_upper_servo_interfaces_local as refit
from repair_motion_clearance_current import detached, all_leaves, face_facts, volume
from full_motion_current import TARGET, _leaves, measured_axes, is_moving, LIMITS

WORK = Path('/tmp/flush-lower-brackets')
BASELINE = 'a938eff0c7d57ff634a32114734a3dcb57dae7845653025cccc716c15155a936'
OUTER_Y = 43.15
INNER_Y = 17.35


def main():
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == BASELINE
    WORK.mkdir(exist_ok=True)
    robot = read_step(TARGET)
    items = _leaves(robot)
    axes = measured_axes(robot)
    replacements = {}
    report = dict(input_sha256=BASELINE, parts=[])
    for item in items:
        if item['label'] != 'roll_motor_bottom':
            continue
        original = detached(item['shape'])
        sign = 1 if original.bounding_box().center().Y > 0 else -1
        joint = 'S07' if sign > 0 else 'S02'
        lower, upper = (OUTER_Y, 100) if sign > 0 else (-100, -OUTER_Y)
        cutter = refit._world_box(-100, 100, lower, upper, -100, 100)
        lower, upper = (-100, INNER_Y) if sign > 0 else (-INNER_Y, 100)
        inner_cutter = refit._world_box(-100, 100, lower, upper, -100, 100)
        fresh = refit._cut(original, [cutter, inner_cutter]).clean()
        removed = refit._shape(original.cut(fresh))
        validity = check_occurrence_shape(fresh.wrapped)
        assert validity['solidCount'] == 1 and not validity['reasons']
        assert volume(fresh.cut(original)) < .001
        bounds = fresh.bounding_box()
        outer = bounds.max.Y if sign > 0 else -bounds.min.Y
        assert abs(outer-OUTER_Y) < 1e-5
        inner = bounds.min.Y if sign > 0 else -bounds.max.Y
        assert abs(inner-INNER_Y) < 1e-5
        for face in original.faces():
            if face.geom_type != bd.GeomType.PLANE:
                continue
            centre = face.center()
            normal = face.normal_at(centre)
            if abs(abs(centre.Y)-20.25) < 1e-5 or abs(abs(centre.Y)-40.25) < 1e-5 or (normal.Z > .99 and -9.58 < centre.Z < -9.56):
                assert sum(piece.area for piece in refit._items(face.intersect(fresh))) >= face.area-.001, 'Servo support face changed'
        for face in original.faces():
            if face.geom_type == bd.GeomType.CYLINDER:
                cylinder = refit.BRepAdaptor_Surface(face.wrapped).Cylinder()
                radius = cylinder.Radius()
                if radius > 3.3:
                    continue
                axis = cylinder.Axis()
                origin = bd.Vector(axis.Location().X(), axis.Location().Y(), axis.Location().Z())
                direction = bd.Vector(axis.Direction().X(), axis.Direction().Y(), axis.Direction().Z())
                centre = face.center()
                delta = centre-origin
                radial = delta-direction*delta.dot(direction)
                if face.normal_at(centre).dot(radial) > 0:
                    continue
                bounds = face.bounding_box()
                guard = refit._world_box(bounds.min.X-.55, bounds.max.X+.55,
                                         bounds.min.Y-.55, bounds.max.Y+.55,
                                         bounds.min.Z-.55, bounds.max.Z+.55)
                assert volume(removed.intersect(guard)) < .001, ('Screw bearing material changed', radius, str(bounds))
        body = next(entry for entry in items if entry['servo'] == joint and entry['label'] == 'ZP10D_vendor_body')
        assert removed.distance_to(detached(body['shape'])) > .2
        minimum = float('inf')
        for moving in items:
            if not is_moving(moving, joint):
                continue
            mover = detached(moving['shape'])
            for angle in range(LIMITS[joint][0], LIMITS[joint][1]+1):
                posed = mover.rotate(bd.Axis(*axes[joint]), angle)
                if posed.bounding_box().max.Z < fresh.bounding_box().min.Z-.2:
                    continue
                distance = fresh.distance_to(posed)
                assert distance >= .199, (joint, moving['path'], angle, distance)
                minimum = min(minimum, distance)
        filename = WORK / f'{joint}.step'
        bd.export_step(fresh, filename)
        fresh = detached(read_step(filename))
        replacements[id(item['shape'])] = fresh
        report['parts'].append(dict(joint=joint, removed_mm3=removed.volume,
                                    outer_y_abs_mm=OUTER_Y, inner_y_abs_mm=INNER_Y, minimum_gap_mm=minimum,
                                    limits_deg=LIMITS[joint], step_deg=1, validity=validity))
        print(joint, report['parts'][-1], flush=True)
    assert len(replacements) == 2
    def replace(node):
        if node.children:
            result = bd.Compound(children=[replace(child) for child in node.children], label=node.label)
            result.location = node.location
        else:
            result = detached(replacements.get(id(node), node))
            result.label = node.label
        result.color = node.color
        return result
    output = WORK / 'review.step'
    bd.export_step(refit._encode_colours(replace(robot)), output)
    reread = read_step(output)
    before, after = all_leaves(robot), all_leaves(reread)
    assert len(before) == len(after) == 137
    for original, actual in zip(before, after):
        expected = replacements.get(id(original), original)
        assert original.label == actual.label and actual.is_valid
        assert abs(expected.volume-actual.volume) < .01
        first, second = face_facts(expected), face_facts(actual)
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert left[0] == right[0]
            assert max(abs(one-two) for one,two in zip(left[1],right[1])) < .0001
    report['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    report['export_leaf_count'] = 137
    (WORK/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Export verified', report['output_sha256'], flush=True)


if __name__ == '__main__':
    main()
