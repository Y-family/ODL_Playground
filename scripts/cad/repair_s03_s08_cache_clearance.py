"""Clear the S03/S08 cache-to-hip-link motion envelope in the existing covers."""

import gc
import hashlib
from pathlib import Path

from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape

import refit_upper_servo_interfaces_local as refit


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
OUTPUT = Path('/tmp/open_duck_s03_s08_collision_candidate.step')
SOURCE_SHA256 = '945755d7426bbf28b3c6a4287605ddcdbf206a2ae3e8879e663378aa50a43976'


def children(node):
    return list(getattr(node, 'children', None) or [])


def detached(node):
    shape = bd.Compound(node.solids())
    shape.label = node.label
    shape.color = node.color
    return shape


def clear_cover(cover, link, axis, angles):
    original = detached(cover)
    result = detached(cover)
    fixed = detached(link)
    for angle in angles:
        posed = original.rotate(axis, angle)
        overlap_result = posed.intersect(fixed)
        overlap_items = list(overlap_result) if overlap_result else []
        if overlap_items:
            overlap = bd.Compound(overlap_items)
            tool = overlap.rotate(axis, -angle)
            result = refit._cut(result, [tool]).clean()
            del tool
            del overlap
        del posed, overlap_items
        gc.collect()
    result = result.fix()
    solids = list(result.solids())
    if len(solids) > 1:
        volumes = sorted((solid.volume, solid) for solid in solids)
        debris_volume, debris = volumes[0]
        main_volume = volumes[-1][0]
        assert debris_volume < 10.0 and debris_volume < main_volume * 1e-3, (
            'unexpected detached solid', debris_volume, main_volume
        )
        result = bd.Compound([solid for solid in solids if solid is not debris]).clean()
        result = result.fix()
    check = check_occurrence_shape(result.wrapped)
    assert check['solidCount'] == 1 and not check['reasons'], check
    result.label, result.color = cover.label, cover.color
    return result


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    robot = read_step(SOURCE)
    found = {}

    def walk(node):
        if not children(node):
            found[node.label] = node
            return
        for child in children(node):
            walk(child)

    walk(robot)
    axes = {
        'right': bd.Axis((-35.06, -93.925, -0.074061187627), (0, 1, 0)),
        'left': bd.Axis((-35.06, 93.925, -0.074061187627), (0, -1, 0)),
    }
    angles = {
        'right': range(40, 71),
        'left': range(-70, -39),
    }
    changes = {
        'right_cache': clear_cover(found['right_cache'], found['right_roll_to_pitch'], axes['right'], angles['right']),
        'left_cache': clear_cover(found['left_cache'], found['left_roll_to_pitch'], axes['left'], angles['left']),
    }

    def replace(node):
        if not children(node):
            return changes.get(node.label, node)
        fresh = bd.Compound(children=[replace(child) for child in children(node)], label=node.label)
        fresh.location, fresh.color = node.location, node.color
        return fresh

    result = refit._encode_colours(replace(robot))
    bd.export_step(result, OUTPUT)
    exported = read_step(OUTPUT)
    for label in changes:
        node = next(item for item in children(exported) if item.label == label)
        check = check_occurrence_shape(node.wrapped)
        assert check['solidCount'] == 1 and not check['reasons'], (label, check)
    print('candidate', OUTPUT, 'sha256', hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
