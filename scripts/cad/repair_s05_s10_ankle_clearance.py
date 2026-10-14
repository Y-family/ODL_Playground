"""Clear the S05/S10 ankle swept envelope by cutting the moving foot parts.

User decision 2026-10-07: method B - remove material ("削料") from the moving
foot parts so the ankle can reach +/-90 deg.  The fixed ankle plates keep their
servo-mount holes untouched, whereas cutting the fixed plates destroyed the
plate's own mount holes (Z=-115.5 removed entirely, Z=-137.4 counterbores cut in
half) - verified before choosing the side to cut.

Removal set for the moving part M against the fixed parts F over a rotation of
the joint is

    union over angle a of ( M intersect F.rotate(-a) )

evaluated on the running result, so each angle contributes exactly the material
that would still collide.  Sampling is 1 deg from -(90+MARGIN) to +(90+MARGIN);
the normal B-Rep collision check samples integer degrees, so every checked angle
is covered and the MARGIN leaves a real running clearance at the +/-90 deg
software limit instead of a zero-gap touch.

The tool is applied one angle at a time.  Merging all angles into one fused
solid, or intersecting once with a compound tool, is both far slower and
unreliable here (OCC returns a null or a corrupt result); the sequential cut
runs at ~0.1 s per angle.

Only the four printed foot parts change; no new parts are added and the servo
mounting holes are preserved.
"""

import gc
import hashlib
import time
from pathlib import Path

from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape

import refit_upper_servo_interfaces_local as refit
from full_motion_current import _leaves, measured_axes


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
OUTPUT = Path('/tmp/open_duck_s05_s10_ankle_candidate.step')
SOURCE_SHA256 = 'b784f01472a058aaeecfa2e4b466d5f2d05f1d1b18065aea3a821add10838661'

MARGIN_DEG = 2.0
STEP_DEG = 1.0
MIN_TOOL_MM3 = 0.05

# servo -> [(moving part path, [fixed part paths]), ...]
JOBS = {
    'S05': [
        ('/foot_assembly <1>[17]/foot_side[1]',
         ['/knee_and_ankle_assembly <3>[13]/left_knee_to_ankle_right_sheet[3]',
          '/knee_and_ankle_assembly <3>[13]/leg_spacer[0]']),
        ('/foot_assembly <1>[17]/foot_top[2]',
         ['/knee_and_ankle_assembly <3>[13]/left_knee_to_ankle_left_sheet[1]']),
    ],
    'S10': [
        ('/foot_assembly <2>[22]/foot_side[0]',
         ['/knee_and_ankle_assembly <4>[20]/left_knee_to_ankle_right_sheet[0]',
          '/knee_and_ankle_assembly <4>[20]/leg_spacer[2]']),
        ('/foot_assembly <2>[22]/foot_top[2]',
         ['/knee_and_ankle_assembly <4>[20]/left_knee_to_ankle_left_sheet[1]']),
    ],
}


def leaves_by_path(robot):
    return {item['path']: item['shape'] for item in _leaves(robot)}


def detached(shape):
    node = bd.Compound(shape.solids())
    node.label = shape.label
    node.color = shape.color
    return node


def as_compound(inter):
    if inter is None:
        return bd.Compound([])
    items = list(inter) if hasattr(inter, '__iter__') else [inter]
    return bd.Compound([item for item in items if item is not None])


def clear_part(moving, fixed_parts, axis, angles, label):
    result = moving
    cuts = 0
    started = time.time()
    for angle in angles:
        for fixed in fixed_parts:
            tool = as_compound(result.intersect(fixed.rotate(axis, -angle)))
            if tool.volume < MIN_TOOL_MM3:
                continue
            result = refit._cut(result, [tool])
            cuts += 1
        gc.collect()
    result = result.clean().fix()
    solids = list(result.solids())
    if len(solids) > 1:
        volumes = sorted((solid.volume, solid) for solid in solids)
        debris_volume, debris = volumes[0]
        main_volume = volumes[-1][0]
        assert debris_volume < 10.0 and debris_volume < main_volume * 1e-3, (
            'unexpected detached solid', label, debris_volume, main_volume)
        result = bd.Compound([s for s in solids if s is not debris]).clean().fix()
    check = check_occurrence_shape(result.wrapped)
    assert check['solidCount'] == 1 and not check['reasons'], (label, check)
    result.label, result.color = moving.label, moving.color
    removed = moving.volume - result.volume
    print(f'{label:<34} vol={moving.volume:8.1f} removed={removed:7.1f} '
          f'({100 * removed / moving.volume:4.1f}%) cuts={cuts} {time.time() - started:.1f}s',
          flush=True)
    return result


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    robot = read_step(SOURCE)
    by_path = leaves_by_path(robot)
    axes = measured_axes(robot)
    angles = []
    value = -(90.0 + MARGIN_DEG)
    while value <= 90.0 + MARGIN_DEG + 1e-9:
        angles.append(round(value, 6))
        value += STEP_DEG

    replacements = {}
    for servo, jobs in JOBS.items():
        origin, direction = axes[servo]
        axis = bd.Axis(tuple(origin), tuple(direction))
        for moving_path, fixed_paths in jobs:
            moving = detached(by_path[moving_path])
            fixed = [detached(by_path[path]) for path in fixed_paths]
            replacements[moving_path] = clear_part(
                moving, fixed, axis, angles, f'{servo} {moving_path.rsplit("/", 1)[-1]}')
            del moving, fixed
            gc.collect()

    def replace(node, path):
        children = list(getattr(node, 'children', None) or [])
        if path in replacements:
            return replacements[path]
        if not children:
            return node
        fresh = bd.Compound(
            children=[replace(child, path + f'/{child.label}[{index}]')
                      for index, child in enumerate(children)],
            label=node.label,
        )
        fresh.location, fresh.color = node.location, node.color
        return fresh

    top = list(getattr(robot, 'children', None) or [])
    result = bd.Compound(
        children=[replace(child, f'/{child.label}[{index}]') for index, child in enumerate(top)],
        label=robot.label,
    )
    result.location, result.color = robot.location, robot.color
    result = refit._encode_colours(result)
    bd.export_step(result, OUTPUT)
    exported = read_step(OUTPUT)
    exported_paths = leaves_by_path(exported)
    assert set(exported_paths) == set(by_path), 'leaf set changed'
    for path in replacements:
        check = check_occurrence_shape(exported_paths[path].wrapped)
        assert check['solidCount'] == 1 and not check['reasons'], (path, check)
    print('candidate', OUTPUT, 'sha256', hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
