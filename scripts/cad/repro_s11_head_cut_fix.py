"""Try to make the head-side cut robust: bloated tool vs fuzzy boolean."""

import gc
from pathlib import Path

import numpy as np
from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape

import refit_upper_servo_interfaces_local as refit
from full_motion_current import _leaves

SOURCE = Path('/tmp/backup_upper_refit_d43675e2.step')
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
HEAD = '/head_assembly <1>[21]/head[4]'
AXIS = bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0))
ANGLES = [38.5, 38.75, 39.0, 39.25]


def as_compound(value):
    if value is None:
        return bd.Compound([])
    items = list(value) if hasattr(value, '__iter__') else [value]
    return bd.Compound([item for item in items if item is not None])


def detached(shape):
    node = bd.Compound(shape.solids())
    node.label, node.color = shape.label, shape.color
    return node


def check(tag, shape):
    result = check_occurrence_shape(shape.wrapped)
    print(f'{tag:<34} volume={shape.volume:10.2f} solids={result["solidCount"]} '
          f'reasons={result["reasons"]}', flush=True)
    return not result['reasons']


def main():
    robot = read_step(SOURCE)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    del robot
    gc.collect()
    head, body = detached(shapes[HEAD]), detached(shapes[BODY])

    tools = []
    for angle in ANGLES:
        posed = head.rotate(AXIS, angle)
        piece = as_compound(posed.intersect(body))
        del posed
        if piece.volume > 0.02:
            tools.append(piece.rotate(AXIS, -angle))
        gc.collect()
    tool = refit._fuse(*tools) if len(tools) > 1 else tools[0]
    check('tool (as built)', tool)

    for amount in (0.1, 0.2, 0.5):
        try:
            op = BRepOffsetAPI_MakeOffsetShape()
            op.PerformByJoin(tool.wrapped, amount, 1e-4)
            grown = bd.Compound(op.Shape())
            gc.collect()
            ok = check(f'grown tool +{amount}', grown)
            if not ok:
                continue
            for delta in (None, 0.01, 0.05):
                cut = BRepAlgoAPI_Cut(head.wrapped, grown.wrapped)
                if delta:
                    cut.SetFuzzyValue(delta)
                cut.Build()
                result = bd.Compound(cut.Shape())
                check(f'  cut fuzzy={delta} (grow {amount})', result)
                del cut, result
                gc.collect()
            del grown
        except Exception as error:  # noqa: BLE001
            print(f'grow +{amount} failed: {error}', flush=True)
        gc.collect()

    for delta in (0.02, 0.1):
        cut = BRepAlgoAPI_Cut(head.wrapped, tool.wrapped)
        cut.SetFuzzyValue(delta)
        cut.Build()
        result = bd.Compound(cut.Shape())
        check(f'plain cut fuzzy={delta}', result)
        del cut, result
        gc.collect()


if __name__ == '__main__':
    main()
