"""Extract existing numbered right-leg branches without regenerating the robot."""
from pathlib import Path
from cadgen import read_step, step, build123d as bd
from build_right_link_original_walls import right_link_original_walls, BRIDGE_SHORTENING_MM
from build_right_roll_top_walls import right_roll_top_walls
from build_right_roll_bottom_fitted import right_roll_bottom_fitted
from build_right_cache_short import right_cache_short

ROOT = Path(__file__).resolve().parents[2]
BRANCHES = ('right_cache', 'hip_roll_assembly <1>', 'knee_and_ankle_assembly <3>',
            'right_roll_to_pitch_assembly <1>', 'foot_assembly <1>', 'knee_and_ankle_assembly <1>')

@step(out='../../cad/local_fix/open_duck_mini_right_leg_s02_s05.step')
def right_leg():
    source = read_step(ROOT / 'cad/local_fix/open_duck_mini_14_servos_numbered.step')
    branches = [x for x in source.children if x.label in BRANCHES]
    if len(branches) != len(BRANCHES):
        raise ValueError('Missing or duplicated right-leg branch')
    replacement=right_link_original_walls()
    replacement.label='right_roll_to_pitch'
    top=right_roll_top_walls()
    top.label='roll_motor_top'
    bottom=right_roll_bottom_fitted()
    bottom.label='roll_motor_bottom'
    cover=right_cache_short()
    cover.label='right_cache'
    shift=bd.Pos(0,BRIDGE_SHORTENING_MM,0)
    revised=[]
    for branch in branches:
        if branch.label in ('right_roll_to_pitch_assembly <1>', 'hip_roll_assembly <1>'):
            parts=[replacement if p.label=='right_roll_to_pitch' else
                   top if p.label=='roll_motor_top' else
                   bottom if p.label=='roll_motor_bottom' else
                   p.moved(shift) if branch.label=='right_roll_to_pitch_assembly <1>' else p
                   for p in branch.children]
            updated=bd.Compound(children=parts,label=branch.label)
            updated.location=branch.location
            revised.append(updated)
        else:
            revised.append((cover if branch.label=='right_cache' else branch).moved(shift))
    result = bd.Compound(children=revised, label='OPEN_DUCK_MINI_RIGHT_LEG_S02_S05')
    ids=sorted(p.label for b in revised for p in getattr(b,'children',()) or () if p.label.startswith('S0'))
    if ids != ['S02_RIGHT_HIP_ROLL_ZP10D','S03_RIGHT_HIP_PITCH_ZP10D',
               'S04_RIGHT_KNEE_ZP10D','S05_RIGHT_ANKLE_ZP10D']:
        raise ValueError(f'Stable servo IDs changed: {ids}')
    return result

if __name__ == '__main__':
    right_leg()
