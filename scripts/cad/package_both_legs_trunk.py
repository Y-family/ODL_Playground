"""Add the left leg and original trunk interfaces to the approved right leg."""
from cadgen import read_step,step,build123d as bd
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
from build_left_leg_from_approved import left_leg,adapt_mirrored_part,APPROVED
from build_connected_trunk import connected_trunk

@step(out='../../cad/local_fix/open_duck_mini_both_legs_trunk_s01_s10.step')
def both_legs_trunk():
    right=read_step(APPROVED)
    left=left_leg()
    trunk=connected_trunk()
    # Resolve deferred child geometry before reading the assembly hierarchy.
    left.solids()
    trunk.solids()
    trunk.label='trunk_assembly <1>'
    right_yaw=m._one_child(trunk,'S01_RIGHT_HIP_YAW_ZP10D')
    left_yaw=m._one_child(trunk,'S06_LEFT_HIP_YAW_ZP10D')
    vb=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    branches={p.label:p for p in right.children}
    for branch in left.children:
        if branch.label=='hip_roll_assembly <2>':
            top=m._one_child(branch,'roll_motor_top')
            # Remap the existing upper plate's yaw bore pattern. The virtual
            # reflection feeds the same bounded-bore helper; no motor is mirrored.
            revised=adapt_mirrored_part(top.mirror(bd.Plane.XZ),{'yaw':right_yaw},
                                       {'yaw':left_yaw},vb,'roll_motor_top')
            revised.color=top.color
            branch=bd.Compound(children=[revised if p.label=='roll_motor_top' else p
                                         for p in branch.children],label=branch.label)
        branches[branch.label]=branch
    branches['trunk_assembly <1>']=trunk
    original=read_step(m.SOURCE_ROBOT)
    parts=[branches[p.label] for p in original.children if p.label in branches]
    assert len(parts)==13
    def walk(p):
        yield p
        for child in getattr(p,'children',()) or (): yield from walk(child)
    ids=sorted(p.label[:3] for p in parts for p in walk(p) if p.label.startswith('S0') or p.label.startswith('S10_'))
    assert ids==[f'S{i:02}' for i in range(1,11)],ids
    return bd.Compound(children=parts,label='OPEN_DUCK_MINI_BOTH_LEGS_TRUNK_S01_S10')

if __name__=='__main__':
    both_legs_trunk()
