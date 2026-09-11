"""Use the approved left yellow bracket for both hips, mm, X forward/Y left.

Keep S02/S07 and the trunk fixed. Extend only the bridge between the horn
mounts and pitch cradle by 7 mm. Move downstream legs to symmetric positions;
right fore/aft position follows the left reference. No component added/deleted.
Retain lowered spacers, covers, colors and numbered servos. Validate exported
interfaces and both +/-25-degree roll sweeps separately after building.
"""
from cadgen import read_step, step, report, build123d as bd
from fit_right_leg_forward import ROOT, move_tree
from fit_s02_outward import extend_bridge
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

SOURCE=ROOT/'cad/local_fix/open_duck_mini_spacers_lower_review.step'
OUTWARD_MM=7.0
RIGHT_FORE_AFT_MM=-12.185
LEFT_MOVING={'left_cache','knee_and_ankle_assembly <2>',
             'knee_and_ankle_assembly <4>','foot_assembly <2>'}
RIGHT_MOVING={'right_cache','knee_and_ankle_assembly <1>',
              'knee_and_ankle_assembly <3>','foot_assembly <1>'}

def fresh(part,label):
    # A modified imported part must not retain the input occurrence-tree tags.
    result=bd.Solid(part.solids()[0].wrapped)
    result.label=label;result.color=part.color
    assert result.is_valid and result.volume>0
    return result

@step(out='../../cad/local_fix/open_duck_mini_symmetric_hips_review.step')
def symmetric_hips():
    robot=read_step(SOURCE)
    def signature(p):return (p.label,[signature(c) for c in p.children])
    original_tree=signature(robot)
    lg=m._one_child(robot,'left_roll_to_pitch_assembly <1>')
    rg=m._one_child(robot,'right_roll_to_pitch_assembly <1>')
    left_yellow=m._one_child(lg,'left_roll_to_pitch')
    left_pitch=m._one_child(lg,'S08_LEFT_HIP_PITCH_ZP10D')
    right_pitch=m._one_child(rg,'S03_RIGHT_HIP_PITCH_ZP10D')
    left_roll=m._one_child(m._one_child(robot,'hip_roll_assembly <2>'),'S07_LEFT_HIP_ROLL_ZP10D')
    right_roll=m._one_child(m._one_child(robot,'hip_roll_assembly <1>'),'S02_RIGHT_HIP_ROLL_ZP10D')
    lshift=bd.Pos(0,OUTWARD_MM,0)
    # The input's right leg already moved out by 7 mm.
    rshift=bd.Pos(RIGHT_FORE_AFT_MM,7.-OUTWARD_MM,0)
    lp=move_tree(left_pitch,lshift);rp=move_tree(right_pitch,rshift)
    report('Extend left-reference bridge; preserve both existing mounting arms')
    ly=fresh(extend_bridge(left_yellow.mirror(bd.Plane.XZ),OUTWARD_MM).mirror(bd.Plane.XZ),'left_roll_to_pitch')
    vendor=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    report('Restore matching original right bracket; preserve its correct screw pattern')
    original=read_step(ROOT/'cad/local_fix/open_duck_mini_complete_14_zp10d_numbered.step')
    original_right=m._one_child(m._one_child(original,'right_roll_to_pitch_assembly <1>'),'right_roll_to_pitch')
    # This is the left-reference counterpart, before the asymmetric forward
    # edit. Its right-specific screw pattern already passed interface checks.
    ry=fresh(extend_bridge(original_right,OUTWARD_MM),'right_roll_to_pitch')
    for sid,servo,yellow,hip_name in [('S02',right_roll,ry,'hip_roll_assembly <1>'),
                                      ('S07',left_roll,ly,'hip_roll_assembly <2>')]:
        hip=m._one_child(robot,hip_name)
        v._validate_joint(sid,v.JointSpec(servo,tuple(p for p in hip.children if 'ZP10D' not in p.label),(yellow,)),vendor)
    for sid,servo,yellow in [('S03',rp,ry),('S08',lp,ly)]:
        body,passive,active=v._servo_parts(servo)
        v._require_host_lines(sid,(yellow,),v._case_mount_lines(body,vendor),1.)
        v._require_no_interference(sid,body,(yellow,))
        v._require_contact(sid,body,(yellow,))
        assert min(yellow.distance_to(h) for h in (passive,active))>.5
    children=[]
    for p in robot.children:
        if p.label==lg.label:
            children.append(bd.Compound(children=[ly if c.label==left_yellow.label else lp if c.label==left_pitch.label else c for c in p.children],label=p.label))
        elif p.label==rg.label:
            children.append(bd.Compound(children=[ry if c.label=='right_roll_to_pitch' else rp if c.label==right_pitch.label else c for c in p.children],label=p.label))
        elif p.label in LEFT_MOVING:children.append(move_tree(p,lshift))
        elif p.label in RIGHT_MOVING:children.append(move_tree(p,rshift))
        else:children.append(p)
    result=bd.Compound(children=children,label=robot.label)
    assert signature(result)==original_tree
    print('yellow volumes',ly.volume,ry.volume,flush=True)
    return result

if __name__=='__main__':symmetric_hips()
