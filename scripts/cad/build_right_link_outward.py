"""Extend only the original right yellow bridge; verify both servo interfaces."""
from cadgen import read_step,step,build123d as bd
from fit_right_leg_forward import ROOT,move_tree
from fit_s02_outward import extend_bridge
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

OUTWARD_MM=7.0

@step(out='../../cad/local_fix/right_link_outward.step')
def right_link_outward():
    r=read_step(ROOT/'cad/local_fix/open_duck_mini_right_leg_forward_review.step')
    g=m._one_child(r,'right_roll_to_pitch_assembly <1>')
    yellow=extend_bridge(m._one_child(g,'right_roll_to_pitch'),OUTWARD_MM)
    s02=m._one_child(m._one_child(r,'hip_roll_assembly <1>'),'S02_RIGHT_HIP_ROLL_ZP10D')
    axis,center=v._axis(s02)
    for horn in v._servo_parts(s02)[1:]:
        v._require_host_lines('S02 horn',(yellow,),v._horn_pattern_lines(horn,axis,center),1.6)
        assert m._intersection_volume(yellow,horn)<.05
    s03=move_tree(m._one_child(g,'S03_RIGHT_HIP_PITCH_ZP10D'),bd.Pos(0,-OUTWARD_MM,0))
    vendor=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    v._require_host_lines('S03 case',(yellow,),v._case_mount_lines(v._servo_parts(s03)[0],vendor),1.)
    print('S02 8 horn holes and S03 4 case holes preserved',flush=True)
    return yellow

if __name__=='__main__':right_link_outward()
