"""Shorten only the straight middle span; preserve the original rounded end."""
from cadgen import read_step,step,build123d as bd
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

@step(out='../../cad/local_fix/right_cache_short.step')
def right_cache_short():
    r=read_step(m.PROJECT_ROOT/'cad/local_fix/open_duck_mini_14_servos_numbered.step')
    old=m._one_child(r,'right_cache')
    knee=m._one_child(m._one_child(r,'knee_and_ankle_assembly <1>'),'S04_RIGHT_KNEE_ZP10D')
    _,passive,active=v._servo_parts(knee)
    end_z=min(p.bounding_box().min.Z for p in (passive,active))+9
    shortening=end_z-old.bounding_box().min.Z
    # The lower profile is translated intact, not cut off or scaled.
    # Shorten the straight middle region between z=-70 and z=-20.
    pieces=[]
    for lo,hi,new_lo,new_hi in [(-200,-70,-200+shortening,-70+shortening),
                               (-70,-20,-70+shortening,-20),
                               (-20,100,-20,100)]:
        clip=m._axis_aligned_clip_box(old,bd.Vector(0,0,1),lo,hi)
        for p in m._intersect_solids(old,clip):
            factor=(new_hi-new_lo)/(hi-lo)
            if abs(factor-1)>1e-10:
                p=p.scale((1,1,factor),about=(0,0,lo))
            pieces.append(p.moved(bd.Pos(0,0,new_lo-lo)))
    result=m._require_one_solid(pieces[0].fuse(*pieces[1:]),'right_cache',max_disconnected_trim=0)
    result.color=old.color
    return result

if __name__=='__main__':
    right_cache_short()
