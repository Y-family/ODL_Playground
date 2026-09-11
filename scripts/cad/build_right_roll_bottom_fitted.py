"""Fit the existing cradle walls to S02; retain the original bearing journal."""
from cadgen import read_step, step, build123d as bd
import build_zp10d_left_leg_complete as m

@step(out='../../cad/local_fix/right_roll_bottom_fitted.step')
def right_roll_bottom_fitted():
    r=read_step(m.PROJECT_ROOT/'cad/local_fix/open_duck_mini_14_servos_numbered.step')
    hip=m._one_child(r,'hip_roll_assembly <1>')
    old=m._one_child(hip,'roll_motor_bottom')
    motor=m._one_child(hip,'S02_RIGHT_HIP_ROLL_ZP10D')
    base_z=-10.1840611876
    base=m._intersect_solids(old,m._axis_aligned_clip_box(old,bd.Vector(0,0,1),-30,base_z))
    # The unused rear base overhang crosses the yellow link's roll sweep.
    # Shorten that outside edge; the R9.95 bearing journal lies inside it.
    pieces=[]
    for p in base:
        pieces.extend(m._intersect_solids(p,m._axis_aligned_clip_box(p,bd.Vector(0,1,0),-48.5,0)))
    # The original rear wall widens at its top and tapers back at its lower
    # exterior edge, clearing the roll sweep without moving its case face.
    rear_profile=bd.Plane.YZ*bd.Polygon((-45,base_z-.01),(-45,3),
                                      (-51.5,3),(-48,base_z-.01),align=None)
    pieces.append(bd.extrude(rear_profile,amount=29).moved(bd.Pos(-33.5,0,0)))
    pieces.append(bd.Box(29,2.9,21.42,align=(bd.Align.MIN,)*3).moved(bd.Pos(-33.5,-25,base_z-.01)))
    pieces.append(bd.Box(29,20,.63,align=(bd.Align.MIN,)*3).moved(bd.Pos(-33.5,-45,base_z-.01)))
    result=pieces[0].fuse(*pieces[1:])
    result=m._cut_sequential(result,list(m._children(motor)))
    result=m._require_one_solid(result,'roll_motor_bottom',max_disconnected_trim=0)
    result.color=old.color
    return result

if __name__=='__main__':
    right_roll_bottom_fitted()
