"""Extend the original S02 holder walls without cylindrical mounting pads."""
from cadgen import read_step,step,build123d as bd
import build_zp10d_left_leg_complete as m

@step(out='../../cad/local_fix/right_roll_top_walls.step')
def right_roll_top_walls():
    r=read_step(m.PROJECT_ROOT/'cad/local_fix/open_duck_mini_14_servos_numbered.step')
    hip=m._one_child(r,'hip_roll_assembly <1>')
    old=m._one_child(hip,'roll_motor_top')
    servo=m._one_child(hip,'S02_RIGHT_HIP_ROLL_ZP10D')
    box=old.bounding_box()
    cap_bottom=35.201
    cap=m._axis_aligned_clip_box(old,bd.Vector(0,0,1),cap_bottom,box.max.Z+1)
    pieces=m._intersect_solids(old,cap)
    # Preserve the existing top interface; change the continuous side-wall dimensions.
    for x0,x1,z0 in [(-40.875,-37.875,16.9259388124),
                      (2.125,5.125,23.9259388124),(0,2.125,cap_bottom)]:
        # Shorten the outside edge to clear the yellow link at -25 degrees.
        y_min=-44.2 if x0 < 0 else -45.8
        pieces.append(bd.Box(x1-x0,-20.64-y_min,box.max.Z-z0,align=(bd.Align.MIN,)*3).moved(
            bd.Pos(x0,y_min,z0)))
    result=pieces[0].fuse(*pieces[1:])
    original=read_step(m.SOURCE_ROBOT)
    p=m._servo_placement(m._one_prefix_child(m._one_child(original,'hip_roll_assembly <1>'),'STS32155'),'hip_roll')
    holes=[p.pose*m._local_x_cylinder(x-5,x+5,y,z,1) for x,y,z in m.ZP10D_CASE_HOLES]
    result=m._cut_sequential(result,[*m._children(servo),*holes])
    result=m._require_one_solid(result,'roll_motor_top',max_disconnected_trim=0)
    result.color=old.color
    return result

if __name__=='__main__':
    right_roll_top_walls()
