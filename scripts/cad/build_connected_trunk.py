"""Original trunk frame and bearing seats adapted to the two hip-yaw servos."""
from cadgen import read_step,step,build123d as bd
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
from package_open_duck_symmetric_10_zp10d import _aligned_left_servo
from build_left_leg_from_approved import cylinder

@step(out='../../cad/local_fix/trunk_connected_s01_s06.step')
def connected_trunk():
    original=read_step(m.SOURCE_ROBOT)
    ot=m._one_child(original,'trunk_assembly <1>')
    numbered=read_step(m.PROJECT_ROOT/'cad/local_fix/open_duck_mini_14_servos_numbered.step')
    nt=m._one_child(numbered,'trunk_assembly <1>')
    right=m._one_child(nt,'S01_RIGHT_HIP_YAW_ZP10D')
    old_left=m._one_child(nt,'S06_LEFT_HIP_YAW_STS3215')
    lp=_aligned_left_servo(old_left,right,'left_yaw')
    left=m._servo_compound(lp,'S06_LEFT_HIP_YAW_ZP10D')
    top=m._one_child(ot,'trunk_top')
    old_floor=44.7909388123765
    lower_seat=43.86593881238
    upper_seat=83.86593881238
    split_y=22.6302467083
    pieces=m._intersect_solids(top,m._axis_aligned_clip_box(top,bd.Vector(0,1,0),-split_y,split_y))
    for y0,y1 in [(-100,-split_y),(split_y,100)]:
        sides=m._intersect_solids(top,m._axis_aligned_clip_box(top,bd.Vector(0,1,0),y0,y1))
        for side in sides:
            for lo,hi,nlo,nhi in [(old_floor,old_floor+5,lower_seat-3,lower_seat),
                                 (old_floor+5,old_floor+36,lower_seat,upper_seat),
                                 (old_floor+36,old_floor+41,upper_seat,upper_seat+3)]:
                clip=m._axis_aligned_clip_box(side,bd.Vector(0,0,1),lo-1e-7,hi+1e-7)
                for p in m._intersect_solids(side,clip):
                    p=p.scale((1,1,(nhi-nlo)/(hi-lo)),about=(0,0,lo))
                    pieces.append(p.moved(bd.Pos(0,0,nlo-lo)))
    # Resize the original bay floors as full walls, never isolated screw pads.
    for y0 in (-50.36,split_y):
        for z0 in (lower_seat-3,upper_seat):
            pieces.append(bd.Box(27.11,50.36-split_y,3,align=(bd.Align.MIN,)*3).moved(bd.Pos(-59.11,y0,z0)))
    revised=pieces[0].fuse(*pieces[1:]).clean()
    vendor_body=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    cutters=[]
    for servo in (right,left):
        body,passive,active=v._servo_parts(servo)
        cutters.append(body)
        # Original output-shaft openings remain open around the rotating horns.
        for horn in (passive,active):
            box=horn.bounding_box(); center=box.center()
            cutters.append(cylinder(center,bd.Vector(0,0,1),box.min.Z-.3,box.max.Z+.3,10.5))
        for point,axis in v._case_mount_lines(body,vendor_body):
            cutters.append(cylinder(point,axis,point.Z-5,point.Z+5,1))
            if point.Z<60:
                cutters.append(cylinder(point,axis,lower_seat-3.01,lower_seat-1.5,2))
            else:
                cutters.append(cylinder(point,axis,upper_seat+1.5,upper_seat+3.01,2))
    revised=m._require_one_solid(m._cut_sequential(revised,cutters),'trunk_top',max_disconnected_trim=0)
    revised.color=top.color
    bottom=m._one_child(ot,'trunk_bottom')
    # Narrow the original middle channel to clear the wider hip plates at
    # +/-30 degrees. Preserve the upper screw interface and lower bearing seats.
    frame=[]
    for lo,hi,factor in [(-100,15.9,1),(15.9,39.4,10/12.6302467083),(39.4,100,1)]:
        for p in m._intersect_solids(bottom,m._axis_aligned_clip_box(bottom,bd.Vector(0,0,1),lo,hi)):
            if factor!=1: p=p.scale((1,factor,1),about=(0,0,0))
            frame.append(p)
    fitted_bottom=m._require_one_solid(frame[0].fuse(*frame[1:]),'trunk_bottom',max_disconnected_trim=0)
    fitted_bottom.color=bottom.color
    bearings=[p for p in ot.children if p.label=='roll_bearing']
    return bd.Compound(children=[revised,fitted_bottom,*bearings,right,left],label='trunk_assembly <1>')

if __name__=='__main__':
    connected_trunk()
