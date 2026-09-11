"""Measure clearance, not just zero-volume intersections, on the changed links."""
from cadgen import read_step
import sys
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

def main():
    r=read_step('cad/local_fix/open_duck_mini_right_leg_s02_s05.step')
    hip=m._one_child(r,'hip_roll_assembly <1>')
    pitch=m._one_child(r,'right_roll_to_pitch_assembly <1>')
    link=m._one_child(pitch,'right_roll_to_pitch')
    bottom=m._one_child(hip,'roll_motor_bottom')
    top=m._one_child(hip,'roll_motor_top')
    roll=m._one_child(hip,'S02_RIGHT_HIP_ROLL_ZP10D')
    s03=m._one_child(pitch,'S03_RIGHT_HIP_PITCH_ZP10D')
    upper=m._one_child(r,'knee_and_ankle_assembly <1>')
    plates=[p for p in upper.children if p.label.startswith('left_knee_to_ankle_')]
    print('neutral cradle/yellow',bottom.distance_to(link),flush=True)
    for p in plates:
        print('neutral black/yellow',p.label,p.distance_to(link),flush=True)
    failures=[]
    for label,motor,moving,fixed,limits,minimum in [
        ('S02',roll,[link],[bottom,top],m.ROLL_RANGE_DEG,.5),
        ('S03',s03,plates,[link],m.PITCH_RANGE_DEG,1.0)]:
        if '--s02-only' in sys.argv and label=='S03':
            continue
        body,passive,active=v._servo_parts(motor)
        if label=='S02':
            fixed=[*fixed,body]
        axis,center=v._axis(motor)
        placement=m.ServoPlacement(m.bd.Location(),axis,center,body,passive,active)
        nearby={}
        if label=='S03':
            # Rotation about Y preserves each plate's Y interval. Material
            # outside this slab is at least 2 mm away at every angle.
            for p in moving:
                box=p.bounding_box()
                clip=m._axis_aligned_clip_box(link,m.bd.Vector(0,1,0),box.min.Y-2,box.max.Y+2)
                nearby[p.label]=m._intersect_solids(link,clip)
        worst=(float('inf'),None)
        for angle in m._rotation_angles(*limits):
            for p in moving:
                posed=m._rotate_about_servo_axis(p,placement,angle)
                for host in fixed:
                    if label=='S03':
                        gap=min([2.0,*[posed.distance_to(part) for part in nearby[p.label]]])
                    else:
                        gap=posed.distance_to(host)
                    if gap<worst[0]: worst=(gap,(angle,p.label,host.label))
                    if gap<minimum:
                        print('CLEARANCE FAIL',label,gap,angle,p.label,host.label,flush=True)
                        failures.append((label,gap,angle,p.label,host.label))
        print(label,'sampled minimum',worst,flush=True)
    raise SystemExit(bool(failures))

if __name__=='__main__':
    main()
