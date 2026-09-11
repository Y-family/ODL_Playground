"""Measure only the replacement yellow part against fixed S02/S03 interfaces."""
from cadgen import read_step
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
from build_right_link_original_walls import BRIDGE_SHORTENING_MM

def main():
    r=read_step('cad/local_fix/open_duck_mini_right_leg_s02_s05.step')
    link=read_step('cad/local_fix/right_link_original_walls.step')
    vendor=read_step(v.VENDOR)
    vb=max(m._children(vendor),key=lambda s:s.volume)
    for branch,sid in [('hip_roll_assembly <1>','S02_RIGHT_HIP_ROLL_ZP10D'),
                       ('right_roll_to_pitch_assembly <1>','S03_RIGHT_HIP_PITCH_ZP10D')]:
        servo=m._one_child(m._one_child(r,branch),sid)
        body,passive,active=v._servo_parts(servo)
        for label,shape in [('body',body),('passive',passive),('active',active)]:
            gap=shape.distance_to(link)
            overlap=m._intersection_volume(shape,link)
            print(sid,label,'gap',gap,'overlap',overlap,flush=True)
            assert overlap < .05
            if sid.startswith('S03') and label!='body':
                required=2.388678568757369 if label=='passive' else 1.438092164322595
                assert gap >= required-.01
            if sid.startswith('S02') and label!='body':
                assert gap < .01
        if sid.startswith('S03'):
            lines=v._case_mount_lines(body,vb)
            found=[v._host_has_line((link,),line,v.HOST_CASE_HOLE_RADIUS_MM) for line in lines]
            print('S03 case holes',found,flush=True)
            assert all(found)
            placement=m.ServoPlacement(m.bd.Location(),m.bd.Vector(0,-1,0),
                m.bd.Vector(-35.06,-90.725+BRIDGE_SHORTENING_MM,-.0740611876),body,passive,active)
            upper=m._one_child(r,'knee_and_ankle_assembly <1>')
            plates=[s for s in m._children(upper) if s.label.startswith('left_knee_to_ankle_')]
            m._require_rotation_clearance('S03 black plates/yellow',plates,[link],placement,m.PITCH_RANGE_DEG)
            m._require_rotation_clearance('S03 horns/yellow',[passive,active],[link],placement,m.PITCH_RANGE_DEG)
            print('S03 rotation -70..30, 5-degree samples PASS',flush=True)
        else:
            axis,center=v._axis(servo)
            for horn in (passive,active):
                found=[v._host_has_line((link,),line,v.HOST_HORN_HOLE_RADIUS_MM)
                       for line in v._horn_pattern_lines(horn,axis,center)]
                print('S02 horn holes',found,flush=True)
                assert all(found)

if __name__=='__main__':
    main()
