"""Enforce the user's cradle fit, smaller gap, and knee-cover requirements."""
from cadgen import read_step
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

def main():
    r=read_step('cad/local_fix/open_duck_mini_right_leg_s02_s05.step')
    h=m._one_child(r,'hip_roll_assembly <1>')
    b=m._one_child(h,'roll_motor_bottom')
    s=m._one_child(h,'S02_RIGHT_HIP_ROLL_ZP10D')
    body,passive,active=v._servo_parts(s)
    link=m._one_child(m._one_child(r,'right_roll_to_pitch_assembly <1>'),'right_roll_to_pitch')
    assert m._intersection_volume(b,body)<.001
    for y,z in [(-45,0),(-25,0),(-35,-9.5740611876)]:
        point=m.bd.Vertex(-30,y,z)
        assert max(b.distance_to(point),body.distance_to(point))<.001
    gap=b.distance_to(link)
    assert .5<gap<4.5,gap
    cover=m._one_child(r,'right_cache')
    knee=m._one_child(m._one_child(r,'knee_and_ankle_assembly <1>'),'S04_RIGHT_KNEE_ZP10D')
    _,kp,ka=v._servo_parts(knee)
    end=min(p.bounding_box().min.Z for p in (kp,ka))+9
    assert abs(cover.bounding_box().min.Z-end)<.001
    axis,center=v._axis(s)
    placement=m.ServoPlacement(m.bd.Location(),axis,center,body,passive,active)
    m._require_rotation_clearance('S02 body / compact yellow',[link],[body],placement,m.ROLL_RANGE_DEG)
    print(f'PASS: cradle contact, zero overlap; yellow gap {gap:.3f} mm; cover shortened another 10 mm; sampled S02 body clearance',flush=True)

if __name__=='__main__':
    main()
