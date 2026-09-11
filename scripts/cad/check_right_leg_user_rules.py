"""Read-only user-rule gate. A failed gate must never be reported as acceptance."""
from pathlib import Path
import json
from cadgen import read_step
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

ROOT = Path(__file__).resolve().parents[2]
USER_CLEARANCE_SERVO_ID = 'S03_RIGHT_HIP_PITCH_ZP10D'

def walk(shape):
    yield shape
    for child in getattr(shape, 'children', ()) or ():
        yield from walk(child)

def one(shape, label):
    matches = [s for s in walk(shape) if s.label == label]
    if len(matches) != 1:
        raise ValueError(f'{label}: expected one, found {len(matches)}')
    return matches[0]

def main():
    robot = read_step(ROOT / 'cad/local_fix/open_duck_mini_right_leg_s02_s05.step')
    link = one(robot, 'right_roll_to_pitch')
    ids = sorted(s.label for s in walk(robot) if s.label.startswith('S0'))
    failures = []
    if ids != ['S02_RIGHT_HIP_ROLL_ZP10D','S03_RIGHT_HIP_PITCH_ZP10D',
               'S04_RIGHT_KNEE_ZP10D','S05_RIGHT_ANKLE_ZP10D']:
        failures.append(f'Expected four leg servos; found {len(ids)}: {ids}')
    servo = one(robot, USER_CLEARANCE_SERVO_ID)
    gaps = {}
    for label in ('ZP10D_vendor_passive_horn', 'ZP10D_vendor_active_horn'):
        gap = one(servo, label).distance_to(link)
        gaps[label] = gap
        required = 2.388678568757369 if 'passive' in label else 1.438092164322595
        if gap < required - 0.01:
            failures.append(f'{USER_CLEARANCE_SERVO_ID}/{label} touches yellow link')
    hip=one(robot,'hip_roll_assembly <1>')
    s02=one(hip,'S02_RIGHT_HIP_ROLL_ZP10D')
    top=one(hip,'roll_motor_top'); bottom=one(hip,'roll_motor_bottom')
    vendor_body=max(m._children(read_step(v.VENDOR)),key=lambda s:s.volume)
    v._validate_joint('S02',v.JointSpec(s02,(top,bottom),(link,)),vendor_body)
    upper=one(robot,'knee_and_ankle_assembly <1>')
    plates=tuple(s for s in upper.children if s.label.startswith('left_knee_to_ankle_'))
    v._validate_joint('S03',v.JointSpec(servo,(link,),plates),vendor_body)
    for motor,moving,fixed,limits in [(s02,[link,servo],[top,bottom],m.ROLL_RANGE_DEG),
                                    (servo,list(plates),[link],m.PITCH_RANGE_DEG)]:
        body,passive,active=v._servo_parts(motor)
        axis,center=v._axis(motor)
        placement=m.ServoPlacement(m.bd.Location(),axis,center,body,passive,active)
        m._require_rotation_clearance(motor.label,moving,fixed,placement,limits)
    print(json.dumps({'local_geometry_checks_passed': not failures,
                      'mechanical_acceptance': 'pending screw engagement and physical fit review',
                      'servo_ids': ids,
                      'horn_to_yellow_mm': gaps, 'failures': failures}, indent=2))
    raise SystemExit(bool(failures))

if __name__ == '__main__':
    main()
