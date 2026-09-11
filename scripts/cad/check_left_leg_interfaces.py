"""Check the newly mirrored leg's real motor mounting interfaces."""
from cadgen import read_step
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

def main():
    r=read_step('cad/local_fix/open_duck_mini_left_leg_s07_s10.step')
    hip=m._one_child(r,'hip_roll_assembly <2>')
    pitch=m._one_child(r,'left_roll_to_pitch_assembly <1>')
    upper=m._one_child(r,'knee_and_ankle_assembly <2>')
    lower=m._one_child(r,'knee_and_ankle_assembly <4>')
    foot=m._one_child(r,'foot_assembly <2>')
    sheets=lambda branch:tuple(p for p in branch.children if p.label.startswith('left_knee_to_ankle_'))
    vb=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    for sid,branch,case,horns in [
        ('S07',hip,tuple(m._one_child(hip,n) for n in ('roll_motor_top','roll_motor_bottom')),(m._one_child(pitch,'left_roll_to_pitch'),)),
        ('S08',pitch,(m._one_child(pitch,'left_roll_to_pitch'),),sheets(upper)),
        ('S09',upper,sheets(upper),sheets(lower)),
        ('S10',lower,sheets(lower),tuple(m._one_child(foot,n) for n in ('foot_top','foot_side')))]:
        servo=next(p for p in branch.children if p.label.startswith(sid+'_'))
        v._validate_joint(sid,v.JointSpec(servo,case,horns),vb)

if __name__=='__main__':
    main()
