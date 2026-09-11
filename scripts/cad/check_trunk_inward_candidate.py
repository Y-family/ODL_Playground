"""Local interface and sampled clearance checks; no full-robot rebuild."""
from cadgen import read_step
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
import check_trunk_leg_connections as connections

TARGET='cad/local_fix/open_duck_mini_trunk_inward_candidate.step'

def main():
    connections.TARGET=TARGET
    connections.main()
    root=read_step(TARGET)
    trunk=m._one_child(root,'trunk_assembly <1>')
    hosts=[m._one_child(trunk,n) for n in ('trunk_top','trunk_bottom')]
    vendor=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    for i,side,sid in [(1,'right','S02'),(2,'left','S07')]:
        hip=m._one_child(root,f'hip_roll_assembly <{i}>')
        motor=next(p for p in hip.children if p.label.startswith(sid+'_'))
        branch=m._one_child(root,f'{side}_roll_to_pitch_assembly <1>')
        link=m._one_child(branch,f'{side}_roll_to_pitch')
        v._validate_joint(sid,v.JointSpec(motor,tuple(m._one_child(hip,n) for n in ('roll_motor_top','roll_motor_bottom')),(link,)),vendor)
        pitch_id='S03' if i==1 else 'S08'
        pitch_motor=next(p for p in branch.children if p.label.startswith(pitch_id+'_'))
        upper=m._one_child(root,f'knee_and_ankle_assembly <{i}>')
        sheets=tuple(p for p in upper.children if p.label.startswith('left_knee_to_ankle_'))
        v._validate_joint(pitch_id,v.JointSpec(pitch_motor,(link,),sheets),vendor)
        body,passive,active=v._servo_parts(motor)
        axis,center=v._axis(motor)
        placement=m.ServoPlacement(m.bd.Location(),axis,center,body,passive,active)
        worst=(float('inf'),None)
        for angle in [25,-25,0,20,-20,15,-15,10,-10,5,-5]:
            posed=m._rotate_about_servo_axis(link,placement,angle)
            for host in hosts:
                gap=posed.distance_to(host)
                print(sid,angle,host.label,round(gap,4),flush=True)
                if gap<worst[0]:worst=(gap,(angle,host.label))
                assert gap>.5,(sid,angle,host.label,gap)
        print(sid,'yellow/trunk sampled clearance',worst,flush=True)

if __name__=='__main__':
    main()
