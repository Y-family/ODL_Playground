"""Validate the final S02/S03 interfaces and yellow/trunk roll clearance."""
import json,math,trimesh
from cadgen import read_step,build123d as bd
from fit_right_leg_forward import ROOT
from scan_s01_s14_motion_collisions import _mesh
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

def main():
    r=read_step(ROOT/'cad/local_fix/open_duck_mini_spacers_lower_review.step')
    hip=m._one_child(r,'hip_roll_assembly <1>')
    pitch=m._one_child(r,'right_roll_to_pitch_assembly <1>')
    upper=m._one_child(r,'knee_and_ankle_assembly <1>')
    yellow=m._one_child(pitch,'right_roll_to_pitch')
    s02=m._one_child(hip,'S02_RIGHT_HIP_ROLL_ZP10D')
    s03=m._one_child(pitch,'S03_RIGHT_HIP_PITCH_ZP10D')
    vendor=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    v._validate_joint('S02',v.JointSpec(s02,tuple(p for p in hip.children if p.label in ('roll_motor_top','roll_motor_bottom')),(yellow,)),vendor)
    v._validate_joint('S03',v.JointSpec(s03,(yellow,),tuple(p for p in upper.children if 'sheet' in p.label)),vendor)
    axis,c=v._axis(s02)
    fixed=[p for p in m._one_child(r,'trunk_assembly <1>').children if p.label in ('trunk_bottom','roll_bearing')]
    bm=trimesh.collision.CollisionManager();cm=trimesh.collision.CollisionManager()
    for i,p in enumerate(fixed):bm.add_object(str(i),_mesh(p,.08,.15))
    cm.add_object('yellow',_mesh(yellow,.08,.15));gaps=[]
    for angle in range(-25,26):
        cm.set_transform('yellow',trimesh.transformations.rotation_matrix(math.radians(angle),list(axis),list(c)))
        assert not cm.in_collision_other(bm),angle
        gaps.append((cm.min_distance_other(bm),angle))
    gap,angle=min(gaps);posed=yellow.rotate(bd.Axis(c,axis),angle)
    exact=min(posed.distance_to(p) for p in fixed)
    overlap=sum(m._intersection_volume(posed,p) for p in fixed)
    assert exact>.5 and overlap<.05
    result=dict(s02_range_deg=[-25,25],step_deg=1,minimum_gap_mm=exact,
                closest_angle_deg=angle,overlap_mm3=overlap,s02_s03_interfaces='passed')
    (ROOT/'cad/review/current_hip_clearance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
