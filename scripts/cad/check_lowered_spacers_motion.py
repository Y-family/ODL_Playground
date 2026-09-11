"""Measure each lowered spacer against its parent servo over the review range."""
import json,math,trimesh
from cadgen import read_step, build123d as bd
from fit_right_leg_forward import ROOT
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
from scan_s01_s14_motion_collisions import _mesh

def main():
    r=read_step(ROOT/'cad/local_fix/open_duck_mini_spacers_lower_review.step')
    rows=[]
    specs=[('knee_and_ankle_assembly <1>','S03','right_roll_to_pitch_assembly <1>',(-30,70)),
           ('knee_and_ankle_assembly <3>','S04','knee_and_ankle_assembly <1>',(-90,90)),
           ('knee_and_ankle_assembly <2>','S08','left_roll_to_pitch_assembly <1>',(-70,30)),
           ('knee_and_ankle_assembly <4>','S09','knee_and_ankle_assembly <2>',(-90,90))]
    for branch,sid,host,limits in specs:
        p=m._one_child(m._one_child(r,branch),'leg_spacer')
        s=next(c for c in m._one_child(r,host).children if c.label.startswith(sid+'_'))
        body,_,_=v._servo_parts(s);axis,center=v._axis(s)
        a=trimesh.collision.CollisionManager();b=trimesh.collision.CollisionManager()
        a.add_object('p',_mesh(p,.08,.1));b.add_object('body',_mesh(body,.08,.1))
        gaps=[];hits=[]
        for angle in range(limits[0],limits[1]+1):
            a.set_transform('p',trimesh.transformations.rotation_matrix(math.radians(angle),list(axis),list(center)))
            if a.in_collision_other(b):hits.append(angle)
            gaps.append((a.min_distance_other(b),angle))
        gap,angle=min(gaps)
        posed=p.rotate(bd.Axis(center,axis),angle)
        exact_gap=posed.distance_to(body)
        overlap=m._intersection_volume(posed,body)
        row=dict(joint=sid,range_deg=limits,step_deg=1,min_mesh_gap_mm=gap,
                 mesh_collision_angles_deg=hits,
                 closest_angle_deg=angle,brep_gap_at_closest_sample_mm=exact_gap,
                 overlap_at_closest_sample_mm3=overlap)
        rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'cad/review/lowered_spacers_motion.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
