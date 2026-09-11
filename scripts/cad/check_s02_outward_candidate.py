"""Choose outward bridge extension by the measured S02/trunk sweep."""
import json,math,trimesh
from cadgen import read_step,build123d as bd
from fit_right_leg_forward import ROOT
from fit_s02_outward import extend_bridge
from scan_s01_s14_motion_collisions import _mesh
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

def main():
    r=read_step(ROOT/'cad/local_fix/open_duck_mini_spacers_lower_review.step')
    y=m._one_child(m._one_child(r,'right_roll_to_pitch_assembly <1>'),'right_roll_to_pitch')
    s=m._one_child(m._one_child(r,'hip_roll_assembly <1>'),'S02_RIGHT_HIP_ROLL_ZP10D')
    axis,c=v._axis(s)
    fixed=[p for p in m._one_child(r,'trunk_assembly <1>').children if p.label in ('trunk_bottom','roll_bearing')]
    bm=trimesh.collision.CollisionManager()
    for i,p in enumerate(fixed):bm.add_object(str(i),_mesh(p,.1,.15))
    for distance in (6.,7.,8.):
        q=extend_bridge(y,distance)
        cm=trimesh.collision.CollisionManager();cm.add_object('yellow',_mesh(q,.1,.15))
        gaps=[]
        for angle in range(-25,26):
            cm.set_transform('yellow',trimesh.transformations.rotation_matrix(math.radians(angle),list(axis),list(c)))
            gaps.append((cm.min_distance_other(bm),angle))
        gap,angle=min(gaps);pose=q.rotate(bd.Axis(c,axis),angle)
        exact=min(pose.distance_to(f) for f in fixed)
        overlap=sum(m._intersection_volume(pose,f) for f in fixed)
        row=dict(outward_mm=distance,angle=angle,min_mesh_gap_mm=gap,brep_gap_mm=exact,overlap_mm3=overlap)
        print(json.dumps(row),flush=True)
        if exact>.5:
            (ROOT/'cad/review/s02_outward_candidate.json').write_text(json.dumps(row,indent=2));break

if __name__=='__main__':main()
