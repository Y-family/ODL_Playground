"""Measure left-reference interfaces and symmetric bridge extension candidates."""
import math
from cadgen import read_step, build123d as bd
import trimesh
from fit_right_leg_forward import ROOT
from fit_s02_outward import extend_bridge
from scan_s01_s14_motion_collisions import _mesh
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

def main():
    r=read_step(ROOT/'cad/local_fix/open_duck_mini_spacers_lower_review.step')
    hip=m._one_child(r,'hip_roll_assembly <2>')
    pitch=m._one_child(r,'left_roll_to_pitch_assembly <1>')
    y=m._one_child(pitch,'left_roll_to_pitch')
    s=m._one_child(pitch,'S08_LEFT_HIP_PITCH_ZP10D')
    vendor=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    v._validate_joint('S08 reference',v.JointSpec(s,(y,),tuple(p for p in m._one_child(r,'knee_and_ankle_assembly <2>').children if 'sheet' in p.label)),vendor)
    axis,c=v._axis(m._one_child(hip,'S07_LEFT_HIP_ROLL_ZP10D'))
    fixed=[p for p in m._one_child(r,'trunk_assembly <1>').children if p.label in ('trunk_bottom','roll_bearing')]
    fixed.extend(p for p in hip.children if 'ZP10D' not in p.label)
    cm=trimesh.collision.CollisionManager()
    for i,p in enumerate(fixed):cm.add_object(str(i),_mesh(p,.12,.2))
    for distance in (0.,7.,10.):
        q=y if not distance else extend_bridge(y.mirror(bd.Plane.XZ),distance).mirror(bd.Plane.XZ)
        print('candidate',distance,'volume',q.volume,'bbox',q.bounding_box(),flush=True)
        ym=trimesh.collision.CollisionManager();ym.add_object('yellow',_mesh(q,.12,.2))
        values=[]
        for angle in range(-25,26):
            ym.set_transform('yellow',trimesh.transformations.rotation_matrix(math.radians(angle),list(axis),list(c)))
            values.append((ym.min_distance_other(cm),angle))
        gap,angle=min(values)
        posed=q.rotate(bd.Axis(c,axis),angle)
        print('sweep',distance,gap,angle,[(p.label,posed.distance_to(p),m._intersection_volume(posed,p)) for p in fixed],flush=True)

if __name__=='__main__':main()
