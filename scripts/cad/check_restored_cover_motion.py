"""One-degree S03 cover/fixed-parts sweep; confirm mesh hits with BRep."""
import json, math
import trimesh
from cadgen import read_step, build123d as bd
from fit_right_leg_forward import ROOT
from scan_s01_s14_motion_collisions import _mesh
from review_all_servo_motion import walk
import build_zp10d_left_leg_complete as m

TARGET=ROOT/'cad/local_fix/open_duck_mini_spacers_lower_review.step'

def main():
    robot=read_step(TARGET)
    nodes=dict(walk(robot))
    joints=json.loads(TARGET.with_name(TARGET.name+'.js').read_text().splitlines()[0][15:-1])
    j=next(j for j in joints if j['id']=='S03')
    cover=nodes['o1.6']
    moving=set(j['refs'])
    fixed={r:p for r,p in nodes.items() if not p.children and r not in moving
           and int(r.split('.')[1]) not in {3,4,8,12,19}}
    cb=cover.bounding_box()
    # Rotation about Y preserves the cover's Y interval exactly.
    fixed={r:p for r,p in fixed.items() if p.bounding_box().max.Y>=cb.min.Y
           and p.bounding_box().min.Y<=cb.max.Y}
    cm=trimesh.collision.CollisionManager(); fm=trimesh.collision.CollisionManager()
    cm.add_object('cover',_mesh(cover,.08,.1))
    for r,p in fixed.items():
        print('mesh',r,p.label,flush=True)
        fm.add_object(r,_mesh(p,.08,.1))
    candidates={}
    for angle in range(-30,71):
        transform=trimesh.transformations.rotation_matrix(math.radians(angle),j['axis'],j['origin'])
        cm.set_transform('cover',transform)
        hit,pairs=cm.in_collision_other(fm,return_names=True)
        for pair in pairs:
            ref=next(r for r in pair if r!='cover')
            candidates.setdefault(ref,[]).append(angle)
    exact=[]
    for ref,angles in candidates.items():
        for angle in angles:
            posed=cover.rotate(bd.Axis(j['origin'],j['axis']),angle)
            overlap=m._intersection_volume(posed,fixed[ref])
            if overlap>.05:exact.append(dict(ref=ref,label=fixed[ref].label,angle=angle,overlap_mm3=overlap))
    report=dict(joint='S03',range_deg=[-30,70],step_deg=1,
                fixed_parts=len(fixed),mesh_candidates=candidates,confirmed_clashes=exact)
    (ROOT/'cad/review/restored_cover_motion.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
