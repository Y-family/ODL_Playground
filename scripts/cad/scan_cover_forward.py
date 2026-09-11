"""Find a symmetric +X cover offset for existing hip/knee animation ranges."""
import argparse,json,math
import numpy as np
import trimesh
from cadgen import read_step,build123d as bd
from fit_right_leg_forward import ROOT
from review_all_servo_motion import walk
from scan_s01_s14_motion_collisions import _mesh

SOURCE=ROOT/'cad/local_fix/open_duck_mini_symmetric_hips_review.step'

def scan(target,offsets,sides):
    nodes=dict(walk(read_step(target)))
    joints=json.loads(target.with_name(target.name+'.js').read_text().splitlines()[0][15:-1])
    leaves={r:p for r,p in nodes.items() if not p.children and int(r.split('.')[1]) not in {3,4,8,12,19}}
    meshes={}
    def mesh(ref):
        if ref not in meshes:
            print('mesh',ref,leaves[ref].label,flush=True)
            meshes[ref]=_mesh(leaves[ref],.1,.15)
        return meshes[ref]
    rows=[]
    for side,cover_ref,ids in [('right','o1.6',('S03','S04')),('left','o1.1',('S08','S09'))]:
        if side not in sides:continue
        cover=leaves[cover_ref];cb=cover.bounding_box()
        print('cover',side,'bounds',cb,flush=True)
        for sid in ids:
            j=next(j for j in joints if j['id']==sid);moving=set(j['refs'])
            cover_moves=cover_ref in moving
            others={r:p for r,p in leaves.items() if r!=cover_ref and ((r not in moving) if cover_moves else (r in moving))}
            others={r:p for r,p in others.items() if p.bounding_box().max.Y>=cb.min.Y and p.bounding_box().min.Y<=cb.max.Y}
            cm=trimesh.collision.CollisionManager();om=trimesh.collision.CollisionManager()
            cm.add_object(cover_ref,mesh(cover_ref))
            for ref in others:om.add_object(ref,mesh(ref))
            transforms=[(angle,trimesh.transformations.rotation_matrix(math.radians(angle),j['axis'],j['origin'])) for angle in range(j['limits'][0],j['limits'][1]+1)]
            for offset in offsets:
                shift=np.eye(4);shift[0,3]=offset
                hits={};distances=[]
                for angle,rotation in transforms:
                    cm.set_transform(cover_ref,rotation@shift if cover_moves else shift)
                    if not cover_moves:
                        for ref in others:om.set_transform(ref,rotation)
                    hit,pairs=cm.in_collision_other(om,return_names=True)
                    for pair in pairs:
                        ref=next(r for r in pair if r!=cover_ref)
                        hits.setdefault(ref,[]).append(angle)
                    distance,names=cm.min_distance_other(om,return_names=True)
                    distances.append((distance,angle,list(names)))
                gap,angle,names=min(distances)
                row={'joint':sid,'offset_x_mm':offset,'minimum_mesh_gap_mm':gap,'closest_angle_deg':angle,
                     'closest_parts':names,'hits':[{'ref':r,'label':others[r].label,'angles':angles} for r,angles in hits.items()]}
                rows.append(row);print(json.dumps(row),flush=True)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',default=str(SOURCE));p.add_argument('--offsets',default='0,4,8,12,16,20');p.add_argument('--sides',default='right,left');p.add_argument('--out',default='cad/review/cover_forward_scan.json');a=p.parse_args()
    rows=scan(__import__('pathlib').Path(a.target),[float(x) for x in a.offsets.split(',')],a.sides.split(','))
    (ROOT/a.out).write_text(json.dumps(rows,indent=2))
