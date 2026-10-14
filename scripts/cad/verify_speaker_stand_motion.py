"""Local motion check for the repaired speaker stand only."""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import numpy as np
import trimesh
from cadgen import build123d as bd, read_step
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from full_motion_current import LIMITS, _leaves, measured_axes
from repair_minor_motion_clearance import _mesh as bounded_mesh, detached
from repair_speaker_layout_local import overlap

TARGET=Path('/tmp/head-smooth-fix/review.step')
OUT=Path('/tmp/head-smooth-fix/motion.json')

def main(head_added=None):
    robot=read_step(TARGET); rows={r['path']:r for r in _leaves(robot)}
    axes=measured_axes(robot)
    one=lambda label: next(p for p,r in rows.items() if r['label']==label)
    stand=one('speaker_stand')
    neighbors=[p for p,r in rows.items() if r['root']=='head_assembly <1>' or r['servo']=='S13'
               or r['label'] in ('head_pitch_to_yaw','head_yaw_to_roll','neck_left_sheet','neck_right_sheet')]
    neighbors += [p for p,r in rows.items() if r['label'] in ('antenna','left_antenna_holder','right_antenna_holder')]
    geometry={p:detached(r['shape']) for p,r in rows.items()}
    changed=[stand]
    if head_added:
        added='head::relocated_speaker_seats'
        rows[added]={'label':'head_relocated_speaker_seats','root':'','servo':None}
        geometry[added]=detached(bd.import_brep(head_added))
        changed.append(added)
    result={'source_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),'static':[],'motion':[],
            'sampling':'single joint, other joints zero, 1 degree steps',
            'distance_method':'Conservative rotated mesh bounding boxes minus measured mesh deflections; near threshold uses exact BRep distance and intersection.'}
    for p in changed:
        for q in neighbors:
            if q==p or (p!=stand and rows[q]['label']=='head'): continue
            a,b=geometry[p],geometry[q]; gap=a.distance_to(b)
            common=overlap(a,b) if gap<1e-7 else 0.0
            contact=(p==stand and rows[q]['label'] in ('head','speaker_interface')) or (p!=stand and q==stand)
            passed=common<1e-6 and (contact or gap>=.2-1e-7)
            row={'moving':p,'fixed':q,'fixed_label':rows[q]['label'],'gap_mm':gap,'overlap_mm3':common,'passed':passed}
            result['static'].append(row); assert passed,row
    meshes={}
    def mesh(p):
        if p not in meshes: meshes[p]=bounded_mesh(geometry[p],.01)
        return meshes[p]
    jobs=[]
    fixed_roll=[one('head_yaw_to_roll'),one('head_pitch_to_yaw')]
    fixed_roll += [p for p,r in rows.items() if r['servo']=='S13']
    jobs += [('S14',p,q,-30,30,axes['S14']) for p,q in itertools.product(changed,fixed_roll)]
    jobs += [('S13',p,one('head_pitch_to_yaw'),*LIMITS['S13'],axes['S13']) for p in changed]
    for joint,lo,hi in (('S11',-20,39),('S12',-45,45)):
        fixed=[p for p,r in rows.items() if (joint=='S11' and r['label'] in ('trunk_top','body_front','body_back','body_middle_top')) or
               (joint=='S12' and (r['label'] in ('neck_left_sheet','neck_right_sheet') or (r['servo']=='S12' and r['label']=='ZP10D_vendor_body')))]
        jobs += [(joint,p,q,lo,hi,axes[joint]) for p,q in itertools.product(changed,fixed)]
    for q,r in rows.items():
        if r['label'] in ('antenna','left_antenna_holder','right_antenna_holder'):
            positive=geometry[q].center().Y>.1
            servo=next(p for p,rr in rows.items() if rr['label']=='sg90' and (geometry[p].center().Y>.1)==positive)
            shaft=[]
            for face in geometry[servo].faces():
                ad=BRepAdaptor_Surface(face.wrapped,True)
                if ad.GetType()==GeomAbs_Cylinder and abs(ad.Cylinder().Radius()-2.425)<1e-6:
                    cy=ad.Cylinder(); shaft=(list(cy.Location().Coord()),list(cy.Axis().Direction().Coord())); break
            if shaft: jobs += [('antenna',q,p,-90,90,shaft) for p in changed]
    for joint,moving,fixed,lo,hi,axis in jobs:
        mm,mf=mesh(moving),mesh(fixed)
        manager=trimesh.collision.CollisionManager(); manager.add_object('fixed',mf)
        err=mm.metadata['maximum_deflection_mm']+mf.metadata['maximum_deflection_mm']
        corners=trimesh.bounds.corners(mm.bounds); minimum=(float('inf'),None); exact_min=float('inf')
        for angle in range(lo,hi+1):
            trans=trimesh.transformations.rotation_matrix(np.deg2rad(angle),axis[1],axis[0])
            posed=trimesh.transform_points(corners,trans)
            sep=np.maximum(np.maximum(posed.min(axis=0)-mf.bounds[1],mf.bounds[0]-posed.max(axis=0)),0)
            gap=float(np.linalg.norm(sep))-err
            if gap<=.2:
                posed_shape=geometry[moving].rotate(bd.Axis(*axis),angle)
                gap=posed_shape.distance_to(geometry[fixed])
                common=overlap(posed_shape,geometry[fixed]) if gap<1e-7 else 0.0
                assert common<1e-6 and gap>=.2-1e-7,(joint,rows[fixed]['label'],angle,gap,common)
            minimum=min(minimum,(gap,angle)); exact_min=min(exact_min,gap)
        result['motion'].append({'joint':joint,'moving':rows[moving]['label'],'fixed':rows[fixed]['label'],'range_deg':[lo,hi],
                                 'minimum_clearance_lower_bound_mm':minimum[0],'minimum_angle_deg':minimum[1],
                                 'passed':True})
        print('passed',joint,rows[fixed]['label'],minimum,flush=True)
    result['passed']=True; OUT.write_text(json.dumps(result,indent=2)+'\n')
    print('stand motion passed',len(result['static']),len(result['motion']),flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',type=Path,default=TARGET)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--head-added',type=Path)
    args=parser.parse_args()
    TARGET,OUT=args.target,args.out
    main(args.head_added)
