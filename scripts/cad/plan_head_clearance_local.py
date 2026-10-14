"""Head-clearance proposal, mm, current STEP world coordinates.

Keep the accepted R50.75/R38.5 ring, outer shell and main servo placements.
Explore only the internal speaker/SG90 installation layout and original C
frame dimensions. No added parts. Preserve original antenna fore-aft motion,
side-facing output directions and original coupling. Vertical-output layouts
and ring lowering are rejected. The latest user instruction authorizes upright
SG90 placement toward the side edges and resizing its existing seats. Holders
and antennas follow that local translation; their motion stays the same.
Run serially through run_memory_limited.py, under the 4 GB address-space cap.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

from cadgen import build123d as bd
from OCP.OSD import OSD_Parallel, OSD_ThreadPool
from repair_minor_motion_clearance import detached, bounds
from full_motion_current import TARGET

WORK=Path('/tmp/head-plan')
SOURCE_SHA='b074352e6a76f692a58eac8475f9aace7f0efb510677e884e2e205e731584c64'
EDGE_OFFSET_MM=(-10.,1.,-2.)
EDGE_CLOCK_DEG=86.1991237358816
SPEAKER_SOURCE_SHA='f5cd5d1d43d93cfad9c208f2d0910d0c0436a06dbed7609ae2ecef83ced7dea1'
SPEAKER_WORK=Path('/tmp/speaker-forward')
SPEAKER_OFFSET_MM=(45.,3.,0.)
OSD_Parallel.SetUseOcctThreads_s(True)
OSD_ThreadPool.DefaultPool_s().Init(1)


def cache():
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==SOURCE_SHA
    current=WORK/'current-cache.json'
    if current.exists():
        c=json.loads(current.read_text())
        assert c['source_sha256']==SOURCE_SHA
        return c
    c=json.loads(Path('/tmp/head-motion/cache.json').read_text())
    for row in c['parts'].values():
        if row['label']=='head_pitch_to_yaw':
            row['file']='/tmp/head-motion/cross-candidate.brep'
    return c


def prepare_current():
    """Cache current local geometry, not a superseded head repair input."""
    from cadgen import read_step
    from full_motion_current import _leaves, measured_axes
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==SOURCE_SHA
    WORK.mkdir(exist_ok=True)
    robot=read_step(TARGET)
    c=dict(source_sha256=SOURCE_SHA,axes=measured_axes(robot),parts={})
    for i,row in enumerate(_leaves(robot)):
        p=detached(row['shape']);file=WORK/f'current-{i}.brep'
        bd.export_brep(p,file)
        c['parts'][row['path']]={k:row[k] for k in ('label','root','servo')}
        c['parts'][row['path']].update(file=str(file),bounds=bounds(p))
        print('Cached',row['path'],flush=True)
    (WORK/'current-cache.json').write_text(json.dumps(c,indent=2)+'\n')


def shape(row):
    return detached(bd.import_brep(row['file']))


def speaker_prepare():
    """CAD brief: move speaker and its original stand toward eyes (+X), mm.

    Keep the ring, servo poses, antenna repair and exterior fixed. Select the
    a small forward displacement with 0.2 mm local motion clearance;
    retain existing fastening relationships without adding any components.
    Temporary geometry is separate from the unchanged formal STEP until checks
    pass. This user authorization concerns the speaker installation only.
    """
    from cadgen import read_step
    from full_motion_current import _leaves, measured_axes
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==SPEAKER_SOURCE_SHA
    SPEAKER_WORK.mkdir(exist_ok=True)
    robot=read_step(TARGET)
    c=dict(source_sha256=SPEAKER_SOURCE_SHA,axes=measured_axes(robot),parts={})
    for i,r in enumerate(_leaves(robot)):
        if r['root']=='head_assembly <1>' or 'antenna' in r['root'] or r['label'] in ('head_pitch_to_yaw','head_yaw_to_roll') or r['servo']=='S13':
            a=detached(r['shape']);file=SPEAKER_WORK/f'part-{i}.brep';bd.export_brep(a,file)
            c['parts'][r['path']]={k:r[k] for k in ('label','root','servo')}
            c['parts'][r['path']].update(file=str(file),bounds=bounds(a))
    (SPEAKER_WORK/'cache.json').write_text(json.dumps(c,indent=2)+'\n')
    print('Prepared current speaker-local geometry',len(c['parts']),flush=True)


def speaker_probe():
    import numpy as np
    import trimesh
    from scan_s01_s14_motion_collisions import _mesh
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from exact_motion_current import _overlap
    c=json.loads((SPEAKER_WORK/'cache.json').read_text())
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==c['source_sha256']==SPEAKER_SOURCE_SHA
    parts={p:shape(r) for p,r in c['parts'].items()}
    one=lambda label:next(p for p,r in c['parts'].items() if r['label']==label)
    speakers=[one(label) for label in ('speaker_stand','speaker_interface','speaker_bottom','speaker_top')]
    stand=parts[one('speaker_stand')]
    for f in stand.faces():
        ad=BRepAdaptor_Surface(f.wrapped,True)
        if ad.GetType()==GeomAbs_Cylinder:
            cy=ad.Cylinder()
            print('STAND_CYLINDER',dict(radius=cy.Radius(),origin=list(cy.Location().Coord()),direction=list(cy.Axis().Direction().Coord()),bounds=bounds(f)),flush=True)
    meshes={p:_mesh(a,.2,.3) for p,a in parts.items()}
    ring=trimesh.collision.CollisionManager();ring.add_object('ring',meshes[one('head_yaw_to_roll')])
    fixed=trimesh.collision.CollisionManager()
    for p,m in meshes.items():
        if p not in speakers:fixed.add_object(p,m)
    result=[]
    for dx in (20,30,35,40,45,50,55):
        translation=np.eye(4);translation[0,3]=dx
        r=dict(dx_mm=dx,ring=[],static=[])
        for p in speakers:
            a=meshes[p]
            gaps=[]
            for angle in range(-30,31,5):
                transform=trimesh.transformations.rotation_matrix(np.deg2rad(angle),c['axes']['S14'][1],c['axes']['S14'][0])@translation
                gaps.append((ring.min_distance_single(a,transform=transform),angle))
            r['ring'].append(dict(part=p,min_sample=min(gaps)))
            hit,names=fixed.in_collision_single(a,transform=translation,return_names=True)
            r['static'].append(dict(part=p,hits=sorted(names),min_mesh_gap=fixed.min_distance_single(a,transform=translation)))
        result.append(r)
        (SPEAKER_WORK/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
        print('SPEAKER_PROBE',json.dumps(r),flush=True)


def speaker_search():
    import itertools
    import numpy as np
    import trimesh
    from scan_s01_s14_motion_collisions import _mesh
    c=json.loads((SPEAKER_WORK/'cache.json').read_text())
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==c['source_sha256']==SPEAKER_SOURCE_SHA
    parts=c['parts'];one=lambda label:next(p for p,r in parts.items() if r['label']==label)
    labels=('speaker_stand','speaker_interface','speaker_bottom','speaker_top')
    movers=[one(label) for label in labels]
    meshes={}
    for p,r in parts.items():
        print('Speaker search mesh',r['label'],flush=True)
        meshes[p]=_mesh(shape(r),.35 if r['label']=='head' else .2,.3)
    fixed=trimesh.collision.CollisionManager()
    for p,m in meshes.items():
        if p not in movers and parts[p]['root']=='head_assembly <1>':fixed.add_object(p,m)
    ring=trimesh.collision.CollisionManager();ring.add_object('ring',meshes[one('head_yaw_to_roll')])
    feasible=[]
    for dx,dy in ((45,3),(43,3),(45,2),(48,3),(50,3),(45,4)):
        t=np.eye(4);t[:3,3]=[dx,dy,0]
        print('Speaker search trial',dx,dy,flush=True)
        if any(fixed.in_collision_single(meshes[p],transform=t) for p in movers[1:]):continue
        bodygap=min(fixed.min_distance_single(meshes[p],transform=t) for p in movers[1:])
        if bodygap<.3:continue
        ringgap=min((0. if ring.in_collision_single(meshes[p],transform=trimesh.transformations.rotation_matrix(
            np.deg2rad(a),c['axes']['S14'][1],c['axes']['S14'][0])@t) else ring.min_distance_single(meshes[p],transform=trimesh.transformations.rotation_matrix(
            np.deg2rad(a),c['axes']['S14'][1],c['axes']['S14'][0])@t)
            )
            for p in movers for a in range(-30,31,5))
        if ringgap<.4:continue
        feasible.append(dict(offset_mm=[dx,float(dy),0],body_mesh_gap_mm=bodygap,ring_mesh_gap_mm=ringgap))
        print('Speaker feasible coarse layout',feasible[-1],flush=True)
        break
    assert feasible,'No forward body layout passed coarse local clearance'
    feasible.sort(key=lambda r:(sum(abs(v) for v in r['offset_mm']),r['offset_mm'][0]))
    report=dict(source_sha256=SPEAKER_SOURCE_SHA,feasible=feasible,selected=feasible[0],mounts_verified=False)
    (SPEAKER_WORK/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SPEAKER_FORWARD_SELECTED',json.dumps(report['selected']),len(feasible),flush=True)


def speaker_build():
    from compact_head_local import clip,affine
    import refit_upper_servo_interfaces_local as refit
    from cadgen.validity import check_occurrence_shape
    from repair_minor_motion_clearance import protect_mounts
    c=json.loads((SPEAKER_WORK/'cache.json').read_text())
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==c['source_sha256']==SPEAKER_SOURCE_SHA
    selected=json.loads((SPEAKER_WORK/'search.json').read_text())['selected']
    dx,dy,dz=selected['offset_mm'];assert dz==0
    assert (dx,dy,dz)==SPEAKER_OFFSET_MM
    parts=c['parts'];one=lambda label:next(p for p,r in parts.items() if r['label']==label)
    old=shape(parts[one('speaker_stand')]);roof=clip(old,2,223.5,250)
    fixed=clip(roof,1,-41.8,150);arm=clip(roof,1,-150,-41.8)
    low=clip(arm,0,-150,40);mid=clip(arm,0,40,41);high=clip(arm,0,41,150)
    # Stretch only the existing curved arm. Its original four roof feet/holes
    # stay attached to the same shell datums; no new bracket or spacer.
    matrix=[[dx+1,0,0,-40*dx],[dy,1,0,-40*dy],[0,0,1,0],[0,0,0,1]]
    extended=refit._fuse(low,affine(mid,matrix),high.moved(bd.Pos(dx,dy,0)))
    lower=clip(old,2,170,223.5).moved(bd.Pos(dx,dy,0))
    stand=refit._fuse(fixed,extended,lower).clean()
    corner=refit._world_box(48.3,50.6,-31.63,-28.05,225.7,229.1)
    stand=refit._cut(stand,[corner]).clean()
    modified={one('speaker_stand'):stand}
    for label in ('speaker_interface','speaker_bottom','speaker_top'):
        p=one(label);modified[p]=shape(parts[p]).moved(bd.Pos(dx,dy,0))
    ip,bp=one('speaker_interface'),one('speaker_bottom')
    original=modified[ip].clean();before=original.volume
    contact=refit._shape(original.intersect(modified[bp]))
    modified[ip]=refit._cut(original,[modified[bp]]).clean()
    canonical=SPEAKER_WORK/'interface.step';bd.export_step(modified[ip],canonical)
    modified[ip]=detached(bd.import_step(canonical))
    protect_mounts(original,modified[ip],contact)
    validity={};files={}
    for p,a in modified.items():
        a.label=parts[p]['label'];file=SPEAKER_WORK/(parts[p]['label']+'.brep')
        bd.export_brep(a,file);files[p]=str(file);validity[p]=check_occurrence_shape(a.wrapped)
        assert validity[p]['solidCount']==1 and not validity[p]['reasons'],(p,validity[p])
        print('Built forward speaker part',p,validity[p],flush=True)
    report=dict(source_sha256=SPEAKER_SOURCE_SHA,offset_mm=[dx,dy,0],modified=files,validity=validity,
                interface_mating_trim_mm3=before-modified[ip].volume,installed=False,
                roof_fixing_points_unchanged=True,ring_unchanged=True)
    (SPEAKER_WORK/'build.json').write_text(json.dumps(report,indent=2)+'\n')


def speaker_verify():
    import itertools
    import numpy as np
    import trimesh
    from repair_minor_motion_clearance import _mesh as bounded_mesh
    from exact_motion_current import _overlap
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    import refit_upper_servo_interfaces_local as refit
    c=json.loads((SPEAKER_WORK/'cache.json').read_text());build=json.loads((SPEAKER_WORK/'build.json').read_text())
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==c['source_sha256']==build['source_sha256']==SPEAKER_SOURCE_SHA
    parts=c['parts'];one=lambda label:next(p for p,r in parts.items() if r['label']==label)
    geometry={p:detached(bd.import_brep(build['modified'][p])) if p in build['modified'] else shape(r) for p,r in parts.items()}
    modified=list(build['modified']);meshes={}
    report=dict(source_sha256=SPEAKER_SOURCE_SHA,offset_mm=build['offset_mm'],static=[],motion=[],mounts=[],completed=False,passed=False)
    def save():(SPEAKER_WORK/'verify.json').write_text(json.dumps(report,indent=2)+'\n')
    def mesh(p):
        if p not in meshes:meshes[p]=bounded_mesh(geometry[p],.01)
        return meshes[p]
    for p in modified:
        a=geometry[p]
        for q,b in geometry.items():
            if p==q or (q in modified and modified.index(q)<modified.index(p)):continue
            gap=a.distance_to(b)
            # Fixed fastening/mating surfaces may touch; every measured
            # zero-distance pair still needs zero solid intersection.
            print('Speaker static distance',parts[p]['label'],parts[q]['label'],gap,flush=True)
            overlap=_overlap(a,b) if gap<1e-6 else 0.
            intended=q in modified or (parts[p]['label']=='speaker_stand' and parts[q]['label']=='head')
            passed=overlap<1e-7 and (intended or gap>=.2-1e-7)
            row=dict(moving=p,fixed=q,gap_mm=gap,overlap_mm3=overlap,intended_contact=intended,passed=passed)
            report['static'].append(row);save()
            assert passed,row
    stand=geometry[one('speaker_stand')];oldstand=shape(parts[one('speaker_stand')]);head=geometry[one('head')]
    dx,dy,dz=build['offset_mm']
    lost_cache={}
    for face in oldstand.faces():
        ad=BRepAdaptor_Surface(face.wrapped,True)
        if ad.GetType()!=GeomAbs_Cylinder:continue
        cy=ad.Cylinder();radius=cy.Radius()
        if radius>2.1:continue
        origin=bd.Vector(cy.Location().Coord());direction=bd.Vector(cy.Axis().Direction().Coord())
        roof=abs(direction.Z)>.99
        translated=bd.Pos(0,0,0) if roof else bd.Pos(dx,dy,dz)
        expected=face.moved(translated)
        candidates=[]
        for f in stand.faces():
            newad=BRepAdaptor_Surface(f.wrapped,True)
            if newad.GetType()!=GeomAbs_Cylinder:continue
            newcy=newad.Cylinder()
            if abs(newcy.Radius()-radius)<1e-6 and expected.distance_to(f)<1e-7:candidates.append(f)
        assert candidates,('Missing original fixing cylinder',radius,list(origin))
        box=expected.bounding_box();guard=refit._world_box(box.min.X-.55,box.max.X+.55,box.min.Y-.55,box.max.Y+.55,box.min.Z-.55,box.max.Z+.55)
        if roof not in lost_cache:
            reference=oldstand.moved(translated);lost_cache[roof]=refit._shape(reference.cut(stand))
        lost=lost_cache[roof]
        removed=_overlap(lost,guard);assert removed<1e-6,(radius,removed)
        row=dict(radius_mm=radius,old_origin=list(origin),translation_mm=[0,0,0] if roof else build['offset_mm'],missing_bearing_material_mm3=removed)
        if roof and abs(radius-1.35)<1e-6:
            pilots=[]
            for f in head.faces():
                ha=BRepAdaptor_Surface(f.wrapped,True)
                if ha.GetType()!=GeomAbs_Cylinder:continue
                hc=ha.Cylinder();point=bd.Vector(hc.Location().Coord());vector=point-origin
                if abs(hc.Radius()-1)<1e-6 and abs(bd.Vector(hc.Axis().Direction().Coord()).dot(direction))>.99999:
                    radial=(vector-direction*vector.dot(direction)).length
                    if radial<1e-6:pilots.append((radial,bounds(f)))
            assert len(pilots)==1,(list(origin),pilots)
            row.update(head_pilot_axis_error_mm=pilots[0][0],head_pilot_bounds=pilots[0][1],roof_top_z_mm=expected.bounding_box().max.Z)
        report['mounts'].append(row)
    # Existing shell screw axes are untouched because all four roof feet stay
    # in place. Verify their counterbores and retained bearing faces above.
    assert sum(abs(r['radius_mm']-1.35)<1e-6 for r in report['mounts'])==4
    save()
    fixed_roll=[one('head_yaw_to_roll'),one('head_pitch_to_yaw')]+[p for p,r in parts.items() if r['servo']=='S13']
    jobs=[('S14',p,q,-30,30) for p,q in itertools.product(modified,fixed_roll)]
    jobs += [('S13',p,one('head_pitch_to_yaw'),-160,160) for p in modified]
    for joint,p,q,low,high in jobs:
        moving,fixed=mesh(p),mesh(q)
        manager=trimesh.collision.CollisionManager();manager.add_object('fixed',fixed)
        error=moving.metadata['maximum_deflection_mm']+fixed.metadata['maximum_deflection_mm']
        gaps=[];fallback=[]
        axis=bd.Axis(*c['axes'][joint])
        for angle in range(low,high+1):
            t=trimesh.transformations.rotation_matrix(np.deg2rad(angle),c['axes'][joint][1],c['axes'][joint][0])
            gap=manager.min_distance_single(moving,transform=t)-error
            if gap<.2:
                posed=geometry[p].rotate(axis,angle);exact=posed.distance_to(geometry[q]);common=_overlap(posed,geometry[q]) if exact<1e-7 else 0
                fallback.append(dict(angle_deg=angle,gap_mm=exact,overlap_mm3=common));assert exact>=.2-1e-7 and common<1e-7,fallback[-1]
                gap=exact
            gaps.append((gap,angle))
        minimum,angle=min(gaps);posed=geometry[p].rotate(axis,angle)
        exact=posed.distance_to(geometry[q]);common=_overlap(posed,geometry[q]) if exact<1e-7 else 0
        assert exact>=.2-1e-7 and common<1e-7,(p,q,angle,exact,common)
        row=dict(joint=joint,moving=p,fixed=q,range_deg=[low,high],step_deg=1,min_gap_lower_bound_mm=minimum,minimum_angle_deg=angle,exact_minimum_sample_gap_mm=exact,overlap_mm3=common,mesh_error_mm=error,exact_fallback=fallback)
        report['motion'].append(row);save();print('Speaker motion passed',joint,parts[p]['label'],parts[q]['label'],minimum,angle,flush=True)
    report.update(completed=True,passed=True,modified_sha256={p:hashlib.sha256(Path(f).read_bytes()).hexdigest() for p,f in build['modified'].items()})
    save();print('Speaker local checks PASSED',flush=True)


def speaker_export():
    from cadgen import read_step
    from full_motion_current import _leaves,measured_axes
    from remove_s01_s06_upper_horns import leaves,compare
    import refit_upper_servo_interfaces_local as refit
    build=json.loads((SPEAKER_WORK/'build.json').read_text());verify=json.loads((SPEAKER_WORK/'verify.json').read_text())
    assert verify['completed'] and verify['passed']
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==SPEAKER_SOURCE_SHA
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==verify['modified_sha256'][p] for p,f in build['modified'].items())
    robot=read_step(TARGET);before=leaves(robot);axes=measured_axes(robot)
    originals={r['path']:r['shape'] for r in _leaves(robot)};replacements={}
    for p,file in build['modified'].items():
        fresh=detached(bd.import_brep(file));old=originals[p];fresh.label,fresh.color=old.label,old.color;replacements[id(old)]=fresh
    def replace(node):
        if node.children:
            fresh=bd.Compound(children=[replace(ch) for ch in node.children],label=node.label);fresh.location=node.location
        else:fresh=detached(replacements.get(id(node),node));fresh.label=node.label
        fresh.color=node.color;return fresh
    candidate=SPEAKER_WORK/'review.step';bd.export_step(refit._encode_colours(replace(robot)),candidate)
    exported=read_step(candidate);after=leaves(exported);assert len(before)==len(after)==135
    for (oldpath,old),(newpath,new) in zip(before,after):
        assert oldpath==newpath
        compare(replacements.get(id(old),old),new)
        assert (old.color is None)==(new.color is None)
        if old.color is not None:assert max(abs(a-b) for a,b in zip(old.color,new.color))<1e-6
    for j,(point,direction) in axes.items():
        p,d=measured_axes(exported)[j]
        assert max(abs(a-b) for a,b in zip(point,p))<1e-5
        assert max(abs(a-b) for a,b in zip(direction,d))<1e-5
    build.update(candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),export_leaf_count=135,unchanged_leaf_count=131,main_axes_preserved=14)
    (SPEAKER_WORK/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    print('Speaker assembly export compared',build['candidate_sha256'],flush=True)


def speaker_parent_verify():
    """Verify only moved speaker leaves against parent interfaces/antennas."""
    import itertools
    import numpy as np
    import trimesh
    from cadgen import read_step
    from full_motion_current import _leaves
    from repair_minor_motion_clearance import _mesh as bounded_mesh
    from exact_motion_current import _overlap
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    c=json.loads((SPEAKER_WORK/'cache.json').read_text());build=json.loads((SPEAKER_WORK/'build.json').read_text())
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==c['source_sha256']==SPEAKER_SOURCE_SHA
    robot=read_step(TARGET);parts={r['path']:r for r in _leaves(robot)}
    originals={p:detached(r['shape']) for p,r in parts.items()}
    changed={p:detached(bd.import_brep(f)) for p,f in build['modified'].items()}
    geometry={**originals,**changed};meshes={}
    def mesh(p):
        if p not in meshes:meshes[p]=bounded_mesh(geometry[p],.02)
        return meshes[p]
    jobs=[]
    for j,low,high in (('S11',-20,39),('S12',-45,45)):
        fixed=[p for p,r in parts.items() if (j=='S11' and r['label'] in ('trunk_top','body_front','body_back','body_middle_top')) or
               (j=='S12' and (r['label'] in ('neck_left_sheet','neck_right_sheet') or (r['servo']=='S12' and r['label']=='ZP10D_vendor_body')))]
        for p,q in itertools.product(changed,fixed):jobs.append((j,p,q,low,high,c['axes'][j]))
    for p,r in parts.items():
        if r['label'] not in ('antenna','left_antenna_holder','right_antenna_holder'):continue
        positive=geometry[p].center().Y>.1
        servo=next(q for q,r2 in parts.items() if r2['label']=='sg90' and (geometry[q].center().Y>.1)==positive)
        shaft=[]
        for f in geometry[servo].faces():
            ad=BRepAdaptor_Surface(f.wrapped,True)
            if ad.GetType()==GeomAbs_Cylinder and abs(ad.Cylinder().Radius()-2.425)<1e-6:
                cy=ad.Cylinder();shaft.append((list(cy.Location().Coord()),list(cy.Axis().Direction().Coord())))
        assert shaft
        for q in changed:jobs.append(('antenna',p,q,-90,90,shaft[0]))
    report=dict(source_sha256=SPEAKER_SOURCE_SHA,checks=[],completed=False,passed=False,antenna_range_diagnostic_only=True)
    def save():(SPEAKER_WORK/'parent-verify.json').write_text(json.dumps(report,indent=2)+'\n')
    for joint,p,q,low,high,axis in jobs:
        moving,fixed=mesh(p),mesh(q);manager=trimesh.collision.CollisionManager();manager.add_object('fixed',fixed)
        error=moving.metadata['maximum_deflection_mm']+fixed.metadata['maximum_deflection_mm'];gaps=[];fallback=[]
        for angle in range(low,high+1):
            t=trimesh.transformations.rotation_matrix(np.deg2rad(angle),axis[1],axis[0])
            gap=manager.min_distance_single(moving,transform=t)-error
            if gap<.2:
                posed=geometry[p].rotate(bd.Axis(*axis),angle);gap=posed.distance_to(geometry[q])
                common=_overlap(posed,geometry[q]) if gap<1e-7 else 0
                fallback.append(dict(angle_deg=angle,gap_mm=gap,overlap_mm3=common));assert gap>=.2-1e-7 and common<1e-7,fallback[-1]
            gaps.append((gap,angle))
        gap,angle=min(gaps);posed=geometry[p].rotate(bd.Axis(*axis),angle);exact=posed.distance_to(geometry[q])
        assert exact>=.2-1e-7
        row=dict(joint=joint,moving=p,fixed=q,range_deg=[low,high],step_deg=1,min_gap_lower_bound_mm=gap,minimum_angle_deg=angle,exact_minimum_sample_gap_mm=exact,overlap_mm3=0,exact_fallback=fallback)
        report['checks'].append(row);save();print('Speaker parent/antenna passed',joint,parts[p]['label'],parts[q]['label'],gap,flush=True)
    report.update(completed=True,passed=True);save()


def speaker_install():
    import shutil
    import tempfile
    build=json.loads((SPEAKER_WORK/'build.json').read_text())
    verify=json.loads((SPEAKER_WORK/'verify.json').read_text())
    parent=json.loads((SPEAKER_WORK/'parent-verify.json').read_text())
    validation=json.loads((SPEAKER_WORK/'step-validation.json').read_text())
    assert verify['completed'] and verify['passed'] and parent['completed'] and parent['passed']
    assert validation['ok'] and validation['occurrenceCount']==4 and validation['failureCount']==0
    assert build['export_leaf_count']==135 and build['unchanged_leaf_count']==131 and build['main_axes_preserved']==14
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==verify['modified_sha256'][p] for p,f in build['modified'].items())
    assert hashlib.sha256(Path(build['snapshot_file']).read_bytes()).hexdigest()==build['snapshot_sha256']
    source=Path(build['candidate']);assert hashlib.sha256(source.read_bytes()).hexdigest()==build['candidate_sha256']
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==SPEAKER_SOURCE_SHA
    shutil.copyfile(TARGET,SPEAKER_WORK/'source.step')
    fd,temp=tempfile.mkstemp(prefix='.upper-refit-',suffix='.tmp',dir=TARGET.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            with source.open('rb') as s:shutil.copyfileobj(s,f)
            f.flush();os.fsync(f.fileno())
        assert hashlib.sha256(Path(temp).read_bytes()).hexdigest()==build['candidate_sha256']
        os.replace(temp,TARGET)
    finally:
        if Path(temp).exists():Path(temp).unlink()
    for file in (WORK/TARGET.name,WORK/'review.step'):shutil.copyfile(TARGET,file)
    build['installed']=True;verify['installed']=True
    (SPEAKER_WORK/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    (SPEAKER_WORK/'verify.json').write_text(json.dumps(verify,indent=2)+'\n')
    print('INSTALLED forward speaker into same formal STEP',build['candidate_sha256'],flush=True)


def edge_search():
    """User-authorized upright, outward SG90 placement; original ring fixed.

    CAD brief: mm, +Z up. Rotate the two existing bodies about their original
    side-facing output direction, then move toward the side walls. Prefer
    sliding along that same output line so the antenna pivots stay put.
    Original integrated seats/holders may be resized; no extra components.
    This mesh search selects placements, not a verified assembly or install.
    """
    import itertools
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    from scan_s01_s14_motion_collisions import _mesh
    c=cache();parts=c['parts'];facts=json.loads((WORK/'facts.json').read_text())
    local={p:r for p,r in parts.items() if r['root']=='head_assembly <1>' or
           'antenna' in r['root'] or r['label'] in ('head_yaw_to_roll','head_pitch_to_yaw') or r['servo']=='S13'}
    meshes={}
    for p,r in local.items():
        file=WORK/('mesh-'+Path(r['file']).stem+'.npz')
        if file.exists():
            data=np.load(file);m=trimesh.Trimesh(data['vertices'],data['faces'],process=False)
        else:
            m=_mesh(shape(r),.15,.3);np.savez(file,vertices=m.vertices,faces=m.faces)
        meshes[p]=m
    one=lambda label:next(p for p,r in local.items() if r['label']==label)
    hull=ConvexHull(meshes[one('head')].vertices)
    ring=trimesh.collision.CollisionManager();ring.add_object('ring',meshes[one('head_yaw_to_roll')])
    frame=trimesh.collision.CollisionManager();frame.add_object('frame',meshes[one('head_pitch_to_yaw')])
    other=trimesh.collision.CollisionManager()
    for p,r in local.items():
        if r['servo']=='S13':other.add_object(p,meshes[p])
    poses={j:[(a,trimesh.transformations.rotation_matrix(np.deg2rad(a),c['axes'][j][1],c['axes'][j][0]))
              for a in range(-30,31,5) if j=='S14'] if j=='S14' else
             [(a,trimesh.transformations.rotation_matrix(np.deg2rad(a),c['axes'][j][1],c['axes'][j][0]))
              for a in range(-160,161,10)] for j in ('S14','S13')}
    p=one('sg90');f=next(x for x in facts if x['path']==p)
    shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
    origin=np.array(shaft['origin']);axis=np.array(shaft['direction'])
    fixed=trimesh.collision.CollisionManager()
    for q,r in local.items():
        if r['root']=='head_assembly <1>' and r['label'] not in ('sg90','head'):fixed.add_object(q,meshes[q])
    body=meshes[p];vertices=body.convex_hull.vertices
    # The output shaft is intentionally outside the skin in its original
    # opening. Envelope-check the housing/ears behind that interface instead.
    vertices=vertices[(vertices-origin)@axis>=7.49]
    report=dict(source_sha256=SOURCE_SHA,ring_unchanged=True,installed=False,mode='same output line outward',trials=[],feasible=[])
    # Negative original shaft direction is outward; previous searches used
    # positive (inward) offsets and never covered this authorized side region.
    broad='broad' in sys.argv[2:]
    offsets=[(-float(s)*axis).tolist() for s in np.arange(0,40.01,.5)]
    if broad:
        offsets=[list(v) for v in itertools.product(np.arange(-20,20.1,2.5),np.arange(0,30.1,2.5),np.arange(-5,20.1,2.5))]
        report['mode']='outward Y with local X/Z grid'
    if 'fine' in sys.argv[2:]:
        offsets=[list(v) for v in itertools.product(np.arange(-12,-6.01,.5),np.arange(.5,5.01,.5),np.arange(-4,1.01,.5))]
        report['mode']='outward Y, original side-facing axis; fine local grid'
    for turn in (86.1991237358816,266.1991237358816):
        rotation=trimesh.transformations.rotation_matrix(np.deg2rad(turn),axis,origin)
        for delta in offsets:
            t=rotation.copy();t[:3,3]+=delta
            pts=trimesh.transform_points(vertices,t)
            outside=float(np.max(pts@hull.equations[:,:3].T+hull.equations[:,3]))
            row=dict(turn_deg=turn,offset_mm=delta,transform=t.tolist(),outside_hull_mm=outside)
            if outside>-.2:row['reject']='outer shell envelope'
            elif fixed.min_distance_single(body,transform=t)<.3:row['reject']='head internal component'
            else:
                row['ring_min_coarse_gap_mm']=min(ring.min_distance_single(body,transform=pose@t) for a,pose in poses['S14'])
                if row['ring_min_coarse_gap_mm']<.3:row['reject']='S14 ring'
                elif min(other.min_distance_single(body,transform=pose@t) for a,pose in poses['S14'])<.3:row['reject']='S14 yaw servo'
                elif min(frame.min_distance_single(body,transform=pose@t) for a,pose in poses['S14']+poses['S13'])<.3:row['reject']='S14/S13 C frame'
                else:report['feasible'].append(row)
            report['trials'].append(row)
        print('Outward upright search',turn,'feasible',len(report['feasible']),flush=True)
    report['feasible'].sort(key=lambda v:sum(abs(x) for x in v['offset_mm']))
    (WORK/'edge-search.json').write_text(json.dumps(report,indent=2)+'\n')
    from collections import Counter
    print('Outward search results',dict(Counter(v.get('reject','feasible') for v in report['trials'])),flush=True)
    print('Best outward placements',json.dumps(report['feasible'][:5]),flush=True)


def edge_probe():
    """Check both side placements as solids, and inspect the existing holders."""
    from exact_motion_current import _overlap
    c=cache();facts=json.loads((WORK/'facts.json').read_text())
    trials=json.loads((WORK/'edge-search.json').read_text())['feasible']
    trials=sorted(trials,key=lambda v:(-v['offset_mm'][1],sum(abs(x) for x in v['offset_mm'])))[:6]
    head=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    ring=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    report=dict(source_sha256=SOURCE_SHA,ring_unchanged=True,checks=[],installed=False)
    for v in trials:
        row=dict(proposal=v,sides=[])
        for p,r in c['parts'].items():
            if r['label']!='sg90':continue
            sign=1 if '[13]' in p else -1
            f=next(x for x in facts if x['path']==p);s=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
            delta=[v['offset_mm'][0],sign*v['offset_mm'][1],v['offset_mm'][2]]
            body=shape(r).rotate(bd.Axis(s['origin'],s['direction']),sign*v['turn_deg']).moved(bd.Pos(*delta))
            holder=shape(next(q for q in c['parts'].values() if q['label']==('left_antenna_holder' if sign==1 else 'right_antenna_holder'))).moved(bd.Pos(*delta))
            common=body.intersect(head)
            entry=dict(path=p,delta_mm=delta,body_bounds=bounds(body),shell_overlap_mm3=_overlap(body,head),
                       overlap_bounds=bounds(bd.Compound(list(common))) if isinstance(common,list) and common else bounds(common) if common else None,
                       moved_holder_shell_overlap_mm3=_overlap(holder,head),body_holder_overlap_mm3=_overlap(body,holder),
                       ring_extreme_gap_mm=body.rotate(bd.Axis(*c['axes']['S14']),sign*30).distance_to(ring),
                       holder_bounds=bounds(holder))
            row['sides'].append(entry);print('Outward candidate exact',json.dumps(entry),flush=True)
        report['checks'].append(row)
        (WORK/'edge-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    # Original holder facts guide resizing its existing spline cup rather than
    # introducing a linkage or changing the antenna's fore-aft motion.
    f=next(x for x in facts if 'sg90[13]' in x['path']);s=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
    plane=bd.Plane(origin=s['origin'],z_dir=s['direction'])
    a=shape(next(r for r in c['parts'].values() if r['label']=='left_antenna_holder')).moved(plane.location.inverse())
    print('Holder local bounds',bounds(a),flush=True)
    for face in a.faces():
        if face.geom_type==bd.GeomType.CYLINDER:
            print('Holder local cylinder',face.radius,bounds(face),list(face.axis_of_rotation.position),list(face.axis_of_rotation.direction),flush=True)


def edge_build():
    """Modify existing ribs/windows for a fixed-height-ring side-edge layout."""
    import numpy as np
    import refit_upper_servo_interfaces_local as refit
    from cadgen.validity import check_occurrence_shape
    from exact_motion_current import _overlap
    c=cache();facts=json.loads((WORK/'facts.json').read_text());modified={};notes=[]
    shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    for p,r in c['parts'].items():
        if r['label']!='sg90':continue
        sign=1 if '[13]' in p else -1
        delta=(EDGE_OFFSET_MM[0],sign*EDGE_OFFSET_MM[1],EDGE_OFFSET_MM[2])
        f=next(x for x in facts if x['path']==p);shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
        holes=[x for x in f['cylinders'] if x['radius']==1]
        d=bd.Vector(shaft['direction']);e=bd.Vector(np.subtract(holes[1]['origin'],holes[0]['origin'])).normalized()
        plane=bd.Plane(origin=shaft['origin'],x_dir=e,z_dir=d)
        turn=sign*EDGE_CLOCK_DEG
        body=shape(r).rotate(bd.Axis(shaft['origin'],shaft['direction']),turn).moved(bd.Pos(*delta));modified[p]=body
        roof=bd.Plane(origin=(-11.05044013810311,37.3795063893792,52.01500213706816),
                      z_dir=(-.14640927609171459,.9464755095294123,-.2876602748648824))
        cavity=(roof*bd.Box(400,400,200,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MAX))).moved(bd.Pos(*(-.02*roof.z_dir)))
        tools=[];ribs=[]
        for low,high,hole in ((-10.625,-5.925,-8.075),(17.175,21.975,19.425)):
            region=refit._world_box(low-.002,high+.002,-6.652,30,7.519,11.772)
            tools.append(plane*refit._shape(region.intersect(cavity)))
            rib=refit._world_box(low,high,-6.65,8,7.49,11.77)
            bore=refit._rod(bd.Vector(hole,0,0),bd.Vector(0,0,1),7.06,11.78,.75)
            ribs.append((plane*refit._cut(rib,[bore]).rotate(bd.Axis.Z,turn)).moved(bd.Pos(*delta)))
        clearance=body.solids()[0].offset_3d([], .2)
        if not clearance.solids():clearance=bd.Solid(clearance)
        assert clearance.is_valid and len(clearance.solids())==1
        # Restore the original 3-mm sidewall in its original plane at the
        # obsolete opening, then cut only the relocated upright aperture.
        # This regenerates the wall thickness, rather than stacking a plate.
        original_skin=plane*refit._world_box(-5.945,8.595,-5.945,5.945,4.52,7.52)
        shell=refit._cut(refit._fuse(refit._cut(shell,tools),original_skin),[clearance])
        shell=refit._fuse(shell,*ribs)
        # Reopen these original pilots after fusing the moved ribs into skin.
        bores=[]
        for x in (-8.075,19.425):
            bore=refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),9.25,11.78,.75)
            bores.append((plane*bore.rotate(bd.Axis.Z,turn)).moved(bd.Pos(*delta)))
            for low,high,radius in ((10.27,14.2701,.75),(14.2701,40,1.5)):
                tool=refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),low,high,radius)
                bores.append((plane*tool.rotate(bd.Axis.Z,turn)).moved(bd.Pos(*delta)))
        shell=refit._cut(shell,bores)
        for holder_label in (('left_antenna_holder',) if sign==1 else ('right_antenna_holder',)):
            q=next(q for q,v in c['parts'].items() if v['label']==holder_label)
            modified[q]=shape(c['parts'][q]).moved(bd.Pos(*delta))
        q=next(q for q,v in c['parts'].items() if v['label']=='antenna' and (v['bounds'][0][1]>.1 if sign==1 else v['bounds'][1][1]<.1))
        modified[q]=shape(c['parts'][q]).moved(bd.Pos(*delta))
        notes.append(dict(part=p,turn_deg=turn,delta_mm=list(delta),body_shell_overlap_mm3=_overlap(body,shell),
                          shell_solids=[s.volume for s in shell.solids()],original_mount_ribs=2,window_clearance_mm=.2))
        print('Side-edge integrated seat',json.dumps(notes[-1]),flush=True)
    head=next(p for p,r in c['parts'].items() if r['label']=='head');modified[head]=shell
    files={}
    for p,a in modified.items():
        file=WORK/f'edge-{Path(c["parts"][p]["file"]).stem}.brep';bd.export_brep(a,file);files[p]=str(file)
    validity={p:check_occurrence_shape(a.wrapped) for p,a in modified.items()}
    report=dict(source_sha256=SOURCE_SHA,ring_unchanged=True,ring_drop_mm=0.,modified=files,seat_changes=notes,
                validity=validity,installed=False,antenna_motion='Original side-facing axis directions and fore-aft movement retained; pivots translated with original holders and antennas',
                stage='actual side-edge seat geometry built; interfaces and motion not yet verified')
    (WORK/'edge-build.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Side-edge build validity',json.dumps(validity),flush=True)


def edge_finish():
    """Close only the obsolete apertures and retain one integrated head."""
    import numpy as np
    import refit_upper_servo_interfaces_local as refit
    from cadgen.validity import check_occurrence_shape
    c=cache();report=json.loads((WORK/'edge-build.json').read_text());facts=json.loads((WORK/'facts.json').read_text())
    hp=next(p for p,r in c['parts'].items() if r['label']=='head')
    head=detached(bd.import_brep(report['modified'][hp]))
    for seat in report['seat_changes']:
        p=seat['part'];f=next(x for x in facts if x['path']==p)
        s=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001);holes=[x for x in f['cylinders'] if x['radius']==1]
        plane=bd.Plane(origin=s['origin'],x_dir=bd.Vector(np.subtract(holes[1]['origin'],holes[0]['origin'])),z_dir=s['direction'])
        patch=plane*refit._world_box(-5.945,8.595,-5.945,5.945,4.52,7.52)
        body=detached(bd.import_brep(report['modified'][p]))
        clearance=body.solids()[0].offset_3d([], .2)
        if not clearance.solids():clearance=bd.Solid(clearance)
        head=refit._fuse(head,refit._cut(patch,[clearance]))
        # Keep the original bearing surfaces at local Z=11.77. A clearance
        # cut after the rib fusion would mistakenly remove their top 0.2 mm.
        ribs=[]
        for low,high,x in ((-10.625,-5.925,-8.075),(17.175,21.975,19.425)):
            rib=refit._world_box(low,high,-6.65,8,7.49,11.77)
            rib=refit._cut(rib,[refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),7.06,11.78,.75)])
            ribs.append((plane*rib.rotate(bd.Axis.Z,seat['turn_deg'])).moved(bd.Pos(*seat['delta_mm'])))
        head=refit._fuse(head,*ribs)
        bores=[(plane*refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),9.25,11.78,.75).rotate(bd.Axis.Z,seat['turn_deg'])).moved(bd.Pos(*seat['delta_mm'])) for x in (-8.075,19.425)]
        for x in (-8.075,19.425):
            for low,high,radius in ((10.27,14.2701,.75),(14.2701,40,1.5)):
                tool=refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),low,high,radius)
                bores.append((plane*tool.rotate(bd.Axis.Z,seat['turn_deg'])).moved(bd.Pos(*seat['delta_mm'])))
        head=refit._cut(head,bores)
    assert len(head.solids())==1
    bd.export_brep(head,report['modified'][hp]);report['validity'][hp]=check_occurrence_shape(head.wrapped)
    report['obsolete_apertures_restored']=True
    (WORK/'edge-build.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Obsolete windows restored; head validity',json.dumps(report['validity'][hp]),flush=True)


def edge_normalize_head():
    """Canonicalize two tiny cylinder trims before the strict STEP comparison.

    OCCT changes UV parameterization at STEP import; two 0.025-mm2 trim faces
    keep area/bounds but differ in default numerical centroid integration.
    Do not weaken the export tolerances: use one single-part STEP round-trip
    in the generator, then verify that actual geometry again.
    """
    import tempfile
    from cadgen import read_step
    c=cache();b=json.loads((WORK/'edge-build.json').read_text())
    hp=next(p for p,r in c['parts'].items() if r['label']=='head')
    head=detached(bd.import_brep(b['modified'][hp]))
    with tempfile.TemporaryDirectory(dir=WORK) as directory:
        step=Path(directory)/'head.step';bd.export_step(head,step)
        head=detached(read_step(step))
    assert len(head.solids())==1 and head.volume>0
    bd.export_brep(head,b['modified'][hp]);b['head_step_trim_normalized']=True
    b['normalized_head_sha256']=hashlib.sha256(Path(b['modified'][hp]).read_bytes()).hexdigest()
    (WORK/'edge-build.json').write_text(json.dumps(b,indent=2)+'\n')
    print('Canonical head trim written; local re-verification required',b['normalized_head_sha256'],flush=True)


def edge_verify():
    """Only moved SG90s, antennas, original seats and adjacent interfaces."""
    import numpy as np
    import trimesh
    import refit_upper_servo_interfaces_local as refit
    from repair_minor_motion_clearance import _mesh as bounded_mesh
    from exact_motion_current import _overlap
    c=cache();parts=c['parts'];build=json.loads((WORK/'edge-build.json').read_text());modified=build['modified'];meshes={}
    def geometry(p):return detached(bd.import_brep(modified[p])) if p in modified else shape(parts[p])
    def one(label):return next(p for p,r in parts.items() if r['label']==label)
    ring=one('head_yaw_to_roll');frame=one('head_pitch_to_yaw');head=one('head')
    head_only='head_only' in sys.argv[2:]
    mounts_only='mounts_only' in sys.argv[2:]
    resume=head_only or mounts_only or 'static_only' in sys.argv[2:]
    report=json.loads((WORK/'edge-verify.json').read_text()) if resume else dict(source_sha256=SOURCE_SHA,ring_unchanged=True,installed=False,motion=[],static=[],mounts=[],completed=False)
    report['completed']=False
    if resume:
        if head_only:report['motion']=[r for r in report['motion'] if r['moving']!=head]
        if head_only:report['static']=[r for r in report['static'] if r['part']!=head and r['fixed']!=head]
        elif not mounts_only:report['static']=[]
        report['mounts']=[]
    def save():
        report['modified_sha256']={p:hashlib.sha256(Path(f).read_bytes()).hexdigest() for p,f in modified.items()}
        (WORK/'edge-verify.json').write_text(json.dumps(report,indent=2)+'\n')
    def mesh(p):
        if p not in meshes:
            a=geometry(p)
            if p==head:
                box=geometry(ring).bounding_box()
                a=bd.Compound([f for f in a.faces() if f.bounding_box().max.X>=box.min.X-.6 and f.bounding_box().min.X<=box.max.X+.6])
            meshes[p]=bounded_mesh(a,.01);del a;gc.collect()
        return meshes[p]
    def motion(joint,p,q,low,high):
        mm,fm=mesh(p),mesh(q);manager=trimesh.collision.CollisionManager();manager.add_object('fixed',fm)
        error=mm.metadata['maximum_deflection_mm']+fm.metadata['maximum_deflection_mm'];minimum=(float('inf'),0);hits=[];exact_fallback=[]
        if p==head:
            full=geometry(p);fixed_shape=geometry(q);box=fixed_shape.bounding_box()
            near_head=bd.Compound([f for f in full.faces() if f.bounding_box().max.X>=box.min.X-.6 and f.bounding_box().min.X<=box.max.X+.6])
        for angle in range(low,high+1):
            t=trimesh.transformations.rotation_matrix(np.deg2rad(angle),c['axes'][joint][1],c['axes'][joint][0])
            raw=manager.min_distance_single(mm,transform=t)
            if p==head:raw=min(.6,raw)
            gap=raw-error
            if p==head and gap<.199:
                # This original cylindrical interface has nominal 0.2-mm
                # clearance. Mesh error alone cannot certify that margin.
                a=near_head.rotate(bd.Axis(*c['axes'][joint]),angle)
                exact=min(.6,a.distance_to(fixed_shape));exact_fallback.append(dict(angle_deg=angle,gap_mm=exact))
                gap=exact-1e-7;del a;gc.collect()
            minimum=min(minimum,(gap,angle))
            if gap<.199:hits.append(dict(angle_deg=angle,gap_lower_bound_mm=gap))
        a=geometry(p).rotate(bd.Axis(*c['axes'][joint]),minimum[1]);b=geometry(q)
        row=dict(joint=joint,moving=p,fixed=q,step_deg=1,min_gap_lower_bound_mm=minimum[0],minimum_angle_deg=minimum[1],
                 mesh_error_mm=error,hits=hits,exact_fallback=exact_fallback,minimum_pose_overlap_mm3=_overlap(a,b),exact_minimum_pose_gap_mm=a.distance_to(b))
        report['motion'].append(row);save();print('Side-edge motion',json.dumps({k:v for k,v in row.items() if k!='hits'}),'failed samples',len(hits),flush=True)
    if 'static_only' not in sys.argv[2:] and not mounts_only:
        for p in ([] if head_only else modified):
            if p==head:continue
            motion('S14',p,ring,-30,30);motion('S14',p,frame,-30,30)
            if parts[p]['label']=='sg90':
                for q,r in parts.items():
                    if r['servo']=='S13':motion('S14',p,q,-30,30)
            motion('S13',p,frame,-160,160)
        motion('S14',head,ring,-30,30)
    shell=geometry(head)
    motors=[p for p in modified if parts[p]['label']=='sg90']
    for p in ([] if mounts_only else modified):
        if p==head:continue
        a=geometry(p)
        for q,r in parts.items():
            if head_only and q!=head:continue
            if q==p or (q not in modified and r['root']!='head_assembly <1>'):continue
            b=geometry(q);gap=a.distance_to(b)
            row=dict(part=p,fixed=q,gap_mm=gap)
            if gap<.6 or q==head:
                row['overlap_mm3']=_overlap(a,b);report['static'].append(row);save();print('Side-edge static',json.dumps(row),flush=True)
            del b
        del a;gc.collect()
    facts=json.loads((WORK/'facts.json').read_text())
    for seat in build['seat_changes']:
        p=seat['part'];f=next(x for x in facts if x['path']==p);s=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
        holes=[x for x in f['cylinders'] if x['radius']==1]
        plane=bd.Plane(origin=s['origin'],x_dir=bd.Vector(np.subtract(holes[1]['origin'],holes[0]['origin'])),z_dir=s['direction'])
        def tool(a):return (plane*a.rotate(bd.Axis.Z,seat['turn_deg'])).moved(bd.Pos(*seat['delta_mm']))
        body=geometry(p)
        local=shell.moved(bd.Pos(*[-x for x in seat['delta_mm']])).moved(plane.location.inverse()).rotate(bd.Axis.Z,-seat['turn_deg'])
        for x in (-8.075,19.425):
            pressure=tool(refit._cut(refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),11.57,11.77,2.1),[
                refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),11.56,11.78,1)]))
            missing=pressure.cut(shell)
            missing_volume=sum(s.volume for s in missing.solids()) if missing else 0.
            pilot=tool(refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),10.27,14.27,.70))
            passage=tool(refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),14.2701,40,1.5))
            cylinders=[]
            for face in local.faces():
                if face.geom_type!=bd.GeomType.CYLINDER:continue
                cylinder=refit.BRepAdaptor_Surface(face.wrapped).Cylinder();point=cylinder.Axis().Location()
                if abs(cylinder.Radius()-.75)<1e-5 and abs(point.X()-x)<1e-5 and abs(point.Y())<1e-5:cylinders.append(face)
            bottom=min(f.bounding_box().min.Z for f in cylinders)
            entry=dict(part=p,hole_local_x_mm=x,pressure_ring_missing_mm3=missing_volume,
                       pilot_blockage_mm3=_overlap(pilot,shell),tool_shell_overlap_mm3=_overlap(passage,shell),tool_body_overlap_mm3=_overlap(passage,body),
                       ear_mm=2.5,nominal_seat_entry_mm=1.5,bore_bottom_measured_z_mm=bottom,seat_depth_mm=11.77-bottom,nominal_bottom_reserve_mm=10.27-bottom,
                       actual_thread_depth_and_strength_accepted=False)
            report['mounts'].append(entry);print('Side-edge 4-mm geometric mount',json.dumps(entry),flush=True)
    report['completed']=True
    report['motion_passed']=bool(report['motion']) and all(not r['hits'] and r['minimum_pose_overlap_mm3']<1e-5 for r in report['motion'])
    report['static_passed']=all(r.get('overlap_mm3',0)<1e-5 for r in report['static'])
    report['mounts_passed']=all(max(r['pressure_ring_missing_mm3'],r['pilot_blockage_mm3'],r['tool_shell_overlap_mm3'],r['tool_body_overlap_mm3'])<1e-5 for r in report['mounts'])
    report['modified_sha256']={p:hashlib.sha256(Path(f).read_bytes()).hexdigest() for p,f in modified.items()}
    save();print('Side-edge local result',report['motion_passed'],report['static_passed'],report['mounts_passed'],flush=True)


def fixed_axis_search():
    """Scan body clocking with the original output line AND pivot fixed.

    Shell hits are recorded separately: the existing integrated seat may need
    resizing, while the outer shell and antenna linkage must remain intact.
    Mesh collision gates select candidates; they do not approve installation.
    """
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    from scan_s01_s14_motion_collisions import _mesh
    c=cache();facts=json.loads((WORK/'facts.json').read_text());meshes={}
    local={p:r for p,r in c['parts'].items() if r['root']=='head_assembly <1>' or
           r['root'] in ('antenna','left_antenna_holder','right_antenna_holder') or
           r['label'] in ('head_yaw_to_roll','head_pitch_to_yaw') or 'S13_HEAD_YAW' in p}
    for p,r in local.items():
        file=WORK/('mesh-'+Path(r['file']).stem+'.npz')
        if file.exists():
            d=np.load(file);m=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
        else:
            a=shape(r);m=_mesh(a,.15,.3);del a
            np.savez(file,vertices=m.vertices,faces=m.faces)
        meshes[p]=m
    shellpath=next(p for p,r in local.items() if r['label']=='head')
    ringpath=next(p for p,r in local.items() if r['label']=='head_yaw_to_roll')
    hull=ConvexHull(meshes[shellpath].vertices)
    shell=trimesh.collision.CollisionManager();shell.add_object(shellpath,meshes[shellpath])
    ring=trimesh.collision.CollisionManager();ring.add_object(ringpath,meshes[ringpath])
    fixed=trimesh.collision.CollisionManager();stationary=trimesh.collision.CollisionManager()
    for p,r in local.items():
        if r['label'] in ('sg90','head','head_yaw_to_roll'):continue
        manager=stationary if r['label']=='head_pitch_to_yaw' or 'S13_HEAD_YAW' in p else fixed
        manager.add_object(p,meshes[p])
    poses=[(a,trimesh.transformations.rotation_matrix(np.deg2rad(a),c['axes']['S14'][1],c['axes']['S14'][0]))
           for a in range(-30,31,5)]
    reverse=len(sys.argv)>2 and sys.argv[2]=='reverse'
    axial=len(sys.argv)>2 and sys.argv[2] in ('axial','reverse')
    report=dict(source_sha256=SOURCE_SHA,preserves_output_axis_line=True,
                shaft_tip_fixed=not axial,clocking_step_deg=3 if axial else 1,
                s14_step_deg=5,installed=False,servos={})
    for p,r in local.items():
        if r['label']!='sg90':continue
        f=next(x for x in facts if x['path']==p)
        shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
        holes=[x for x in f['cylinders'] if abs(x['radius']-1)<.001]
        ear=np.subtract(holes[1]['origin'],holes[0]['origin']);ear/=np.linalg.norm(ear)
        axis=np.array(shaft['direction']);origin=np.array(shaft['origin'])
        moving=trimesh.collision.CollisionManager();moving.add_object('motor',meshes[p])
        vertices=meshes[p].convex_hull.vertices;trials=[]
        import itertools
        flip=trimesh.transformations.rotation_matrix(np.pi,ear,origin) if reverse else np.eye(4)
        for shift,angle in itertools.product(range(0,41) if axial else (0,),range(0,360,3 if axial else 1)):
            t=trimesh.transformations.rotation_matrix(np.deg2rad(angle),axis,origin)
            t=t@flip
            t[:3,3]+=shift*axis
            assert np.allclose(trimesh.transform_points([origin],t)[0],origin+shift*axis)
            assert np.allclose(t[:3,:3]@axis,-axis if reverse else axis)
            points=trimesh.transform_points(vertices,t);moving.set_transform('motor',t)
            outside=float(np.max(points@hull.equations[:,:3].T+hull.equations[:,3]))
            shellhit=shell.in_collision_other(moving);statichit=fixed.in_collision_other(moving)
            ringhits=[];otherhits=[];ringgaps=[]
            for a,pose in poses:
                moving.set_transform('motor',pose@t)
                if ring.in_collision_other(moving):ringhits.append(a)
                else:ringgaps.append((ring.min_distance_other(moving),a))
                if stationary.in_collision_other(moving):otherhits.append(a)
            row=dict(turn_deg=angle,axial_shift_mm=shift,shaft_origin=origin.tolist(),shaft_direction=axis.tolist(),
                     transform=t.tolist(),ear_vertical_component=float(abs((t[:3,:3]@ear)[2])),
                     shell_hit=shellhit,static_hit=statichit,outside_hull_mm=outside,
                     s14_ring_hit_angles=ringhits,s14_other_hit_angles=otherhits,
                     ring_min_coarse_gap_mm=min(ringgaps)[0] if not ringhits else 0.)
            trials.append(row)
        feasible=[v for v in trials if not v['s14_ring_hit_angles'] and not v['s14_other_hit_angles']
                  and not v['static_hit'] and v['outside_hull_mm']<.01]
        feasible.sort(key=lambda v:(v['shell_hit'],v['axial_shift_mm'],-v['ear_vertical_component']))
        report['servos'][p]=dict(trials=trials,feasible=feasible,
            ring_clear_clockings=[v['turn_deg'] for v in trials if not v['s14_ring_hit_angles']],
            shell_clear_clockings=[v['turn_deg'] for v in trials if not v['shell_hit']])
        (WORK/('same-line-reverse-search.json' if reverse else 'same-line-axial-search.json' if axial else 'fixed-axis-search.json')).write_text(json.dumps(report,indent=2)+'\n')
        print('Fixed axis',p,'ring-clear',len(report['servos'][p]['ring_clear_clockings']),
              'shell-clear',len(report['servos'][p]['shell_clear_clockings']),'feasible',len(feasible),
              'best',[(v['turn_deg'],v['axial_shift_mm'],v['shell_hit'],v['ring_min_coarse_gap_mm']) for v in feasible[:6]],flush=True)


def fixed_axis_diagnose():
    """Locate exact interference and test an axisymmetric contained core."""
    import numpy as np
    from exact_motion_current import _overlap
    c=cache();facts=json.loads((WORK/'facts.json').read_text())
    ring=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    report=dict(source_sha256=SOURCE_SHA,installed=False,checks=[],invariant_cores=[])
    for p,r in c['parts'].items():
        if r['label']!='sg90':continue
        a=shape(r);f=next(x for x in facts if x['path']==p)
        shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
        axis=bd.Axis(shaft['origin'],shaft['direction']);side=1 if '[13]' in p else -1
        for angle in (0,side*86.1991237358816,side*266.1991237358816):
            b=a.rotate(axis,angle)
            for roll in (0,side*30):
                posed=b.rotate(bd.Axis(*c['axes']['S14']),roll)
                pieces=posed.intersect(ring)
                common=bd.Compound(children=list(pieces)) if isinstance(pieces,list) else pieces
                row=dict(path=p,clocking_deg=angle,s14_deg=roll,ring_overlap_mm3=sum(s.volume for s in common.solids()) if common else 0.,
                    intersection_bounds=bounds(common) if common else None,
                    shell_overlap_mm3=_overlap(b,shell) if roll==0 else None)
                report['checks'].append(row);print('Exact fixed axis',json.dumps(row),flush=True)
        for radius,low,high in ((5.5,8,29),(5.,10,28),(4.,12,26)):
            start=np.array(shaft['origin'])+low*np.array(shaft['direction'])
            core=bd.Plane(origin=tuple(start),z_dir=shaft['direction'])*bd.Cylinder(radius,high-low,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
            excess=core.cut(a);outside=sum(s.volume for s in excess.solids()) if excess else 0.
            hit=_overlap(core.rotate(bd.Axis(*c['axes']['S14']),side*30),ring)
            row=dict(path=p,radius_mm=radius,axial_span_mm=[low,high],outside_servo_mm3=outside,ring_overlap_at_s14_extreme_mm3=hit,
                     proves_all_clockings_collide=outside<.00001 and hit>.001)
            report['invariant_cores'].append(row);print('Invariant core',json.dumps(row),flush=True)
            if row['proves_all_clockings_collide']:
                bd.export_brep(core,WORK/f'fixed-axis-invariant-{side}.brep');break
        (WORK/'fixed-axis-diagnose.json').write_text(json.dumps(report,indent=2)+'\n')


def side_shell_exact():
    """BRep-check selected side-output candidates that only failed the shell gate."""
    from exact_motion_current import _overlap
    c=cache();p=next(p for p,r in c['parts'].items() if r['label']=='sg90')
    body=shape(c['parts'][p]);shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    candidates=[]
    for name in ('side-clock-search.json','side-rear-search.json'):
        s=json.loads((WORK/name).read_text())['servos'][p]['shell_only']
        keys=[lambda v:v['score'],lambda v:v['bounds'][1][2],
              lambda v:(-v['offset_mm'][0],v['score']),
              lambda v:(-v['offset_mm'][0],v['bounds'][1][2])]
        for key in keys:
            candidates.extend(sorted(s,key=key)[:12])
    seen=set();results=[]
    for v in candidates:
        key=(v['turn_deg'],tuple(v['offset_mm']))
        if key in seen:continue
        seen.add(key)
        a=body.rotate(bd.Axis(v['shaft_origin'],v['rotation_axis']),v['turn_deg']).moved(bd.Pos(*v['offset_mm']))
        pieces=a.intersect(shell);common=bd.Compound(children=list(pieces.solids())) if pieces else None
        row=dict(proposal=v,shell_overlap_mm3=sum(s.volume for s in common.solids()) if common else 0.,
                 overlap_bounds=bounds(common) if common else None,shell_gap_mm=a.distance_to(shell))
        results.append(row)
        print('Side-output shell',v['turn_deg'],v['offset_mm'],row['shell_overlap_mm3'],flush=True)
        report=dict(source_sha256=SOURCE_SHA,installed=False,checks=sorted(results,key=lambda v:v['shell_overlap_mm3']))
        (WORK/'side-shell-exact.json').write_text(json.dumps(report,indent=2)+'\n')


def side_failure_exact():
    """Verify that a tempting shell-clear pair is still mechanically blocked."""
    from exact_motion_current import _overlap
    c=cache();trial=json.loads((WORK/'side-pair.json').read_text())['trials'][0]
    v=trial['proposal'];motors=[(p,r) for p,r in c['parts'].items() if r['label']=='sg90']
    bodies=[];report=dict(source_sha256=SOURCE_SHA,proposal=v,installed=False,checks=[])
    for i,(p,r) in enumerate(motors):
        axis=v['rotation_axis'] if i==0 else [-v['rotation_axis'][0],v['rotation_axis'][1],-v['rotation_axis'][2]]
        origin=v['shaft_origin'] if i==0 else [v['shaft_origin'][0],.2-v['shaft_origin'][1],v['shaft_origin'][2]]
        delta=v['offset_mm'] if i==0 else [v['offset_mm'][0],-v['offset_mm'][1],v['offset_mm'][2]]
        a=shape(r).rotate(bd.Axis(origin,axis),v['turn_deg']).moved(bd.Pos(*delta))
        bodies.append(a);bd.export_brep(a,WORK/f'side-blocked-{i}.brep')
    pair=_overlap(*bodies);report['pair_overlap_mm3']=pair
    print('Body pair exact overlap',pair,flush=True)
    bridge=shape(next(r for r in c['parts'].values() if r['label']=='head_pitch_to_yaw'))
    shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    for i,a in enumerate(bodies):
        print('Body shell exact',i,_overlap(a,shell),a.distance_to(shell),flush=True)
        for angle in trial['s13_bridge_angles'][i]:
            posed=a.rotate(bd.Axis(*c['axes']['S13']),angle)
            row=dict(motor=i,s13_deg=angle,bridge_overlap_mm3=_overlap(posed,bridge))
            report['checks'].append(row);print('Side body yaw exact',json.dumps(row),flush=True)
    (WORK/'side-failure-exact.json').write_text(json.dumps(report,indent=2)+'\n')


def fixed_axis_review():
    """Review exact fixed-pivot interference; no proposed installation."""
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    import trimesh
    from scan_s01_s14_motion_collisions import _mesh
    c=cache();facts=json.loads((WORK/'facts.json').read_text())
    p=next(p for p,r in c['parts'].items() if r['label']=='sg90')
    shaft=next(x for f in facts if f['path']==p for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
    ring=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    body=shape(c['parts'][p]).rotate(bd.Axis(shaft['origin'],shaft['direction']),86.1991237358816)
    posed=body.rotate(bd.Axis(*c['axes']['S14']),30)
    core=detached(bd.import_brep(WORK/'fixed-axis-invariant-1.brep')).rotate(bd.Axis(*c['axes']['S14']),30)
    common=bd.Compound(children=list(posed.intersect(ring).solids()))
    geometry=[(ring,'#287caa',.18),(posed,'#e28a25',.32),(core,'#9a38af',.65),(common,'#e33838',1.)]
    meshes=[(_mesh(a,.08,.2),color,alpha) for a,color,alpha in geometry]
    fig=plt.figure(figsize=(12,6),layout='constrained');iso=fig.add_subplot(121,projection='3d');section=fig.add_subplot(122)
    for m,color,alpha in meshes:
        iso.add_collection3d(Poly3DCollection(m.triangles,facecolors=color,edgecolors='none',alpha=alpha,rasterized=True))
        segments=trimesh.intersections.mesh_plane(m,plane_origin=[0,28,0],plane_normal=[0,1,0])
        if len(segments):section.add_collection(LineCollection(segments[:,:,[0,2]],colors=color,linewidths=2))
    iso.set(xlim=(-50,-10),ylim=(15,48),zlim=(166,206),xlabel='X / mm',ylabel='Y / mm',zlabel='Z / mm',title='S14 +30 deg: body / original ring')
    iso.set_box_aspect((40,33,40));iso.view_init(elev=23,azim=-125)
    section.set(xlim=(-50,-10),ylim=(166,206),xlabel='X / mm',ylabel='Z / mm',title='Actual section Y=28 mm');section.set_aspect('equal');section.grid(alpha=.2)
    fig.suptitle('Fixed original antenna output axis / pivot: upright body still collides\nBlue: original ring; orange: SG90; purple: contained axisymmetric core; red: exact overlap\nBody/ring 231.281 mm3; contained core/ring 88.681 mm3. Formal STEP unchanged.',fontsize=11)
    output=TARGET.parent/'HEAD_UPRIGHT_REVIEW.png';fig.savefig(output,dpi=150);plt.close(fig);print(output,flush=True)


def seat_facts():
    """Measure the existing integrated SG90 seat in its mounting datum."""
    import numpy as np
    c=cache();facts=json.loads((WORK/'facts.json').read_text())
    f=next(r for r in facts if 'sg90[13]' in r['path']);shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
    holes=[x for x in f['cylinders'] if x['radius']==1]
    d=bd.Vector(shaft['direction']);e=bd.Vector(np.subtract(holes[1]['origin'],holes[0]['origin'])).normalized()
    plane=bd.Plane(origin=shaft['origin'],x_dir=e,z_dir=d)
    old=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    local=old.moved(plane.location.inverse())
    report={'hole_axes_local':[list(plane.to_local_coords(bd.Vector(x['origin']))) for x in holes], 'seat_faces':[]}
    for face in local.faces():
        box=face.bounding_box()
        if box.min.X>32 or box.max.X<-15 or box.min.Z>16 or box.max.Z<3 or box.min.Y>38 or box.max.Y<-12:continue
        row=dict(type=str(face.geom_type),area_mm2=face.area,bounds=bounds(face),centre=list(face.center()))
        if face.geom_type==bd.GeomType.PLANE:row['normal']=list(face.normal_at())
        if face.geom_type==bd.GeomType.CYLINDER:row.update(radius=face.radius,axis_origin=list(face.axis_of_rotation.position),axis_direction=list(face.axis_of_rotation.direction))
        report['seat_faces'].append(row)
    (WORK/'seat-facts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    bd.export_brep(local,WORK/'shell-seat-local.brep')


def build_original_axis():
    """Build actual original-seat and original-ring revisions, not a layout.

    Antennas, holder axes and output pivots remain exactly fixed. Rotate the
    SG90 bodies and their original integrated mounting ribs about that axis.
    Ring dimensions stay R50.75/R38.5 x 5.774; lower its plane 12 mm, extend
    the existing walls with a sloping front transition. Review before install.
    """
    import numpy as np
    import refit_upper_servo_interfaces_local as refit
    from cadgen.validity import check_occurrence_shape
    from exact_motion_current import _overlap
    c=cache();facts=json.loads((WORK/'facts.json').read_text());modified={};notes=[]
    shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    for p,r in c['parts'].items():
        if r['label']!='sg90':continue
        f=next(x for x in facts if x['path']==p);shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
        holes=[x for x in f['cylinders'] if x['radius']==1]
        d=bd.Vector(shaft['direction']);e=bd.Vector(np.subtract(holes[1]['origin'],holes[0]['origin'])).normalized()
        plane=bd.Plane(origin=shaft['origin'],x_dir=e,z_dir=d)
        turn=86.1991237358816*(1 if '[13]' in p else -1)
        axis=bd.Axis(shaft['origin'],shaft['direction'])
        body=shape(r).rotate(axis,turn);modified[p]=body
        local_body=body.moved(plane.location.inverse())
        # Remove the two original ribs only on the cavity side of the roof.
        # The old exterior roof skin is retained away from the servo opening.
        roof=bd.Plane(origin=(-11.05044013810311,37.3795063893792,52.01500213706816),
                      z_dir=(-.14640927609171459,.9464755095294123,-.2876602748648824))
        cavity=roof*bd.Box(400,400,200,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MAX))
        cavity=cavity.moved(bd.Pos(*(-.02*roof.z_dir)))
        ribs=[];cutters=[]
        for low,high,hole in ((-10.625,-5.925,-8.075),(17.175,21.975,19.425)):
            region=refit._world_box(low-.002,high+.002,-6.652,30,7.519,11.772)
            cutters.append(refit._shape(region.intersect(cavity)))
            rib=refit._world_box(low,high,-6.65,8,7.49,11.77)
            bore=refit._rod(bd.Vector(hole,0,0),bd.Vector(0,0,1),7.06,11.78,.75)
            ribs.append(refit._cut(rib,[bore]).rotate(bd.Axis.Z,turn))
        clearance=local_body.solids()[0].offset_3d([], .2)
        if not clearance.solids():clearance=bd.Solid(clearance)
        assert clearance.is_valid and len(clearance.solids())==1
        # Operate on the existing world shape. Round-tripping the entire shell
        # through a rotated datum and unifying every remote face damaged two
        # untouched front-hole trims. Only the new tools need transformations.
        shell=refit._cut(shell,[plane*tool for tool in cutters+[clearance]])
        shell=refit._fuse(shell,*[plane*rib for rib in ribs])
        overlap=_overlap(body,shell)
        notes.append(dict(part=p,turn_deg=turn,rib_count=2,rib_volumes=[a.volume for a in ribs],
                          body_shell_overlap_mm3=overlap,window_clearance_mm=.2))
        print('Rotated original seat',json.dumps(notes[-1]),flush=True)
    print('Both original integrated seats built',flush=True)
    shellpath=next(p for p,r in c['parts'].items() if r['label']=='head');modified[shellpath]=shell
    old=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    drop=12.;z0=171.87687762474;z1=177.65087762474
    region=refit._world_box(-100,100,-100,100,z0-.001,177.87787762474)
    annulus=refit._cut(refit._rod(bd.Vector(9,.1,0),bd.Vector(0,0,1),z0-drop,z1-drop,50.75),
                      [refit._rod(bd.Vector(9,.1,0),bd.Vector(0,0,1),z0-drop-.01,z1-drop+.01,38.5)])
    top=177.88787762474
    front_outline=[(47.8,z0-drop),(49.95,z0-drop),(49.95,top),
                   (39.5,top),(39.5,175.5),(44.5,z0),(47.8,z1-drop)]
    front=bd.extrude(bd.Plane.XZ*bd.Polygon(*front_outline,align=None),amount=25.12,dir=(0,1,0)).moved(bd.Pos(0,-12.46,0))
    rear=refit._world_box(-30.05,-27.05,-13.9,14.1,z0-drop,top)
    ring=refit._fuse(refit._cut(old,[region]),annulus,front,rear).clean()
    ringpath=next(p for p,r in c['parts'].items() if r['label']=='head_yaw_to_roll');modified[ringpath]=ring
    for p,a in modified.items():bd.export_brep(a,WORK/f'original-axis-{Path(c["parts"][p]["file"]).stem}.brep')
    validity={p:check_occurrence_shape(a.wrapped) for p,a in modified.items()}
    report=dict(source_sha256=SOURCE_SHA,stage='actual original-axis geometry built; validation pending',
                antenna_pivots_and_holders_unchanged=True,ring_drop_mm=drop,ring_dimensions_mm=[101.5,77,5.774],
                modified={p:str(WORK/f'original-axis-{Path(c["parts"][p]["file"]).stem}.brep') for p in modified},
                validity=validity,seat_changes=notes,installed=False)
    (WORK/'original-axis-build.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Actual revised geometry built',json.dumps(validity),flush=True)


def seat_debug():
    """Locate the actual seat/window interference in the original local frame."""
    import numpy as np
    import refit_upper_servo_interfaces_local as refit
    from cadgen.validity import check_occurrence_shape
    c=cache();facts=json.loads((WORK/'facts.json').read_text())
    p,r=next((p,r) for p,r in c['parts'].items() if r['label']=='sg90')
    f=next(x for x in facts if x['path']==p)
    shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
    holes=[x for x in f['cylinders'] if x['radius']==1]
    plane=bd.Plane(origin=shaft['origin'],x_dir=bd.Vector(np.subtract(holes[1]['origin'],holes[0]['origin'])),z_dir=shaft['direction'])
    shell=shape(next(r for r in c['parts'].values() if r['label']=='head')).moved(plane.location.inverse())
    body=shape(r).moved(plane.location.inverse()).rotate(bd.Axis.Z,86.1991237358816)
    print('Original validity',json.dumps(check_occurrence_shape(shell.wrapped)),flush=True)
    print('Rotated local body',bounds(body),flush=True)
    cuts=[refit._world_box(low-.002,high+.002,-6.652,30,7.519,11.772)
          for low,high in ((-10.625,-5.925),(17.175,21.975))]
    shell=refit._cut(shell,cuts)
    bd.export_brep(shell,WORK/'seat-debug-stripped.brep')
    print('Removed old ribs validity',json.dumps(check_occurrence_shape(shell.wrapped)),flush=True)
    window=refit._world_box(-5.925,8.575,-5.925,5.925,4.5,7.54).rotate(bd.Axis.Z,86.1991237358816)
    shell=refit._cut(shell,[window])
    print('Changed window validity',json.dumps(check_occurrence_shape(shell.wrapped)),flush=True)
    common=body.intersect(shell)
    if common and common.solids():
        common=refit._shape(common)
        bd.export_brep(common,WORK/'seat-interference.brep')
        print('Body/shell overlap',common.volume,bounds(common),flush=True)
        for a in common.solids():print('Overlap piece',a.volume,bounds(a),flush=True)
    bd.export_brep(shell,WORK/'seat-debug-shell.brep')
    bd.export_brep(body,WORK/'seat-debug-body.brep')
    reverse=body.rotate(bd.Axis.Z,180)
    common=reverse.intersect(shell)
    if common and common.solids():
        print('Opposite upright overlap',sum(a.volume for a in common.solids()),flush=True)
        for a in common.solids():print('Opposite overlap piece',a.volume,bounds(a),flush=True)
    else:print('Opposite upright overlap 0',flush=True)


def original_axis_verify():
    """Check the revised seats and nearby single-joint interfaces only."""
    import numpy as np
    import trimesh
    from repair_minor_motion_clearance import _mesh as bounded_mesh
    from exact_motion_current import _overlap
    c=cache();build=json.loads((WORK/'original-axis-build.json').read_text())
    assert build['source_sha256']==SOURCE_SHA
    for p,v in build['validity'].items():
        assert v['solidCount']==1 and v['volumes'][0]>0
        if c['parts'][p]['label']!='head':assert not v['reasons'],(p,v)
    parts=c['parts'];modified=build['modified'];meshes={}
    def geometry(p):
        return detached(bd.import_brep(modified[p])) if p in modified else shape(parts[p])
    def one(label):return next(p for p,r in parts.items() if r['label']==label)
    def mesh(p):
        if p not in meshes:
            a=geometry(p)
            if parts[p]['label']=='head':
                box=geometry(one('head_yaw_to_roll')).bounding_box()
                a=bd.Compound([f for f in a.faces() if f.bounding_box().max.X>=box.min.X-.6 and f.bounding_box().min.X<=box.max.X+.6])
            meshes[p]=bounded_mesh(a,.01);del a;gc.collect()
        return meshes[p]
    ring=one('head_yaw_to_roll');frame=one('head_pitch_to_yaw')
    motors=[p for p,r in parts.items() if r['label']=='sg90']
    head_only='head_only' in sys.argv[2:]
    report=json.loads((WORK/'original-axis-verify.json').read_text()) if head_only else dict(source_sha256=SOURCE_SHA,modified=modified,motion=[],static=[],
                geometry_validity=build['validity'],
                original_head_cap_warning='Two unchanged original B-spline cap faces also report selfIntersecting; no full-head validity pass claimed.',
                function='original output lines, shaft pivots, antenna shapes, holder shapes and antenna trajectories retained',
                installed=False,assembly_passed=False)
    report['geometry_validity']=build['validity']
    if head_only:
        report['motion']=[r for r in report['motion'] if r['moving']!=one('head')]
    def save(): (WORK/'original-axis-verify.json').write_text(json.dumps(report,indent=2)+'\n')
    def motion(joint,moving,fixed,low,high):
        mm=mesh(moving);fm=mesh(fixed)
        manager=trimesh.collision.CollisionManager();manager.add_object('fixed',fm)
        error=mm.metadata['maximum_deflection_mm']+fm.metadata['maximum_deflection_mm']
        minimum=(float('inf'),0);hits=[]
        for angle in range(low,high+1):
            t=trimesh.transformations.rotation_matrix(np.deg2rad(angle),c['axes'][joint][1],c['axes'][joint][0])
            raw=manager.min_distance_single(mm,transform=t)
            if parts[moving]['label']=='head':raw=min(.6,raw)
            gap=raw-error
            minimum=min(minimum,(gap,angle))
            if gap<.199:hits.append(dict(angle_deg=angle,gap_lower_bound_mm=gap))
        a=geometry(moving).rotate(bd.Axis(*c['axes'][joint]),minimum[1]);b=geometry(fixed)
        overlap=_overlap(a,b)
        row=dict(joint=joint,moving=moving,fixed=fixed,step_deg=1,
                 min_gap_lower_bound_mm=minimum[0],minimum_angle_deg=minimum[1],
                 mesh_error_mm=error,hits=hits,minimum_pose_overlap_mm3=overlap,
                 exact_minimum_pose_gap_mm=a.distance_to(b))
        report['motion'].append(row);save()
        print('Revised local motion',json.dumps({k:v for k,v in row.items() if k!='hits'}),'failed samples',len(hits),flush=True)
    if head_only:
        motion('S14',one('head'),ring,-30,30)
    for p in ([] if head_only else motors):
        motion('S14',p,ring,-30,30)
        motion('S14',p,frame,-30,30)
        for q,r in parts.items():
            if r['servo']=='S13':motion('S14',p,q,-30,30)
        motion('S13',p,frame,-160,160)
    for label in (() if head_only else ('speaker_stand','speaker_interface','speaker_bottom','speaker_top','head_bot_sheet','head')):
        motion('S14',one(label),ring,-30,30)
        if label not in ('head','head_bot_sheet'):
            motion('S14',one(label),frame,-30,30)
            motion('S13',one(label),frame,-160,160)
    for p,r in parts.items():
        if not head_only and r['servo']=='S14' and ('body' in r['label'] or 'passive_horn' in r['label']):
            motion('S14',p,ring,-30,30)
    for label in (() if head_only else ('neck_left_sheet','neck_right_sheet')):
        motion('S12',ring,one(label),-45,45)
    for p,r in parts.items():
        if not head_only and r['servo']=='S12' and 'body' in r['label']:motion('S12',ring,p,-45,45)
    for label in (() if head_only else ('body_middle_top','body_front')):
        motion('S11',ring,one(label),-20,39)
    if not head_only:motion('S13',ring,frame,-160,160)
    # The integrated head seat shares the servo's bearing planes; treat that
    # intentional contact separately from running clearance.
    head=geometry(one('head'))
    for p in motors:
        a=geometry(p)
        report['static'].append(dict(part=p,fixed=one('head'),overlap_mm3=_overlap(a,head),
                                     gap_mm=a.distance_to(head),contact='original mounting ears on rotated integrated ribs'))
        for q,r in parts.items():
            if r['root']=='head_assembly <1>' and q not in motors and r['label']!='head':
                b=geometry(q);gap=a.distance_to(b)
                if gap<.6:
                    row=dict(part=p,fixed=q,overlap_mm3=_overlap(a,b),gap_mm=gap)
                    report['static'].append(row);print('Near motor static',json.dumps(row),flush=True)
                del b
    report['completed']=True
    report['local_motion_passed']=all(not r['hits'] and r['minimum_pose_overlap_mm3']<1e-5 for r in report['motion'])
    report['s14_passed']=all(not r['hits'] and r['minimum_pose_overlap_mm3']<1e-5 for r in report['motion'] if r['joint']=='S14')
    report['s13_ring_mechanical_90_passed']=all(abs(x['angle_deg'])>90 for r in report['motion'] if r['joint']=='S13' and r['moving']==ring for x in r['hits'])
    report['modified_sha256']={p:hashlib.sha256(Path(f).read_bytes()).hexdigest() for p,f in modified.items()}
    report['final_head_sha256']=report['modified_sha256'][one('head')]
    save();print('Local verification completed',report['local_motion_passed'],flush=True)


def relocate_original_floor_fixings(head, floor):
    """Move the two existing floor seats and their matching holes together."""
    import refit_upper_servo_interfaces_local as refit
    facts=json.loads((WORK/'facts.json').read_text())
    posts=[];tools=[];records=[]
    for x,y in ((-6.938816346202,62.16023589767),(-6.938816346191,-61.96023589767)):
        side=1 if y>.1 else -1
        newx,newy=x-20,y-6*side
        zone=refit._rod(bd.Vector(x,y,0),bd.Vector(0,0,1),174.876878,196,4)
        post=refit._shape(head.intersect(zone))
        assert 880<post.volume<883
        f=next(q for q in facts if ('sg90[13]' if side==1 else 'sg90[22]') in q['path'])
        shaft=next(q for q in f['cylinders'] if abs(q['radius']-2.425)<.001)
        plane=bd.Plane(origin=shaft['origin'],z_dir=shaft['direction'])
        # Retain the original exterior skin; remove only the seat inside it.
        cavity=plane*refit._world_box(-500,500,-500,500,7.53,200)
        tools.append(refit._shape(zone.intersect(cavity)))
        posts.append(post.moved(bd.Pos(-20,-6*side,0)))
        fill=refit._rod(bd.Vector(x,y,0),bd.Vector(0,0,1),171.87687762474,174.87687762474,3.251)
        floor=refit._fuse(floor,fill)
        floor=refit._cut(floor,[
            refit._rod(bd.Vector(newx,newy,0),bd.Vector(0,0,1),171.8,175,1.6),
            refit._rod(bd.Vector(newx,newy,0),bd.Vector(0,0,1),171.8,172.876878,3.25)])
        records.append(dict(original=[x,y],revised=[newx,newy],translation_mm=[-20,-6*side,0],
                            existing_post_volume_mm3=post.volume))
    with bd.SkipClean():head=refit._fuse(refit._cut(head,tools),*posts)
    solids=sorted(head.solids(),key=lambda s:s.volume,reverse=True)
    # OCCT leaves two tolerance-scale chips at the old seats, not detached
    # fastening seats. Both moved seats already belong to the main shell.
    assert all(0<s.volume<1e-5 for s in solids[1:]),[s.volume for s in solids]
    removed=sum(s.volume for s in solids[1:])
    return solids[0],floor.clean(),records,removed


def finish_original_mount_bores(head, ring):
    """Keep the original lower seats and open pilots after the final fusions."""
    import numpy as np
    import refit_upper_servo_interfaces_local as refit
    old=shape(next(r for r in cache()['parts'].values() if r['label']=='head_yaw_to_roll'))
    lower=refit._shape(old.intersect(refit._world_box(24,44,-13.91,14.11,171.87687762474,177.87687762474)))
    ring=refit._fuse(ring,lower)
    cutters=[]
    for y in (-10.15,10.35):
        origin=bd.Vector(41.75,y,0)
        cutters.extend([refit._rod(origin,bd.Vector(0,0,1),171.87,174.87687762474,2),
                        refit._rod(origin,bd.Vector(0,0,1),174.87687762474,177.9,1.25)])
    ring=refit._cut(ring,cutters).clean()
    facts=json.loads((WORK/'facts.json').read_text());cutters=[]
    for f in facts:
        if '/sg90[' not in f['path']:continue
        shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
        holes=[x for x in f['cylinders'] if x['radius']==1]
        plane=bd.Plane(origin=shaft['origin'],x_dir=bd.Vector(np.subtract(holes[1]['origin'],holes[0]['origin'])),z_dir=shaft['direction'])
        turn=86.1991237358816*(1 if '[13]' in f['path'] else -1)
        for x in (-8.075,19.425):
            bore=refit._rod(bd.Vector(x,0,0),bd.Vector(0,0,1),9.25,11.78,.75)
            cutters.append(plane*bore.rotate(bd.Axis.Z,turn))
    with bd.SkipClean():head=refit._cut(head,cutters)
    return head,ring,lower


def original_axis_export():
    """Export the actual reviewed geometry, preserving every assembly leaf."""
    from cadgen import read_step
    from full_motion_current import _leaves, measured_axes
    from remove_s01_s06_upper_horns import leaves, compare
    import refit_upper_servo_interfaces_local as refit
    edge='edge' in sys.argv[2:];prefix='edge' if edge else 'original-axis'
    c=cache();build=json.loads((WORK/f'{prefix}-build.json').read_text())
    verification=json.loads((WORK/f'{prefix}-verify.json').read_text())
    if edge:
        assert verification['motion_passed'] and verification['static_passed'] and verification['mounts_passed']
        assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==verification['modified_sha256'][p] for p,f in build['modified'].items())
    else:
        assert verification['s14_passed']
        assert all(not r['hits'] for r in verification['motion'] if r['joint']!='S13')
        assert verification['final_head_sha256']==hashlib.sha256(Path(next(f for p,f in build['modified'].items() if c['parts'][p]['label']=='head')).read_bytes()).hexdigest()
    assert all(v['solidCount']==1 and v['volumes'][0]>0 and
               (not v['reasons'] or (c['parts'][p]['label'] in ('head','left_antenna_holder','right_antenna_holder') and set(v['reasons'])<= {'selfIntersecting'}))
               for p,v in build['validity'].items())
    robot=read_step(TARGET);before=leaves(robot);axes=measured_axes(robot)
    originals={r['path']:r['shape'] for r in _leaves(robot)}
    replacements={}
    for p,file in build['modified'].items():
        fresh=detached(bd.import_brep(file));old=originals[p]
        fresh.label,fresh.color=old.label,old.color
        replacements[id(old)]=fresh
    assert len(replacements)==(7 if edge else 9)
    def replace(node):
        if node.children:
            fresh=bd.Compound(children=[replace(child) for child in node.children],label=node.label)
            fresh.location=node.location
        else:
            fresh=detached(replacements.get(id(node),node));fresh.label=node.label
        fresh.color=node.color
        return fresh
    candidate=WORK/'review.step'
    bd.export_step(refit._encode_colours(replace(robot)),candidate)
    exported=read_step(candidate);after=leaves(exported)
    assert len(before)==len(after)==135
    for (oldpath,old),(newpath,new) in zip(before,after):
        assert oldpath==newpath
        expected=replacements.get(id(old),old)
        compare(expected,new)
        assert (old.color is None)==(new.color is None)
        if old.color is not None:assert max(abs(a-b) for a,b in zip(old.color,new.color))<1e-6
    newaxes=measured_axes(exported)
    for servo,(point,direction) in axes.items():
        assert max(abs(a-b) for a,b in zip(point,newaxes[servo][0]))<1e-5
        assert max(abs(a-b) for a,b in zip(direction,newaxes[servo][1]))<1e-5
    build.update(candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                 stage='actual assembly exported and reread; final review pending',
                 export_leaf_count=135,unchanged_leaf_count=135-len(replacements),main_axes_preserved=14,
                 modified_sha256={p:hashlib.sha256(Path(f).read_bytes()).hexdigest() for p,f in build['modified'].items()})
    verification['candidate_sha256']=build['candidate_sha256']
    verification['known_remaining_pairs']={'S13':2,'S14':3 if edge else 0}
    (WORK/f'{prefix}-verify.json').write_text(json.dumps(verification,indent=2)+'\n')
    (WORK/f'{prefix}-build.json').write_text(json.dumps(build,indent=2)+'\n')
    print('Actual assembly export verified',build['candidate_sha256'],135,'leaves',flush=True)



def original_axis_review():
    """Draw the exported STEP when the standard browser snapshot fails."""
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    from cadgen import read_step
    from full_motion_current import _leaves
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    speaker='speaker' in sys.argv[2:]
    robot=read_step((SPEAKER_WORK if speaker else WORK)/'review.step');rows=_leaves(robot);mesh=[]
    for r in rows:
     a=r['shape'];verts,faces=a.tessellate(.35,.3)
     if not verts or not faces:continue
     vertices=np.asarray([list(p) for p in verts]);tri=vertices[np.asarray(faces)]
     if speaker and r['label'].startswith('speaker'):
      print('SPEAKER_REVIEW_MESH_BOUNDS',r['label'],vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist(),flush=True)
     color=tuple(a.color)[:3] if a.color else (.65,.65,.65)
     mesh.append((r,tri,color));print('REVIEW_MESH',r['label'],len(faces),flush=True)
    fig=plt.figure(figsize=(14,10),facecolor='white');axes=[fig.add_subplot(2,2,i+1,projection='3d') for i in range(4)]
    for i,ax in enumerate(axes):
     internal=i>=2
     for r,tri,color in mesh:
      alpha=1
      if internal:
       if r['root'] not in ('head_assembly <1>','neck_yaw_assembly <1>','head_pitch_to_yaw','left_antenna_holder','right_antenna_holder','antenna'):continue
       if r['label']=='head_bot_sheet':continue
       if r['label']=='head':alpha=.07;color=(.7,.7,.7)
       if r['label']=='sg90':color=(.94,.43,.12)
       if r['label']=='head_yaw_to_roll':color=(.15,.42,.76)
       if r['label']=='speaker_stand':color=(.45,.27,.6)
       tri=tri[(tri[:,:,2].min(axis=1)>=145)&(tri[:,:,2].max(axis=1)<=242)]
      if len(tri)==0:continue
      normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
      light=np.array([.35,-.45,.82]);light/=np.linalg.norm(light);shading=.5+.5*np.abs(normal@light)
      artist=Poly3DCollection(tri,facecolors=np.asarray(color)[None,:]*shading[:,None],alpha=alpha,zsort='average');artist.set_edgecolor('none');ax.add_collection3d(artist)
     if internal:
      ax.set_xlim(-74,114);ax.set_ylim(-93,93);ax.set_zlim(145,243);ax.set_box_aspect((188,186,98));ax.set_title('Head internals - translucent shell / floor hidden',fontsize=10)
     else:
      allv=np.concatenate([tri.reshape(-1,3) for _,tri,_ in mesh]);lo=allv.min(axis=0);hi=allv.max(axis=0);centre=(lo+hi)/2;span=(hi-lo)*1.08
      ax.set_xlim(centre[0]-span[0]/2,centre[0]+span[0]/2);ax.set_ylim(centre[1]-span[1]/2,centre[1]+span[1]/2);ax.set_zlim(centre[2]-span[2]/2,centre[2]+span[2]/2);ax.set_box_aspect(span);ax.set_title('Actual modified STEP - exterior',fontsize=10)
     ax.view_init(elev=25 if internal else 18,azim=-145 if i%2==0 else 35);ax.set_axis_off()
    title=('Upright SG90s toward rear side edges / original antenna fore-aft motion retained\n'
           'Ring dimensions and height UNCHANGED. Motor/holder/antenna shift: -X10, outward Y1, -Z2 mm. Orange: SG90; blue: ring.') if 'edge' in sys.argv[2:] else ('Actual nine-part review candidate / original antenna output axes and pivots retained\nRing OD 101.5, ID 77, width 12.25, thickness 5.774 mm; ring -Z 12 mm. Orange: SG90; blue: ring; purple: original speaker stand.')
    if speaker:
        title=('Speaker / lower cradle shifted toward eyes: +X45, +Y3, Z unchanged (mm)\n'
               'Original roof fixing points retained; existing curved arm extended. Purple: speaker stand; blue: original ring.')
    fig.suptitle(title,fontsize=11)
    fig.tight_layout(pad=1.5)
    file=TARGET.parent/'HEAD_UPRIGHT_REVIEW.png';fig.savefig(file,dpi=170);print('ACTUAL_REVIEW_SAVED',file,hashlib.sha256(file.read_bytes()).hexdigest(),flush=True)
    if speaker:
        build=json.loads((SPEAKER_WORK/'build.json').read_text())
        build.update(snapshot_file=str(file),snapshot_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        (SPEAKER_WORK/'build.json').write_text(json.dumps(build,indent=2)+'\n')


def original_axis_finish():
    """Finish the same review candidate, retaining existing roof fasteners."""
    import refit_upper_servo_interfaces_local as refit
    from compact_head_local import clip,affine
    from cadgen.validity import check_occurrence_shape
    from exact_motion_current import _overlap
    c=cache();report=json.loads((WORK/'original-axis-build.json').read_text())
    def row(label):return next((p,r) for p,r in c['parts'].items() if r['label']==label)
    modified={}
    # Resize the original curved arm; keep the entire roof fixing region and
    # its four original holes, instead of moving its feet into the roll mount.
    p,r=row('speaker_stand');old=shape(r)
    roof=clip(old,2,223.5,250);fixed=clip(roof,1,-41.8,150);arm=clip(roof,1,-150,-41.8)
    low=clip(arm,0,-150,40);mid=clip(arm,0,40,41);high=clip(arm,0,41,150)
    matrix=[[51,0,0,-2000],[3,1,0,-120],[0,0,1,0],[0,0,0,1]]
    arm=refit._fuse(low,affine(mid,matrix),high.moved(bd.Pos(50,3,0)))
    lower=clip(old,2,170,223.5).moved(bd.Pos(50,3,0))
    stand=refit._fuse(fixed,arm,lower).clean()
    # Also remove the existing 2.644 mm3 roof-corner interference at the roll
    # fixing pad; the four original bearing footprints are outside this cut.
    corner=refit._world_box(48.3,50.6,-31.63,-28.05,225.7,229.1)
    modified[p]=refit._cut(stand,[corner]).clean()
    for label in ('speaker_interface','speaker_bottom','speaker_top'):
        p,r=row(label);modified[p]=shape(r).moved(bd.Pos(50,3,0))
    ip,_=row('speaker_interface');bp,_=row('speaker_bottom')
    before_trim=modified[ip].volume
    original_interface=modified[ip].clean()
    contact_material=refit._shape(original_interface.intersect(modified[bp]))
    modified[ip]=refit._cut(modified[ip],[modified[bp]]).clean()
    # Normalize the new cone trims to their STEP representation before the
    # assembly export; the original cut carries inconsistent UV trim areas.
    canonical=WORK/'speaker-interface-canonical.step'
    bd.export_step(modified[ip],canonical)
    modified[ip]=detached(bd.import_step(canonical))
    from repair_minor_motion_clearance import protect_mounts
    protect_mounts(original_interface,modified[ip],contact_material)
    report['speaker_interface_mating_trim_mm3']=before_trim-modified[ip].volume
    p,r=row('head_bot_sheet');floor=shape(r)
    opening=bd.extrude(bd.Plane(origin=(9,.1,171))*bd.Ellipse(64,60),amount=5)
    modified[p]=refit._cut(floor,[opening]).clean()
    p,r=row('head_yaw_to_roll');old=shape(r);z0=171.87687762474;z1=177.65087762474;drop=12.;top=177.88787762474
    removed=refit._world_box(-100,100,-100,100,z0-.001,177.87787762474)
    annulus=refit._cut(refit._rod(bd.Vector(9,.1,0),bd.Vector(0,0,1),z0-drop,z1-drop,50.75),
                      [refit._rod(bd.Vector(9,.1,0),bd.Vector(0,0,1),z0-drop-.01,z1-drop+.01,38.5)])
    outline=[(47.8,z0-drop),(49.95,z0-drop),(49.95,top),(39.5,top),(39.5,175.5),(44.5,z0),(47.8,z1-drop)]
    front=bd.extrude(bd.Plane.XZ*bd.Polygon(*outline,align=None),amount=25.12,dir=(0,1,0)).moved(bd.Pos(0,-12.46,0))
    rear=refit._world_box(-30.05,-27.05,-13.9,14.1,z0-drop,top)
    modified[p]=refit._fuse(refit._cut(old,[removed]),annulus,front,rear).clean()
    hp,_=row('head');fp,_=row('head_bot_sheet')
    head=detached(bd.import_brep(report['modified'][hp]))
    modified[hp],modified[fp],records,chips=relocate_original_floor_fixings(head,modified[fp])
    rp,_=row('head_yaw_to_roll')
    modified[hp],modified[rp],_=finish_original_mount_bores(modified[hp],modified[rp])
    report['floor_fixing_relocations']=records
    report['removed_boolean_chips_mm3']=chips
    for p,a in modified.items():
        file=WORK/f'original-axis-{Path(c["parts"][p]["file"]).stem}.brep'
        bd.export_brep(a,file);report['modified'][p]=str(file)
        report['validity'][p]=check_occurrence_shape(a.wrapped)
        print('Finished existing interface',p,json.dumps(report['validity'][p]),flush=True)
    head=detached(bd.import_brep(report['modified'][row('head')[0]]))
    report['speaker_translation_mm']=[50,3,0]
    report['speaker_roof_holes_unchanged']=True
    report['floor_opening_semiaxes_mm']=[64,60]
    report['stage']='actual nine-part review candidate built; final local verification pending'
    report['speaker_stand_head_overlap_mm3']=_overlap(modified[row('speaker_stand')[0]],head)
    (WORK/'original-axis-build.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Finished original local interfaces',report['speaker_stand_head_overlap_mm3'],flush=True)


def upright_fit():
    """Fit original SG90 bodies upright inside the unchanged head, mm.

    Preserve side-facing output axes and the antenna's original motion first.
    Search placements only; shell, ring, vendor bodies and formal STEP remain
    unchanged until the existing mounts and antenna connections also pass.
    Coarse meshes are a selection gate, never installation acceptance.
    """
    import itertools
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from scan_s01_s14_motion_collisions import _mesh
    mode=sys.argv[2] if len(sys.argv)>2 else ''
    clock=mode in ('clock','rear','refine','sidelevel')
    if len(sys.argv)>2 and sys.argv[2]=='shaft':
        raise SystemExit('Rejected by user: vertical output changes antenna function. Use side-facing clock search.')
    c=cache();facts=json.loads((WORK/'facts.json').read_text());meshes={}
    local={p:r for p,r in c['parts'].items() if r['root']=='head_assembly <1>' or
           r['label'] in ('head_yaw_to_roll','head_pitch_to_yaw') or 'S13_HEAD_YAW' in p}
    for p,r in local.items():
        file=WORK/('mesh-'+Path(r['file']).stem+'.npz')
        if file.exists():
            d=np.load(file);m=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
        else:
            a=shape(r);m=_mesh(a,.15,.3);del a
            np.savez(file,vertices=m.vertices,faces=m.faces)
        meshes[p]=m
        print('Mesh ready',p,len(m.faces),flush=True)
        gc.collect()
    shellpath=next(p for p,r in local.items() if r['label']=='head')
    hull=ConvexHull(meshes[shellpath].vertices)
    shell=trimesh.collision.CollisionManager();shell.add_object(shellpath,meshes[shellpath])
    fixed=trimesh.collision.CollisionManager()
    for p,r in local.items():
        if r['label'] in ('sg90','head','head_yaw_to_roll','head_pitch_to_yaw') or 'S13_HEAD_YAW' in p:continue
        fixed.add_object(p,meshes[p])
    stationary=trimesh.collision.CollisionManager()
    for p,r in local.items():
        if r['label'] in ('head_yaw_to_roll','head_pitch_to_yaw') or 'S13_HEAD_YAW' in p:
            stationary.add_object(p,meshes[p])
    poses=[trimesh.transformations.rotation_matrix(np.deg2rad(a),c['axes']['S14'][1],c['axes']['S14'][0])
           for a in range(-30,31,5)]
    report=dict(source_sha256=SOURCE_SHA,stage='body-only coarse search; mounting and antenna linkage not passed',servos={})
    for p,r in local.items():
        if r['label']!='sg90':continue
        if mode in ('refine','sidelevel') and '[22]' in p:continue
        vertical=len(sys.argv)>2 and sys.argv[2]=='shaft'
        if vertical and '[22]' in p:continue
        f=next(x for x in facts if x['path']==p)
        shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
        holes=[x for x in f['cylinders'] if abs(x['radius']-1)<.001]
        ear=np.subtract(holes[1]['origin'],holes[0]['origin']);ear/=np.linalg.norm(ear)
        direction=np.array(shaft['direction']);up=np.array((0.,0.,1.));up-=direction*up.dot(direction);up/=np.linalg.norm(up)
        signed=np.rad2deg(np.arctan2(direction.dot(np.cross(ear,up)),ear.dot(up)))
        sign=1 if '[13]' in p else -1
        own=meshes[p];vertices=own.convex_hull.vertices
        moving=trimesh.collision.CollisionManager();moving.add_object('motor',own)
        found=[];shell_only=[];count=0;rejected={};near=[]
        offsets=list(itertools.product(range(0,61,5),range(0,51,5),range(-15,36,5)))
        if mode=='rear':offsets=list(itertools.product(np.arange(-25,0.1,2.5),np.arange(0,25.1,2.5),np.arange(-10,15.1,2.5)))
        if mode=='refine':offsets=list(itertools.product(range(10,23),range(10,30),range(5,23)))
        if mode=='sidelevel':offsets=list(itertools.product(np.arange(0,50.1,2.5),np.arange(0,35.1,2.5),np.arange(0,30.1,2.5)))
        orientations=[]
        angles=[signed+d for d in range(-45,46,15)]+[signed+180+d for d in range(-45,46,15)] if clock else (signed,signed+180)
        if mode=='refine':angles=[signed+d for d in range(5,31,5)]+[signed+180+d for d in range(5,31,5)]
        for angle in angles:
            rotation=trimesh.transformations.rotation_matrix(np.deg2rad(angle),direction,shaft['origin'])
            orientations.append((angle,rotation,direction))
        if mode=='sidelevel':
            orientations=[]
            source=np.column_stack((ear,direction,np.cross(ear,direction)))
            for earsign,shaftsign in itertools.product((1,-1),(1,-1)):
                target_axis=np.array((0.,-sign*shaftsign,0.))
                target_ear=np.array((0.,0.,earsign))
                target=np.column_stack((target_ear,target_axis,np.cross(target_ear,target_axis)))
                vector=Rotation.from_matrix(target@source.T).as_rotvec()
                angle=float(np.rad2deg(np.linalg.norm(vector)));axis=vector/np.linalg.norm(vector)
                rotation=trimesh.transformations.rotation_matrix(np.deg2rad(angle),axis,shaft['origin'])
                orientations.append((angle,rotation,axis))
        if vertical:
            orientations=[]
            source=np.column_stack((ear,direction,np.cross(ear,direction)))
            for shaftsign,yaw in itertools.product((1,-1),(0,90,180,270)):
                target_axis=np.array((0.,0.,shaftsign))
                target_ear=np.array((np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw)),0.))
                target=np.column_stack((target_ear,target_axis,np.cross(target_ear,target_axis)))
                vector=Rotation.from_matrix(target@source.T).as_rotvec()
                angle=float(np.rad2deg(np.linalg.norm(vector)));axis=vector/np.linalg.norm(vector)
                rotation=trimesh.transformations.rotation_matrix(np.deg2rad(angle),axis,shaft['origin'])
                orientations.append((angle,rotation,axis))
        for angle,rotation,axis in orientations:
            for dx,inside,dz in offsets:
                delta=(dx,-sign*inside,dz);t=trimesh.transformations.translation_matrix(delta)@rotation
                points=trimesh.transform_points(vertices,t)
                if np.max(points@hull.equations[:,:3].T+hull.equations[:,3])>-.25:continue
                count+=1
                moving.set_transform('motor',t)
                if fixed.in_collision_other(moving):continue
                hit=False
                for pose in poses:
                    moving.set_transform('motor',pose@t)
                    if stationary.in_collision_other(moving):
                        hit=True
                        break
                if hit:
                    continue
                v=dict(turn_deg=float(angle),rotation_axis=list(axis),shaft_origin=shaft['origin'],
                       shaft_direction=shaft['direction'],offset_mm=list(delta),transform=t.tolist(),
                       shaft_direction_after=list(rotation[:3,:3]@direction),
                       score=abs(dx)+abs(inside)+abs(dz),bounds=[points.min(axis=0).tolist(),points.max(axis=0).tolist()])
                moving.set_transform('motor',t)
                if shell.in_collision_other(moving):shell_only.append(v)
                else:found.append(v)
            print('Orientation checked',p,angle,'inside',count,'clear',len(found),'shell-only',len(shell_only),flush=True)
            report['servos'][p]=dict(inside_hull_trials=count,feasible=sorted(found,key=lambda v:v['score']),
                                    shell_only=sorted(shell_only,key=lambda v:v['score']),search_complete=False)
            (WORK/('sidelevel-search.json' if mode=='sidelevel' else 'side-refine-search.json' if mode=='refine' else 'side-rear-search.json' if mode=='rear' else 'side-clock-search.json' if clock else 'upright-fit-shaft.json' if vertical else 'upright-fit.json')).write_text(json.dumps(report,indent=2)+'\n')
        report['servos'][p]=dict(inside_hull_trials=count,feasible=sorted(found,key=lambda v:v['score']),
                                shell_only=sorted(shell_only,key=lambda v:v['score']),rejected_by=rejected,near=near)
        print('Fit search',p,'inside',count,'body clear',len(found),'shell-only',len(shell_only),
              'best',report['servos'][p]['feasible'][:3],flush=True)
        (WORK/('sidelevel-search.json' if mode=='sidelevel' else 'side-refine-search.json' if mode=='refine' else 'side-rear-search.json' if mode=='rear' else 'side-clock-search.json' if clock else 'upright-fit-shaft.json' if vertical else 'upright-fit.json')).write_text(json.dumps(report,indent=2)+'\n')


def upright_pair():
    """Screen both existing motors and the directly adjacent S13 C frame."""
    import numpy as np
    import trimesh
    side=len(sys.argv)>2 and sys.argv[2] in ('side','fine','level','level_fine','level_margin')
    fine=len(sys.argv)>2 and sys.argv[2] in ('fine','level_fine','level_margin')
    level_fine=len(sys.argv)>2 and sys.argv[2] in ('level_fine','level_margin')
    margin=.215 if len(sys.argv)>2 and sys.argv[2]=='level_margin' else .35
    level=len(sys.argv)>2 and sys.argv[2]=='level'
    if not side:raise SystemExit('Vertical-output candidates are rejected; use upright_pair side.')
    c=cache();search=json.loads((WORK/('sidelevel-search.json' if level else 'side-refine-search.json')).read_text())
    meshes={}
    for p,r in c['parts'].items():
        f=WORK/('mesh-'+Path(r['file']).stem+'.npz')
        if f.exists():
            d=np.load(f);meshes[p]=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
    motors=[p for p,r in c['parts'].items() if r['label']=='sg90']
    fixed=trimesh.collision.CollisionManager()
    for p,m in meshes.items():
        if c['parts'][p]['root']=='head_assembly <1>' and p not in motors:fixed.add_object(p,m)
    bridge=trimesh.collision.CollisionManager()
    cp=next(p for p,r in c['parts'].items() if r['label']=='head_pitch_to_yaw')
    bridge.add_object(cp,meshes[cp])
    mirror=np.diag([1.,-1.,1.,1.]);mirror[1,3]=.2
    report=dict(source_sha256=SOURCE_SHA,stage='both-side and S13 coarse gate',trials=[],feasible=[])
    candidates=next(iter(search['servos'].values()))['feasible']
    if fine:
        import itertools
        base=json.loads((WORK/'side-pair.json').read_text())['feasible'][0]['proposal'] if level_fine else candidates[0]
        axis=base['shaft_direction'];origin=base['shaft_origin']
        candidates=[]
        grid=itertools.product([base['turn_deg']],np.arange(23,25.1,.25),np.arange(21,24.1,.25),np.arange(12,14.1,.25)) if level_fine else itertools.product(np.arange(81.1991237358816,109,2.5),np.arange(15,20.1,.5),np.arange(23,28.1,.5),np.arange(0,7.1,.5))
        for angle,dx,inside,dz in grid:
            t=trimesh.transformations.rotation_matrix(np.deg2rad(angle),base['rotation_axis'] if level_fine else axis,origin)
            delta=[float(dx),float(-inside),float(dz)];t[:3,3]+=delta
            candidates.append(dict(turn_deg=float(angle),rotation_axis=base['rotation_axis'] if level_fine else axis,shaft_origin=origin,shaft_direction=axis,
                shaft_direction_after=base['shaft_direction_after'] if level_fine else axis,offset_mm=delta,transform=t.tolist(),score=float(dx+inside+dz)))
        candidates.sort(key=lambda v:v['score'])
    rollfixed=trimesh.collision.CollisionManager()
    for p,m in meshes.items():
        r=c['parts'][p]
        if r['label'] in ('head_yaw_to_roll','head_pitch_to_yaw') or 'S13_HEAD_YAW' in p:rollfixed.add_object(p,m)
    rollposes=[trimesh.transformations.rotation_matrix(np.deg2rad(a),c['axes']['S14'][1],c['axes']['S14'][0]) for a in range(-30,31,5)]
    moving=[]
    for p in motors:
        m=trimesh.collision.CollisionManager();m.add_object('motor',meshes[p]);moving.append(m)
    for n,v in enumerate(candidates):
        transforms=[np.array(v['transform']),mirror@np.array(v['transform'])@mirror]
        if fine:
            for m,t in zip(moving,transforms):m.set_transform('motor',t)
            if moving[0].in_collision_other(moving[1]):continue
            if any(fixed.in_collision_other(m) for m in moving):continue
            if level_fine and any(fixed.min_distance_other(m)<margin for m in moving):continue
            rollhit=False
            for m,t in zip(moving,transforms):
                for pose in rollposes:
                    m.set_transform('motor',pose@t)
                    if rollfixed.in_collision_other(m):rollhit=True;break
                    if level_fine and rollfixed.min_distance_other(m)<margin:rollhit=True;break
                if rollhit:break
            if rollhit:continue
        static=[];yaw=[]
        for i,(p,m,t) in enumerate(zip(motors,moving,transforms)):
            m.set_transform('motor',t)
            static.append(fixed.in_collision_other(m))
            bad=[]
            for a in range(-160,161,5):
                pose=trimesh.transformations.rotation_matrix(np.deg2rad(a),c['axes']['S13'][1],c['axes']['S13'][0])
                m.set_transform('motor',pose@t)
                if bridge.in_collision_other(m) or level_fine and bridge.min_distance_other(m)<margin:bad.append(a)
            yaw.append(bad)
            m.set_transform('motor',t)
        pair_hit=moving[0].in_collision_other(moving[1])
        trial=dict(proposal=v,transforms=[t.tolist() for t in transforms],static_hits=static,s13_bridge_angles=yaw,pair_hit=pair_hit)
        report['trials'].append(trial)
        if not any(static) and not any(yaw) and not pair_hit:report['feasible'].append(trial)
        print('Paired gate',v['offset_mm'],v['shaft_direction_after'],'static',static,'yaw samples',[len(x) for x in yaw],flush=True)
        (WORK/'side-pair.json').write_text(json.dumps(report,indent=2)+'\n')
        if fine and len(report['feasible'])>=10:break
    print('Paired feasible',len(report['feasible']),flush=True)
    report.update(search_complete=True,candidate_count=len(candidates),fine_grid=fine)
    (WORK/('side-pair-fine.json' if fine else 'side-pair.json')).write_text(json.dumps(report,indent=2)+'\n')


def upright_fit_exact():
    """Exact static fit and bounded one-degree motion checks for a body pair."""
    import numpy as np
    import trimesh
    from cadgen.validity import check_occurrence_shape
    from exact_motion_current import _overlap
    from repair_minor_motion_clearance import _mesh as bounded_mesh
    side=len(sys.argv)>2 and sys.argv[2]=='side'
    if not side:raise SystemExit('Vertical-output candidates are rejected; use upright_fit_exact side INDEX.')
    c=cache();paired=json.loads((WORK/'side-pair.json').read_text())
    trial=paired['feasible'][int(sys.argv[3]) if len(sys.argv)>3 else 0]
    v=trial['proposal'];motors=[(p,r) for p,r in c['parts'].items() if r['label']=='sg90']
    bodies={};report=dict(source_sha256=SOURCE_SHA,proposal=v,static=[],motion=[],validity={},
                          body_pair_passed=False,installation_passed=False,installed=False)
    for i,(p,r) in enumerate(motors):
        axis=v['rotation_axis'] if i==0 else [-v['rotation_axis'][0],v['rotation_axis'][1],-v['rotation_axis'][2]]
        origin=v['shaft_origin'] if i==0 else [v['shaft_origin'][0],.2-v['shaft_origin'][1],v['shaft_origin'][2]]
        delta=v['offset_mm'] if i==0 else [v['offset_mm'][0],-v['offset_mm'][1],v['offset_mm'][2]]
        a=shape(r).rotate(bd.Axis(origin,axis),v['turn_deg']).moved(bd.Pos(*delta))
        bodies[p]=a;report['validity'][p]=check_occurrence_shape(a.wrapped)
        assert not report['validity'][p]['reasons'] and report['validity'][p]['solidCount']==1
        bd.export_brep(a,WORK/f'side-selected-{i}.brep')
    neighbours={p:r for p,r in c['parts'].items() if (r['root']=='head_assembly <1>' or r['root'] in
        ('antenna','left_antenna_holder','right_antenna_holder')) and r['label']!='sg90'}
    for p,a in bodies.items():
        ab=np.array(bounds(a))
        for q,r in neighbours.items():
            bb=np.array(r['bounds']);boxgap=float(np.linalg.norm(np.maximum(np.maximum(ab[0]-bb[1],bb[0]-ab[1]),0)))
            if boxgap>=.25:gap=boxgap;common=0.;method='AABB lower bound'
            else:
                b=shape(r);gap=a.distance_to(b);common=_overlap(a,b);method='BRep'
            result=dict(moving=p,fixed=q,gap_mm=gap,overlap_mm3=common,method=method)
            report['static'].append(result)
            print('Static body fit',q,gap,common,method,flush=True)
            assert gap>=.199 and common<.001,result
        (WORK/'side-fit-exact.json').write_text(json.dumps(report,indent=2)+'\n')
    mesh={p:bounded_mesh(a,.01) for p,a in bodies.items()}
    fixed={p:r for p,r in c['parts'].items() if r['label'] in ('head_pitch_to_yaw','head_yaw_to_roll') or 'S13_HEAD_YAW' in p}
    for joint in ('S13','S14'):
        for q,r in fixed.items():
            if joint=='S13' and r['label']!='head_pitch_to_yaw':continue
            b=shape(r);bm=bounded_mesh(b,.01);manager=trimesh.collision.CollisionManager();manager.add_object(q,bm)
            low,high=(-160,160) if joint=='S13' else (-30,30)
            for p,m in mesh.items():
                errors=m.metadata['maximum_deflection_mm']+bm.metadata['maximum_deflection_mm'];gaps=[]
                for angle in range(low,high+1):
                    t=trimesh.transformations.rotation_matrix(np.deg2rad(angle),c['axes'][joint][1],c['axes'][joint][0])
                    gaps.append((manager.min_distance_single(m,transform=t)-errors,angle))
                gap,angle=min(gaps);common=_overlap(bodies[p].rotate(bd.Axis(*c['axes'][joint]),angle),b)
                row=dict(joint=joint,moving=p,fixed=q,minimum_gap_lower_bound_mm=gap,angle_deg=angle,
                         mesh_error_mm=errors,overlap_at_minimum_mm3=common,range_deg=[low,high],step_deg=1)
                report['motion'].append(row);print('Motion body fit',json.dumps(row),flush=True)
                (WORK/'side-fit-exact.json').write_text(json.dumps(report,indent=2)+'\n')
                assert gap>=.199 and common<.001,row
    a,b=bodies.values();assert a.distance_to(b)>=.199
    report['body_pair_passed']=True
    report['unresolved']=['Original shell mounting seats and antenna holders are not redesigned',
                          'Side-facing output preserved; original holder dimensions and resulting antenna sweep require checks',
                          'Existing speaker and S13 collisions are not repaired by these body checks']
    (WORK/'side-fit-exact.json').write_text(json.dumps(report,indent=2)+'\n')


def upright_mount_facts():
    c=cache();report={}
    shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    motors=[(p,r) for p,r in c['parts'].items() if r['label']=='sg90']
    for i,(p,r) in enumerate(motors):
        old=shape(r);fresh=detached(bd.import_brep(WORK/f'upright-selected-{i}.brep'))
        cylinders=[]
        for name,a in [('old_motor',old),('new_motor',fresh),('head',shell)]:
            for f in a.faces():
                if f.geom_type!=bd.GeomType.CYLINDER:continue
                axis=f.axis_of_rotation
                if name=='head':
                    if abs(abs(axis.direction.dot(bd.Vector(.263083740059,-.935408853542 if i==0 else .935408853542,-.236214356955)))-1)>.001:continue
                    if f.distance_to(old)>3:continue
                cylinders.append(dict(part=name,radius=f.radius,origin=list(axis.position),direction=list(axis.direction),
                                      face_bounds=bounds(f),centre=list(f.center())))
        report[p]=cylinders;print(p,json.dumps(cylinders),flush=True)
    (WORK/'upright-mount-facts.json').write_text(json.dumps(report,indent=2)+'\n')


def upright_fit_review():
    """Show measured body positions, explicitly excluding installation approval."""
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    c=cache();verified=json.loads((WORK/'upright-fit-exact.json').read_text())
    assert verified['body_pair_passed']
    def mesh(p):
        r=c['parts'][p];d=np.load(WORK/('mesh-'+Path(r['file']).stem+'.npz'))
        return trimesh.Trimesh(d['vertices'],d['faces'],process=False)
    sp=next(p for p,r in c['parts'].items() if r['label']=='head');shell=mesh(sp)
    motors=[p for p,r in c['parts'].items() if r['label']=='sg90']
    proposed=[]
    for i,p in enumerate(motors):
        m=mesh(p);v=verified['proposal'];t=np.array(v['transform'])
        if i:
            mirror=np.diag([1.,-1.,1.,1.]);mirror[1,3]=.2;t=mirror@t@mirror
        m.apply_transform(t);proposed.append(m)
    ring=mesh(next(p for p,r in c['parts'].items() if r['label']=='head_yaw_to_roll'))
    bridge=mesh(next(p for p,r in c['parts'].items() if r['label']=='head_pitch_to_yaw'))
    fig=plt.figure(figsize=(14,10),layout='constrained')
    top=fig.add_subplot(221);oldside=fig.add_subplot(222);newside=fig.add_subplot(223);iso=fig.add_subplot(224,projection='3d')
    for m,color,label in [(shell,'#999999','Shell envelope'),(ring,'#237da0','Original ring'),(bridge,'#8b63a6','C frame')]:
        points=m.vertices[:,:2];h=ConvexHull(points);outline=points[np.r_[h.vertices,h.vertices[0]]]
        top.plot(*outline.T,color=color,label=label)
    for i,p in enumerate(motors):
        for m,color,label in [(mesh(p),'#328ac1','Original SG90'),(proposed[i],'#e58a20','Upright candidate')]:
            pts=m.vertices[:,:2];h=ConvexHull(pts);outline=pts[np.r_[h.vertices,h.vertices[0]]]
            top.plot(*outline.T,color=color,label=label if i==0 else None)
    top.set(title='Top projection: original / proposed motor bodies',xlabel='X / mm (+ front)',ylabel='Y / mm');top.set_aspect('equal');top.legend(fontsize=8)
    for ax,own,title in [(oldside,mesh(motors[0]),'CURRENT'),(newside,proposed[0],'BODY-FIT CANDIDATE')]:
        y=float(own.bounds.mean(axis=0)[1])
        for m,color in [(shell,'#999999'),(ring,'#237da0'),(bridge,'#8b63a6'),(own,'#e58a20')]:
            segments=trimesh.intersections.mesh_plane(m,plane_origin=[0,y,0],plane_normal=[0,1,0])
            if len(segments):ax.add_collection(LineCollection(segments[:,:,[0,2]],colors=color,linewidths=1.4))
        ax.set(title=f'{title}: actual section at Y={y:.2f} mm',xlabel='X / mm (+ front)',ylabel='Z / mm (+ up)',xlim=(-67,110),ylim=(167,241))
        ax.set_aspect('equal');ax.grid(alpha=.15)
    for m,color,alpha in [(shell,'#aaaaaa',.06),(ring,'#237da0',.65),(bridge,'#8b63a6',.6),*[(m,'#e58a20',.85) for m in proposed]]:
        iso.add_collection3d(Poly3DCollection(m.triangles,facecolors=color,edgecolors='none',alpha=alpha,rasterized=True))
    iso.set(xlim=(-67,110),ylim=(-90,90),zlim=(165,241),title='Candidate bodies inside unchanged shell',xlabel='X',ylabel='Y',zlabel='Z')
    iso.set_box_aspect((177,180,76));iso.view_init(elev=25,azim=-135)
    fig.suptitle('SG90 upright BODY-FIT candidate; formal STEP unchanged\nShaft datum: +40 X, 30 inward, +25 Z mm; output axis vertical\n'
                 'Shell / existing head parts and sampled S13 / S14 body clearances passed; mounts and antenna linkage NOT redesigned',fontsize=12)
    path=TARGET.parent/'HEAD_UPRIGHT_REVIEW.png';fig.savefig(path,dpi=150);plt.close(fig)
    print(path,flush=True)


def upright():
    """SG90 upright layout brief: millimetres, +Z up; ring untouched.

    First keep the output shaft's existing point and axis, rotating the motor
    housing around it to put its long dimension along projected +Z/-Z.
    Thus antenna and holder stay at their existing poses. Only if this fails,
    inspect small translations as diagnostics, explicitly marking changed
    output points. No formal STEP or installation seats are modified.
    Validate both SG90s against ring during S14 and current local head parts.
    Coarse meshes select candidates; positive 0.2-mm clearance needs separate
    measured-error bounds or BRep verification before acceptance.
    """
    import itertools
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation
    c=cache();facts=json.loads((WORK/'facts.json').read_text())
    def mesh(row):
        d=np.load(WORK/('mesh-'+Path(row['file']).stem+'.npz'))
        return trimesh.Trimesh(d['vertices'],d['faces'],process=False)
    ringrow=next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll')
    ring=trimesh.collision.CollisionManager();ring.add_object('ring',mesh(ringrow))
    poses=[trimesh.transformations.rotation_matrix(np.deg2rad(a),c['axes']['S14'][1],c['axes']['S14'][0])
           for a in range(-30,31)]
    strict=len(sys.argv)>2 and sys.argv[2]=='strict'
    inner=len(sys.argv)>2 and sys.argv[2]=='shaft_inner'
    shaft_up=len(sys.argv)>2 and sys.argv[2] in ('shaft','shaft_inner')
    hull=None
    if inner:
        from scipy.spatial import ConvexHull
        hull=ConvexHull(mesh(next(r for r in c['parts'].values() if r['label']=='head')).vertices)
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,ring_unchanged=True,
                strict_vertical=strict,
                output_axis_vertical=shaft_up,
                stage='Upright coarse search; original antenna/holder poses kept for zero-offset cases',servos={})
    for p,r in c['parts'].items():
        if r['label']!='sg90':continue
        f=next(x for x in facts if x['path']==p)
        shaft=next(x for x in f['cylinders'] if abs(x['radius']-2.425)<.001)
        holes=[x for x in f['cylinders'] if abs(x['radius']-1)<.001]
        long=np.array(holes[1]['origin'])-np.array(holes[0]['origin']);long/=np.linalg.norm(long)
        direction=np.array(shaft['direction']);up=np.array((0.,0.,1.));up-=direction*np.dot(up,direction);up/=np.linalg.norm(up)
        signed=np.rad2deg(np.arctan2(np.dot(direction,np.cross(long,up)),np.dot(long,up)))
        fixed=trimesh.collision.CollisionManager()
        for q,rr in c['parts'].items():
            if q==p or rr['root']!='head_assembly <1>':continue
            fixed.add_object(q,mesh(rr))
        own=mesh(r);variants=[];feasible=[]
        sign=1 if '[13]' in p else -1
        offsets=[(0,0,0)]+[(dx,dy*sign,dz) for dx,dy,dz in itertools.product((0,5,10,15,20),(0,-3,-6),(3,6,9,12,15,18))]
        if shaft_up:offsets=[(0,0,0)]+[(dx,dy*sign,dz) for dx,dy,dz in itertools.product((0,5,10,15),(0,-3,-6),(-9,-6,-3,0,3,6,9)) if (dx,dy,dz)!=(0,0,0)]
        if inner:offsets=[(dx,dy*sign,dz) for dx,dy,dz in itertools.product((0,10,20),(-10,-15,-20,-25,-30,-35),(-12,-6,0,6,12))]
        orientations=[]
        for angle in (signed,signed+180):
            rotate=trimesh.transformations.rotation_matrix(np.deg2rad(angle),direction,shaft['origin'])
            rotation_axis=direction
            if strict or shaft_up:
                target_long=np.array((0.,0.,1. if angle==signed else -1.))
                target_shaft=direction.copy();target_shaft[2]=0.;target_shaft/=np.linalg.norm(target_shaft)
                if shaft_up:
                    target_shaft=np.array((0.,0.,1. if angle==signed else -1.))
                    target_long=long.copy();target_long[2]=0.;target_long/=np.linalg.norm(target_long)
                source_basis=np.column_stack((long,direction,np.cross(long,direction)))
                target_basis=np.column_stack((target_long,target_shaft,np.cross(target_long,target_shaft)))
                vector=Rotation.from_matrix(target_basis@source_basis.T).as_rotvec()
                angle=float(np.rad2deg(np.linalg.norm(vector)));rotation_axis=vector/np.linalg.norm(vector)
                rotate=trimesh.transformations.rotation_matrix(np.deg2rad(angle),rotation_axis,shaft['origin'])
            orientations.append((angle,rotate,rotation_axis))
        if inner:
            base=orientations[0][1];orientations=[]
            for yaw in (0,90,180,270):
                rotate=trimesh.transformations.rotation_matrix(np.deg2rad(yaw),(0,0,1),shaft['origin'])@base
                vector=Rotation.from_matrix(rotate[:3,:3]).as_rotvec()
                angle=float(np.rad2deg(np.linalg.norm(vector)));rotation_axis=vector/np.linalg.norm(vector)
                orientations.append((angle,rotate,rotation_axis))
        for angle,rotate,rotation_axis in orientations:
            for delta in offsets:
                transform=trimesh.transformations.translation_matrix(delta)@rotate
                if hull is not None:
                    transformed=trimesh.transform_points(own.vertices,transform)
                    if any(np.max(points@hull.equations[:,:3].T+hull.equations[:,3])>.05 for points in np.array_split(transformed,15)):continue
                static_hit,contacts=fixed.in_collision_single(own,transform=transform,return_names=True)
                roll_hits=[a-30 for a,pose in enumerate(poses) if ring.in_collision_single(own,transform=pose@transform)]
                v=dict(turn_deg=float(angle),offset_mm=list(delta),shaft_origin=shaft['origin'],shaft_direction=shaft['direction'],
                       rotation_axis=list(rotation_axis),shaft_direction_after=list(rotate[:3,:3]@direction),
                       long_axis_before=list(long),long_axis_after=list(rotate[:3,:3]@long),
                       transform=transform.tolist(),static_collisions=sorted(contacts),s14_ring_collision_angles=roll_hits,
                       keeps_antenna_axis_and_point=delta==(0,0,0) and not strict and not shaft_up,score=sum(abs(x) for x in delta))
                variants.append(v)
                if not static_hit and not roll_hits:feasible.append(v)
                if delta==(0,0,0):print('Upright anchored',p,json.dumps(v),flush=True)
        report['servos'][p]=dict(anchored=[v for v in variants if v['offset_mm']==[0,0,0]],
                                 feasible=sorted(feasible,key=lambda x:x['score']),all_trials=variants)
        print('Upright feasible',p,len(feasible),'best',sorted(feasible,key=lambda x:x['score'])[:2],flush=True)
        filename='upright-inner-search.json' if inner else 'upright-shaft-search.json' if shaft_up else 'upright-strict-search.json' if strict else 'upright-search.json'
        (WORK/filename).write_text(json.dumps(report,indent=2)+'\n')


def upright_verify():
    import numpy as np
    import trimesh
    import refit_upper_servo_interfaces_local as refit
    from exact_motion_current import _overlap
    from repair_minor_motion_clearance import _mesh as bounded_mesh,volume
    strict=len(sys.argv)>2 and sys.argv[2]=='strict'
    shaft_up=len(sys.argv)>2 and sys.argv[2]=='shaft'
    inner=len(sys.argv)>2 and sys.argv[2]=='inner'
    prefix='upright-inner' if inner else 'upright-shaft' if shaft_up else 'upright-strict' if strict else 'upright'
    c=cache();search=json.loads((WORK/(prefix+'-search.json')).read_text())
    out=WORK/(prefix+'-verify.json')
    shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    ring=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    rm=bounded_mesh(ring,.01);manager=trimesh.collision.CollisionManager();manager.add_object('ring',rm)
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,ring_unchanged=True,
                trials=[],assembly_passed=False)
    for p,data in search['servos'].items():
        clear=[v for v in data['all_trials'] if not v['s14_ring_collision_angles']]
        selected=data['anchored'][:1]+sorted(clear,key=lambda x:x['score'])[:2]
        if shaft_up:selected=data['anchored']+sorted(data['feasible'],key=lambda x:x['score'])[:2]
        if inner:
            best=json.loads((WORK/'upright-inner-probe.json').read_text())['best']['proposal']
            selected=[v for v in data['all_trials'] if v['offset_mm'][0]==best['offset_mm'][0]
                      and abs(v['offset_mm'][1])==abs(best['offset_mm'][1]) and v['offset_mm'][2]==best['offset_mm'][2]
                      and abs(v['long_axis_after'][0]-best['long_axis_after'][0])<.001]
        for v in selected:
            fresh=shape(c['parts'][p]).rotate(bd.Axis(v['shaft_origin'],v.get('rotation_axis',v['shaft_direction'])),v['turn_deg']).moved(bd.Pos(*v['offset_mm']))
            raw=fresh.intersect(shell)
            common=refit._shape(raw) if refit._items(raw) else None
            sv=volume(common) if common else 0.
            fm=bounded_mesh(fresh,.01);error=fm.metadata['maximum_deflection_mm']+rm.metadata['maximum_deflection_mm']
            gaps=[];hits=[]
            for angle in range(-30,31):
                transform=trimesh.transformations.rotation_matrix(np.deg2rad(angle),c['axes']['S14'][1],c['axes']['S14'][0])
                gap=manager.min_distance_single(fm,transform=transform)-error;gaps.append((gap,angle))
                if gap<.199:hits.append(dict(angle_deg=angle,gap_lower_bound_mm=gap))
            maximum=30 if '[13]' in p else -30
            result=dict(path=p,proposal=v,bounds=bounds(fresh),
                        shell_overlap_mm3=sv,shell_overlap_solids=[dict(volume_mm3=volume(s),bounds=bounds(s)) for s in common.solids()] if sv>.001 else [],
                        s14_min_gap_lower_bound_mm=min(gaps)[0],s14_min_angle_deg=min(gaps)[1],
                        s14_failed_samples=hits,step_deg=1,mesh_error_bound_mm=error,
                        s14_extreme_overlap_mm3=_overlap(fresh.rotate(bd.Axis(*c['axes']['S14']),maximum),ring))
            report['trials'].append(result)
            print('Upright exact',json.dumps({k:val for k,val in result.items() if k not in ('proposal','s14_failed_samples')}),flush=True)
            out.write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True
    out.write_text(json.dumps(report,indent=2)+'\n')


def upright_probe():
    from exact_motion_current import _overlap
    inner=len(sys.argv)>2 and sys.argv[2]=='inner'
    filename='upright-inner-search.json' if inner else 'upright-strict-search.json'
    out=WORK/('upright-inner-probe.json' if inner else 'upright-probe.json')
    c=cache();search=json.loads((WORK/filename).read_text())
    shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,trials=[])
    for p,data in search['servos'].items():
        variants=[v for v in data['all_trials'] if not v['s14_ring_collision_angles'] and v['offset_mm'][0]>=5
                  and v['long_axis_after'][2]>.9]
        if inner:variants=[v for v in data['all_trials'] if not v['s14_ring_collision_angles'] and all(q.endswith('/head[4]') for q in v['static_collisions'])]
        variants.sort(key=lambda v:v['score'])
        for v in variants:
            a=shape(c['parts'][p]).rotate(bd.Axis(v['shaft_origin'],v['rotation_axis']),v['turn_deg']).moved(bd.Pos(*v['offset_mm']))
            overlap=_overlap(a,shell)
            result=dict(path=p,proposal=v,shell_overlap_mm3=overlap,bounds=bounds(a))
            report['trials'].append(result)
            print('Upright placement probe',p,v['offset_mm'],overlap,flush=True)
            out.write_text(json.dumps(report,indent=2)+'\n')
        # The opposite side is mirrored in this input; verify its selected
        # minimum separately rather than repeating all exploratory placements.
        break
    report['best']=min(report['trials'],key=lambda x:(x['shell_overlap_mm3'],x['proposal']['score'])) if report['trials'] else None
    report['completed']=True
    out.write_text(json.dumps(report,indent=2)+'\n')


def upright_local():
    import numpy as np
    from exact_motion_current import _overlap
    c=cache();search=json.loads((WORK/'upright-shaft-search.json').read_text())
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,stage='Upright body static interfaces; original holders and antennas unmodified',trials=[])
    for p,data in search['servos'].items():
        for v in data['feasible'][:2]:
            a=shape(c['parts'][p]).rotate(bd.Axis(v['shaft_origin'],v['rotation_axis']),v['turn_deg']).moved(bd.Pos(*v['offset_mm']))
            ab=np.array(bounds(a));trial=dict(path=p,proposal=v,checks=[])
            for q,r in c['parts'].items():
                if q==p:continue
                if r['root']!='head_assembly <1>' and r['label'] not in ('left_antenna_holder','right_antenna_holder','antenna','head_pitch_to_yaw','head_yaw_to_roll') and r['servo']!='S13':continue
                bb=np.array(r['bounds']);lower=float(np.linalg.norm(np.maximum(np.maximum(ab[0]-bb[1],bb[0]-ab[1]),0)))
                # Cached C bounds are conservative after installed S12 relief.
                if lower>=.6:
                    gap=lower;overlap=0.;method='AABB lower bound'
                else:
                    b=shape(r);gap=a.distance_to(b);overlap=_overlap(a,b) if gap<1e-6 else 0.;method='BRep distance and intersection'
                result=dict(neighbor=q,gap_mm=gap,overlap_mm3=overlap,method=method)
                trial['checks'].append(result)
                if gap<.199:print('Upright local failure',p,v['offset_mm'],json.dumps(result),flush=True)
            trial['passed']=all(x['gap_mm']>=.199 and x['overlap_mm3']<.001 for x in trial['checks'])
            print('Upright local complete',p,v['offset_mm'],'passed',trial['passed'],'minimum',min(x['gap_mm'] for x in trial['checks']),flush=True)
            report['trials'].append(trial)
            (WORK/'upright-local.json').write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True
    report['assembly_passed']=False
    report['unresolved']=['SG90 original side output connection is no longer coaxial with vertical output',
                          'Original SG90 seats and antenna holder connection must be redesigned',
                          'Antenna output motion direction must be confirmed',
                          'Speaker S14 three pairs and S13 two pairs remain in formal STEP']
    (WORK/'upright-local.json').write_text(json.dumps(report,indent=2)+'\n')


def upright_envelope():
    import numpy as np
    from scipy.spatial import ConvexHull
    c=cache();row=next(r for r in c['parts'].values() if r['label']=='head')
    shell=np.load(WORK/('mesh-'+Path(row['file']).stem+'.npz'))['vertices']
    hull=ConvexHull(shell);search=json.loads((WORK/'upright-shaft-search.json').read_text())
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,
                method='Tessellated shell convex hull violation diagnoses exterior protrusion; passing does not prove cavity containment',checks=[])
    for p,data in search['servos'].items():
        row=c['parts'][p];vertices=np.load(WORK/('mesh-'+Path(row['file']).stem+'.npz'))['vertices']
        for v in data['feasible'][:2]:
            matrix=np.asarray(v['transform']);moved=vertices@matrix[:3,:3].T+matrix[:3,3]
            maximum=(-float('inf'),None)
            for points in np.array_split(moved,50):
                residual=points@hull.equations[:,:3].T+hull.equations[:,3]
                where=np.unravel_index(residual.argmax(),residual.shape)
                if residual[where]>maximum[0]:maximum=(float(residual[where]),points[where[0]].tolist())
            result=dict(path=p,offset_mm=v['offset_mm'],maximum_outside_convex_hull_mm=maximum[0],point=maximum[1])
            report['checks'].append(result);print('Upright exterior envelope',json.dumps(result),flush=True)
    (WORK/'upright-envelope.json').write_text(json.dumps(report,indent=2)+'\n')


def upright_review():
    import numpy as np
    import trimesh
    from scan_s01_s14_motion_collisions import _mesh
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    c=cache();path=next(p for p,r in c['parts'].items() if r['label']=='sg90' and '[13]' in p)
    variants=[None,
              next(x['proposal'] for x in json.loads((WORK/'upright-shaft-verify.json').read_text())['trials'] if x['path']==path and x['proposal']['offset_mm']==[0,0,3]),
              json.loads((WORK/'upright-inner-probe.json').read_text())['best']['proposal']]
    row=c['parts'][path];d=np.load(WORK/('mesh-'+Path(row['file']).stem+'.npz'))
    original=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
    shellrow=next(r for r in c['parts'].values() if r['label']=='head')
    d=np.load(WORK/('mesh-'+Path(shellrow['file']).stem+'.npz'))
    shell=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
    ringrow=next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll')
    d=np.load(WORK/('mesh-'+Path(ringrow['file']).stem+'.npz'))
    ring=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for col,v in enumerate(variants):
        candidate=original.copy()
        if v:candidate.apply_transform(np.array(v['transform']))
        for line in range(2):
            ax=axes[line,col]
            pose=trimesh.transformations.rotation_matrix(np.deg2rad(30 if line else 0),c['axes']['S14'][1],c['axes']['S14'][0])
            centre=trimesh.transform_points(candidate.bounds.mean(axis=0)[None,:],pose)[0]
            normal=(0,1,0);point=(0,float(centre[1]),0);coordinates=(0,2)
            title=('S14 +30 deg' if line else 'Neutral pose')+f'; section Y={centre[1]:.1f} mm'
            for m,color in ((shell,'#777777'),(ring,'#377fba'),(candidate,'#e69530')):
                mm=m.copy()
                if m is not ring:mm.apply_transform(pose)
                lines=trimesh.intersections.mesh_plane(mm,plane_normal=normal,plane_origin=point)
                if len(lines):ax.add_collection(LineCollection(lines[:,:,coordinates],colors=color,linewidths=1.4))
            ax.set_aspect('equal');ax.set_xlim(-75,50);ax.set_ylim(150,245)
            ax.grid(alpha=.12);ax.set_xlabel('X mm (front +)');ax.set_ylabel('Z mm (up +)')
            ax.set_title(('CURRENT' if not v else 'Shaft Z; +3 Z; OUTSIDE SHELL' if col==1 else 'Shaft Z; +20 X / -35 Y / -6 Z; SHELL HIT')+'\n'+title,fontsize=10)
    fig.suptitle('SG90 upright diagnostics; formal STEP unchanged\nGrey: shell section; blue: original ring; orange: servo section\n'
                 'Shaft Z / +3 Z clears ring but protrudes outside; inner candidate clears ring but intersects shell\n'
                 'Installation seats, antenna linkage and full assembly NOT passed',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.89));out=TARGET.parent/'HEAD_UPRIGHT_REVIEW.png'
    fig.savefig(out,dpi=150);plt.close(fig);print(out,flush=True)


def inspect():
    WORK.mkdir(exist_ok=True);c=cache();facts=[]
    for path,row in c['parts'].items():
        if row['label'] not in ('sg90','speaker_stand','right_antenna_holder',
                                'left_antenna_holder','head'):continue
        p=shape(row);cyl=[]
        for f in p.faces():
            if f.geom_type==bd.GeomType.CYLINDER:
                axis=f.axis_of_rotation
                cyl.append(dict(radius=f.radius,origin=list(axis.position),
                                direction=list(axis.direction),bounds=bounds(f)))
        facts.append(dict(path=path,bounds=bounds(p),cylinders=cyl))
        print(path, 'bounds',bounds(p),'cylinders',len(cyl),flush=True)
        if row['label']!='head':print(json.dumps(cyl),flush=True)
        del p;gc.collect()
    (WORK/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')


def search():
    import itertools
    import numpy as np
    import trimesh
    from scan_s01_s14_motion_collisions import _mesh
    c=cache();WORK.mkdir(exist_ok=True)
    head={p:r for p,r in c['parts'].items() if r['root']=='head_assembly <1>'}
    # Only local head neighbours; triangulations serve candidate selection.
    meshes={}
    needed=set(head)|{p for p,r in c['parts'].items()
                     if r['label'] in ('head_yaw_to_roll','head_pitch_to_yaw')}
    for p in needed:
        r=c['parts'][p];file=WORK/('mesh-'+Path(r['file']).stem+'.npz')
        if file.exists():
            d=np.load(file);mesh=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
        else:
            a=shape(r);mesh=_mesh(a,.25,.3);del a
            np.savez(file,vertices=mesh.vertices,faces=mesh.faces)
        meshes[p]=mesh
        print('meshed',p,len(mesh.faces),flush=True);gc.collect()
    ringpath=next(p for p,r in c['parts'].items() if r['label']=='head_yaw_to_roll')
    ring=trimesh.collision.CollisionManager();ring.add_object('ring',meshes[ringpath])
    poses=[trimesh.transformations.rotation_matrix(np.deg2rad(a),c['axes']['S14'][1],c['axes']['S14'][0])
           for a in range(-30,31,5)]
    report=dict(source_sha256=SOURCE_SHA,stage='coarse mesh search only',ring_unchanged=True,sg90={},speaker=[])
    if len(sys.argv)>2 and sys.argv[2]=='speaker':
        previous=json.loads((WORK/'search.json').read_text())
        report['sg90']=previous['sg90'];report['sg90_ring_only']=previous.get('sg90_ring_only',{})
    elif (WORK/'search.json').exists():
        report['speaker']=json.loads((WORK/'search.json').read_text()).get('speaker',[])
    for p,r in head.items():
        if r['label']!='sg90':continue
        if len(sys.argv)>2 and sys.argv[2]=='speaker':continue
        facts=next(f for f in json.loads((WORK/'facts.json').read_text()) if f['path']==p)
        output=next(f for f in facts['cylinders'] if abs(f['radius']-2.425)<.001)
        fixed=trimesh.collision.CollisionManager()
        for q,s in head.items():
            if q==p or s['label']=='glass':continue
            fixed.add_object(q,meshes[q])
        found=[];ring_only=[]
        poses_to_try=[]
        sign=1 if '[13]' in p else -1
        poses_to_try += [(a,(dx,dy*sign,dz)) for a,dx,dy,dz in
                        itertools.product((0,-90),range(60,101,10),(0,),(0,5))]
        poses_to_try += [(a,(dx,dy*sign,dz)) for a,dx,dy,dz in
                        itertools.product((90,-90),(10,20,25,30),(0,-5,-10),(15,20,22))]
        for angle,offset in poses_to_try:
            transform=(trimesh.transformations.translation_matrix(offset)@
                       trimesh.transformations.rotation_matrix(np.deg2rad(angle),output['direction'],output['origin']))
            gaps=[]
            for pose in poses:
                if ring.in_collision_single(meshes[p],transform=pose@transform):
                    gaps.append(0.);break
                gaps.append(1.)
            if min(gaps)<.3:continue
            gap=0. if fixed.in_collision_single(meshes[p],transform=transform) else 1.
            ring_only.append(dict(angle_deg=angle,offset_mm=offset,static_mesh_collision=gap==0.,roll_mesh_collision=False))
            if gap<.3:continue
            found.append(dict(angle_deg=angle,offset_mm=offset,
                              score=sum(abs(x) for x in offset)+abs(angle)/10,
                              shaft_origin=output['origin'],shaft_direction=output['direction'],
                              static_mesh_collision=False,roll_mesh_collision=False,transform=transform.tolist()))
        report['sg90'][p]=sorted(found,key=lambda x:x['score'])
        report.setdefault('sg90_ring_only',{})[p]=ring_only
        print('SG90 feasible',p,len(found),report['sg90'][p][:3],flush=True)
        (WORK/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    if len(sys.argv)>2 and sys.argv[2]=='sg90':
        print('Ring-only alternatives',report.get('sg90_ring_only'),flush=True);return
    speakerpaths=[p for p,r in head.items() if r['label'] in ('speaker_interface','speaker_bottom','speaker_top')]
    source=trimesh.util.concatenate([meshes[p] for p in speakerpaths]);centre=source.bounds.mean(axis=0)
    fixed=trimesh.collision.CollisionManager()
    for p,r in head.items():
        if p in speakerpaths or r['label']=='speaker_stand':continue
        fixed.add_object(p,meshes[p])
    for turn in (0,90,-90,180):
        rotation=trimesh.transformations.rotation_matrix(np.deg2rad(turn),(0,0,1),centre)
        found=[]
        for dx,dy,dz in itertools.product(range(0,81,10),range(-20,61,10),(0,5,-5)):
            transform=trimesh.transformations.translation_matrix((dx,dy,dz))@rotation
            points=trimesh.transform_points(source.bounding_box.vertices,transform)
            if (points.min(axis=0)<[-63,-84,177]).any() or (points.max(axis=0)>[104,84,234]).any():continue
            gaps=[]
            for pose in poses:
                if ring.in_collision_single(source,transform=pose@transform):
                    gaps.append(0.);break
                gaps.append(1.)
            if min(gaps)<.3:continue
            if fixed.in_collision_single(source,transform=transform):continue
            found.append(dict(turn_deg=turn,offset_mm=[dx,dy,dz],roll_mesh_collision=False,
                              score=abs(dx)+abs(dy)+abs(dz)+abs(turn)/10,transform=transform.tolist()))
        report['speaker'].extend(found);report['speaker'].sort(key=lambda x:x['score'])
        print('Speaker turn',turn,'feasible',len(found),'best',report['speaker'][:3],flush=True)
        (WORK/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Search complete; formal STEP unchanged.',flush=True)


def frame():
    import numpy as np
    import trimesh
    from scan_s01_s14_motion_collisions import _mesh
    import refit_upper_servo_interfaces_local as refit
    from exact_motion_current import _overlap
    from cadgen.validity import check_occurrence_shape
    c=cache();old=shape(next(r for r in c['parts'].values() if r['label']=='head_pitch_to_yaw'))
    z=lambda h:h+.00093881241
    rest=refit._cut(old,[refit._world_box(-100,-5.1,-10.9,11.1,z(171.575),z(227))])
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,stage='C-frame size proposal; no servo motion',trials=[])
    for outer in (-45.6,-54.1,-60.1,-64.1):
        inner=outer+3
        outline=[(outer,z(170.075)),(outer,z(225.65)),(-5.1,z(225.65)),
                 (-5.1,z(223.575)),(inner,z(223.575)),(inner,z(171.575)),
                 (-5.1,z(171.575)),(-5.1,z(170.075))]
        arm=bd.extrude(bd.Plane.XZ*bd.Polygon(*outline,align=None),amount=22,dir=(0,1,0)).moved(bd.Pos(0,-10.9,0))
        fresh=refit._fuse(rest,arm).clean()
        validity=check_occurrence_shape(fresh.wrapped)
        if validity['solidCount']!=1 or validity['reasons']:
            print('Frame invalid',outer,validity,flush=True);continue
        bd.export_brep(fresh,WORK/'frame-candidate.brep')
        mesh=_mesh(fresh,.25,.3);fixed=trimesh.collision.CollisionManager();fixed.add_object('frame',mesh)
        collisions=[]
        for p,r in c['parts'].items():
            if r['root'] not in ('head_assembly <1>','neck_yaw_assembly <1>'):continue
            if r['servo']=='S13' and 'horn' in r['label']:continue
            file=WORK/('mesh-'+Path(r['file']).stem+'.npz')
            if file.exists():
                d=np.load(file);other=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
            else:
                a=shape(r);other=_mesh(a,.25,.3);del a
            hits=[]
            for angle in range(-160,161,10):
                pose=trimesh.transformations.rotation_matrix(np.deg2rad(angle),c['axes']['S13'][1],c['axes']['S13'][0])
                if fixed.in_collision_single(other,transform=pose):hits.append(angle)
            if hits:collisions.append(dict(path=p,angles=hits))
        trial=dict(wall_outer_x_mm=outer,wall_inner_x_mm=inner,validity=validity,coarse_collisions=collisions)
        report['trials'].append(trial);print('Frame trial',json.dumps(trial),flush=True)
        (WORK/'frame-report.json').write_text(json.dumps(report,indent=2)+'\n');gc.collect()


def diagnose():
    import numpy as np
    import trimesh
    from exact_motion_current import _overlap
    c=cache();report=json.loads((WORK/'search.json').read_text());rows=[]
    for p,variants in report['sg90_ring_only'].items():
        old=shape(c['parts'][p]);facts=next(f for f in json.loads((WORK/'facts.json').read_text()) if f['path']==p)
        shaft=next(f for f in facts['cylinders'] if abs(f['radius']-2.425)<.001)
        fixed=trimesh.collision.CollisionManager()
        for q,r in c['parts'].items():
            if q==p or r['root']!='head_assembly <1>':continue
            d=np.load(WORK/('mesh-'+Path(r['file']).stem+'.npz'))
            fixed.add_object(q,trimesh.Trimesh(d['vertices'],d['faces'],process=False))
        data=np.load(WORK/('mesh-'+Path(c['parts'][p]['file']).stem+'.npz'))
        mesh=trimesh.Trimesh(data['vertices'],data['faces'],process=False)
        variants.sort(key=lambda v:sum(abs(x) for x in v['offset_mm'])+abs(v['angle_deg'])/10)
        for v in variants[:12]:
            turn=v['angle_deg'];delta=v['offset_mm']
            transform=trimesh.transformations.translation_matrix(delta)@trimesh.transformations.rotation_matrix(
                np.deg2rad(turn),shaft['direction'],shaft['origin'])
            hit,names=fixed.in_collision_single(mesh,transform=transform,return_names=True)
            fresh=old.rotate(bd.Axis(shaft['origin'],shaft['direction']),turn).moved(bd.Pos(*delta))
            actual=[]
            for q in sorted(names):
                other=shape(c['parts'][q]);overlap=_overlap(fresh,other)
                if overlap>.001:actual.append(dict(path=q,overlap_mm3=overlap))
                del other
            row=dict(path=p,turn_deg=turn,offset_mm=delta,bounds=bounds(fresh),interferences=actual)
            rows.append(row);print('SG90 diagnostic',json.dumps(row),flush=True)
            (WORK/'diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n');gc.collect()


def section():
    import numpy as np
    import trimesh
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    c=cache();fig,axes=plt.subplots(1,2,figsize=(14,7))
    for ax,height in zip(axes,(195,222)):
        for label,color in (('head','#555555'),('sg90','#ef9c37'),('head_yaw_to_roll','#5072ad')):
            for p,r in c['parts'].items():
                if r['label']!=label:continue
                d=np.load(WORK/('mesh-'+Path(r['file']).stem+'.npz'))
                mesh=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
                lines=trimesh.intersections.mesh_plane(mesh,plane_origin=(0,0,height),plane_normal=(0,0,1))
                ax.add_collection(LineCollection(lines[:,:,:2],colors=color,linewidths=1))
        ax.set_aspect('equal');ax.set_xlim(-75,120);ax.set_ylim(-95,95)
        ax.set_title('Original shell / servo section Z='+str(height)+' mm')
        ax.set_xlabel('X mm (front +)');ax.set_ylabel('Y mm')
    fig.tight_layout();fig.savefig(WORK/'section.png',dpi=150);plt.close(fig)


def exact():
    from exact_motion_current import _overlap
    c=cache();ring=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    axis=bd.Axis(*c['axes']['S14']);report=dict(source_sha256=SOURCE_SHA,checks=[],formal_step_unchanged=True)
    for path,row in c['parts'].items():
        if row['label'] not in ('sg90','speaker_interface','speaker_bottom','speaker_top'):continue
        old=shape(row)
        if row['label']=='sg90':
            fact=next(f for f in json.loads((WORK/'facts.json').read_text()) if f['path']==path)
            shaft=next(f for f in fact['cylinders'] if abs(f['radius']-2.425)<.001)
            sign=1 if '[13]' in path else -1
            variants=[(-90*sign,(25,-5*sign,22)),(0,(100,0,0))]
        else:variants=[(0,(50,10,0))]
        for turn,delta in variants:
            fresh=old
            if turn:fresh=fresh.rotate(bd.Axis(shaft['origin'],shaft['direction']),turn)
            fresh=fresh.moved(bd.Pos(*delta))
            checks=[];hits=[]
            for angle in range(-30,31):
                posed=fresh.rotate(axis,angle);gap=posed.distance_to(ring)
                checks.append((gap,angle))
                if gap<.199:
                    overlap=_overlap(posed,ring);hits.append(dict(angle_deg=angle,gap_mm=gap,overlap_mm3=overlap))
                    if len(hits)>=3:break
            minimum,angle=min(checks)
            static=[]
            for p,r in c['parts'].items():
                if p==path or r['root']!='head_assembly <1>':continue
                if r['label'] in ('speaker_interface','speaker_bottom','speaker_top','speaker_stand') and row['label'].startswith('speaker'):continue
                other=shape(r);ba,bb=fresh.bounding_box(),other.bounding_box()
                if any(list(ba.min)[i]>list(bb.max)[i] or list(bb.min)[i]>list(ba.max)[i] for i in range(3)):continue
                overlap=_overlap(fresh,other)
                if overlap>.001:static.append(dict(path=p,overlap_mm3=overlap))
                del other
            result=dict(path=path,turn_deg=turn,offset_mm=delta,bounds=bounds(fresh),
                        s14_min_gap_mm=minimum,minimum_angle_deg=angle,
                        samples_completed=len(checks),s14_failures=hits,static_interferences=static)
            report['checks'].append(result);print('Exact proposal',json.dumps(result),flush=True)
            if not hits and not static:
                bd.export_brep(fresh,WORK/(Path(row['file']).stem+'-candidate.brep'))
            (WORK/'exact-report.json').write_text(json.dumps(report,indent=2)+'\n');gc.collect()
    frame=detached(bd.import_brep(WORK/'frame-candidate.brep'))
    report['frame_static_exact']=[]
    for label in ('head','head_bot_sheet','head_roll_mount'):
        other=shape(next(r for r in c['parts'].values() if r['label']==label))
        report['frame_static_exact'].append(dict(label=label,overlap_mm3=_overlap(frame,other)))
    (WORK/'exact-report.json').write_text(json.dumps(report,indent=2)+'\n')


def layout():
    from exact_motion_current import _overlap
    c=cache();head={p:r for p,r in c['parts'].items() if r['root']=='head_assembly <1>'}
    ring=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    servos={p:shape(r).moved(bd.Pos(120,0,0)) for p,r in head.items() if r['label']=='sg90'}
    labels=('speaker_interface','speaker_bottom','speaker_top')
    speakers={p:shape(r) for p,r in head.items() if r['label'] in labels}
    box=bd.Compound(list(speakers.values())).bounding_box();centre=box.center()
    fixed={p:shape(r) for p,r in head.items() if r['label'] not in (*labels,'speaker_stand','sg90')}
    fixed.update(servos)
    axis=bd.Axis(*c['axes']['S14']);report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,variants=[])
    for turn,dx,dy,dz in ((180,20,80,0),(180,30,80,0),(180,20,70,0),
                         (0,20,80,0),(90,30,80,0),(-90,30,80,0)):
        trial={p:s.rotate(bd.Axis(centre,(0,0,1)),turn).moved(bd.Pos(dx,dy,dz)) for p,s in speakers.items()}
        hits=[];gaps=[]
        for p,s in trial.items():
            for angle in (-30,-20,-10,0,10,20,30):
                a=s.rotate(axis,angle);gap=a.distance_to(ring)
                if gap<.199:hits.append(dict(path=p,angle_deg=angle,gap_mm=gap,overlap_mm3=_overlap(a,ring)));break
                gaps.append(gap)
            if hits:break
        static=[]
        if not hits:
            for p,s in trial.items():
                for q,other in fixed.items():
                    ba,bb=s.bounding_box(),other.bounding_box()
                    if any(list(ba.min)[i]>list(bb.max)[i] or list(bb.min)[i]>list(ba.max)[i] for i in range(3)):continue
                    v=_overlap(s,other)
                    if v>.001:static.append(dict(part=p,other=q,overlap_mm3=v))
        result=dict(speaker_turn_deg=turn,speaker_offset_mm=[dx,dy,dz],
                    sg90_offset_mm=[120,0,0],s14_failures=hits,static_interferences=static,
                    sampled_gap_mm=min(gaps) if gaps else None)
        report['variants'].append(result);print('Combined layout',json.dumps(result),flush=True)
        (WORK/'layout-report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not hits and not static:
            report['selected']=result
            report['rotation_centre']=list(centre)
            for p,s in {**servos,**trial}.items():
                bd.export_brep(s,WORK/(Path(head[p]['file']).stem+'-layout.brep'))
            (WORK/'layout-report.json').write_text(json.dumps(report,indent=2)+'\n');break


def layout_search():
    """Select a simultaneous layout; this does not validate mountings."""
    import itertools
    import numpy as np
    import trimesh
    c=cache();head={p:r for p,r in c['parts'].items() if r['root']=='head_assembly <1>'}
    def mesh(row):
        d=np.load(WORK/('mesh-'+Path(row['file']).stem+'.npz'))
        return trimesh.Trimesh(d['vertices'],d['faces'],process=False)
    labels=('speaker_interface','speaker_bottom','speaker_top')
    source=trimesh.util.concatenate([mesh(r) for r in head.values() if r['label'] in labels])
    centre=source.bounds.mean(axis=0)
    shell_bounds=np.asarray(next(r['bounds'] for r in head.values() if r['label']=='head'))
    fixed=trimesh.collision.CollisionManager()
    for p,r in head.items():
        if r['label'] in (*labels,'speaker_stand'):continue
        m=mesh(r)
        if r['label']=='sg90':m.apply_translation((100,0,0))
        fixed.add_object(p,m)
    ringrow=next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll')
    ring=trimesh.collision.CollisionManager();ring.add_object('ring',mesh(ringrow))
    for p,r in c['parts'].items():
        if r['label'] in ('left_antenna_holder','right_antenna_holder'):
            file=WORK/('mesh-'+Path(r['file']).stem+'.npz')
            if file.exists():m=mesh(r)
            else:
                from scan_s01_s14_motion_collisions import _mesh
                m=_mesh(shape(r),.25,.3);np.savez(file,vertices=m.vertices,faces=m.faces)
            m.apply_translation((100,0,0));fixed.add_object(p,m)
    poses=[trimesh.transformations.rotation_matrix(np.deg2rad(a),c['axes']['S14'][1],c['axes']['S14'][0])
           for a in range(-30,31,5)]
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,ring_unchanged=True,
                sg90_offset_mm=[100,0,0],stage='coarse simultaneous layout; mounts/antenna exits unverified',
                rotation_centre=list(centre),variants=[])
    flat=len(sys.argv)>2 and sys.argv[2]=='flat'
    orientations=[((1,0,0),90),((1,0,0),-90),((0,1,0),90),((0,1,0),-90)] if flat else [((0,0,1),a) for a in (0,90,-90,180)]
    offsets=itertools.product(range(0,61,10),(-10,0,10),(0,3,6,9,12)) if flat else itertools.product(range(10,81,10),range(10,101,10),(0,-3,3))
    offsets=list(offsets)
    for direction,turn in orientations:
        rotation=trimesh.transformations.rotation_matrix(np.deg2rad(turn),direction,centre)
        for dx,dy,dz in offsets:
            transform=trimesh.transformations.translation_matrix((dx,dy,dz))@rotation
            proposed_bounds=trimesh.transform_points(source.bounding_box.vertices,transform)
            if (proposed_bounds.min(axis=0)<shell_bounds[0]).any() or (proposed_bounds.max(axis=0)>shell_bounds[1]).any():continue
            if fixed.in_collision_single(source,transform=transform):continue
            if any(ring.in_collision_single(source,transform=pose@transform) for pose in poses):continue
            result=dict(speaker_turn_deg=turn,speaker_axis=list(direction),speaker_offset_mm=[dx,dy,dz],transform=transform.tolist(),
                        score=abs(dx)+abs(dy)+abs(dz)+abs(turn)/10)
            report['variants'].append(result)
        report['variants'].sort(key=lambda x:x['score'])
        print('Simultaneous coarse layout',direction,turn,'feasible',len(report['variants']),
              'best',report['variants'][:2],flush=True)
        (WORK/'layout-search.json').write_text(json.dumps(report,indent=2)+'\n')


def layout_verify():
    import numpy as np
    import trimesh
    from repair_minor_motion_clearance import _mesh as bounded_mesh
    c=cache();proposal=json.loads((WORK/'layout-search.json').read_text())
    labels=('speaker_interface','speaker_bottom','speaker_top')
    head={p:r for p,r in c['parts'].items() if r['root']=='head_assembly <1>'}
    ring=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    fixed_neck=[ring]
    for r in c['parts'].values():
        if r['label']=='head_pitch_to_yaw' or r['servo']=='S13':fixed_neck.append(shape(r))
    ringmesh=bounded_mesh(bd.Compound(fixed_neck),.02);ringmanager=trimesh.collision.CollisionManager()
    ringmanager.add_object('ring',ringmesh)
    fixed={};shell=None
    for p,r in head.items():
        if r['label'] in (*labels,'speaker_stand','sg90'):continue
        if r['label']=='head':
            shell=(p,shape(r));continue
        a=shape(r);fixed[p]=bounded_mesh(a,.02);del a;gc.collect()
    holders={}
    for p,r in c['parts'].items():
        if r['label'] in ('left_antenna_holder','right_antenna_holder'):
            holders[p]=bounded_mesh(shape(r).moved(bd.Pos(100,0,0)),.02)
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,ring_unchanged=True,
                scope='S14 single joint and static body layout only; stand, antenna links and exits unverified',
                s14_fixed_parts=['head_yaw_to_roll','head_pitch_to_yaw','S13 body and horns'],
                trials=[])
    for variant in proposal['variants'][:6]:
        candidates={}
        for p,r in head.items():
            if r['label'] not in (*labels,'sg90'):continue
            a=shape(r)
            if r['label']=='sg90':a=a.moved(bd.Pos(100,0,0))
            else:a=a.rotate(bd.Axis(proposal['rotation_centre'],variant['speaker_axis']),variant['speaker_turn_deg']).moved(bd.Pos(*variant['speaker_offset_mm']))
            candidates[p]=(a,bounded_mesh(a,.02))
        trial=dict(variant=variant,static=[],motion=[],body_layout_passed=True)
        for p,(a,m) in candidates.items():
            worst=(float('inf'),None);failures=[]
            for q,n in {**fixed,**{q:mm for q,(aa,mm) in candidates.items() if q!=p}}.items():
                if head[p]['label'] in labels and q in candidates and head[q]['label'] in labels:continue
                manager=trimesh.collision.CollisionManager();manager.add_object(q,n)
                error=m.metadata['maximum_deflection_mm']+n.metadata['maximum_deflection_mm']
                gap=manager.min_distance_single(m)-error
                worst=min(worst,(gap,q))
                if gap<.199:failures.append(dict(other=q,gap_lower_bound_mm=gap))
            result=dict(part=p,min_gap_lower_bound_mm=worst[0],closest_part=worst[1],failures=failures)
            if head[p]['label'] in labels:
                for q,n in holders.items():
                    manager=trimesh.collision.CollisionManager();manager.add_object(q,n)
                    gap=manager.min_distance_single(m)-m.metadata['maximum_deflection_mm']-n.metadata['maximum_deflection_mm']
                    if gap<.199:result['failures'].append(dict(other=q,gap_lower_bound_mm=gap))
                    if gap<result['min_gap_lower_bound_mm']:
                        result['min_gap_lower_bound_mm']=gap;result['closest_part']=q
            # The complete shell fails fine tessellation on a remote screw face.
            # Use every exact face within 0.6 mm of this body's bounding box;
            # omitted faces have an AABB distance of at least 0.6 mm.
            sp,ss=shell;body_bounds=np.asarray(bounds(a));selected=[]
            for face in ss.faces():
                fb=np.asarray(bounds(face))
                box_gap=float(np.linalg.norm(np.maximum(np.maximum(body_bounds[0]-fb[1],fb[0]-body_bounds[1]),0)))
                if box_gap<.6:selected.append(face)
            if selected:
                sm=bounded_mesh(bd.Compound(selected),.02)
                manager=trimesh.collision.CollisionManager();manager.add_object(sp,sm)
                gap=min(.6,manager.min_distance_single(m))-m.metadata['maximum_deflection_mm']-sm.metadata['maximum_deflection_mm']
            else:gap=.6-m.metadata['maximum_deflection_mm']
            result['shell_gap_lower_bound_mm']=gap
            result['shell_selected_faces']=len(selected)
            if gap<.199:result['failures'].append(dict(other=sp,gap_lower_bound_mm=gap))
            trial['static'].append(result)
            print('Layout static bounded',json.dumps(result),flush=True)
            error=m.metadata['maximum_deflection_mm']+ringmesh.metadata['maximum_deflection_mm']
            minimum=(float('inf'),0);failures=[]
            for angle in range(-30,31):
                pose=trimesh.transformations.rotation_matrix(np.deg2rad(angle),c['axes']['S14'][1],c['axes']['S14'][0])
                gap=ringmanager.min_distance_single(m,transform=pose)-error
                minimum=min(minimum,(gap,angle))
                if gap<.199:failures.append(dict(angle_deg=angle,gap_lower_bound_mm=gap))
            result=dict(part=p,min_gap_lower_bound_mm=minimum[0],minimum_angle_deg=minimum[1],
                        step_deg=1,mesh_error_bound_mm=error,failures=failures)
            trial['motion'].append(result)
            print('Layout S14 bounded',json.dumps(result),flush=True)
            if trial['static'][-1]['failures'] or failures:trial['body_layout_passed']=False
        report['trials'].append(trial)
        (WORK/'layout-verify.json').write_text(json.dumps(report,indent=2)+'\n')
        if trial['body_layout_passed']:
            report['selected']=trial;report['rotation_centre']=proposal['rotation_centre']
            for p,(a,m) in candidates.items():bd.export_brep(a,WORK/(Path(head[p]['file']).stem+'-layout.brep'))
            (WORK/'layout-verify.json').write_text(json.dumps(report,indent=2)+'\n')
            break
        del candidates;gc.collect()


def guards():
    import refit_upper_servo_interfaces_local as refit
    from exact_motion_current import _overlap
    from repair_minor_motion_clearance import volume
    c=cache();ring=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    # Three larger simple volumes enclose the complete original ring connector.
    zones=[refit._rod(bd.Vector(9,.1,0),bd.Vector(0,0,1),171.8768,177.878,50.751),
           refit._world_box(23.999,65,-12.461,12.661,171.8768,222.278),
           refit._world_box(-45,-17.299,-13.901,14.101,171.8768,222.278)]
    enclosing=refit._fuse(*zones).clean()
    covered=volume(refit._shape(ring.intersect(enclosing)))
    assert abs(volume(ring)-covered)<.001,(volume(ring),covered)
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,
                ring_guard_coverage_mm3=covered,checks=[],s13_frame_checks=[])
    proposed={}
    for path,row in c['parts'].items():
        if row['label'] not in ('sg90','speaker_interface','speaker_bottom','speaker_top'):continue
        delta=(100,0,0) if row['label']=='sg90' else (50,10,0)
        fresh=shape(row).moved(bd.Pos(*delta));gaps=[]
        for angle in range(-30,31):
            a=fresh.rotate(bd.Axis(*c['axes']['S14']),angle)
            gap=min(a.distance_to(b) for b in zones);gaps.append((gap,angle))
        minimum,angle=min(gaps)
        assert minimum>=.199,(path,minimum,angle)
        result=dict(path=path,offset_mm=delta,min_gap_lower_bound_mm=minimum,
                    minimum_angle_deg=angle,step_deg=1,
                    distance_method='Distance to three simple volumes enclosing complete original ring')
        report['checks'].append(result);proposed[path]=fresh
        print('Conservative local check',json.dumps(result),flush=True)
        (WORK/'guard-report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['combined_interferences']=[]
    for p,a in proposed.items():
        if c['parts'][p]['label']!='sg90':continue
        for q,b in proposed.items():
            if not c['parts'][q]['label'].startswith('speaker'):continue
            v=_overlap(a,b)
            if v>.001:report['combined_interferences'].append(dict(a=p,b=q,overlap_mm3=v))
    frame=detached(bd.import_brep(WORK/'frame-candidate.brep'))
    for label in ('head','head_bot_sheet','head_roll_mount'):
        other=shape(next(r for r in c['parts'].values() if r['label']==label))
        v=_overlap(frame,other)
        report['s13_frame_checks'].append(dict(part=label,angle_deg=0,overlap_mm3=v))
        print('Frame exact neutral',label,v,flush=True)
    report['assembly_passed']=False
    (WORK/'guard-report.json').write_text(json.dumps(report,indent=2)+'\n')


def drawing():
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon, Circle
    c=cache();fig,axes=plt.subplots(1,2,figsize=(14,7))
    shellrow=next(r for r in c['parts'].values() if r['label']=='head')
    d=np.load(WORK/('mesh-'+Path(shellrow['file']).stem+'.npz'))
    mesh=trimesh.Trimesh(d['vertices'],d['faces'],process=False)
    lines=trimesh.intersections.mesh_plane(mesh,plane_origin=(0,0,195),plane_normal=(0,0,1))
    for panel,ax in enumerate(axes):
        ax.add_collection(LineCollection(lines[:,:,:2],colors='#555555',linewidths=1))
        for radius in (50.75,38.5):
            ax.add_patch(Circle((9,.1),radius,fill=False,color='#3b6eb2',linewidth=2))
        for p,r in c['parts'].items():
            selected=r['label']=='sg90' if panel else r['label'] in ('speaker_interface','speaker_bottom','speaker_top')
            if not selected:continue
            d=np.load(WORK/('mesh-'+Path(r['file']).stem+'.npz'));xy=d['vertices'][:,:2]
            hull=xy[ConvexHull(xy).vertices]
            delta=np.array([100,0] if panel else [50,10])
            ax.add_patch(Polygon(hull,fill=False,color='#999999',linestyle='--'))
            ax.add_patch(Polygon(hull+delta,facecolor='#e5a353',edgecolor='#a5671e',alpha=.55))
            centre=hull.mean(axis=0);ax.annotate('',xy=centre+delta,xytext=centre,
                                             arrowprops={'arrowstyle':'->','color':'#a5671e'})
        ax.set_aspect('equal');ax.set_xlim(-75,120);ax.set_ylim(-95,95)
        ax.set_xlabel('X mm (front +)');ax.set_ylabel('Y mm')
        ax.set_title('SG90 bodies: +100 mm X; antenna positions change' if panel else
                     'Speaker bodies: +50 mm X, +10 mm Y')
    fig.suptitle('Separate placement checks, NOT a verified assembly\nRing stays 101.5 / 77 mm; mounts and antenna connections unresolved')
    fig.tight_layout();out=TARGET.parent/'HEAD_LAYOUT_PROPOSAL.png';fig.savefig(out,dpi=150);plt.close(fig)
    print(out,flush=True)


def layout_review():
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    from exact_motion_current import _overlap
    from scan_s01_s14_motion_collisions import _mesh
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    c=cache();verified=json.loads((WORK/'layout-verify.json').read_text())
    chosen=verified['selected']['variant'];transform=np.asarray(chosen['transform'])
    shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,antenna_offset_mm=[100,0,0],
                stage='Antenna route diagnostic; shell holes and installation seats not modified',checks=[])
    for p,r in c['parts'].items():
        if r['label'] not in ('antenna','left_antenna_holder','right_antenna_holder'):continue
        old=shape(r);new=old.moved(bd.Pos(100,0,0))
        result=dict(path=p,baseline_shell_overlap_mm3=_overlap(old,shell),
                    proposed_shell_overlap_mm3=_overlap(new,shell))
        report['checks'].append(result)
        print('Antenna route',json.dumps(result),flush=True)
        (WORK/'antenna-route.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(14,11))
    labels=('sg90','speaker_interface','speaker_bottom','speaker_top','antenna',
            'left_antenna_holder','right_antenna_holder','head','head_yaw_to_roll')
    for p,r in c['parts'].items():
        if r['label'] not in labels:continue
        file=WORK/('mesh-'+Path(r['file']).stem+'.npz')
        if file.exists():vertices=np.load(file)['vertices']
        else:
            a=shape(r);m=_mesh(a,.25,.3);vertices=m.vertices;del a,m
        proposed=vertices
        label=r['label']
        if label in ('sg90','antenna','left_antenna_holder','right_antenna_holder'):
            proposed=vertices+np.array((100,0,0))
        elif label.startswith('speaker_'):proposed=trimesh.transform_points(vertices,transform)
        color='#f19a38' if label=='sg90' else '#8658aa' if label.startswith('speaker_') else '#39875d' if 'antenna' in label else '#3975ae'
        for row,coordinates in enumerate(((0,1),(0,2))):
            for col,points in enumerate((vertices,proposed)):
                ax=axes[row,col];xy=points[:,coordinates];hull=xy[ConvexHull(xy).vertices]
                if label=='head':
                    ax.add_patch(Polygon(hull,fill=False,edgecolor='#888888',linewidth=1.5))
                elif label=='head_yaw_to_roll' and row==0:
                    for radius in (50.75,38.5):ax.add_patch(Circle((9,.1),radius,fill=False,edgecolor=color,linewidth=1.5))
                else:ax.add_patch(Polygon(hull,facecolor=color,edgecolor=color,alpha=.32))
    for row in range(2):
        for col in range(2):
            ax=axes[row,col];ax.set_aspect('equal');ax.set_xlim(-80,125)
            ax.set_ylim((-100,110) if row==0 else (140,335))
            ax.set_xlabel('X mm (front +)');ax.set_ylabel('Y mm' if row==0 else 'Z mm')
            ax.set_title(('CURRENT' if col==0 else 'CANDIDATE BODIES')+(' - top projection' if row==0 else ' - side projection'))
            ax.grid(alpha=.15)
    fig.suptitle('Ring stays 101.5 / 77 mm, width 12.25 mm, original height\n'
                 f"Orange: SG90 +100 X; purple: speaker {chosen['speaker_turn_deg']} deg about {chosen['speaker_axis']}, offset {chosen['speaker_offset_mm']} mm\n"
                 'Green: antenna route shown at +100 X, NOT validated; mounting seats/stand NOT designed\n'
                 'Projection hulls are layout aids; formal STEP unchanged',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.89));out=TARGET.parent/'HEAD_LAYOUT_PROPOSAL.png'
    fig.savefig(out,dpi=150);plt.close(fig);print(out,flush=True)


def layout_routes():
    from exact_motion_current import _overlap
    c=cache();shell=shape(next(r for r in c['parts'].values() if r['label']=='head'))
    report=json.loads((WORK/'antenna-route.json').read_text())
    bodies={p:detached(bd.import_brep(WORK/(Path(r['file']).stem+'-layout.brep')))
            for p,r in c['parts'].items() if r['label'] in ('sg90','speaker_interface','speaker_bottom','speaker_top')}
    report['body_shell_exact']=[]
    for p,a in bodies.items():
        result=dict(path=p,overlap_mm3=_overlap(a,shell))
        report['body_shell_exact'].append(result);print('Exact body shell',json.dumps(result),flush=True)
        (WORK/'antenna-route.json').write_text(json.dumps(report,indent=2)+'\n')
    report['holder_speaker_exact']=[]
    for p,r in c['parts'].items():
        if r['label'] not in ('left_antenna_holder','right_antenna_holder'):continue
        a=shape(r).moved(bd.Pos(100,0,0))
        for q,b in bodies.items():
            if c['parts'][q]['label']=='sg90':continue
            result=dict(holder=p,speaker=q,overlap_mm3=_overlap(a,b))
            report['holder_speaker_exact'].append(result)
            print('Exact holder speaker',json.dumps(result),flush=True)
            (WORK/'antenna-route.json').write_text(json.dumps(report,indent=2)+'\n')
    report['assembly_passed']=False
    report['unresolved']=['Moved antennas intersect unchanged shell; exit holes must be relocated',
                          'Original speaker stand and SG90 installation seats have not been redesigned',
                          'Antenna linkage and motion clearances have not passed',
                          'S13 two existing primary collisions remain; other joint checks of this layout pending']
    (WORK/'antenna-route.json').write_text(json.dumps(report,indent=2)+'\n')


def combined():
    from exact_motion_current import _overlap
    c=cache();report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,
                         assembly_passed=False,combined_interferences=[],s13_frame_checks=[])
    proposed={p:shape(r).moved(bd.Pos(100,0,0) if r['label']=='sg90' else bd.Pos(50,10,0))
              for p,r in c['parts'].items() if r['label'] in
              ('sg90','speaker_interface','speaker_bottom','speaker_top')}
    for p,a in proposed.items():
        if c['parts'][p]['label']!='sg90':continue
        for q,b in proposed.items():
            if not c['parts'][q]['label'].startswith('speaker'):continue
            v=_overlap(a,b)
            if v>.001:report['combined_interferences'].append(dict(a=p,b=q,overlap_mm3=v))
            print('Combined pair',p,q,v,flush=True)
    frame=detached(bd.import_brep(WORK/'frame-candidate.brep'))
    for label in ('head','head_bot_sheet','head_roll_mount'):
        other=shape(next(r for r in c['parts'].values() if r['label']==label))
        v=_overlap(frame,other)
        report['s13_frame_checks'].append(dict(part=label,angle_deg=0,overlap_mm3=v))
        print('Frame exact neutral',label,v,flush=True)
    (WORK/'combined-report.json').write_text(json.dumps(report,indent=2)+'\n')


def level():
    import refit_upper_servo_interfaces_local as refit
    from cadgen.validity import check_occurrence_shape
    from exact_motion_current import _overlap
    from scan_s01_s14_motion_collisions import _mesh
    import numpy as np
    import trimesh
    c=cache();old=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    z0,z1=171.87687762474,177.65087762474
    drop=float(sys.argv[2]) if len(sys.argv)>2 else 26
    screen=len(sys.argv)>3 and sys.argv[3]=='screen'
    removed=refit._world_box(-100,100,-100,100,z0-.001,177.87787762474)
    feet=refit._shape(old.intersect(refit._fuse(
        refit._world_box(24,65,-12.46,12.66,z0-.01,177.888),
        refit._world_box(-45,-17.3,-13.9,14.1,z0-.01,177.888))))
    annulus=refit._cut(refit._rod(bd.Vector(9,.1,0),bd.Vector(0,0,1),z0-drop,z1-drop,50.75),
                      [refit._rod(bd.Vector(9,.1,0),bd.Vector(0,0,1),z0-drop-.01,z1-drop+.01,38.5)])
    # Extend the original front/back wall in its own plane down to the ring;
    # these are proposed dimensions, not newly added mounting parts.
    front=refit._world_box(39.5,49.95,-12.46,12.66,z0-drop,z0+.01)
    rear=refit._world_box(-30.05,-27.05,-13.9,14.1,z0-drop,z0+.01)
    fresh=refit._fuse(refit._cut(old,[removed]),feet,annulus,front,rear).clean()
    validity=check_occurrence_shape(fresh.wrapped)
    print('Ring level candidate',validity,bounds(fresh),flush=True)
    report=dict(source_sha256=SOURCE_SHA,ring_level_delta_mm=-drop,
                radii_mm=[50.75,38.5],width_mm=12.25,formal_step_unchanged=True,
                validity=validity,bounds=bounds(fresh),checks=[],static=[])
    (WORK/'level-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if validity['solidCount']!=1 or validity['reasons']:return
    bd.export_brep(fresh,WORK/'level-candidate.brep')
    if screen:
        from repair_minor_motion_clearance import _mesh as bounded_mesh
        mesh=bounded_mesh(fresh,.01)
    else:
        mesh=_mesh(fresh,.15,.2)
    fixed=trimesh.collision.CollisionManager();fixed.add_object('ring',mesh)
    report['motion_coarse']=[]
    for joint in ('S12','S14'):
        rows=[(p,r) for p,r in c['parts'].items() if
              (joint=='S14' and r['label'] in ('sg90','speaker_stand','speaker_interface','speaker_bottom')) or
              (joint=='S12' and (r['label'] in ('neck_left_sheet','neck_right_sheet') or r['servo']=='S12' and 'body' in r['label']))]
        for p,r in rows:
            other=shape(r);axis=bd.Axis(*c['axes'][joint]);hits=[]
            if screen:
                om=bounded_mesh(other,.01);minimum=(float('inf'),0)
                error=mesh.metadata['maximum_deflection_mm']+om.metadata['maximum_deflection_mm']
                end=45 if joint=='S12' else 30
                from repair_minor_motion_clearance import relevant_mesh
                coordinate=1 if joint=='S12' else 0
                moving_mesh=relevant_mesh(mesh if joint=='S12' else om,
                                          om if joint=='S12' else mesh,coordinate)
                manager=trimesh.collision.CollisionManager()
                manager.add_object('fixed',om if joint=='S12' else mesh)
                for angle in range(-end,end+1):
                    transform=trimesh.transformations.rotation_matrix(np.deg2rad(angle),
                        c['axes'][joint][1],c['axes'][joint][0])
                    raw=min(.6,manager.min_distance_single(moving_mesh,transform=transform))
                    gap=raw-error
                    minimum=min(minimum,(gap,angle))
                    if gap<.199:hits.append(dict(angle_deg=angle,gap_lower_bound_mm=gap))
                result=dict(joint=joint,neighbor=p,hits=hits,
                            min_gap_lower_bound_mm=minimum[0],minimum_angle_deg=minimum[1],
                            mesh_error_bound_mm=error,step_deg=1,
                            method='FCL distance minus summed measured OCCT tessellation deflections')
                report['motion_coarse'].append(result)
                print('Ring level bounded',json.dumps(result),flush=True)
                (WORK/'level-report.json').write_text(json.dumps(report,indent=2)+'\n')
                continue
            elif joint=='S12':
                for angle in range(-45,46,5):
                    a=fresh.rotate(axis,angle);v=_overlap(a,other)
                    if v>.001:hits.append(dict(angle_deg=angle,overlap_mm3=v))
            else:
                om=_mesh(other,.15,.2)
                for angle in range(-30,31):
                    transform=trimesh.transformations.rotation_matrix(np.deg2rad(angle),
                        c['axes'][joint][1],c['axes'][joint][0])
                    if fixed.in_collision_single(om,transform=transform):
                        v=_overlap(other.rotate(axis,angle),fresh)
                        if v>.001:hits.append(dict(angle_deg=angle,overlap_mm3=v))
            result=dict(joint=joint,neighbor=p,hits=hits)
            report['motion_coarse'].append(result);print('Ring level local',json.dumps(result),flush=True)
            (WORK/'level-report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['motion_coarse_completed']=True
    (WORK/'level-report.json').write_text(json.dumps(report,indent=2)+'\n')


def level_diagnose():
    import refit_upper_servo_interfaces_local as refit
    from repair_minor_motion_clearance import volume
    c=cache();fresh=detached(bd.import_brep(WORK/'level-candidate.brep'))
    other=shape(next(r for r in c['parts'].values() if r['label']=='neck_left_sheet'))
    posed=fresh.rotate(bd.Axis(*c['axes']['S12']),45)
    common=refit._shape(posed.intersect(other))
    gap,pa,pb=posed.distance_to_with_closest_points(other)
    result=dict(ring_level_delta_mm=json.loads((WORK/'level-report.json').read_text())['ring_level_delta_mm'],
                angle_deg=45,overlap_mm3=volume(common),gap_mm=gap,
                closest_points=[list(pa),list(pb)],
                overlap_bounds=bounds(common) if common and volume(common)>1e-8 else None)
    print('Ring neck diagnostic',json.dumps(result),flush=True)
    (WORK/'level-diagnose.json').write_text(json.dumps(result,indent=2)+'\n')


def level_local():
    import refit_upper_servo_interfaces_local as refit
    from exact_motion_current import _overlap
    c=cache();fresh=detached(bd.import_brep(WORK/'level-candidate.brep'))
    old=shape(next(r for r in c['parts'].values() if r['label']=='head_yaw_to_roll'))
    report=dict(source_sha256=SOURCE_SHA,formal_step_unchanged=True,
                ring_level_delta_mm=json.loads((WORK/'level-report.json').read_text())['ring_level_delta_mm'],
                checks=[])
    tasks=[('S14','speaker_stand',-30),('S14','speaker_stand',-29),
           ('S11','body_middle_top',39),('S11','body_front',39),
           ('S11','body_middle_top',-20),('static','head_pitch_to_yaw',0)]
    for joint,label,angle in tasks:
        a=shape(next(r for r in c['parts'].values() if r['label']==label))
        if joint=='S14':
            a=a.rotate(bd.Axis(*c['axes'][joint]),angle);b=fresh;before=old
        elif joint=='S11':
            b=fresh.rotate(bd.Axis(*c['axes'][joint]),angle)
            before=old.rotate(bd.Axis(*c['axes'][joint]),angle)
        else:b=fresh;before=old
        v=_overlap(a,b);baseline=_overlap(a,before)
        common=refit._shape(a.intersect(b)) if v>.001 else None
        result=dict(joint=joint,part=label,angle_deg=angle,overlap_mm3=v,
                    baseline_overlap_mm3=baseline,
                    overlap_bounds=bounds(common) if common else None)
        report['checks'].append(result)
        print('Ring extra local',json.dumps(result),flush=True)
        (WORK/'level-local-report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True
    (WORK/'level-local-report.json').write_text(json.dumps(report,indent=2)+'\n')


def publish():
    import re
    cache()
    read=lambda name:json.loads((WORK/name).read_text())
    verified=read('layout-verify.json');routes=read('antenna-route.json')
    assert verified.get('selected') and len(routes['body_shell_exact'])==5 and len(routes['holder_speaker_exact'])==6
    assert all(x['overlap_mm3']<.001 for x in routes['body_shell_exact']+routes['holder_speaker_exact'])
    selected=verified['selected'];variant=selected['variant']
    assert variant['speaker_turn_deg']==-90 and variant['speaker_offset_mm']==[10,10,12]
    static_min=min(min(x['min_gap_lower_bound_mm'],x['shell_gap_lower_bound_mm']) for x in selected['static'])
    motion_min=min(x['min_gap_lower_bound_mm'] for x in selected['motion'])
    peaks={}
    for label in ('layout-search','layout-verify','layout-review','layout-routes','level'):
        log=Path('/tmp/head-plan-'+label+'.log')
        values=re.findall(r'CAD peak process RSS: ([0-9.]+) MB',log.read_text()) if log.exists() else []
        if values:peaks[label]=float(values[-1])
    report=dict(date='2026-10-10',target=str(TARGET),source_sha256=SOURCE_SHA,
                formal_step_unchanged=True,formal_known_remaining_pairs={'S13':2,'S14':5},
                ring={'outer_diameter_mm':101.5,'inner_diameter_mm':77,'radial_width_mm':12.25,
                      'axial_thickness_mm':5.774,'original_height_preserved':True},
                selected_body_layout=verified,antenna_route=routes,
                minimum_static_gap_lower_bound_mm=static_min,
                minimum_s14_gap_lower_bound_mm=motion_min,
                assembly_passed=False,installed=False,user_confirmation='pending: antenna exits +100 mm X',
                unresolved=routes['unresolved'],
                rejected_combination=read('combined-report.json'),
                rejected_ring_level_23mm=read('level-local-report.json'),
                rejected_ring_level_22mm=read('level-report.json'),
                rejected_c_frame_trials=read('frame-report.json'),
                source_script='scripts/cad/plan_head_clearance_local.py',
                review='cad/local_fix/HEAD_LAYOUT_PROPOSAL.png',
                review_sha256=hashlib.sha256((TARGET.parent/'HEAD_LAYOUT_PROPOSAL.png').read_bytes()).hexdigest(),
                snapshot_skipped_reason='No STEP created or visibly updated in this planning round; only temporary BREP bodies and a diagnostic projection were created',
                resources={'address_space_cap_bytes':4_000_000_000,'workspace_limit_bytes':5_000_000_000,
                           'serial':True,'native_threads':1,'daemon':False,'peak_process_rss_mb':peaks})
    (TARGET.parent/'HEAD_LAYOUT_PLAN_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    block=f'''<!-- HEAD_LAYOUT_PLAN_START -->
## 2026-10-10 保留原圆环的布局测算：候选待确认，正式模型未改

用户同意按内部布局方向测算，同时保留此前“具体改法先确认”的要求。正式 STEP 指纹仍为 `{SOURCE_SHA}`；已知 S13 两对、S14 五对仍未在正式文件中修好。不修改限位，不缩径/削细圆环，不削两块颈板。

- U02 / S14 本体候选：扬声器三个原器件整组绕世界 X 轴 −90°，旋转中心来自本轮几何包络，再平移 +X10、+Y10、+Z12 mm；两颗 SG90 保持原朝向、各向 +X100 mm。圆环外径 101.5、内径 77、环带 12.25、厚度 5.774 mm，原高度不动。
- 五个器件本体对其余当前头内器件的静态间隙下界最小 {static_min:.6f} mm；三扬声器件对同步前移的左右原天线夹持件也满足 0.2 mm。五件对头壳的实体交集各为 0，两个夹持件对三扬声器件的六个实体配对交集各为 0。SG90 与原同侧夹持件的固定关系不以 0.2 mm 非接触判据判断，安装接口尚未验收。
- S14 −30°…+30°、每 1°，上述五器件本体对原圆环、当前 C 架、S13 壳体及两舵盘的间隙下界最小 {motion_min:.6f} mm。方法为 FCL 距离减去两个网格各自实测 OCCT 最大弦高误差。静态头壳细网格在一处远端螺孔面失败，改为按完整原面包围盒筛出本体 0.6 mm 内所有面，省略面距离下界 0.6 mm，所得值封顶后扣实际误差；另用完整头壳 B-Rep 交集复核。不是完整装配或多关节组合通过。
- 天线与两个原夹持件若同步前移 +X100 mm：夹持件/当前头壳实体交集为 0；两天线各与当前头壳交集约 7.873717 mm³。因此原出口不能沿用，必须调整孔位及原 SG90 安装座，外部天线位置会明显变化。已向用户询问是否允许前移 100 mm，尚未确认；不能继续把出口改动当授权。
- 原 speaker_stand 尚未重新设计。固定座、连接孔及承压材料、天线传动/摆动、其它关节和组合姿态、4 mm 螺丝接口均未通过。不能把上述本体结果称为完整 S14 修复。
- U01/U03 / S13：直边 C 架后移尺寸候选仍失败。最后 −64.1 mm 壁候选在零位与头壳/底板交集 2359.717775 / 128.857209 mm³；不采用。S13 翻转或单轴没有授权或实施，S01/S06 例外不扩展至 S13。
- 已排除：原竖向扬声器 +X50/+Y10 与两 SG90 +X100 的组合（负侧 SG90 与三扬声器件交集 373.120848 / 210.426582 / 610.321587 mm³）；扬声器 +90° 横放、+10/+10/+9 的首候选也碰右天线夹持件，后换为上述 −90° 候选并重新验证。
- 降环方案未采用：下移 26 mm 在 S12 ±45° 碰颈板；下移 24 mm 左颈板交集 0.005425 mm³；下移 23 mm 虽颈板间隙通过，仍在 S14 −30° 碰 speaker_stand 12.453991 mm³，S11 +39° 又碰 body_middle_top 0.555561 mm³（原配对为 0）。下移 22 mm 仍未消除支架擦碰，不把降低圆环作为已决定方案。

实际查看 `HEAD_LAYOUT_PROPOSAL.png`，其中灰为头壳投影凸包，蓝为原圆环/连接件投影，橙为 SG90，紫为扬声器，绿为天线及夹持件；图明确标注安装结构未验证，是布局图，不是正式成品。报告 `HEAD_LAYOUT_PLAN_REPORT.json` 保存本轮完整数据；临时 BREP 位于 `/tmp/head-plan`，源 `scripts/cad/plan_head_clearance_local.py`。本轮未创建或更新 STEP，因此不重跑正式 STEP 截图。全部重型步骤经导入前 4 GB 地址空间硬限制、单线程及串行锁；本轮最终检查峰值 RSS {peaks.get('layout-verify',0):.1f} MB。用户确认具体天线位置后，才继续对应原安装结构设计并逐项验证；正式安装还须全部验证通过。
<!-- HEAD_LAYOUT_PLAN_END -->

'''
    for name in ('UPPER_REFIT_ISSUES.md','MOTION_COLLISION_REGISTER.md'):
        path=TARGET.parent/name;text=path.read_text()
        if '<!-- HEAD_LAYOUT_PLAN_START -->' in text:
            start=text.index('<!-- HEAD_LAYOUT_PLAN_START -->');end=text.index('<!-- HEAD_LAYOUT_PLAN_END -->',start)+len('<!-- HEAD_LAYOUT_PLAN_END -->')
            text=text[:start]+text[end:].lstrip('\n')
        anchor='## 2026-10-10 用户要求还原圆环'
        position=text.index(anchor);text=text[:position]+block+text[position:]
        path.write_text(text)
    print('Saved HEAD_LAYOUT_PLAN_REPORT.json and synchronized two issue registers; formal STEP unchanged.',flush=True)


if __name__=='__main__':
    if sys.argv[1] in ('speaker_prepare','speaker_probe','speaker_search','speaker_build','speaker_verify','speaker_export','speaker_parent_verify','speaker_install'):
        with bd.SkipClean():globals()[sys.argv[1]]()
        sys.exit()
    if sys.argv[1] in ('build_original_axis','original_axis_finish','original_axis_verify'):
        raise SystemExit('Ring-lowering layout rejected by user; use the edge layout commands.')
    if sys.argv[1] in ('original_axis_export','original_axis_review') and 'edge' not in sys.argv[2:] and 'speaker' not in sys.argv[2:]:
        raise SystemExit('Ring-lowering candidate rejected; the current assembly requires the edge argument.')
    if sys.argv[1]=='edge_search':
        edge_search()
        sys.exit()
    if sys.argv[1]=='edge_probe':
        edge_probe()
        sys.exit()
    if sys.argv[1]=='edge_build':
        with bd.SkipClean():edge_build()
        sys.exit()
    if sys.argv[1]=='edge_finish':
        with bd.SkipClean():edge_finish()
        sys.exit()
    if sys.argv[1]=='edge_normalize_head':
        edge_normalize_head()
        sys.exit()
    if sys.argv[1]=='edge_verify':
        edge_verify()
        sys.exit()
    if sys.argv[1]=='original_axis_review':
        original_axis_review()
        sys.exit()
    if sys.argv[1]=='original_axis_export':
        original_axis_export()
        sys.exit()
    if sys.argv[1]=='original_axis_finish':
        original_axis_finish()
        sys.exit()
    if sys.argv[1]=='original_axis_verify':
        original_axis_verify()
        sys.exit()
    if sys.argv[1]=='seat_debug':
        seat_debug()
        sys.exit()
    extra={'build_original_axis':build_original_axis,'seat_facts':seat_facts,'fixed_axis_search':fixed_axis_search,'fixed_axis_diagnose':fixed_axis_diagnose,'fixed_axis_review':fixed_axis_review,'side_shell_exact':side_shell_exact,'side_failure_exact':side_failure_exact,'prepare_current':prepare_current,'upright_fit':upright_fit,'upright_pair':upright_pair,'upright_fit_exact':upright_fit_exact,
           'upright_mount_facts':upright_mount_facts,'upright_fit_review':upright_fit_review}
    if sys.argv[1] in extra:
        if sys.argv[1]=='build_original_axis':
            with bd.SkipClean():build_original_axis()
        else:extra[sys.argv[1]]()
        sys.exit()
    {'inspect':inspect,'search':search,'frame':frame,'diagnose':diagnose,'section':section,'exact':exact,'layout':layout,'layout_search':layout_search,'layout_verify':layout_verify,'layout_review':layout_review,'layout_routes':layout_routes,'guards':guards,'drawing':drawing,'combined':combined,'level':level,'level_diagnose':level_diagnose,'level_local':level_local,'publish':publish,'upright':upright,'upright_verify':upright_verify,'upright_probe':upright_probe,'upright_review':upright_review,'upright_local':upright_local,'upright_envelope':upright_envelope}[sys.argv[1]]()
