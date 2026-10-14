"""Local, subtractive motion relief; run through run_memory_limited.py.

CAD brief: millimetres, current STEP coordinates (+X forward). Preserve all
servo poses, assembly occurrences, colours, mounting cylinders and enclosing
shells. First inspect S03/S08's existing relief and the internal S12/S13/S14
contacts. Require 0.2 mm running clearance, intact mounting material and one
valid solid before installing any candidate into the single official STEP.
"""

import gc
import hashlib
import json
import math
import os
import shutil
import runpy
import sys
from pathlib import Path

import trimesh
import numpy as np
from OCP.OSD import OSD_Parallel, OSD_ThreadPool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from cadgen import build123d as bd, read_step
from full_motion_current import TARGET, MJCF, _leaves, measured_axes
from exact_motion_current import _overlap
from repair_motion_clearance_current import all_leaves, face_facts
from cadgen.validity import check_occurrence_shape
import refit_upper_servo_interfaces_local as refit

BASE_SHA256 = '0398653cb440736ace1acd58a0548ae62ca953c9fe0ce13dc5240a1301043d71'
WORK = Path('/tmp/minor-clearance')
MESH_TOLERANCE_MM=.01
VOLUME_AUDIT_TOLERANCE_MM3=.01
os.environ['CADGEN_JOBS']='1'
OSD_Parallel.SetUseOcctThreads_s(True)
OSD_ThreadPool.DefaultPool_s().Init(1)


def _mesh(shape,tolerance=MESH_TOLERANCE_MM):
    BRepTools.Clean_s(shape.wrapped)
    mesher=BRepMesh_IncrementalMesh(shape.wrapped,tolerance,False,.2,False)
    deflections=[]
    missing=[]
    for face in shape.faces():
        triangulation=BRep_Tool.Triangulation_s(face.wrapped,TopLoc_Location())
        if triangulation is None:
            missing.append((face.area,bounds(face)))
        else:
            deflections.append(triangulation.Deflection())
    print('mesh',shape.label,'done',mesher.IsDone(),'maximum deflection',max(deflections,default=0),'missing',missing,flush=True)
    assert mesher.IsDone() and not missing,(mesher.GetStatusFlags(),missing)
    actual=max(deflections)+1e-8
    assert BRepTools.Triangulation_s(shape.wrapped,max(tolerance,actual)*1.01)
    vertices,faces=shape.tessellate(max(tolerance,actual)*1.01,.2)
    mesh=trimesh.Trimesh(vertices=np.asarray([list(p) for p in vertices]),
                         faces=np.asarray(faces),process=False)
    mesh.metadata['maximum_deflection_mm']=actual
    return mesh


def relevant_mesh(mesh,other,coordinate,padding=.6):
    """Discard triangles separated by at least padding on an invariant axis."""
    values=mesh.triangles[:,:,coordinate]
    keep=(values.max(axis=1)>=other.bounds[0,coordinate]-padding)&(values.min(axis=1)<=other.bounds[1,coordinate]+padding)
    result=trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces[keep],process=False)
    result.remove_unreferenced_vertices()
    result.metadata.update(mesh.metadata)
    return result


def conservative_mesh_gap(manager,moving,fixed,error_bound):
    gaps=np.maximum(np.maximum(moving.bounds[0]-fixed.bounds[1],
                               fixed.bounds[0]-moving.bounds[1]),0)
    box_gap=float(np.linalg.norm(gaps))
    # Omitted triangles were at least 0.6 mm away. Cap the bound accordingly.
    measured=.6 if box_gap>.6 else min(.6,manager.min_distance_single(moving))
    return measured-error_bound


def detached(shape):
    result = bd.Compound(shape.solids())
    result.label, result.color = shape.label, shape.color
    return result


def volume(shape):
    # Audit spline trims with adaptive integration rather than default
    # quadrature, whose result depends on how the fillet is trimmed.
    props=GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped,props,Eps=1e-10,OnlyClosed=True,SkipShared=False)
    return props.Mass()


def bounds(shape):
    box = shape.bounding_box()
    return [list(box.min), list(box.max)]


def cylinder_signature(shape, mounting_only=False):
    return sorted((round(f.area, 5), *[round(x, 5) for x in f.center()],
                   *[round(x, 5) for row in bounds(f) for x in row])
                  for f in shape.faces() if f.geom_type == bd.GeomType.CYLINDER
                  and (not mounting_only or is_mounting_cylinder(f)))


def is_mounting_cylinder(face):
    cylinder=refit.BRepAdaptor_Surface(face.wrapped).Cylinder()
    if cylinder.Radius()>3.3:
        return False
    axis=cylinder.Axis()
    origin=bd.Vector(axis.Location().X(),axis.Location().Y(),axis.Location().Z())
    direction=bd.Vector(axis.Direction().X(),axis.Direction().Y(),axis.Direction().Z())
    delta=face.center()-origin
    radial=delta-direction*delta.dot(direction)
    return face.normal_at(face.center()).dot(radial)<0


def load_inspection():
    data = json.loads((WORK / 'inspection.json').read_text())
    assert data['source_sha256'] == BASE_SHA256
    return data


def cached(data, path):
    return detached(bd.import_brep(WORK / data['parts'][path]['file']))


def probe_mounts():
    data = load_inspection()
    report = []
    for row in data['pairs']:
        if row['joint'] not in ('S12', 'S13', 'S14'):
            continue
        moving, fixed = cached(data,row['moving']), cached(data,row['fixed'])
        axis = bd.Axis(*data['axes'][row['joint']])
        intersection = moving.rotate(axis,row['angle']).intersect(fixed)
        removed = bd.Compound(intersection.solids())
        target = moving if row['joint']=='S12' else fixed
        if row['joint']=='S12':
            removed = removed.rotate(axis,-row['angle'])
        mounts = []
        for face in target.faces():
            if face.geom_type != bd.GeomType.CYLINDER:
                continue
            radius = refit.BRepAdaptor_Surface(face.wrapped).Cylinder().Radius()
            if radius > 3.3:
                continue
            box=face.bounding_box()
            guard=refit._world_box(box.min.X-.55,box.max.X+.55,
                                   box.min.Y-.55,box.max.Y+.55,
                                   box.min.Z-.55,box.max.Z+.55)
            volume=_overlap(removed,guard)
            if volume>.001:
                mounts.append({'radius':radius,'center':list(face.center()),'guard_overlap_mm3':volume})
        result={'joint':row['joint'],'moving':row['moving'],'fixed':row['fixed'],
                'neutral_contact_bounds':bounds(removed),'mounts_at_risk':mounts}
        report.append(result)
        print(json.dumps(result),flush=True)
        del moving,fixed,removed,intersection
        gc.collect()
    (WORK/'mount-probe.json').write_text(json.dumps(report,indent=2)+'\n')


def protect_mounts(original, fresh, removed=None):
    assert cylinder_signature(original,True) == cylinder_signature(fresh,True), 'Mounting cylinder changed'
    if removed is None:
        removed = refit._shape(original.cut(fresh))
    for face in original.faces():
        if face.geom_type != bd.GeomType.CYLINDER:
            continue
        if not is_mounting_cylinder(face):
            continue
        box = face.bounding_box()
        guard = refit._world_box(box.min.X-.55, box.max.X+.55,
                                 box.min.Y-.55, box.max.Y+.55,
                                 box.min.Z-.55, box.max.Z+.55)
        assert _overlap(removed, guard) < .001, 'Mounting material changed'
    return removed


def cache_relief():
    data = load_inspection()
    report = {'source_sha256': BASE_SHA256, 'modified': {}}
    for row in data['pairs']:
        joint = row['joint']
        if joint not in ('S03', 'S08'):
            continue
        original = cached(data, row['moving'])
        fixed = cached(data, row['fixed'])
        # The existing sweep terminates in two planar relief faces. Move
        # those surfaces into the cover, preserving the original structure.
        relief=[]
        for face in original.faces():
            if face.geom_type != bd.GeomType.PLANE or face.area<1:
                continue
            center,normal=face.center(),face.normal_at()
            if abs(abs(center.Y)-117.425)<1e-5 and abs(normal.Y)>.999:
                relief.append(face)
            elif abs(normal.X-.9396926208)<1e-6 and abs(normal.Z+.3420201433)<1e-6:
                relief.append(face)
        assert len(relief)==2,[(f.area,list(f.center())) for f in relief]
        print(joint,'recessing existing relief faces by 0.35 mm and opening the existing pocket profile by 1.20 mm',flush=True)
        axis = bd.Axis(*data['axes'][joint])
        low, high = (-30, 70) if joint == 'S03' else (-70, 30)
        fresh = original
        tools=[]
        for face in relief:
            normal=face.normal_at()
            profile_expansion=1.2 if abs(normal.Y)>.999 else .5
            wire=face.outer_wire().offset_2d(profile_expansion,kind=bd.Kind.INTERSECTION)
            lead=(abs(face.center().Y)-93.75+.02) if abs(normal.Y)>.999 else .02
            expanded_face=bd.Face(wire).moved(bd.Pos(*(normal*lead)))
            tool=bd.extrude(expanded_face,amount=lead+.35,dir=-normal)
            tools.append(tool)
        cuts=len(relief)
        combined=refit._fuse(*tools)
        fresh = refit._cut(original,[combined])
        bd.export_brep(fresh,WORK/(joint+'-relief.brep'))
        validity = check_occurrence_shape(fresh.wrapped)
        assert validity['solidCount'] == 1 and not validity['reasons'], validity
        bd.export_brep(combined,WORK/(joint+'-tool.brep'))
        removed=refit._shape(original.intersect(combined))
        bd.export_brep(removed,WORK/(joint+'-removed.brep'))
        removed = protect_mounts(original, fresh, removed)
        removed_volume=volume(removed)
        original_volume,fresh_volume=volume(original),volume(fresh)
        volume_delta=original_volume-fresh_volume
        assert abs(removed_volume-volume_delta)<VOLUME_AUDIT_TOLERANCE_MM3, ('Unreliable subtraction',removed_volume,volume_delta)
        print(joint,'mounts retained; removed',removed_volume,'mm3; checking motion',flush=True)
        assert fresh_volume < original_volume, 'Expected a subtractive repair'
        # The enclosing dimensions and visible long cover faces stay in place.
        assert max(abs(a-b) for x,y in zip(bounds(original), bounds(fresh))
                   for a,b in zip(x,y)) < 1e-5, 'Exterior envelope changed'
        outer=lambda shape:sorted(round(f.area,5) for f in shape.faces()
                                   if f.geom_type==bd.GeomType.PLANE
                                   and abs(abs(f.center().Y)-133.75)<1e-5)
        assert outer(original)==outer(fresh),'Visible outer side face changed'
        minimum = float('inf')
        minimum_angle = None
        manager=trimesh.collision.CollisionManager()
        fixed_mesh=_mesh(fixed)
        mesh=_mesh(fresh)
        error_bound=fixed_mesh.metadata['maximum_deflection_mm']+mesh.metadata['maximum_deflection_mm']
        full_fixed,full_moving=fixed_mesh,mesh
        fixed_mesh=relevant_mesh(full_fixed,full_moving,1)
        mesh=relevant_mesh(full_moving,full_fixed,1)
        manager.add_object('fixed',fixed_mesh)
        for half_degree in range(low*2, high*2+1):
            angle = half_degree/2
            posed=mesh.copy()
            posed.apply_transform(trimesh.transformations.rotation_matrix(
                math.radians(angle),data['axes'][joint][1],data['axes'][joint][0]))
            # Both absolute-deflection meshes contribute to the uncertainty.
            distance=conservative_mesh_gap(manager,posed,fixed_mesh,error_bound)
            if distance < minimum:
                minimum, minimum_angle = distance, angle
            assert distance >= .2-1e-5, (joint, angle, distance)
            if half_degree%20==0:
                print(joint,'clearance checked through',angle,'minimum',minimum,flush=True)
        filename = joint + '-relief.brep'
        exact_overlap=_overlap(fresh.rotate(axis,minimum_angle),fixed)
        assert exact_overlap<.001,(joint,minimum_angle,exact_overlap)
        bd.export_brep(fresh, WORK / filename)
        removed_file=joint+'-removed.brep'
        bd.export_brep(removed,WORK/removed_file)
        report['modified'][row['moving']] = {
            'joint': joint, 'file': filename,'removed_file':removed_file,
            'file_sha256':hashlib.sha256((WORK/filename).read_bytes()).hexdigest(),
            'removed_mm3': removed_volume,
            'removed_fraction': removed_volume/original_volume, 'cuts': cuts,
            'pocket_profile_expansion_mm':1.2,'normal_recession_mm':.35,
            'volume_integration_relative_tolerance':1e-10,
            'volume_audit_tolerance_mm3':VOLUME_AUDIT_TOLERANCE_MM3,
            'volume_audit_difference_mm3':abs(removed_volume-volume_delta),
            'minimum_gap_lower_bound_mm': minimum, 'minimum_angle_deg': minimum_angle,
            'limits_deg': [low,high], 'verification_step_deg': .5,
            'mesh_absolute_tolerance_mm':MESH_TOLERANCE_MM,
            'mesh_pair_error_bound_mm':error_bound,
            'exact_sample_angle_deg':minimum_angle,'exact_sample_overlap_mm3':exact_overlap,
            'validity': validity, 'mounting_guard_mm': .55,
            'cylinders_preserved': True, 'bounds_preserved': True,
            'visible_outer_side_faces_preserved':True}
        (WORK / 'cache-report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(joint, report['modified'][row['moving']], flush=True)
        del original, fixed, fresh, removed
        gc.collect()


def speaker_corner():
    data=load_inspection()
    row=next(p for p in data['pairs'] if p['joint']=='S13' and 'speaker_stand' in p['moving'])
    original=cached(data,row['fixed'])
    moving=cached(data,row['moving'])
    # The stand's horizontal lip starts at Z=225.9009388. Lower only the
    # bracket's hidden rear upper edge to 225.65, with a sloping transition
    # back into its existing top plate, clear of all four mounting holes.
    bottom=225.65093881235
    outline=bd.Wire.make_polygon([(-10,-12,bottom),(-5.1,-12,bottom),
                                   (-3.1,-12,bottom+2),(-3.1,-12,230),
                                   (-10,-12,230)],close=True)
    tool=bd.extrude(bd.Face(outline),amount=25,dir=(0,1,0))
    fresh=refit._cut(original,[tool]).clean().fix()
    validity=check_occurrence_shape(fresh.wrapped)
    assert validity['solidCount']==1 and not validity['reasons'],validity
    removed=protect_mounts(original,fresh,refit._shape(original.intersect(tool)))
    assert abs(removed.volume-(original.volume-fresh.volume))<.001, 'Unreliable subtraction'
    print('S13 speaker corner; removed',removed.volume,'mm3; checking motion',flush=True)
    assert removed.volume/original.volume<.01
    minimum=float('inf')
    minimum_angle=None
    axis=bd.Axis(*data['axes']['S13'])
    manager=trimesh.collision.CollisionManager()
    fixed_mesh=_mesh(fresh)
    mesh=_mesh(moving)
    error_bound=fixed_mesh.metadata['maximum_deflection_mm']+mesh.metadata['maximum_deflection_mm']
    full_fixed,full_moving=fixed_mesh,mesh
    fixed_mesh=relevant_mesh(full_fixed,full_moving,2)
    mesh=relevant_mesh(full_moving,full_fixed,2)
    manager.add_object('modified bracket',fixed_mesh)
    for angle in range(-160,161):
        mesh_posed=mesh.copy()
        mesh_posed.apply_transform(trimesh.transformations.rotation_matrix(
            math.radians(angle),data['axes']['S13'][1],data['axes']['S13'][0]))
        distance=conservative_mesh_gap(manager,mesh_posed,fixed_mesh,error_bound)
        if distance<minimum:
            minimum,minimum_angle=distance,angle
        assert distance>=.2-1e-5,(angle,distance)
        if angle%30==0:
            print('S13 speaker checked through',angle,'minimum',minimum,flush=True)
    filename='S13-speaker-corner.brep'
    exact_overlap=_overlap(moving.rotate(axis,-90),fresh)
    assert exact_overlap<.001,exact_overlap
    bd.export_brep(fresh,WORK/filename)
    removed_file='S13-speaker-removed.brep'
    bd.export_brep(removed,WORK/removed_file)
    result={'source_sha256':BASE_SHA256,'modified':{row['fixed']:{
        'joint':'S13 speaker_stand pair only','file':filename,'removed_file':removed_file,
        'file_sha256':hashlib.sha256((WORK/filename).read_bytes()).hexdigest(),
        'removed_mm3':removed.volume,
        'removed_fraction':removed.volume/original.volume,'removed_bounds':bounds(removed),
        'minimum_gap_lower_bound_mm':minimum,'minimum_angle_deg':minimum_angle,
        'mesh_absolute_tolerance_mm':MESH_TOLERANCE_MM,
        'mesh_pair_error_bound_mm':error_bound,
        'exact_sample_angle_deg':-90,'exact_sample_overlap_mm3':exact_overlap,
        'limits_deg':[-160,160],'verification_step_deg':1,
        'validity':validity,'cylinders_preserved':True,'mounting_guard_mm':.55}}}
    (WORK/'speaker-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def probe_gap():
    data=load_inspection()
    row=next(p for p in data['pairs'] if p['joint']=='S03')
    fixed=cached(data,row['fixed'])
    fresh=detached(bd.import_brep(WORK/'S03-relief.brep'))
    axis=bd.Axis(*data['axes']['S03'])
    posed=fresh.rotate(axis,43.5)
    distance,cover_point,link_point=posed.distance_to_with_closest_points(fixed)
    neutral=bd.Vertex(*cover_point).rotate(axis,-43.5).center()
    result={'angle':43.5,'distance_mm':distance,'cover_point_world':list(cover_point),
            'cover_point_neutral':list(neutral),'link_point_world':list(link_point)}
    print(json.dumps(result),flush=True)
    (WORK/'gap-probe.json').write_text(json.dumps(result,indent=2)+'\n')


def export_candidate():
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==BASE_SHA256
    reports=[json.loads((WORK/name).read_text())
             for name in ('cache-report.json','speaker-report.json')]
    changes={path:entry for report in reports for path,entry in report['modified'].items()}
    assert len(changes)==3,changes
    robot=read_step(TARGET)
    before=all_leaves(robot)
    originals={item['path']:item['shape'] for item in _leaves(robot)}
    replacements={}
    for path,entry in changes.items():
        original=originals[path]
        assert hashlib.sha256((WORK/entry['file']).read_bytes()).hexdigest()==entry['file_sha256']
        fresh=detached(bd.import_brep(WORK/entry['file']))
        fresh.label,fresh.color=original.label,original.color
        assert fresh.is_valid and len(fresh.solids())==1
        removed=detached(bd.import_brep(WORK/entry['removed_file']))
        protect_mounts(detached(original),fresh,removed)
        assert abs(volume(original)-volume(fresh)-entry['removed_mm3'])<VOLUME_AUDIT_TOLERANCE_MM3
        replacements[id(original)]=fresh
    metadata=[]
    for node in before:
        expected=replacements.get(id(node),node)
        adaptive=id(node) in replacements
        metadata.append({'label':node.label,'color':list(node.color) if node.color else None,
                         'volume':volume(expected) if adaptive else expected.volume,
                         'adaptive_volume':adaptive,'faces':face_facts(expected)})
    def replace(node):
        if node.children:
            fresh=bd.Compound(children=[replace(child) for child in node.children],label=node.label)
            fresh.location=node.location
        else:
            fresh=detached(replacements.get(id(node),node))
            fresh.label=node.label
        fresh.color=node.color
        return fresh
    candidate=WORK/'review.step'
    bd.export_step(refit._encode_colours(replace(robot)),candidate)
    after=all_leaves(read_step(candidate))
    assert len(before)==len(after)
    for expected,actual in zip(metadata,after):
        assert actual.label==expected['label']
        actual_volume=volume(actual) if expected['adaptive_volume'] else actual.volume
        assert abs(actual_volume-expected['volume'])<.001
        if expected['color'] is not None:
            assert max(abs(a-b) for a,b in zip(actual.color,expected['color']))<1e-6
        facts=face_facts(actual)
        assert len(facts)==len(expected['faces']),actual.label
        for first,second in zip(facts,expected['faces']):
            assert first[0]==second[0],actual.label
            assert max(abs(a-b) for a,b in zip(first[1],second[1]))<.0001,actual.label
        if actual.label in ('right_cache','left_cache','head_pitch_to_yaw'):
            validity=check_occurrence_shape(actual.wrapped)
            assert validity['solidCount']==1 and not validity['reasons'],(actual.label,validity)
    report={'source_sha256':BASE_SHA256,'output_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
            'mjcf_sha256':hashlib.sha256(MJCF.read_bytes()).hexdigest(),
            'axes':load_inspection()['axes'],
            'clearance_method':'FCL mesh separation minus measured OCCT deflection bounds; 0.6 mm invariant-axis pruning; sampled angles only.',
            'candidate':str(candidate),'modified':changes,'export_leaf_count':len(after),
            'unchanged_leaf_count':len(after)-len(changes),
            'export_checks':'All occurrence labels, colours and per-face geometry match expected parts.',
            'mount_probe':json.loads((WORK/'mount-probe.json').read_text()),
            'scope':'Modified parts and direct interfaces only. Single-joint samples; no combined-motion certification.'}
    (WORK/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('export verified',len(after),'leaves',report['output_sha256'],flush=True)


def install():
    report=json.loads((WORK/'report.json').read_text())
    candidate=Path(report['candidate'])
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==report['source_sha256']
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==report['output_sha256']
    review=TARGET.parent/'MINOR_CLEARANCE_REVIEW.png'
    assert review.exists() and review.stat().st_mtime>=candidate.stat().st_mtime
    backup=WORK/'source.step'
    shutil.copyfile(TARGET,backup)
    temporary=TARGET.with_suffix('.step.tmp')
    shutil.copyfile(candidate,temporary)
    os.replace(temporary,TARGET)
    report['installed']=str(TARGET)
    report['visual_review_file']=str(review)
    (TARGET.parent/'MINOR_CLEARANCE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('installed',TARGET,report['output_sha256'],flush=True)


def snapshot():
    sys.argv=['cadgen','step','snapshot',str(TARGET),
              str(TARGET.parent/'MINOR_CLEARANCE_SNAPSHOT.png'),'--json']
    runpy.run_module('cadgen.cli',run_name='__main__')


def viewer():
    os.chdir(TARGET.parent)
    sys.argv=['cadgen.viewer','--host','127.0.0.1','--json',*sys.argv[2:]]
    runpy.run_module('cadgen.viewer',run_name='__main__')


def review_image():
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    data=load_inspection()
    report=json.loads((WORK/'report.json').read_text())
    fig=plt.figure(figsize=(14,11))
    def draw(ax,shape,color,alpha=1):
        mesh=_mesh(shape,.2)
        collection=Poly3DCollection(mesh.vertices[mesh.faces],facecolors=color,
                                    alpha=alpha,shade=True)
        collection.set_edgecolor('none')
        ax.add_collection3d(collection)
    def frame(ax,shape,title,azimuth):
        box=shape.bounding_box()
        center=np.array(list(box.center()))
        radius=max(box.size)*.56
        ax.set_xlim(center[0]-radius,center[0]+radius)
        ax.set_ylim(center[1]-radius,center[1]+radius)
        ax.set_zlim(center[2]-radius,center[2]+radius)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=30,azim=azimuth)
        ax.set_axis_off()
        ax.set_title(title,fontsize=11)
    for row,(path,entry) in enumerate(report['modified'].items()):
        original=cached(data,path)
        fresh=detached(bd.import_brep(WORK/entry['file']))
        removed=detached(bd.import_brep(WORK/entry['removed_file']))
        azimuth=60 if 'cache' in path else -130
        label=path.split('/')[-1]
        for column in range(3):
            ax=fig.add_subplot(3,3,row*3+column+1,projection='3d')
            if column==0:
                draw(ax,original,'#aeb7c3')
                frame(ax,original,label+' — before',azimuth)
            elif column==1:
                draw(ax,fresh,'#6b99ba')
                frame(ax,original,'After; mounts and exterior dimensions retained',azimuth)
            else:
                draw(ax,removed,'#f28b30')
                frame(ax,removed,f'Removed material: {volume(removed):.2f} mm³',azimuth)
        gc.collect()
    fig.suptitle('Local clearance repairs — orange shows actual removed material',fontsize=15)
    fig.tight_layout()
    output=TARGET.parent/'MINOR_CLEARANCE_REVIEW.png'
    fig.savefig(output,dpi=150,facecolor='white')
    print('review image',output,flush=True)


def finish_candidate():
    cache_relief()
    gc.collect()
    export_candidate()
    gc.collect()
    review_image()


def inspect():
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == BASE_SHA256
    WORK.mkdir(exist_ok=True)
    robot = read_step(TARGET)
    leaves = _leaves(robot)
    by_path = {row['path']: row['shape'] for row in leaves}
    axes = measured_axes(robot)
    exact = json.loads((TARGET.parent / 'FULL_MOTION_EXACT_REPORT.json').read_text())
    assert exact['target_sha256'] == BASE_SHA256
    report = {'source_sha256': BASE_SHA256, 'axes': axes, 'pairs': []}
    saved = set()
    for row in exact['checks']:
        if row['joint'] not in ('S03', 'S08', 'S12', 'S13', 'S14'):
            continue
        moving = detached(by_path[row['moving']])
        fixed = detached(by_path[row['fixed']])
        angles = row.get('collision_samples', []) or row['exact']
        worst = max(angles, key=lambda sample: sample['overlap_mm3'])
        axis = bd.Axis(*axes[row['joint']])
        posed = moving.rotate(axis, worst['angle_deg'])
        overlap = posed.intersect(fixed)
        pieces = list(overlap.solids()) if overlap is not None else []
        print(row['joint'], row['moving'], row['fixed'], flush=True)
        item = {'joint': row['joint'], 'moving': row['moving'], 'fixed': row['fixed'],
                'angle': worst['angle_deg'], 'overlap_mm3': _overlap(posed, fixed),
                'moving_volume': moving.volume, 'fixed_volume': fixed.volume,
                'moving_bounds': bounds(moving), 'fixed_bounds': bounds(fixed),
                'contact_bounds': [bounds(piece) for piece in pieces]}
        report['pairs'].append(item)
        for path, shape in ((row['moving'], moving), (row['fixed'], fixed)):
            if path in saved:
                continue
            filename = hashlib.sha256(path.encode()).hexdigest()[:12] + '.brep'
            bd.export_brep(shape, WORK / filename)
            report.setdefault('parts', {})[path] = {
                'file': filename, 'volume': shape.volume, 'bounds': bounds(shape),
                'cylinders': cylinder_signature(shape),
                'planes': [{'area': round(f.area, 3), 'center': list(f.center()),
                            'normal': list(f.normal_at()), 'bounds': bounds(f)}
                           for f in shape.faces() if f.geom_type == bd.GeomType.PLANE]}
            saved.add(path)
        del moving, fixed, posed, overlap, pieces
        gc.collect()
    (WORK / 'inspection.json').write_text(json.dumps(report, indent=2) + '\n')
    print('inspection saved', WORK / 'inspection.json', flush=True)


if __name__ == '__main__':
    {'inspect': inspect, 'cache': cache_relief, 'probe':probe_mounts,
     'speaker':speaker_corner,'export':export_candidate,'install':install,
     'snapshot':snapshot,'review':review_image,'viewer':viewer,
     'finish':finish_candidate,'gap':probe_gap}[sys.argv[1]]()
