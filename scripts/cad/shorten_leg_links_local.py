"""Shorten the four original leg links by 10 mm; keep all end interfaces.

Run each stage through run_memory_limited.py. Purchased parts are translated,
never scaled. Candidates remain in /tmp until local checks permit publication.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

from cadgen import build123d as bd, read_step
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
import numpy as np

from full_motion_current import _leaves, measured_axes, is_moving, LIMITS
from validate_s01_s10_interfaces_local import _cylinder_lines
import refit_upper_servo_interfaces_local as refit
from refit_case_bracket_width_local import export_difference

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT/'cad/local_fix/open_duck_mini_upper_refit.step'
WORK = Path('/tmp/open-duck-leg-shorten')
MANIFEST = WORK/'manifest.json'
REPORT = ROOT/'cad/local_fix/LEG_LINK_SHORTENING_REPORT.json'
SHORTEN_MM = 10.0
MIDDLE_MOUNT_SHIFT_MM = 4.0
SHEETS = {'left_knee_to_ankle_left_sheet', 'left_knee_to_ankle_right_sheet'}
THIGH_ROOTS = {'knee_and_ankle_assembly <1>', 'knee_and_ankle_assembly <2>'}
SHANK_ROOTS = {'knee_and_ankle_assembly <3>', 'knee_and_ankle_assembly <4>'}
FOOT_ROOTS = {'foot_assembly <1>', 'foot_assembly <2>'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(data):
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def load():
    return json.loads(MANIFEST.read_text())


def bounds(shape):
    box = shape.bounding_box()
    return [list(box.min), list(box.max)]


def volume(shape):
    if shape is None:
        return 0.
    if isinstance(shape, (list, tuple, bd.ShapeList)):
        return sum(volume(s) for s in shape)
    return sum(s.volume for s in shape.solids())


def cylinders(shape):
    rows = []
    for face in shape.faces():
        surface = BRepAdaptor_Surface(face.wrapped, True)
        if surface.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = surface.Cylinder()
        rows.append(dict(radius_mm=cylinder.Radius(),
                         origin=list(cylinder.Location().Coord()),
                         direction=list(cylinder.Axis().Direction().Coord()),
                         bounds=bounds(face), area_mm2=face.area))
    return rows


def clip(shape, low, high):
    box = shape.bounding_box()
    return refit._shape(shape.intersect(refit._world_box(
        box.min.X-2, box.max.X+2, box.min.Y-2, box.max.Y+2, low, high)))


def classification(row):
    if row['root'] in THIGH_ROOTS:
        if row['label']=='leg_spacer':
            return 'translate',MIDDLE_MOUNT_SHIFT_MM
        return ('shorten_thigh' if row['label'] in SHEETS else 'translate'), SHORTEN_MM
    if row['root'] in SHANK_ROOTS:
        if row['label']=='leg_spacer':
            return 'translate',SHORTEN_MM+MIDDLE_MOUNT_SHIFT_MM
        return ('shorten_shank' if row['label'] in SHEETS else 'translate'), 2*SHORTEN_MM
    if row['root'] in FOOT_ROOTS:
        return 'translate', 2*SHORTEN_MM
    if row['label'] in ('left_cache', 'right_cache'):
        return 'shorten_cover', SHORTEN_MM
    return 'unchanged', 0.


def prepare():
    WORK.mkdir(exist_ok=True)
    (WORK/'parts').mkdir(exist_ok=True)
    if MANIFEST.exists():
        assert load()['source_sha256'] == digest(TARGET), 'Fixed input changed'
        print('Fixed input already prepared', flush=True)
        return
    source = WORK/'input.step'
    shutil.copyfile(TARGET, source)
    robot = read_step(source)
    axes = measured_axes(robot)
    parts = []
    for index, leaf in enumerate(_leaves(robot)):
        shape = leaf['shape']
        mode, dz = classification(leaf)
        path = WORK/'parts'/f'{index:03d}_original.step'
        # Local checking uses original part documents; unrelated geometry is
        # still read unchanged from the pinned full assembly at assembly time.
        local = mode != 'unchanged' or ('S01' in leaf['servo'] if leaf['servo'] else False)
        local |= any(word in leaf['root'] for word in ('roll_to_pitch', 'hip_roll', 'trunk_assembly'))
        if local:
            exported = bd.Solid(shape.solids()[0].wrapped)
            exported.label, exported.color = shape.label, shape.color
            bd.export_step(exported, path)
        row = {k:v for k,v in leaf.items() if k != 'shape'}
        row.update(index=index, original=str(path), mode=mode, shift_z_mm=dz,
                   bounds=bounds(shape), volume_mm3=shape.volume,
                   face_count=len(shape.faces()), solid_count=len(shape.solids()),
                   rgba=list(shape.color) if shape.color else None)
        if mode.startswith('shorten') or row['label'] == 'leg_spacer':
            row['cylinders'] = cylinders(shape)
        parts.append(row)
        if mode != 'unchanged':
            print(index, mode, row['path'], flush=True)
    data = dict(source=str(source), source_sha256=digest(source), parts=parts,
                source_axes=axes, shortened_by_mm=SHORTEN_MM, stage='prepared')
    assert digest(TARGET) == data['source_sha256']
    save(data)
    print('Prepared', len(parts), 'physical parts;',
          sum(p['mode'] != 'unchanged' for p in parts), 'affected', flush=True)


def inspect_sections(index):
    row = load()['parts'][index]
    shape = read_step(row['original'])
    joint_z = 0. if row['mode'] != 'shorten_shank' else -78.65
    for z in [-42.,-44.,-45.,-46.,-48.,-50.,-52.,-54.,-55.,-56.,-58.,-60.]:
        z += joint_z
        try:
            section = clip(shape, z-.005, z+.005)
            print(round(z, 4), 'area', round(section.volume/.01, 5),
                  'solids', len(section.solids()), 'bounds', bounds(section), flush=True)
        except ValueError:
            print(z, 'empty', flush=True)
    print('cylinders', json.dumps([
        dict(radius=round(c['radius_mm'],4), origin=[round(v,4) for v in c['origin']],
             z=[round(c['bounds'][i][2],4) for i in (0,1)])
        for c in row.get('cylinders', []) if c['radius_mm'] < 4]), flush=True)


def candidate_file(row):
    return WORK/'parts'/f'{row["index"]:03d}_candidate.step'


def align_cover_bores(original, shortened, data, row):
    root = 'knee_and_ankle_assembly <2>' if row['label']=='left_cache' else 'knee_and_ankle_assembly <1>'
    spacer = next(p for p in data['parts'] if p['root']==root and p['label']=='leg_spacer')
    host = read_step(candidate_file(spacer))
    lines = _cylinder_lines(host,1.,bd.Vector(0,1,0))
    assert len(lines)==2
    old_holes = [c for c in cylinders(shortened) if min(abs(c['radius_mm']-r) for r in (1.75,3.25))<1e-4]
    assert len(old_holes)==4
    result = shortened
    for c in old_holes:
        a,b = c['bounds']
        origin = bd.Vector(c['origin'][0],0,c['origin'][2])
        result = refit._fuse(result,refit._rod(origin,bd.Vector(0,1,0),a[1],b[1],c['radius_mm']))
    for c in old_holes:
        p = min(lines,key=lambda line:abs(line[0].X-c['origin'][0]))[0]
        a,b = c['bounds']
        origin = bd.Vector(p.X,0,p.Z)
        result = refit._cut(result,[refit._rod(origin,bd.Vector(0,1,0),a[1]-1e-5,b[1]+1e-5,c['radius_mm'])])
    result = result.clean()
    assert result.is_valid and len(result.solids())==1
    new_lines = _cylinder_lines(result,1.75,bd.Vector(0,1,0))
    assert len(new_lines)==2
    errors = [min(np.linalg.norm(np.array([p.X,p.Z])-np.array([q.X,q.Z])) for q,_ in new_lines) for p,_ in lines]
    assert max(errors)<1e-5
    checks = dict(old_hole_pitch_mm=16.,new_hole_pitch_mm=abs(lines[0][0].X-lines[1][0].X),
                  axis_errors_mm=errors,through_radius_mm=1.75,counterbore_radius_mm=3.25,
                  end_preservation_scope='End regions preserved before the two existing bores are aligned to their current 13mm host pattern.')
    return result,checks


def shorten(row, original, data):
    mode=row['mode']
    if mode=='shorten_cover':
        # Keep the upper hip clearance profile and both original mounting
        # regions. Keep the middle-mount shift small enough for hip running clearance.
        spans=[(-33.,-23.,MIDDLE_MOUNT_SHIFT_MM),(-62.,-42.,SHORTEN_MM-MIDDLE_MOUNT_SHIFT_MM)]
    else:
        sid = ('S04' if row['root'].endswith('<3>') else 'S09') if mode=='shorten_shank' else (
            'S03' if row['root'].endswith('<1>') else 'S08')
        z=data['source_axes'][sid][0][2]
        spans=[(z-26.,z-26.+MIDDLE_MOUNT_SHIFT_MM,MIDDLE_MOUNT_SHIFT_MM),
               (z-49.1,z-42.2,SHORTEN_MM-MIDDLE_MOUNT_SHIFT_MM)]
    for low,high,removed in spans:
        for cylinder in row['cylinders']:
            if cylinder['radius_mm']>4 or abs(cylinder['direction'][2])>.01:
                continue
            assert cylinder['bounds'][1][2]<=low or cylinder['bounds'][0][2]>=high, ('Shortening touches a bore',low,high,cylinder)
        if mode!='shorten_cover' and abs(high-low-removed)<1e-5:
            a=clip(original,high,high+.01)
            b=clip(original,low,low+.01).moved(bd.Pos(0,0,removed))
            error=max(export_difference(a,b),export_difference(b,a))
            assert error<.001, ('Nonuniform joining span',low,high,error)
    zmin,zmax=row['bounds'][0][2]-2,row['bounds'][1][2]+2
    pieces=[];kept=[];previous=zmax;shift=0.
    for low,high,removed in spans:
        region=clip(original,high,previous).moved(bd.Pos(0,0,shift))
        pieces.append(region);kept.append((region,high+shift+.001,previous+shift-.001))
        if high-low>removed+1e-6:
            middle=clip(original,low,high).scale((1,1,(high-low-removed)/(high-low)),about=(0,0,low))
            pieces.append(middle.moved(bd.Pos(0,0,shift+removed)))
        shift+=removed;previous=low
    lower=clip(original,zmin,previous).moved(bd.Pos(0,0,shift))
    pieces.append(lower);kept.append((lower,zmin+shift,previous+shift-.001))
    result=refit._shape(pieces[0].fuse(*pieces[1:],tol=1e-6)).clean()
    assert result.is_valid and len(result.solids())==1 and result.volume>0, 'Invalid shortened solid'
    errors=[]
    for expected,low,high in kept:
        actual=clip(result,low,high);expected=clip(expected,low,high)
        errors.append(max(export_difference(actual,expected),export_difference(expected,actual)))
    assert max(errors)<.001, 'Fixed mounting region changed'
    inherited=SHORTEN_MM if mode=='shorten_shank' else 0.
    result=result.moved(bd.Pos(0,0,inherited))
    return result,dict(original_shortening_spans_z_mm=spans,
                       preserved_mounting_region_difference_mm3=errors,
                       middle_mount_shift_mm=MIDDLE_MOUNT_SHIFT_MM,inherited_shift_z_mm=inherited,
                       valid=True,solid_count=1)


def build(indices=None):
    data = load()
    assert digest(TARGET) == data['source_sha256'], 'Fixed input changed'
    # Cover bores depend on the completed original spacer translations.
    ordered = sorted(data['parts'],key=lambda p:p['mode']=='shorten_cover')
    for row in ordered:
        if row['mode'] == 'unchanged' or (indices is not None and row['index'] not in indices):
            continue
        row['mode'],row['shift_z_mm']=classification(row)
        original = read_step(row['original'])
        if row['mode'] == 'translate':
            result = original.moved(bd.Pos(0,0,row['shift_z_mm']))
            checks = dict(rigid_translation_z_mm=row['shift_z_mm'],valid=result.is_valid,solid_count=len(result.solids()))
        else:
            result, checks = shorten(row,original,data)
            if row['mode']=='shorten_cover':
                result, bore_checks = align_cover_bores(original,result,data,row)
                checks['cover_bores'] = bore_checks
        exported = bd.Solid(result.solids()[0].wrapped)
        exported.label = row['label']
        if row['rgba'] is not None:
            exported.color = bd.Color(*row['rgba'])
        output = candidate_file(row)
        bd.export_step(exported,output)
        row['candidate'] = str(output)
        row['candidate_sha256'] = digest(output)
        row['build_checks'] = checks | dict(bounds=bounds(result),volume_mm3=result.volume)
        save(data)
        print('Built',row['index'],row['mode'],'volume',round(result.volume,3),flush=True)
        del original,result,exported
        gc.collect()
    data['stage'] = 'built' if all('candidate' in p for p in data['parts'] if p['mode']!='unchanged') else 'partially_built'
    save(data)


def assemble():
    data = load()
    assert data['stage']=='built' and digest(TARGET)==data['source_sha256']
    robot = read_step(data['source'])
    by_path = {p['path']:p for p in data['parts']}
    count = 0
    def rebuild(node,path=''):
        nonlocal count
        children = refit._children(node)
        if children:
            fresh = bd.Compound(children=[rebuild(child,path+f'/{child.label}[{i}]')
                                          for i,child in enumerate(children)],label=node.label)
            fresh.location,fresh.color = node.location,node.color
            return fresh
        row = by_path.get(path)
        if row is None or row['mode']=='unchanged':
            return node
        assert digest(row['candidate'])==row['candidate_sha256']
        part = read_step(row['candidate'])
        fresh = bd.Solid(part.solids()[0].wrapped)
        fresh.label,fresh.color = node.label,node.color
        count += 1
        return fresh
    result = refit._encode_colours(rebuild(robot))
    assert count==34
    candidate = WORK/'candidate.step'
    bd.export_step(result,candidate)
    del result,robot
    gc.collect()
    exported = read_step(candidate)
    leaves = _leaves(exported)
    assert len(leaves)==len(data['parts'])==115
    for leaf,row in zip(leaves,data['parts']):
        assert leaf['path']==row['path'] and leaf['label']==row['label'] and leaf['servo']==row['servo']
        shape = leaf['shape']
        expected = row.get('build_checks',{}).get('bounds',row['bounds'])
        assert np.max(np.abs(np.array(bounds(shape))-expected))<1e-4, row['path']
        assert abs(shape.volume-row.get('build_checks',{}).get('volume_mm3',row['volume_mm3']))<.001
        assert len(shape.solids())==row['solid_count']==1
        assert (shape.color is None)==(row['rgba'] is None)
        if shape.color:
            assert max(abs(a-b) for a,b in zip(shape.color,row['rgba']))<1e-5
    axes = measured_axes(exported)
    lengths = []
    for a,b in [('S03','S04'),('S04','S05'),('S08','S09'),('S09','S10')]:
        old = np.linalg.norm(np.array(data['source_axes'][a][0])-data['source_axes'][b][0])
        new = np.linalg.norm(np.array(axes[a][0])-axes[b][0])
        # Axial datum offsets of 0.1/0.125mm are along parallel hinge axes;
        # report both the Euclidean datum spacing and the true axis spacing.
        direction = np.array(axes[a][1])
        delta = np.array(axes[b][0])-axes[a][0]
        distance = np.linalg.norm(np.cross(delta,direction))
        assert abs(distance-68.65)<1e-5
        lengths.append(dict(joints=[a,b],old_datum_distance_mm=float(old),new_datum_distance_mm=float(new),
                            new_axis_distance_mm=float(distance)))
    for sid,(origin,direction) in axes.items():
        offset = 10 if sid in ('S04','S09') else 20 if sid in ('S05','S10') else 0
        expected = np.array(data['source_axes'][sid][0])+[0,0,offset]
        assert np.linalg.norm(np.array(origin)-expected)<1e-5
        assert np.linalg.norm(np.array(direction)-data['source_axes'][sid][1])<1e-8
    data.update(candidate=str(candidate),candidate_sha256=digest(candidate),candidate_axes=axes,
                axis_lengths=lengths,export_checks=dict(physical_parts=115,affected_parts=34,
                reshaped_parts=10,labels_and_colors_preserved=True,unaffected_bounds_and_volume_preserved=True),stage='assembled')
    save(data)
    print('Assembly export and four 68.65mm axis distances verified',flush=True)


def check():
    """Compare only the changed legs and their immediate hip interfaces."""
    import itertools
    import math
    import trimesh
    from exact_motion_current import _overlap

    data = load()
    assert data['stage']=='assembled' and digest(data['candidate'])==data['candidate_sha256']
    local = [p for p in data['parts'] if p['mode']!='unchanged' or p['root'] in
             ('right_roll_to_pitch_assembly <1>','left_roll_to_pitch_assembly <1>')]
    shapes, meshes, managers = {}, {}, {}
    for row in local:
        for version in ('original','candidate'):
            path = row['original'] if version=='original' else row.get('candidate',row['original'])
            shape = read_step(path)
            vertices,faces = shape.tessellate(.15,.2)
            shapes[version,row['index']] = shape
            meshes[version,row['index']] = trimesh.Trimesh(
                np.array([tuple(v) for v in vertices]),np.array(faces),process=False)
            manager=trimesh.collision.CollisionManager()
            manager.add_object(str(row['index']),meshes[version,row['index']])
            managers[version,row['index']]=manager
    affected = {p['index'] for p in local if p['mode']!='unchanged'}
    report = dict(source_sha256=data['source_sha256'],candidate_sha256=data['candidate_sha256'],
                  scope='Changed legs and immediate hip interfaces only; neutral static pose and six independent pitch-joint sweeps at 5 degree steps, other joints neutral. Not a combined-pose certificate.',
                  clearance_mm=.2,mesh_deflection_mm=.15,static=[],motion=[],new_interferences=[])

    def exact(a,b,version,sid=None,angle=0):
        left,right = shapes[version,a['index']],shapes[version,b['index']]
        if sid:
            axes = data['source_axes'] if version=='original' else data['candidate_axes']
            axis = bd.Axis(*axes[sid])
            if is_moving(a,sid):
                left = left.rotate(axis,angle)
            if is_moving(b,sid):
                right = right.rotate(axis,angle)
        gap = left.distance_to(right)
        return dict(gap_mm=gap,overlap_mm3=_overlap(left,right) if gap<1e-5 else 0.)

    def record():
        (WORK/'local_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

    def side(row):
        return 1 if np.mean(np.array(row['bounds'])[:,1])>0 else -1

    for a,b in itertools.combinations(local,2):
        if not ({a['index'],b['index']}&affected) or side(a)!=side(b):
            continue
        # Internal purchased-servo and foot contacts are preserved rigidly;
        # the build/export and centered-geometry checks already cover them.
        if a['mode']==b['mode']=='translate' and a['shift_z_mm']==b['shift_z_mm']:
            continue
        boxes=[np.array(bounds(shapes['candidate',p['index']])) for p in (a,b)]
        if np.max(np.maximum(boxes[0][0]-boxes[1][1],boxes[1][0]-boxes[0][1]))>.21:
            continue
        before,after=exact(a,b,'original'),exact(a,b,'candidate')
        row=dict(parts=[a['index'],b['index']],before=before,after=after)
        report['static'].append(row)
        if after['overlap_mm3']>max(1e-4,before['overlap_mm3']+.001):
            report['new_interferences'].append(dict(kind='static',**row))
        print('Static',row['parts'],'gap',round(after['gap_mm'],4),'overlap',round(after['overlap_mm3'],5),flush=True)
        record()
    for sid in ('S03','S04','S05','S08','S09','S10'):
        rows=[p for p in local if side(p)==(-1 if int(sid[1:])<6 else 1)]
        angles=sorted(set(range(LIMITS[sid][0],LIMITS[sid][1]+1,5))|{0,LIMITS[sid][1]})
        pairs={}
        for angle in angles:
            for version in ('original','candidate'):
                axes=data['source_axes'] if version=='original' else data['candidate_axes']
                origin,direction=axes[sid]
                transform=trimesh.transformations.rotation_matrix(math.radians(angle),direction,origin)
                for row in rows:
                    managers[version,row['index']].set_transform(str(row['index']),
                        transform if is_moving(row,sid) else np.eye(4))
            for a in rows:
                if not is_moving(a,sid):
                    continue
                for b in rows:
                    if is_moving(b,sid) or not ({a['index'],b['index']}&affected):
                        continue
                    gaps={}
                    for version in ('original','candidate'):
                        gaps[version]=float(managers[version,b['index']].min_distance_other(
                            managers[version,a['index']]))
                    if gaps['candidate']<.4:
                        key=(a['index'],b['index'])
                        pair=pairs.setdefault(key,dict(parts=list(key),near_samples=[]))
                        pair['near_samples'].append(dict(angle_deg=angle,mesh_gap_mm=gaps))
        sweep=dict(joint=sid,sampled_angles_deg=angles,near_pairs=[])
        for (i,j),pair in pairs.items():
            a=next(p for p in rows if p['index']==i);b=next(p for p in rows if p['index']==j)
            # Exact BRep checks at the worst candidate gap and every sample
            # whose broad phase changed from clear to near-contact.
            selected={min(pair['near_samples'],key=lambda s:s['mesh_gap_mm']['candidate'])['angle_deg']}
            selected.update(s['angle_deg'] for s in pair['near_samples']
                            if s['mesh_gap_mm']['original']>.35 and s['mesh_gap_mm']['candidate']<.35)
            pair['exact_samples']=[]
            for angle in sorted(selected):
                before,after=exact(a,b,'original',sid,angle),exact(a,b,'candidate',sid,angle)
                sample=dict(angle_deg=angle,before=before,after=after)
                pair['exact_samples'].append(sample)
                if (after['overlap_mm3']>max(1e-4,before['overlap_mm3']+.001) or
                    (before['gap_mm']>=.2 and after['gap_mm']<.2-1e-5)):
                    report['new_interferences'].append(dict(kind='motion',joint=sid,parts=[i,j],**sample))
            sweep['near_pairs'].append(pair)
        report['motion'].append(sweep)
        print('Sweep',sid,len(angles),'angles;',len(pairs),'near pairs;',
              len(report['new_interferences']),'new issues',flush=True)
        record()
    report['cover_hip_refinement']=[]
    for sid,i,j in [('S03',3,38),('S08',0,105)]:
        a=next(p for p in local if p['index']==i);b=next(p for p in local if p['index']==j)
        origin,direction=data['candidate_axes'][sid]
        samples=[]
        managers['candidate',j].set_transform(str(j),np.eye(4))
        for angle in range(LIMITS[sid][0],LIMITS[sid][1]+1):
            transform=trimesh.transformations.rotation_matrix(math.radians(angle),direction,origin)
            managers['candidate',i].set_transform(str(i),transform)
            gap=managers['candidate',j].min_distance_other(managers['candidate',i])
            samples.append(dict(angle_deg=angle,mesh_gap_mm=float(gap)))
        worst=min(samples,key=lambda s:s['mesh_gap_mm'])
        actual=exact(a,b,'candidate',sid,worst['angle_deg'])
        refined=dict(joint=sid,parts=[i,j],step_deg=1,samples=samples,
                     minimum_mesh_sample=worst,exact_at_minimum=actual)
        report['cover_hip_refinement'].append(refined)
        if actual['gap_mm']<.2-1e-7 or actual['overlap_mm3']>1e-4:
            report['new_interferences'].append(dict(kind='hip_refinement',**refined))
        print('Hip cover',sid,'1deg sampled minimum',round(actual['gap_mm'],4),'mm at',worst['angle_deg'],flush=True)
        record()
    # Keep the user's home joint angles and check the resulting shortened-leg
    # geometry together. A single-joint sweep does not cover this bent pose.
    from full_motion_current import JOINT_NAMES
    home=json.loads((TARGET.parent/'open_duck_mini_zp10d_mass_properties.json').read_text())['home_joint_angles_rad']
    transforms={}
    for version in ('original','candidate'):
        axes=data['source_axes'] if version=='original' else data['candidate_axes']
        for row in local:
            matrix=np.eye(4)
            for sid,name in JOINT_NAMES.items():
                if is_moving(row,sid):
                    origin,direction=axes[sid]
                    matrix=matrix@trimesh.transformations.rotation_matrix(home[name],direction,origin)
            transforms[version,row['index']]=matrix
            managers[version,row['index']].set_transform(str(row['index']),matrix)
    report['home_pose']={'angles_rad':home,'near_pairs':[]}
    for a,b in itertools.combinations(local,2):
        if not ({a['index'],b['index']}&affected):
            continue
        chain_a=[sid for sid in JOINT_NAMES if is_moving(a,sid)]
        chain_b=[sid for sid in JOINT_NAMES if is_moving(b,sid)]
        if chain_a==chain_b or (a['servo'] and a['servo']==b['servo']):
            continue
        gap=float(managers['candidate',a['index']].min_distance_other(managers['candidate',b['index']]))
        if gap>=.4:
            continue
        sample=dict(parts=[a['index'],b['index']],mesh_gap_mm=gap)
        for version in ('original','candidate'):
            posed=[]
            for p in (a,b):
                matrix=transforms[version,p['index']]
                plane=bd.Plane(origin=tuple(matrix[:3,3]),x_dir=tuple(matrix[:3,0]),z_dir=tuple(matrix[:3,2]))
                posed.append(shapes[version,p['index']].moved(bd.Location(plane)))
            gap_exact=posed[0].distance_to(posed[1])
            sample[version]=dict(gap_mm=gap_exact,overlap_mm3=_overlap(*posed) if gap_exact<1e-5 else 0.)
        before,after=sample['original'],sample['candidate']
        if (after['overlap_mm3']>max(1e-4,before['overlap_mm3']+.001) or
            (before['gap_mm']>=.2 and after['gap_mm']<.2-1e-5)):
            report['new_interferences'].append(dict(kind='combined_home',**sample))
        report['home_pose']['near_pairs'].append(sample)
    print('Combined home: checked',len(report['home_pose']['near_pairs']),'near pairs',flush=True)
    report['passed']=not report['new_interferences']
    record()
    assert report['passed'], 'New leg interface interference; see /tmp/open-duck-leg-shorten/local_checks.json'
    data['local_checks']=report
    data['stage']='locally_checked'
    save(data)
    print('PASS: local static interfaces and six sampled joint sweeps',flush=True)


def render():
    """Fallback review from the actual candidate BRep, if Chromium fails."""
    import os
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    data=load()
    robot=read_step(data['candidate'])
    triangles=[];colors=[];indices=[]
    for index,row in enumerate(_leaves(robot)):
        shape=row['shape']
        try:
            vertices,faces=shape.tessellate(.5,.35)
        except AttributeError:
            from OCP.BRepTools import BRepTools
            print('Remeshing finer',index,row['label'],flush=True)
            BRepTools.Clean_s(shape.wrapped)
            vertices,faces=shape.tessellate(.35,.3)
        v=np.array([tuple(p) for p in vertices]);f=np.array(faces)
        triangles.extend(v[f]);indices.extend([index]*len(f))
        rgba=list(shape.color) if shape.color else [.65,.68,.72,1]
        colors.extend([rgba[:3]]*len(f))
    triangles=np.array(triangles);colors=np.array(colors);indices=np.array(indices)
    fig,axes=plt.subplots(2,2,figsize=(15,12))
    directions=[([1,-1,.6],'Whole assembly / isometric',False),
                ([0,-1,0],'Shortened legs / front',True),
                ([1,0,0],'Shortened legs / side',True),
                ([-1,1,-.6],'Shortened legs / opposite isometric',True)]
    affected={p['index'] for p in data['parts'] if p['mode']!='unchanged'}
    local=affected|{38,39,40,41,105,106,107,108}
    light=np.array([.4,-.5,1]);light/=np.linalg.norm(light)
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    norm=np.linalg.norm(normal,axis=1);normal/=np.maximum(norm[:,None],1e-15)
    shade=.45+.55*np.abs(normal@light)
    for ax,(direction,title,legs) in zip(axes.flat,directions):
        d=np.array(direction,dtype=float);d/=np.linalg.norm(d)
        horizontal=np.cross(d,[0,0,1]);horizontal/=np.linalg.norm(horizontal)
        vertical=np.cross(horizontal,d)
        selected=np.isin(indices,list(local)) if legs else np.ones(len(triangles),dtype=bool)
        t=triangles[selected];color=colors[selected]*shade[selected,None]
        if legs:
            changed=np.array([i in affected for i in indices[selected]])
            color[changed]=np.array([.95,.58,.18])*shade[selected][changed,None]
        depth=t.mean(1)@d;order=np.argsort(depth)
        points=np.stack([t@horizontal,t@vertical],axis=-1)
        ax.add_collection(PolyCollection(points[order],facecolors=color[order],edgecolors='none'))
        ax.autoscale();ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('mm');ax.set_ylabel('Z [mm]')
        ax.set_facecolor('#f7f8fa')
    fig.suptitle('Leg links: 78.65 → 68.65 mm each / orange = affected leg components')
    fig.tight_layout()
    output=TARGET.parent/'LEG_LINK_SHORTENING_REVIEW.png'
    fig.savefig(output,dpi=130)
    data['review_png']=str(output);data['review_png_sha256']=digest(output)
    save(data)
    print('Saved',output,flush=True)


def animation():
    """Update the existing replay pivots and remeasure its two-foot contact."""
    import math
    import re
    import trimesh
    from exact_motion_current import _overlap
    data=load()
    path=TARGET.with_suffix('.step.js')
    text=path.read_text()
    match=re.search(r'const data = (.*);',text)
    payload=json.loads(match[1])
    assert payload['target_sha256']==data['source_sha256']
    payload['target_sha256']=data['candidate_sha256']
    payload['axes'].update(data['candidate_axes'])
    feet=next(s for s in payload['scenarios'] if s['id']=='feet')
    rows=[data['parts'][i] for i in (44,85)]
    shapes=[bd.Compound(read_step(candidate_file(p)).solids()) for p in rows]
    samples=[]
    def overlap(progress):
        posed=[]
        for shape,row in zip(shapes,rows):
            matrix=np.eye(4)
            # Compose by source kinematic order, independent of JSON order.
            for sid in (f'S{i:02d}' for i in range(1,15)):
                if not is_moving(row,sid):
                    continue
                origin,direction=data['candidate_axes'][sid]
                angle=feet['start_pose_deg'][sid]+progress*(feet['end_pose_deg'][sid]-feet['start_pose_deg'][sid])
                matrix=matrix@trimesh.transformations.rotation_matrix(math.radians(angle),direction,origin)
            plane=bd.Plane(origin=tuple(matrix[:3,3]),x_dir=tuple(matrix[:3,0]),z_dir=tuple(matrix[:3,2]))
            posed.append(shape.moved(bd.Location(plane)))
        gap=posed[0].distance_to(posed[1])
        value=_overlap(*posed) if gap<1e-5 else 0.
        samples.append(dict(progress=progress,gap_mm=gap,overlap_mm3=value))
        return value
    previous=0.
    found=None
    for progress in np.linspace(0,1,21):
        value=overlap(float(progress))
        if value>feet['volume_threshold_mm3']:
            found=[previous,float(progress)]
            break
        previous=float(progress)
    if found:
        for _ in range(12):
            middle=sum(found)*.5
            found[1 if overlap(middle)>feet['volume_threshold_mm3'] else 0]=middle
    else:
        feet['label']='4 缩腿后双脚极限组合复核（未碰撞）'
    feet['collision_detected']=found is not None
    feet['contact_progress']=found[1] if found else 1.
    feet['contact_bracket_progress']=found
    feet['remeasured_for_step_sha256']=data['candidate_sha256']
    text=text[:match.start(1)]+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+text[match.end(1):]
    (WORK/'candidate.step.js').write_text(text)
    data['animation_checks']=dict(updated_axes=['S04','S05','S09','S10'],feet_contact_bracket_progress=found,
                                   feet_collision_detected=found is not None,
                                   feet_minimum_sampled_gap_mm=min(s['gap_mm'] for s in samples),
                                   feet_clip_samples=samples,head_pair_geometry_and_axes_preserved=True)
    save(data)
    print('New feet replay contact bracket',found,'minimum sampled gap',data['animation_checks']['feet_minimum_sampled_gap_mm'],flush=True)


def hip_probe():
    """Fast exact check at the previously found narrow-angle hip contact."""
    data=load()
    cover=read_step(candidate_file(data['parts'][3]))
    hip=read_step(data['parts'][38]['original'])
    axis=bd.Axis(*data['source_axes']['S03'])
    samples=[]
    for angle in range(-22,-14):
        gap=cover.rotate(axis,angle).distance_to(hip)
        samples.append(dict(angle_deg=angle,gap_mm=gap))
        print('Hip probe',angle,round(gap,5),'mm',flush=True)
    (WORK/'hip_probe.json').write_text(json.dumps(samples,indent=2)+'\n')
    assert min(s['gap_mm'] for s in samples)>=.2, 'Hip clearance still too small'


def publish():
    data=load()
    assert data['stage']=='locally_checked' and data['local_checks']['passed']
    assert digest(TARGET)==data['source_sha256']
    assert digest(data['candidate'])==data['candidate_sha256']
    assert digest(data['review_png'])==data['review_png_sha256']
    assert (WORK/'candidate.step.js').exists() and 'animation_checks' in data
    validation=WORK/'geometry_validation.json'
    assert validation.stat().st_mtime>=Path(data['candidate']).stat().st_mtime
    assert json.loads(validation.read_text())['ok']
    # Keep the former dynamics evidence pinned before its same-name update.
    previous=WORK/'previous_dynamics.json'
    if not previous.exists():
        shutil.copyfile(TARGET.parent/'open_duck_mini_zp10d_mass_properties.json',previous)
    temporary=TARGET.with_suffix('.step.tmp')
    shutil.copyfile(data['candidate'],temporary)
    temporary.replace(TARGET)
    data.update(stage='published',output=str(TARGET),output_sha256=digest(TARGET),
                dynamics_status='pending synchronization',old_motion_evidence='Old R10 controls apply to the former STEP only.')
    shutil.copyfile(WORK/'candidate.step.js',TARGET.with_suffix('.step.js'))
    save(data)
    REPORT.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('Published shortened-leg CAD:',data['output_sha256'],flush=True)


def reuse_collisions():
    """Reuse centered collision assets only for exactly retained rigid geometry."""
    import copy
    import trimesh
    from scipy.spatial import cKDTree
    import build_zp10d_mjcf_from_final_step as sync
    cad=load();data=sync.load()
    old=json.loads((WORK/'previous_dynamics.json').read_text())
    assert old['source_step_sha256']==cad['source_sha256']
    reused=[]
    for current,previous,row in zip(data['parts'],old['parts'],cad['parts']):
        assert current['path']==previous['path']==row['path']
        if row['mode'].startswith('shorten'):
            continue
        assert abs(current['volume_mm3']-previous['volume_mm3'])<.001
        assert abs(current['mass_kg']-previous['mass_kg'])<1e-10
        shift=np.array([0,0,row['shift_z_mm']*.001])
        assert np.linalg.norm(np.array(current['com_world_m'])-previous['com_world_m']-shift)<1e-9
        a=trimesh.load(sync.WORK/'assets'/current['mesh'],force='mesh').vertices
        b=trimesh.load(sync.ASSET_DIR/previous['mesh'],force='mesh').vertices
        error=max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max())
        assert error<1e-8,(row['index'],error)
        current['collision']=copy.deepcopy(previous['collision'])
        current['collision']['reuse_centered_visual_error_mm']=float(error*1000)
        current['collision']['reused_from_step_sha256']=old['source_step_sha256']
        for piece in current['collision']['pieces']:
            source=sync.ASSET_DIR/piece['mesh']
            assert digest(source)==piece['sha256']
            shutil.copyfile(source,sync.WORK/'assets'/piece['mesh'])
        reused.append(row['index'])
    data['collision_reuse']={'indices':reused,'source_step_sha256':old['source_step_sha256'],
                             'checks':'Retained CAD geometry, volume, mass, COM shift and bidirectional centered-mesh vertex distance <1e-5 mm.'}
    sync.save(data)
    print('Reused verified centered collision meshes for',len(reused),'rigid parts; rebuild 10 changed shapes.',flush=True)


def finish():
    import copy
    import re
    data=load()
    assert data['stage']=='published' and digest(TARGET)==data['output_sha256']
    dynamics=json.loads((TARGET.parent/'open_duck_mini_zp10d_mass_properties.json').read_text())
    assert dynamics['source_step_sha256']==data['output_sha256'] and dynamics['validation']['synchronization_passed']
    validation=json.loads((WORK/'geometry_validation.json').read_text())
    assert validation['ok'] and validation['failureCount']==0
    data.update(stage='completed',dynamics_status='synchronized_and_validated',
                user_visual_acceptance='pending review of the updated CAD',
                dynamics_summary={k:dynamics[k] for k in ('mjcf','mjcf_sha256','scene_sha256','updated_total_mass_kg','collision_piece_count')},
                standard_modified_part_validation=validation,
                control_status='Old R10 liftoff/torque results apply only to the former model; post-shortening liftoff and load assessment pending.',
                limitations=['Sampled local geometry only, not exhaustive combined-motion clearance.',
                             'Retained vendor/rigid assembly contact intersections are listed separately from new interference.',
                             'PETG 10% shells / 30% other printed parts specified; effective-density mass estimates still need slicer/measured calibration.',
                             'Original home joint angles retained; 60-degree knee stance and 32% torque reduction remain unvalidated estimates.'])
    review_path=TARGET.parent/'FINAL_ASSEMBLY_REVIEW.json'
    old=json.loads(review_path.read_text())
    interfaces=copy.deepcopy(old['interface_validation'])
    for joint in interfaces['joints']:
        shift=10 if joint['servo'] in ('S04','S09') else 20 if joint['servo'] in ('S05','S10') else 0
        for pattern in joint['patterns']:
            for hole in pattern['holes']:
                hole['point_mm'][2]+=shift
        joint['current_evidence']='Preserved original mounting region double BRep difference and rigid servo placement verified this round.' if joint['servo'] in ('S03','S04','S05','S08','S09','S10') else 'Unchanged geometry; historical interface evidence retained.'
    interfaces['scope']='Interface axes transformed with verified rigid mounting regions; original bore diameters and screw-bearing geometry preserved by current BRep comparisons. No new screw engagement/strength certification.'
    text=TARGET.with_suffix('.step.js').read_text()
    payload=json.loads(re.search(r'const data = (.*);',text)[1])
    current=dict(date='2026-10-17',target_sha256=data['output_sha256'],
                 mjcf_sha256=dynamics['mjcf_sha256'],geometry_changed=True,
                 review_scope='R11 changed leg components and immediate interfaces; prior whole-robot review retained under historical_full_review.',
                 axes=dynamics['axes'],limits_deg=dynamics['limits_deg'],parts=dynamics['parts'],
                 leg_shortening_report=str(REPORT.relative_to(ROOT)),
                 static=data['local_checks']['static'],single_joint_review=data['local_checks']['motion'],
                 combined_home=data['local_checks']['home_pose'],standard_validation=validation,
                 interface_validation=interfaces,
                 simulation=dynamics['validation']['rollouts'],
                 dynamics_synchronization=data['dynamics_summary'],
                 motion_demonstration=dict(completed=True,source_sha256=data['output_sha256'],
                     module_file=str(TARGET.with_suffix('.step.js')),module_sha256=digest(TARGET.with_suffix('.step.js')),
                     scenarios=payload['scenarios'],leg_contact_recheck=data['animation_checks']),
                 visual=dict(png=data['review_png'],sha256=data['review_png_sha256']),
                 review_completed=True,local_geometry_passed=True,user_visual_acceptance=data['user_visual_acceptance'],
                 full_motion_review_repeated=False,control_status=data['control_status'],
                 limitations=data['limitations'],user_disposition=old.get('user_disposition'),historical_full_review=old)
    review_path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
    block=f'''<!-- LEG_LINK_SHORTENING_CURRENT_START -->
## 2026-10-17 R11：四段腿长各缩短10 mm，CAD与力学模型已同步

正式STEP已覆盖同名文件，SHA256 `{data['output_sha256']}`。左右大腿/小腿轴距实测均为68.65 mm（原78.65 mm）；8块原连接板和2个腿罩改形，24个原零件仅刚体平移，115个物理件及编号/配色保留。中部固定区移{MIDDLE_MOUNT_SHIFT_MM:g} mm，原孔群/承压区保持，腿罩原两孔改为13 mm孔距并与原支撑同轴；原下端圆弧保留。

新外观尚待用户查看确认。本轮修改件CLI实体检查通过；局部静态、六俯仰关节5°抽样、髋部腿罩接口1°细查及原home组合姿态检查未发现新增干涉。髋侧最小实体间隙见`LEG_LINK_SHORTENING_REPORT.json`。本轮仅验证腿部局部，旧头部/整机组合扫描保留历史记录，未重复验收。标准浏览器快照的本轮结果保存在/tmp；当前直接CAD几何图为`LEG_LINK_SHORTENING_REVIEW.png`，已实际查看。

质量/惯量、关节轴、视觉及碰撞模型已同步，估计总质量{dynamics['updated_total_mass_kg']:.9f} kg。105个保形件经中心化网格对比复用碰撞分解，10个改形件重新生成；同步验证和三项4秒数值检查见质量报告。原home关节角保留。旧R08/R09/R10负载、低高度抬脚及c27737fa诊断均属于旧腿长，不能作为新模型动作验收；下一步复测含地面反力的膝扭矩及双侧抬脚。60°膝姿态、约32%减载尚未验证，PETG外壳10%/其他30%的真实打印质量仍待切片或称重标定。
<!-- LEG_LINK_SHORTENING_CURRENT_END -->

'''
    for name in ('UPPER_REFIT_ISSUES.md','FINAL_ASSEMBLY_REVIEW.md'):
        path=TARGET.parent/name
        old_text=path.read_text()
        old_text=old_text.replace('各缩短10 mm（实施中，未验收）','各缩短10 mm（实施记录，验收见顶部）')
        path.write_text(block+old_text)
    readme=ROOT/'playground/open_duck_mini_zp10d/README.md'
    text=readme.read_text()
    text=text.replace('synchronized and validated on 2026-10-15.',
                      'synchronized and validated on 2026-10-17 after shortening each thigh and shank by 10 mm (78.65 to 68.65 mm).')
    text=text.replace(data['source_sha256'],data['output_sha256'],1)
    text=text.replace('The estimated total mass is 2.097727648 kg.',
                      f'The estimated total mass is {dynamics["updated_total_mass_kg"]:.9f} kg.')
    text=text.replace('0.167787046 m',f'{dynamics["home_free_qpos"][2]:.9f} m')
    start=text.index('## Published validation')
    end=text.index('## Next steps',start)
    rows=[]
    for rollout in dynamics['validation']['rollouts']:
        rows.append(f'| {rollout["mode"]}, 4 s | {max(rollout["max_tracking_error_deg"]):.3f}° | '
                    f'{rollout["max_base_tilt_deg"]:.3f}° | {100*max(rollout["saturation_fraction"]):.2f}% | '
                    f'{sum(rollout["warning_counts"])} |')
    published=f'''## Published validation

R11 has 115 physical parts, 16 joint axes and {dynamics['collision_piece_count']} convex contact meshes.
All synchronization checks passed. The original home joint angles were retained;
only free-base height was realigned to the new feet. Current evidence is in
`cad/local_fix/open_duck_mini_zp10d_mass_properties.json` and
`cad/local_fix/LEG_LINK_SHORTENING_REPORT.json`.

| Numerical check | Maximum tracking error | Maximum base tilt | Actuator saturation | Warnings |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

These are short rigid-body numerical checks, not new walking, liftoff, PETG
strength or continuous-torque validation. The former R08/R09/R10 load/control
results and saved replays below belong to the previous leg lengths. Replay
hash checks reject them for the new model until independently regenerated.
The estimated 32% unloading with a 60-degree knee stance is unvalidated.

The current compiled home pose is exported to
`cad/local_fix/open_duck_mini_zp10d_dynamics.glb` (explicit normals and PBR colors)
and `cad/local_fix/open_duck_mini_zp10d_dynamics.png`.
The CAD review is `cad/local_fix/LEG_LINK_SHORTENING_REVIEW.png`.

'''
    text=text[:start]+published+text[end:]
    readme.write_text(text)
    data['resources']={'address_space_cap_bytes':4_000_000_000,'workspace_cap_bytes':8_000_000_000,'serial':True,'native_threads':1}
    save(data)
    REPORT.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('Recorded R11 CAD/dynamics completion; new liftoff and load assessment remain pending.',flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'prepare'
    if mode == 'prepare':
        prepare()
    elif mode == 'sections':
        for arg in sys.argv[2:]:
            print('PART', arg, flush=True)
            inspect_sections(int(arg))
            gc.collect()
    elif mode == 'build':
        build(set(map(int,sys.argv[2:])) if len(sys.argv)>2 else None)
    elif mode == 'assemble':
        assemble()
    elif mode == 'check':
        check()
    elif mode == 'render':
        render()
    elif mode == 'animation':
        animation()
    elif mode == 'hip-probe':
        hip_probe()
    elif mode == 'publish':
        publish()
    elif mode == 'reuse-collisions':
        reuse_collisions()
    elif mode == 'finish':
        finish()
    elif mode == 'cli':
        import runpy
        sys.argv=['cadgen',*sys.argv[2:]]
        runpy.run_module('cadgen.cli',run_name='__main__')
    else:
        raise SystemExit('usage: prepare | sections INDEX')
