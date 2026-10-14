"""Repair S12 and S14 using the two existing head connectors.

CAD brief: mm, STEP world coordinates. Recess the S12 bridge underside locally
to Z=170.07593881241, with sloping transitions and unchanged screw seats. Reduce the existing yaw
ring to R30.5/R24.5, retaining its case screw seats and rear bearing root.
Servo shapes, poses, smooth neck plates and enclosing shells remain fixed.
No added assembly parts. Validate modified parts and direct interfaces only;
S13 remains a separate unresolved item. Run through run_memory_limited.py.
"""

import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape
from OCP.OSD import OSD_Parallel, OSD_ThreadPool

from full_motion_current import TARGET, MJCF, _leaves, measured_axes, LIMITS, is_moving
from repair_minor_motion_clearance import detached, bounds, volume, is_mounting_cylinder
from remove_s01_s06_upper_horns import compare, leaves
from exact_motion_current import _overlap
import refit_upper_servo_interfaces_local as refit

WORK = Path('/tmp/head-motion')
REPORT = TARGET.parent / 'HEAD_MOTION_REPAIR_REPORT.json'
REVIEW = TARGET.parent / 'HEAD_MOTION_REVIEW.png'
BASE_SHA256 = '579174b3c4ec4748bdf7a085cce242e13e990bb4326f991f284d9c480c675115'
Z0, Z1 = 171.87687762474, 177.65087762474
R_OUTER, R_INNER = 30.5, 24.5
os.environ['CADGEN_JOBS'] = '1'
OSD_Parallel.SetUseOcctThreads_s(True)
OSD_ThreadPool.DefaultPool_s().Init(1)


def load():
    cache = json.loads((WORK/'cache.json').read_text())
    assert cache['source_sha256'] == BASE_SHA256
    return cache


def part(cache, label):
    rows = [row for row in cache['parts'].values() if row['label'] == label]
    assert len(rows) == 1, label
    return detached(bd.import_brep(rows[0]['file']))


def bore_span(face):
    axis = face.axis_of_rotation
    centre, direction = axis.position, axis.direction.normalized()
    box = face.bounding_box()
    projections = [(bd.Vector(x, y, z)-centre).dot(direction)
        for x in (box.min.X, box.max.X) for y in (box.min.Y, box.max.Y)
        for z in (box.min.Z, box.max.Z)]
    return centre, direction, min(projections), max(projections)


def build():
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == BASE_SHA256
    cache = load()
    old = part(cache, 'head_pitch_to_yaw')
    underside = bd.Plane.XZ * bd.Polygon(
        (-2, 167.66687762474), (1, 170.07593881241),
        (17, 170.07593881241), (20, 167.66687762474), align=None)
    relief = bd.extrude(underside, amount=40, dir=(0, 1, 0)).moved(bd.Pos(0, -18.525, 0))
    cross = refit._cut(old, [relief]).clean()
    bd.export_brep(cross, WORK/'cross-candidate.brep')
    old = part(cache, 'head_yaw_to_roll')
    base_region = refit._world_box(-100, 100, -100, 100, Z0-.001, 177.87787762474)
    keep = refit._fuse(
        refit._world_box(24, 65, -12.46, 12.66, Z0-.01, 177.88687762474),
        refit._world_box(-45, -17.3, -13.9, 14.1, Z0-.01, 177.88687762474))
    # Resize the original annulus and extend the existing rear base toward it
    # in the same plane; the existing case seats/bearing remain in place.
    bottom = refit._shape(old.intersect(base_region))
    feet = refit._shape(bottom.intersect(keep))
    outer = refit._rod(bd.Vector(9, .1, 0), bd.Vector(0, 0, 1), Z0, Z1, R_OUTER)
    inner = refit._rod(bd.Vector(9, .1, 0), bd.Vector(0, 0, 1), Z0-.01, Z1+.01, R_INNER)
    annulus = refit._cut(outer, [inner])
    rear_base = refit._world_box(-30, -17.3, -13.9, 14.1, Z0, Z1)
    ring = refit._fuse(refit._cut(old, [base_region]), feet, annulus, rear_base).clean()
    # The resized annulus must retain the original bottom bores/counterbores.
    for face in old.faces():
        if face.geom_type != bd.GeomType.CYLINDER or not is_mounting_cylinder(face): continue
        if face.bounding_box().max.Z > 177.878: continue
        centre, direction, start, end = bore_span(face)
        tool = refit._rod(centre, direction, start-1e-5, end+1e-5, face.radius)
        ring = refit._cut(ring, [tool])
    ring = ring.clean()
    bd.export_brep(ring, WORK/'ring-candidate.brep')
    report = dict(source_sha256=BASE_SHA256, mjcf_sha256=hashlib.sha256(MJCF.read_bytes()).hexdigest(),
                  modified={}, limits_deg=LIMITS, ring_radii_mm=[R_OUTER,R_INNER],
                  scope='S12/S14 connectors and direct head interfaces; S13 unresolved.')
    for label, fresh in [('head_pitch_to_yaw', cross), ('head_yaw_to_roll', ring)]:
        original = part(cache, label)
        validity = check_occurrence_shape(fresh.wrapped)
        assert validity['solidCount'] == 1 and not validity['reasons'], (label, validity)
        report['modified'][label] = dict(bounds=bounds(fresh), volume_before_mm3=volume(original),
            volume_after_mm3=volume(fresh), validity=validity)
        print(label, report['modified'][label], flush=True)
    (WORK/'repair-report.json').write_text(json.dumps(report, indent=2)+'\n')


def verify():
    cache = load()
    report = json.loads((WORK/'repair-report.json').read_text())
    fresh = {'head_pitch_to_yaw': detached(bd.import_brep(WORK/'cross-candidate.brep')),
             'head_yaw_to_roll': detached(bd.import_brep(WORK/'ring-candidate.brep'))}
    report['motion_checks'] = []
    for servo, moving_labels, fixed_labels in (
        ('S12', ('head_pitch_to_yaw',), ('S12_HEAD_PITCH_ZP10D',)),
        ('S12', ('head_yaw_to_roll',), ('S12_HEAD_PITCH_ZP10D', 'neck_left_sheet', 'neck_right_sheet')),
        ('S14', ('sg90', 'speaker_stand', 'speaker_interface', 'speaker_bottom'), ('head_yaw_to_roll',)),
    ):
        if moving_labels == ('head_pitch_to_yaw',):
            change=report['modified']['head_pitch_to_yaw']
            if abs(change['volume_before_mm3']-change['volume_after_mm3'])<.001:
                continue
        moving = [(p, r) for p, r in cache['parts'].items() if r['label'] in moving_labels]
        fixed = [(p, r) for p, r in cache['parts'].items()
                 if r['label'] in fixed_labels or
                 any('/'+label+'/' in p for label in fixed_labels)]
        # Servo assembly names contain additional numbering in the path.
        if servo == 'S12':
            fixed += [(p, r) for p, r in cache['parts'].items()
                      if 'S12_HEAD_PITCH_ZP10D[' in p and r['label']=='ZP10D_vendor_body']
        fixed = list({p:r for p,r in fixed}.items())
        for mp, mr in moving:
            for fp, fr in fixed:
                if not is_moving(mr, servo) or is_moving(fr, servo):
                    continue
                a = fresh.get(mr['label'])
                if a is None: a = detached(bd.import_brep(mr['file']))
                b = fresh.get(fr['label'])
                if b is None: b = detached(bd.import_brep(fr['file']))
                method = 'B-Rep distance'
                if servo=='S12' and mr['label']=='head_yaw_to_roll' and fr['label']=='ZP10D_vendor_body':
                    box=b.bounding_box()
                    b=refit._world_box(box.min.X,box.max.X,box.min.Y,box.max.Y,box.min.Z,box.max.Z)
                    method='Conservative distance to complete servo bounding box'
                axis = bd.Axis(*cache['axes'][servo]); gaps = []
                for angle in range(LIMITS[servo][0], LIMITS[servo][1]+1):
                    posed = a.rotate(axis, angle); gap = posed.distance_to(b)
                    assert gap >= .199, (servo, mp, fp, angle, gap)
                    gaps.append((gap, angle))
                    del posed
                minimum, angle = min(gaps)
                posed = a.rotate(axis, angle)
                assert _overlap(posed, b) < .001
                row = dict(joint=servo, moving=mp, fixed=fp, min_gap_mm=minimum,
                           minimum_angle_deg=angle, step_deg=1, distance_method=method)
                report['motion_checks'].append(row); print(row, flush=True)
                del posed; gc.collect()
    report['local_motion_passed'] = True
    (WORK/'repair-report.json').write_text(json.dumps(report, indent=2)+'\n')


def interfaces():
    cache = load(); report = json.loads((WORK/'repair-report.json').read_text())
    revised = {label: detached(bd.import_brep(WORK/file)) for label, file in (
        ('head_pitch_to_yaw', 'cross-candidate.brep'), ('head_yaw_to_roll', 'ring-candidate.brep'))}
    report['bearing_checks'] = []; report['static_checks'] = []
    for label, fresh in revised.items():
        change=report['modified'][label]
        if abs(change['volume_before_mm3']-change['volume_after_mm3'])<.001:
            continue
        original = part(cache, label)
        for face in original.faces():
            if face.geom_type != bd.GeomType.CYLINDER or not is_mounting_cylinder(face): continue
            if face.radius > 1.8: continue
            centre, direction, start, end = bore_span(face)
            # Preserve the complete original pressure material around each
            # through bore over its original axial bearing thickness.
            guard = refit._rod(centre, direction, start+1e-5,
                               end-1e-5, 3.8 if face.radius>1.4 else 3.05)
            old_material = refit._shape(original.intersect(guard))
            new_material = refit._shape(fresh.intersect(guard))
            before, after = volume(old_material), volume(new_material)
            retained = volume(refit._shape(old_material.intersect(new_material)))
            assert abs(before-retained)<.001, (label, list(face.center()), before, retained)
            bore = refit._rod(centre, direction, start+1e-5, end-1e-5, face.radius-1e-5)
            old_block, new_block = _overlap(original, bore), _overlap(fresh, bore)
            assert new_block <= old_block+.001, (label, 'blocked bore', new_block, old_block)
            report['bearing_checks'].append(dict(part=label, bore_radius_mm=face.radius,
                bore_center=list(face.center()), bearing_before_mm3=before, bearing_after_mm3=after,
                retained_original_material_mm3=retained, bore_block_before_mm3=old_block,
                bore_block_after_mm3=new_block))
        for p, row in cache['parts'].items():
            if row['label'] == label: continue
            lo, hi = row['bounds']; b = fresh.bounding_box()
            if any(lo[i]>list(b.max)[i]+.2 or hi[i]<list(b.min)[i]-.2 for i in range(3)):continue
            other = revised.get(row['label'])
            if other is None: other = detached(bd.import_brep(row['file']))
            before = _overlap(original, other); after = _overlap(fresh, other)
            assert after<=before+.001, (label,p,before,after)
            report['static_checks'].append(dict(part=label, neighbor=p,
                overlap_before_mm3=before, overlap_after_mm3=after))
            del other; gc.collect()
    report['interfaces_passed']=True
    (WORK/'repair-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Bearing regions and static interfaces passed.',flush=True)


def export():
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==BASE_SHA256
    report=json.loads((WORK/'repair-report.json').read_text())
    assert report['local_motion_passed'] and report['interfaces_passed'] and report['added_material_passed']
    robot=read_step(TARGET); before=leaves(robot); axes=measured_axes(robot)
    replacements={label:detached(bd.import_brep(WORK/file)) for label,file in (
        ('head_pitch_to_yaw','cross-candidate.brep'),('head_yaw_to_roll','ring-candidate.brep'))}
    def rebuild(node):
        if node.children:
            result=bd.Compound(children=[rebuild(child) for child in node.children],label=node.label)
            result.location=node.location
        else:
            result=detached(replacements.get(node.label,node));result.label=node.label
        result.color=node.color
        return result
    candidate=WORK/'review.step';bd.export_step(refit._encode_colours(rebuild(robot)),candidate)
    reloaded=read_step(candidate);after=leaves(reloaded)
    assert len(before)==len(after)==135
    for (p,a),(q,b) in zip(before,after):
        assert p==q;compare(replacements.get(a.label,a),b)
        assert (a.color is None)==(b.color is None)
        if a.color is not None:assert max(abs(x-y) for x,y in zip(a.color,b.color))<1e-6
    new_axes=measured_axes(reloaded)
    for servo in axes:
        assert max(abs(a-b) for x,y in zip(axes[servo],new_axes[servo]) for a,b in zip(x,y))<1e-5
    report.update(candidate=str(candidate),output_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  export_leaf_count=135,unchanged_leaf_count=135-sum(
                      abs(row['volume_before_mm3']-row['volume_after_mm3'])>.001
                      for row in report['modified'].values()),all_joint_axes_preserved=True,
                  export_checks='Per-face geometry, hierarchy, colours, leaf order and all joint axes preserved.')
    (WORK/'repair-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Exported',report['output_sha256'],flush=True)


def additions():
    cache=load();report=json.loads((WORK/'repair-report.json').read_text())
    original=part(cache,'head_yaw_to_roll')
    fresh=detached(bd.import_brep(WORK/'ring-candidate.brep'))
    added=refit._shape(fresh.cut(original))
    added_volume=volume(added)
    assert 0<added_volume<8800,added_volume
    bd.export_brep(added,WORK/'ring-added.brep')
    report['added_material_mm3']=added_volume;report['added_material_motion_checks']=[]
    outer=refit._rod(bd.Vector(9,.1,0),bd.Vector(0,0,1),Z0,Z1,R_OUTER)
    inner=refit._rod(bd.Vector(9,.1,0),bd.Vector(0,0,1),Z0-.01,Z1+.01,R_INNER)
    envelope=refit._fuse(refit._cut(outer,[inner]),
                       refit._world_box(-30,-17.3,-13.9,14.1,Z0,Z1)).clean()
    covered=volume(refit._shape(added.intersect(envelope)))
    assert abs(covered-added_volume)<.001,(added_volume,covered)
    report['added_envelope_covered_mm3']=covered
    cross=detached(bd.import_brep(WORK/'cross-candidate.brep'))
    for servo in ('S11','S12','S13','S14'):
        for path,row in cache['parts'].items():
            if servo=='S14':
                relevant=is_moving(row,servo)
            elif servo=='S13':
                relevant=row['label'] in ('head_pitch_to_yaw','neck_left_sheet','neck_right_sheet') or 'S12_HEAD_PITCH_ZP10D[' in path
            elif servo=='S12':
                relevant=(row['label'] in ('neck_left_sheet','neck_right_sheet') or
                          'S12_HEAD_PITCH_ZP10D[' in path) and not is_moving(row, servo)
            else:
                relevant=row['label'] in ('body_front','body_middle_top','body_back','trunk_top')
            if not relevant:continue
            other=cross if row['label']=='head_pitch_to_yaw' else detached(bd.import_brep(row['file']))
            method='AABB lower bound with B-Rep distance fallback'
            if row['label']=='ZP10D_vendor_body':
                box=other.bounding_box()
                other=refit._world_box(box.min.X,box.max.X,box.min.Y,box.max.Y,box.min.Z,box.max.Z)
                method='Conservative complete servo bounding box'
            axis=bd.Axis(*cache['axes'][servo]);gaps=[];exact_count=0
            mover=added
            if row['label']=='head_bot_sheet':
                mover=envelope
                method='Conservative annulus/rear-base envelope (all added material contained)'
            for angle in range(LIMITS[servo][0],LIMITS[servo][1]+1):
                a,b=(mover,other.rotate(axis,angle)) if servo=='S14' else (mover.rotate(axis,angle),other)
                boxa,boxb=a.bounding_box(),b.bounding_box()
                delta=[max(0,list(boxa.min)[i]-list(boxb.max)[i],
                            list(boxb.min)[i]-list(boxa.max)[i]) for i in range(3)]
                gap=sum(d*d for d in delta)**.5
                if gap<.199:
                    gap=a.distance_to(b);exact_count+=1
                assert gap>=.199,(servo,path,angle,gap)
                gaps.append((gap,angle))
                if row['label']=='head_bot_sheet' and angle%10==0:
                    print('Bottom sheet progress',angle,'gap',gap,flush=True)
            minimum,angle=min(gaps)
            result=dict(joint=servo,neighbor=path,min_gap_lower_bound_mm=minimum,
                        minimum_angle_deg=angle,step_deg=1,exact_distance_samples=exact_count,
                        distance_method=method)
            report['added_material_motion_checks'].append(result)
            print('Added material',result,flush=True);gc.collect()
    report['added_material_passed']=True
    (WORK/'repair-report.json').write_text(json.dumps(report,indent=2)+'\n')


def review():
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cache=load();fig=plt.figure(figsize=(13,9))
    for row,(label,file) in enumerate((('head_pitch_to_yaw','cross-candidate.brep'),
                                      ('head_yaw_to_roll','ring-candidate.brep'))):
        old=part(cache,label);fresh=detached(bd.import_brep(WORK/file))
        box=old.bounding_box();centre=np.array(list(box.center()));radius=max(box.size)*.56
        for col,shape in enumerate((old,fresh)):
            ax=fig.add_subplot(2,2,row*2+col+1,projection='3d')
            pts,faces=shape.tessellate(.1,.3)
            mesh=Poly3DCollection(np.asarray([list(p) for p in pts])[faces],
                facecolors='#aeb7c3' if col==0 else '#607586',shade=True)
            mesh.set_edgecolor('none');ax.add_collection3d(mesh)
            ax.set_xlim(centre[0]-radius,centre[0]+radius)
            ax.set_ylim(centre[1]-radius,centre[1]+radius)
            ax.set_zlim(centre[2]-radius,centre[2]+radius);ax.set_box_aspect((1,1,1))
            ax.view_init(elev=-25 if row==0 else 30,azim=-130 if row==0 else 35)
            ax.set_axis_off();ax.set_title(label+(' BEFORE' if col==0 else ' AFTER'))
    fig.tight_layout();fig.savefig(REVIEW,dpi=160);plt.close(fig);print(REVIEW,flush=True)


def install():
    report=json.loads((WORK/'repair-report.json').read_text())
    assert report['local_motion_passed'] and report['interfaces_passed'] and report['added_material_passed'] and REVIEW.exists()
    candidate=Path(report['candidate'])
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==report['output_sha256']
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==BASE_SHA256
    shutil.copyfile(TARGET,WORK/'source.step')
    temporary=TARGET.with_suffix('.step.tmp');shutil.copyfile(candidate,temporary);os.replace(temporary,TARGET)
    report.update(installed=str(TARGET),review=str(REVIEW))
    REPORT.write_text(json.dumps(report,indent=2)+'\n');print('Installed',report['output_sha256'],flush=True)


def snapshot():
    sys.argv=['cadgen','step','snapshot',str(TARGET),str(TARGET.parent/'HEAD_MOTION_SNAPSHOT.png'),'--json']
    runpy.run_module('cadgen.cli',run_name='__main__')


if __name__ == '__main__':
    {'build':build,'verify':verify,'interfaces':interfaces,'additions':additions,'export':export,
     'review':review,'install':install,'snapshot':snapshot}[sys.argv[1]]()
