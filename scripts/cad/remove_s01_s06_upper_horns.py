"""Restore smooth neck edges and give S01/S06 one lower output only.

CAD brief: millimetres, +X forward. Keep the lower output horns, case envelope,
poses, mounting bores and assembly colours. The user's single-output instruction
requires removing both upper horns and their shafts/bosses; trim those features
at the existing case top, Z=83.865938812333 mm (5.5 mm below the old shaft tip).
Extend the original neck faces across the old relief notch;
do not carve a new recess. Also remove both obsolete upper-horn frames,
their roots and horn screw seats from trunk_top; retain the rear servo-case
fastening seats and central S11 mount. Validate local S11 interfaces at the existing
MJCF limits before replacing the same official STEP. Run through the project's
memory-limited launcher, serially.
"""

import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape
from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
from OCP.OSD import OSD_Parallel, OSD_ThreadPool

import refit_upper_servo_interfaces_local as refit
from repair_motion_clearance_current import detached, face_facts
from full_motion_current import TARGET, MJCF, measured_axes, LIMITS
from validate_s01_s10_interfaces_local import SINGLE_OUTPUT_SERVOS

BASE_SHA256 = 'dcb978187096062087c1747a91c76378dc77df510ce7698ed88463c61f3e4ec9'
HORN_ONLY_SHA256 = 'f7a8dced8339929c9c239c37de1227408e0a2b378e1937fadbc73a78fe131675'
CASE_TOP_Z_MM = 83.865938812333
SUPPORT_BASE_SHA256 = '38eed0a3f2c5dbf61d30335866a1c712f69a19d3561b5ca52742976ff2f8a422'
# Remove the obsolete upper-horn frame all the way to its root. The rear
# case screw seats at X=-39 and the central S11 mount remain outside the cut.
SUPPORT_ROOT_X_MM = -32.35  # former R13.35 collar's rear tangent, X=-19
CENTRAL_TRUNK_HALF_Y_MM = 17.8802467083
WORK = Path('/tmp/single-axis')
REPORT = TARGET.parent / 'SINGLE_AXIS_REPORT.json'
REVIEW = TARGET.parent / 'SINGLE_AXIS_REVIEW.png'
NECKS = {'neck_left_sheet', 'neck_right_sheet'}
os.environ['CADGEN_JOBS'] = '1'
OSD_Parallel.SetUseOcctThreads_s(True)
OSD_ThreadPool.DefaultPool_s().Init(1)


def leaves(node, path=()):
    path = (*path, node.label)
    if node.children:
        return [row for child in node.children for row in leaves(child, path)]
    return [(path, node)]


def cylinders(shape):
    return sorted((row for row in face_facts(shape) if 'CYLINDER' in row[0]),
                  key=lambda row: tuple(round(v, 4) for v in row[1]))


def compare(expected, actual):
    first, second = face_facts(expected), face_facts(actual)
    assert len(first) == len(second), actual.label
    for a, b in zip(first, second):
        assert a[0] == b[0] and max(abs(x-y) for x, y in zip(a[1], b[1])) < 1e-4, actual.label
    assert abs(expected.volume - actual.volume) < .001, actual.label


def smooth_neck(original):
    # Remove the two 45-degree notch faces and the left plate's recess floor;
    # OCCT extends the existing outside and side faces back to their intersection.
    faces = [face for face in original.faces()
             if face.geom_type == bd.GeomType.PLANE and face.area < 10
             and face.center().X < -1 and 104 < face.center().Z < 107]
    assert len(faces) == (3 if original.label == 'neck_left_sheet' else 2)
    operation = BRepAlgoAPI_Defeaturing()
    operation.SetShape(original.wrapped)
    operation.SetRunParallel(False)
    for face in faces:
        operation.AddFaceToRemove(face.wrapped)
    operation.Build()
    assert operation.IsDone(), original.label
    fresh = bd.Compound.cast(operation.Shape())
    fresh.label, fresh.color = original.label, original.color
    validity = check_occurrence_shape(fresh.wrapped)
    assert validity['solidCount'] == 1 and not validity['reasons'], validity
    assert 1 < fresh.volume - original.volume < 4
    assert len(fresh.faces()) == 29
    old_bores, new_bores = cylinders(original), cylinders(fresh)
    assert len(old_bores) == len(new_bores)
    for first, second in zip(old_bores, new_bores):
        assert max(abs(a-b) for a,b in zip(first[1], second[1])) < 1e-5
    return fresh, validity


def single_output_body(original):
    bounds = original.bounding_box()
    top = [face for face in original.faces()
           if face.geom_type == bd.GeomType.PLANE and face.area > 500
           and abs(face.center().Z - CASE_TOP_Z_MM) < 1e-5
           and face.normal_at().Z > .999]
    assert len(top) == 1
    assert abs(bounds.max.Z - CASE_TOP_Z_MM - 5.5) < 1e-5
    fresh = refit._cut(detached(original), [refit._world_box(
        bounds.min.X-1, bounds.max.X+1, bounds.min.Y-1, bounds.max.Y+1,
        CASE_TOP_Z_MM, bounds.max.Z+1)])
    fresh.label, fresh.color = original.label, original.color
    validity = check_occurrence_shape(fresh.wrapped)
    assert validity['solidCount'] == 1 and not validity['reasons'], validity
    assert abs(fresh.bounding_box().max.Z - CASE_TOP_Z_MM) < 1e-5
    removed = refit._shape(detached(original).cut(fresh))
    assert removed.bounding_box().min.Z >= CASE_TOP_Z_MM - 1e-5
    assert 250 < removed.volume < 300
    assert abs(original.volume - fresh.volume - removed.volume) < .001

    # Normalize the supplier's split cylinder seams in both shapes before
    # comparison, as the boolean's normal cleanup unifies those seams.
    # Every retained face is identical except the enlarged, flat case top.
    # This also checks the lower shaft and every case fastening cylinder.
    def retained_faces(shape):
        return face_facts(bd.Compound([face for face in shape.faces()
            if face.bounding_box().max.Z <= CASE_TOP_Z_MM + 1e-5
            and not (face.geom_type == bd.GeomType.PLANE and face.area > 500
                     and abs(face.center().Z - CASE_TOP_Z_MM) < 1e-5)]))
    first, second = retained_faces(detached(original).clean()), retained_faces(fresh)
    assert len(first) == len(second), (len(first),len(second))
    for a,b in zip(first,second):
        assert a[0] == b[0] and max(abs(x-y) for x,y in zip(a[1],b[1])) < 1e-4
    return fresh, {'removed_mm3':removed.volume, 'case_top_z_mm':CASE_TOP_Z_MM,
                   'shaft_and_boss_height_mm':5.5, 'faces_before':len(original.faces()),
                   'faces_after':len(fresh.faces()), 'retained_case_and_lower_faces':len(first),
                   'validity':validity}


def build():
    source_sha = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    assert source_sha in (BASE_SHA256,HORN_ONLY_SHA256), source_sha
    completing = source_sha == HORN_ONLY_SHA256
    WORK.mkdir(exist_ok=True)
    source = WORK / ('pre-shaft.step' if completing else 'source.step')
    shutil.copyfile(TARGET, source)
    previous = json.loads(REPORT.read_text()) if completing else None
    if previous:
        assert previous['output_sha256'] == source_sha
        (WORK/'horn-only-report.json').write_text(json.dumps(previous,indent=2)+'\n')
    robot = read_step(TARGET)
    before = leaves(robot)
    axes = measured_axes(robot)
    removed = [(path, node) for path, node in before
               if path[-2] in SINGLE_OUTPUT_SERVOS and node.label == 'ZP10D_vendor_passive_horn']
    assert len(removed) == (0 if completing else 2)
    excluded = {id(node) for _, node in removed}
    replacements = {}
    records = {}
    shaft_records = {}
    for path, original in before:
        if original.label in NECKS and not completing:
            fresh, validity = smooth_neck(original)
            replacements[id(original)] = fresh
            bd.export_brep(detached(original), WORK / (original.label + '.brep'))
            bd.export_brep(fresh, WORK / (original.label + '-smooth.brep'))
            records[original.label] = {'restored_mm3': fresh.volume - original.volume,
                                      'faces_before': len(original.faces()), 'faces_after': len(fresh.faces()),
                                      'cylinders_preserved': True, 'validity': validity}
        if path[-2] in SINGLE_OUTPUT_SERVOS and original.label == 'ZP10D_vendor_body':
            fresh, record = single_output_body(original)
            replacements[id(original)] = fresh
            shaft_records[path[-2]] = record
            print(path[-2], 'upper shaft and boss removed',record['removed_mm3'],'mm3',flush=True)
    assert len(replacements) == (2 if completing else 4)
    def rebuild(node):
        if id(node) in excluded:
            return None
        if node.children:
            children = [child for old in node.children if (child := rebuild(old)) is not None]
            fresh = bd.Compound(children=children, label=node.label)
            fresh.location = node.location
        else:
            fresh = detached(replacements.get(id(node), node))
            fresh.label = node.label
        fresh.color = node.color
        return fresh
    expected = [(path, replacements.get(id(node),node)) for path,node in before if id(node) not in excluded]
    candidate = WORK / 'review.step'
    bd.export_step(refit._encode_colours(rebuild(robot)), candidate)
    exported = read_step(candidate)
    after = leaves(exported)
    assert len(before) == (135 if completing else 137) and len(after) == 135
    for (old_path, old), (new_path, new) in zip(expected, after):
        assert old_path == new_path, (old_path,new_path)
        compare(old, new)
        assert (old.color is None) == (new.color is None)
        if old.color is not None:
            assert max(abs(a-b) for a,b in zip(old.color,new.color)) < 1e-6
    fresh_axes = measured_axes(exported)
    for servo, (point, direction) in axes.items():
        assert max(abs(a-b) for a,b in zip(point,fresh_axes[servo][0])) < 1e-5, servo
        assert max(abs(a-b) for a,b in zip(direction,fresh_axes[servo][1])) < 1e-5, servo
    report = {'source_sha256': source_sha, 'source':str(source),
              'output_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
              'candidate': str(candidate), 'mjcf_sha256': hashlib.sha256(MJCF.read_bytes()).hexdigest(),
              'removed': (previous['removed'] if completing else
                          [{'path': '/'.join(path), 'volume_mm3': node.volume} for path,node in removed]),
              'restored_necks': previous['restored_necks'] if completing else records,
              'previous_horn_and_neck_stage_sha256': HORN_ONLY_SHA256 if completing else None,
              'removed_upper_shafts':shaft_records,
              'leaf_count_before':len(before), 'leaf_count_after': 135,
              'unchanged_leaf_count':len(before)-len(excluded)-len(replacements),
              'export_checks': 'Retained hierarchy, leaf order, per-face geometry, colours and all 14 joint axes verified.',
              'limits_deg': LIMITS['S11'], 'scope': 'Local S11 and the retained S01/S06 interfaces; sampled single-joint motion only.'}
    (WORK / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Candidate exported and compared; both servos have lower outputs only.',flush=True)


def verify():
    report = json.loads((WORK / 'report.json').read_text())
    candidate = Path(report['candidate'])
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == report['output_sha256']
    robot = read_step(candidate)
    rows = leaves(robot)
    necks = [node for _, node in rows if node.label in NECKS]
    bodies = [(path,node) for path,node in rows
              if path[-2] in SINGLE_OUTPUT_SERVOS and node.label == 'ZP10D_vendor_body']
    assert len(bodies) == 2 and len(necks) == 2
    assert not any(path[-2] in SINGLE_OUTPUT_SERVOS and node.label == 'ZP10D_vendor_passive_horn'
                   for path,node in rows)
    for path,node in bodies:
        assert abs(node.bounding_box().max.Z - CASE_TOP_Z_MM) < 1e-5
    report['current_validity'] = {}
    for path,node in [*bodies,*[((),neck) for neck in necks]]:
        validity = check_occurrence_shape(node.wrapped)
        assert validity['solidCount'] == 1 and not validity['reasons'], validity
        report['current_validity']['/'.join(path) if path else node.label] = validity
    report['neck_faces_current'] = {neck.label:len(neck.faces()) for neck in necks}
    assert all(len(neck.faces()) == 29 for neck in necks)
    # Case mounting and the active interface remain exact. Record their local
    # contacts before and after trimming, rather than rerunning the whole robot.
    original = read_step(Path(report['source']))
    originals = {path:node for path,node in leaves(original)}
    neighbours = [(path,node) for path,node in rows if node.label in ('trunk_top','trunk_bottom')]
    report['case_mount_checks'] = []
    for path,body in bodies:
        for neighbour_path,neighbour in neighbours:
            old = originals[path]
            old_gap,new_gap = old.distance_to(neighbour),body.distance_to(neighbour)
            old_overlap = old.intersect(detached(neighbour))
            new_overlap = body.intersect(detached(neighbour))
            old_volume = sum(s.volume for s in refit._items(old_overlap))
            new_volume = sum(s.volume for s in refit._items(new_overlap))
            assert abs(old_gap-new_gap) < 1e-5
            assert abs(old_volume-new_volume) < .001
            report['case_mount_checks'].append({'servo':path[-2], 'host':'/'.join(neighbour_path),
                'gap_before_mm':old_gap,'gap_after_mm':new_gap,
                'overlap_before_mm3':old_volume,'overlap_after_mm3':new_volume})
    fixed = [(path,node) for path,node in rows if node.label in ('trunk_top','body_front','body_middle_top')
             or (path[-2] in SINGLE_OUTPUT_SERVOS and node.label == 'ZP10D_vendor_body')]
    assert len(fixed) == 5
    axis = bd.Axis(*measured_axes(robot)['S11'])
    checks = []
    for neck in necks:
        for path, neighbour in fixed:
            minimum = float('inf')
            angle_at_min = None
            max_overlap = 0.0
            for angle in range(LIMITS['S11'][0], LIMITS['S11'][1]+1):
                posed = detached(neck).rotate(axis,angle)
                gap = posed.distance_to(neighbour)
                if gap < minimum:
                    minimum,angle_at_min = gap,angle
                if gap < .199:
                    overlap = posed.intersect(detached(neighbour))
                    max_overlap = max(max_overlap,sum(s.volume for s in refit._items(overlap)))
                del posed
            checks.append({'moving':neck.label,'fixed':'/'.join(path),'step_deg':1,
                           'min_gap_mm':minimum,'minimum_angle_deg':angle_at_min,'max_overlap_mm3':max_overlap})
            print(neck.label,neighbour.label,'min gap',minimum,'max overlap',max_overlap,flush=True)
    report['motion_checks'] = checks
    report['local_motion_passed'] = all(row['min_gap_mm'] >= .199 for row in checks)
    (WORK / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['local_motion_passed'], checks


def review():
    os.environ['MPLCONFIGDIR'] = str(WORK / 'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    original = read_step(WORK/'source.step')
    candidate = read_step(WORK/'review.step')
    def local(robot):
        return [(path,node) for path,node in leaves(robot) if node.label in NECKS
                or (path[-2] in SINGLE_OUTPUT_SERVOS)
                or (path[-2]=='S11_NECK_PITCH_ZP10D')]
    def draw(ax, node, colour, alpha=1):
        points, faces = node.tessellate(.1,.3)
        collection = Poly3DCollection(np.array([list(p) for p in points])[faces],facecolors=colour,alpha=alpha,shade=True)
        collection.set_edgecolor('none')
        ax.add_collection3d(collection)
    fig = plt.figure(figsize=(12,14))
    for column,(robot,title) in enumerate(((original,'Before: two upper passive horns; notched neck edges'),
                                         (candidate,'After: lower outputs only; original neck edges restored'))):
        ax = fig.add_subplot(4,2,column+1,projection='3d')
        for path,node in local(robot):
            colour = '#ec9429' if path[-2] in SINGLE_OUTPUT_SERVOS and node.label=='ZP10D_vendor_passive_horn' else ('#405b70' if node.label in NECKS else '#aaaaaa')
            draw(ax,node,colour,.8)
        ax.set_xlim(-55,30);ax.set_ylim(-50,50);ax.set_zlim(35,145)
        ax.set_box_aspect((85,100,110));ax.view_init(elev=25,azim=-145)
        ax.set_axis_off();ax.set_title(title,fontsize=10)
        ax = fig.add_subplot(4,2,3+column,projection='3d')
        for path,node in local(robot):
            if path[-2] == 'S06_LEFT_HIP_YAW_ZP10D':
                section = node.intersect(refit._world_box(-50,-8,19,42,77,92))
                if section is not None and refit._items(section):
                    draw(ax,refit._shape(section),'#ec9429' if 'horn' in node.label else '#aaaaaa')
        ax.set_xlim(-50,-8);ax.set_ylim(19,42);ax.set_zlim(77,92)
        ax.set_box_aspect((42,23,15));ax.view_init(elev=25,azim=-40)
        ax.set_axis_off();ax.set_title('Upper end, S06 shown (S01 identical): '+
                                      ('disc + shaft + boss' if column==0 else 'flat case top, no upper output'),fontsize=10)
        for row,name in enumerate(('neck_left_sheet','neck_right_sheet'),2):
            ax = fig.add_subplot(4,2,row*2+column+1,projection='3d')
            node = next(node for path,node in local(robot) if node.label==name)
            # A local section keeps the long plate from hiding its small notch.
            box=node.bounding_box()
            section = node.intersect(refit._world_box(-3,.3,box.min.Y-.1,box.max.Y+.1,103.4,107.2))
            draw(ax,refit._shape(section),'#405b70')
            ax.set_xlim(-3,.3);ax.set_ylim(box.min.Y-.1,box.max.Y+.1);ax.set_zlim(103.4,107.2)
            ax.set_box_aspect((3.3,box.size.Y+.2,3.8))
            ax.view_init(elev=22,azim=(-150 if row==1 else 150))
            ax.set_axis_off();ax.set_title(name+(' — old recess' if column==0 else ' — smooth original side'),fontsize=10)
    fig.tight_layout();fig.savefig(REVIEW,dpi=170);plt.close(fig)
    print('Saved',REVIEW,flush=True)


def install():
    report = json.loads((WORK/'report.json').read_text())
    assert report['local_motion_passed'] and REVIEW.exists()
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == report['source_sha256']
    candidate = Path(report['candidate'])
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == report['output_sha256']
    temporary = TARGET.with_suffix('.step.tmp')
    shutil.copyfile(candidate,temporary);os.replace(temporary,TARGET)
    report['installed'] = str(TARGET)
    report['review'] = str(REVIEW)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print('Installed',report['output_sha256'],flush=True)


def snapshot():
    sys.argv = ['cadgen','step','snapshot',str(TARGET),str(TARGET.parent/'SINGLE_AXIS_SNAPSHOT.png'),'--json']
    runpy.run_module('cadgen.cli',run_name='__main__')


def support_build():
    """CAD brief: remove both disused trunk_top upper-horn frames, in mm.

    Cut the R12 bosses, eight screw/counterbore sets and their forward roots
    above the case top. Preserve the original central beam/S11 mount and all
    four hip-case screw seats. No added material, new parts or joint movement.
    """
    from repair_minor_motion_clearance import volume as adaptive_volume
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == SUPPORT_BASE_SHA256
    work = WORK/'supports'
    work.mkdir(exist_ok=True)
    shutil.copyfile(TARGET,work/'source.step')
    shutil.copyfile(REPORT,work/'source-report.json')
    robot = read_step(TARGET)
    before = leaves(robot)
    axes = measured_axes(robot)
    path,original = next((path,node) for path,node in before if node.label=='trunk_top')
    bosses = [face for face in original.faces() if face.geom_type==bd.GeomType.CYLINDER
              and abs(face.radius-12) < 1e-4 and face.bounding_box().min.Z > 85
              and abs(face.axis_of_rotation.position.X+19) < 1e-4]
    assert len(bosses)==2
    old_bores = [face for face in original.faces() if face.geom_type==bd.GeomType.CYLINDER
                 and any(abs(face.radius-r)<1e-4 for r in (1.6,3.25))
                 and face.bounding_box().min.Z>90 and -30<face.center().X<-8
                 and 18<abs(face.center().Y)<43]
    assert len(old_bores)==16, len(old_bores)
    fresh = detached(original)
    removed_by_side = []
    for sign in (-1,1):
        lower,upper = sorted((sign*CENTRAL_TRUNK_HALF_Y_MM,sign*47))
        tool = refit._world_box(SUPPORT_ROOT_X_MM,0,lower,upper,CASE_TOP_Z_MM,95)
        removed = refit._shape(fresh.intersect(tool))
        removed_by_side.append({'servo':'S01' if sign<0 else 'S06',
            'volume_mm3':adaptive_volume(removed),'bounds':[list(removed.bounding_box().min),list(removed.bounding_box().max)]})
        fresh = refit._cut(fresh,[tool])
        assert sum(s.volume for s in refit._items(fresh.intersect(tool))) < .001
    fragments=sorted(fresh.solids(),key=lambda solid:solid.volume,reverse=True)
    # The old R13.35 collars extend into the central beam's Y band. Removing
    # their roots detaches one tiny crescent per side; these are obsolete
    # collar material, not part of the central beam or a case screw seat.
    for scrap in fragments[1:]:
        b=scrap.bounding_box()
        assert scrap.volume<5 and b.min.Z>85.8 and b.max.Z<86.9
        assert b.min.X>-25 and b.max.X<-18
        assert 16.8<abs(b.center().Y)<CENTRAL_TRUNK_HALF_Y_MM+.001
        sign=1 if b.center().Y>0 else -1
        record=next(row for row in removed_by_side if row['servo']==('S06' if sign>0 else 'S01'))
        record['detached_collar_sliver_mm3']=scrap.volume
        record['volume_mm3']+=scrap.volume
    fresh=fragments[0]
    fresh.label,fresh.color = original.label,original.color
    validity = check_occurrence_shape(fresh.wrapped)
    assert validity['solidCount']==1 and not validity['reasons'],validity
    removed = refit._shape(detached(original).cut(fresh))
    original_volume,fresh_volume,removed_volume=map(adaptive_volume,(original,fresh,removed))
    residual=original_volume-fresh_volume-removed_volume
    print('Adaptive volume balance residual',residual,'mm3',flush=True)
    assert abs(residual) < .01
    assert sum(s.volume for s in refit._items(fresh.cut(detached(original)))) < .001
    # All non-horn cylindrical interfaces are compared after normalizing seams.
    def kept_cylinders(shape):
        return cylinders(bd.Compound([face for face in shape.faces()
            if face.geom_type==bd.GeomType.CYLINDER
            and (face.bounding_box().max.Z < CASE_TOP_Z_MM+1e-5
                 or face.bounding_box().max.X < SUPPORT_ROOT_X_MM)]))
    old_kept,new_kept = kept_cylinders(detached(original).clean()),kept_cylinders(fresh)
    assert len(old_kept)==len(new_kept),(len(old_kept),len(new_kept))
    for a,b in zip(old_kept,new_kept):
        assert max(abs(x-y) for x,y in zip(a[1],b[1])) < 1e-4
    # Preserve the full local material around the four hip-case screw seats.
    guards=[]
    for y in (-37.25,-23.25,23.25,37.25):
        guard=refit._world_box(-42.5,-35.5,y-3.5,y+3.5,CASE_TOP_Z_MM,87)
        assert sum(s.volume for s in refit._items(removed.intersect(guard))) < .001
        guards.append([-39,y])
    def replace(node):
        if node.children:
            result=bd.Compound(children=[replace(child) for child in node.children],label=node.label)
            result.location=node.location
        else:
            result=detached(fresh if node is original else node)
            result.label=node.label
        result.color=node.color
        return result
    candidate=work/'review.step'
    bd.export_step(refit._encode_colours(replace(robot)),candidate)
    exported=read_step(candidate)
    after=leaves(exported)
    assert len(before)==len(after)==135
    for (p,old),(q,new) in zip(before,after):
        assert p==q
        compare(fresh if old is original else old,new)
        assert (old.color is None)==(new.color is None)
        if old.color is not None:
            assert max(abs(a-b) for a,b in zip(old.color,new.color)) < 1e-6
    fresh_axes=measured_axes(exported)
    for servo,(point,direction) in axes.items():
        assert max(abs(a-b) for a,b in zip(point,fresh_axes[servo][0])) < 1e-5
        assert max(abs(a-b) for a,b in zip(direction,fresh_axes[servo][1])) < 1e-5
    report={'source_sha256':SUPPORT_BASE_SHA256,'source':str(work/'source.step'),
        'output_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'candidate':str(candidate),
        'modified':{'path':'/'.join(path),'removed_by_side':removed_by_side,'removed_mm3':removed_volume,
                    'adaptive_volume_balance_residual_mm3':residual,
                    'validity':validity,'obsolete_horn_bore_faces_removed':len(old_bores),
                    'retained_cylinder_faces':len(new_kept),'case_seat_guards_xy':guards},
        'leaf_count_before':135,'leaf_count_after':135,'unchanged_leaf_count':134,
        'export_checks':'Hierarchy, per-face geometry, colours and all 14 joint axes checked.',
        'limits_deg':LIMITS['S11'],'scope':'Only trunk_top upper-horn supports and their direct interfaces.'}
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Removed both upper-horn frames and forward roots:',removed_by_side,flush=True)


def support_verify():
    work=WORK/'supports'
    report=json.loads((work/'report.json').read_text())
    candidate=Path(report['candidate'])
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==report['output_sha256']
    robot=read_step(candidate)
    before=read_step(Path(report['source']))
    rows=leaves(robot)
    trunk=next(node for path,node in rows if node.label=='trunk_top')
    old=next(node for path,node in leaves(before) if node.label=='trunk_top')
    validity=check_occurrence_shape(trunk.wrapped)
    assert validity['solidCount']==1 and not validity['reasons'],validity
    report['current_validity']={'trunk_top':validity}
    report['static_checks']=[]
    for path,node in rows:
        if not path[-2].startswith(('S01_','S06_','S11_')):
            continue
        overlap_before=sum(s.volume for s in refit._items(old.intersect(detached(node))))
        overlap_after=sum(s.volume for s in refit._items(trunk.intersect(detached(node))))
        assert overlap_after<=overlap_before+.001
        report['static_checks'].append({'part':'/'.join(path),'gap_before_mm':old.distance_to(node),
            'gap_after_mm':trunk.distance_to(node),'overlap_before_mm3':overlap_before,'overlap_after_mm3':overlap_after})
    axis=bd.Axis(*measured_axes(robot)['S11'])
    report['motion_checks']=[]
    report['former_support_angles']=[]
    for path,node in rows:
        if node.label not in NECKS:
            continue
        gaps=[]
        for angle in range(LIMITS['S11'][0],LIMITS['S11'][1]+1):
            posed=detached(node).rotate(axis,angle)
            gap=posed.distance_to(trunk)
            assert gap>=.199,(node.label,angle,gap)
            gaps.append((gap,angle))
        minimum,angle=min(gaps)
        report['motion_checks'].append({'moving':node.label,'fixed':'trunk_top',
            'step_deg':1,'min_gap_mm':minimum,'minimum_angle_deg':angle})
        print(node.label,'current limits min gap',minimum,flush=True)
        # Diagnose the angles at which the removed frames obstructed the neck.
        # This is local evidence, not a claim that the full head reaches +65.
        for angle in (46,47,52,55,65):
            posed=detached(node).rotate(axis,angle)
            report['former_support_angles'].append({'neck':node.label,'angle_deg':angle,
                'gap_before_mm':posed.distance_to(old),'gap_after_mm':posed.distance_to(trunk),
                'overlap_before_mm3':sum(s.volume for s in refit._items(posed.intersect(old))),
                'overlap_after_mm3':sum(s.volume for s in refit._items(posed.intersect(trunk)))})
            print(report['former_support_angles'][-1],flush=True)
    report['local_motion_passed']=True
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def support_review():
    work=WORK/'supports'
    os.environ['MPLCONFIGDIR']=str(WORK/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    original=read_step(work/'source.step')
    candidate=read_step(work/'review.step')
    def draw(ax,node,colour,alpha=1):
        pts,faces=node.tessellate(.1,.3)
        collection=Poly3DCollection(np.array([list(p) for p in pts])[faces],
                                   facecolors=colour,alpha=alpha,shade=True)
        collection.set_edgecolor('none');ax.add_collection3d(collection)
    fig=plt.figure(figsize=(13,10))
    for column,(robot,title) in enumerate(((original,'Before: unused upper-horn frames remain'),
                                          (candidate,'After: both frames and roots removed'))):
        for row,(elev,azim) in enumerate(((25,-140),(20,40))):
            ax=fig.add_subplot(2,2,row*2+column+1,projection='3d')
            trunk=next(node for path,node in leaves(robot) if node.label=='trunk_top')
            draw(ax,refit._shape(trunk.intersect(refit._world_box(-60,30,-52,52,43,100))),
                 '#9da8b2' if column==0 else '#607586')
            for path,node in leaves(robot):
                if node.label in NECKS or path[-2].startswith(('S01_','S06_','S11_')):
                    draw(ax,node,'#cccccc',.25)
            ax.set_xlim(-60,30);ax.set_ylim(-52,52);ax.set_zlim(40,145)
            ax.set_box_aspect((90,104,105));ax.view_init(elev=elev,azim=azim)
            ax.set_axis_off();ax.set_title(title)
    fig.tight_layout();fig.savefig(REVIEW,dpi=160);plt.close(fig)
    print('Saved',REVIEW,flush=True)


def support_install():
    work=WORK/'supports'
    report=json.loads((work/'report.json').read_text())
    assert report['local_motion_passed'] and REVIEW.exists()
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==report['source_sha256']
    candidate=Path(report['candidate'])
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==report['output_sha256']
    previous=json.loads((work/'source-report.json').read_text())
    report['prior_single_axis_stage_sha256']=previous['output_sha256']
    for key in ('removed','restored_necks','removed_upper_shafts'):
        report[key]=previous[key]
    temporary=TARGET.with_suffix('.step.tmp')
    shutil.copyfile(candidate,temporary);os.replace(temporary,TARGET)
    report['installed']=str(TARGET);report['review']=str(REVIEW)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print('Installed with both obsolete frames removed:',report['output_sha256'],flush=True)


if __name__ == '__main__':
    {'build':build,'verify':verify,'review':review,'install':install,'snapshot':snapshot,
     'support-build':support_build,'support-verify':support_verify,
     'support-review':support_review,'support-install':support_install}[sys.argv[1]]()
