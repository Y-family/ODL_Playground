"""Compare candidate collisions against the pinned accepted assembly."""
import argparse
import gc
import hashlib
import json
import math
from cadgen import read_step,build123d as bd
import trimesh
from full_motion_current import TARGET,_leaves,_mesh,_depths,measured_axes,is_moving,LIMITS
from fit_6800_narrow_hips import WORK
from repair_motion_clearance_current import detached
from exact_motion_current import _overlap


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['static','broad','exact'])
    args=parser.parse_args()
    spec=json.loads((WORK/'report.json').read_text())
    candidate=WORK/'review.step'
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==spec['input_sha256']
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==spec['output_sha256']
    before=read_step(TARGET)
    after=read_step(candidate)
    old={item['path']:item for item in _leaves(before)}
    new={item['path']:item for item in _leaves(after)}
    assert set(old)==set(new)
    old_axes,new_axes=measured_axes(before),measured_axes(after)
    modifications={row['path']:row for row in spec['parts']}
    def changed(path):
        return modifications.get(path,{}).get('changed',False)
    def shift(path):
        return modifications.get(path,{}).get('shift_y_mm',0)
    def relevant(left,right,joint=None):
        if new[left]['servo'] and new[left]['servo']==new[right]['servo']:
            return False
        if changed(left) or changed(right):
            return True
        if shift(left)!=shift(right):
            return True
        return joint is not None and abs(new_axes[joint][0][1]-old_axes[joint][0][1]-shift(left))>.0001
    if args.phase in ('static','broad'):
        meshes={path:_mesh(item['shape']) for path,item in new.items()}
        if args.phase=='static':
            manager=trimesh.collision.CollisionManager()
            for path,mesh in meshes.items():
                manager.add_object(path,mesh)
            collided,pairs,contacts=manager.in_collision_internal(return_names=True,return_data=True)
            candidates=set()
            for contact in contacts:
                if contact.depth>.03:
                    pair=tuple(sorted(contact.names))
                    if len(pair)==2 and relevant(*pair):
                        candidates.add(pair)
            records=[]
            for left,right in sorted(candidates):
                original=_overlap(detached(old[left]['shape']),detached(old[right]['shape']))
                actual=_overlap(detached(new[left]['shape']),detached(new[right]['shape']))
                records.append(dict(parts=[left,right],before_mm3=original,after_mm3=actual,increase_mm3=actual-original))
                print('STATIC',left,right,round(original,4),round(actual,4),flush=True)
            issues=[row for row in records if row['increase_mm3']>.05]
            result=dict(input_sha256=spec['input_sha256'],output_sha256=spec['output_sha256'],checks=records,issues=issues)
            (WORK/'static.json').write_text(json.dumps(result,indent=2)+'\n')
            print('New static interferences',len(issues),flush=True)
            return
        result=dict(input_sha256=spec['input_sha256'],output_sha256=spec['output_sha256'],step_deg=5,joints=[])
        for joint in LIMITS:
            moving={path for path,item in new.items() if is_moving(item,joint)}
            manager=trimesh.collision.CollisionManager()
            for path,mesh in meshes.items():
                if path not in moving:
                    manager.add_object(path,mesh)
            origin,direction=new_axes[joint]
            pairs={}
            for angle in range(LIMITS[joint][0],LIMITS[joint][1]+1,5):
                matrix=trimesh.transformations.rotation_matrix(math.radians(angle),direction,origin)
                for path in sorted(moving):
                    posed=meshes[path].copy()
                    posed.apply_transform(matrix)
                    for fixed,depth in _depths(manager,posed).items():
                        if depth>.03 and relevant(path,fixed,joint):
                            pairs.setdefault((path,fixed),[]).append(angle)
            result['joints'].append(dict(joint=joint,limits_deg=LIMITS[joint],pairs=[dict(moving=left,fixed=right,angles=angles) for (left,right),angles in sorted(pairs.items())]))
            (WORK/'motion-broad.json').write_text(json.dumps(result,indent=2)+'\n')
            print(joint,'candidate pairs',len(pairs),flush=True)
        return
    broad=json.loads((WORK/'motion-broad.json').read_text())
    assert broad['output_sha256']==spec['output_sha256']
    result=dict(input_sha256=spec['input_sha256'],output_sha256=spec['output_sha256'],step_deg=5,checks=[],issues=[],completed_joints=[])
    for row in broad['joints']:
        joint=row['joint']
        for pair in row['pairs']:
            left,right=pair['moving'],pair['fixed']
            old_mover,new_mover=detached(old[left]['shape']),detached(new[left]['shape'])
            old_fixed,new_fixed=detached(old[right]['shape']),detached(new[right]['shape'])
            samples=[]
            for angle in pair['angles']:
                old_pose=old_mover.rotate(bd.Axis(*old_axes[joint]),angle)
                new_pose=new_mover.rotate(bd.Axis(*new_axes[joint]),angle)
                original,actual=_overlap(old_pose,old_fixed),_overlap(new_pose,new_fixed)
                samples.append(dict(angle_deg=angle,before_mm3=original,after_mm3=actual,increase_mm3=actual-original))
                del old_pose,new_pose
                gc.collect()
            checked=dict(joint=joint,moving=left,fixed=right,samples=samples)
            result['checks'].append(checked)
            failures=[sample for sample in samples if sample['increase_mm3']>.05]
            if failures:
                result['issues'].append(dict(joint=joint,moving=left,fixed=right,samples=failures))
            print(joint,left.rsplit('/',1)[-1],right.rsplit('/',1)[-1],'increases',[(sample['angle_deg'],round(sample['increase_mm3'],3)) for sample in failures],flush=True)
            (WORK/'motion-exact.json').write_text(json.dumps(result,indent=2)+'\n')
        result['completed_joints'].append(joint)
        (WORK/'motion-exact.json').write_text(json.dumps(result,indent=2)+'\n')
    print('New motion interferences',len(result['issues']),flush=True)


if __name__=='__main__':
    main()
