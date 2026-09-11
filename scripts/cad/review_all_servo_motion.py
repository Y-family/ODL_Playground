"""Read the approved STEP; generate individual servo clips and collision report."""
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from cadgen import read_step
import validate_s01_s10_interfaces_local as v
from scan_s01_s14_motion_collisions import MOVING_REFS, _angles, _mesh

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'cad/local_fix/open_duck_mini_complete_14_zp10d_numbered.step'
NAMES = ['right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle',
         'left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
         'neck_pitch','head_pitch','head_yaw','head_roll']
LIMITS = [(-30,30),(-25,25),(-30,70),(-90,90),(-90,90),
          (-30,30),(-25,25),(-70,30),(-90,90),(-90,90),
          (-20,65),(-45,45),(-135,135),(-30,30)]

def walk(shape, ref='o1'):
    yield ref, shape
    for i, child in enumerate(v._children(shape), 1):
        yield from walk(child, f'{ref}.{i}')

def main():
    robot = read_step(TARGET)
    nodes = dict(walk(robot))
    leaves = {r:s for r,s in nodes.items() if not v._children(s)
              and int(r.split('.')[1]) not in {3,4,8,12,19}}
    joints=[]
    for i,(name,limits) in enumerate(zip(NAMES,LIMITS),1):
        sid=f'S{i:02}'
        matches=[(r,s) for r,s in nodes.items() if s.label.startswith(sid+'_')]
        assert len(matches)==1, sid
        servo_ref,servo=matches[0]
        axis,center=v._axis(servo)
        groups=set(MOVING_REFS[name])
        if i in (1,2,3): groups.add(6)
        if i in (6,7,8): groups.add(1)
        moving=[r for r in leaves if int(r.split('.')[1]) in groups]
        horns=[f'{servo_ref}.{n}' for n,s in enumerate(v._children(servo),1)
               if 'horn' in s.label]
        moving=sorted(set(moving+horns))
        joints.append(dict(id=sid,name=name,axis=list(axis),origin=list(center),
                           limits=list(limits),refs=moving))
        print(sid,'axis',list(axis),'origin',list(center),flush=True)

    # Render choreography uses the same measured axes and exact moving leaves
    # as collision testing. Each clip starts and ends at the written rest pose.
    js='const joints = '+json.dumps(joints)+';\n'
    js+='''function pose(t, m, j) {
  const phase = (t % 3) / 3;
  const stops = [0, j.limits[1], 0, j.limits[0], 0];
  const segment = Math.min(3, Math.floor(phase * 4));
  const u = (1 - Math.cos(Math.PI * (phase * 4 - segment))) / 2;
  const angle = stops[segment] + (stops[segment + 1] - stops[segment]) * u;
  for (const ref of j.refs) m.get('#' + ref).rotate(j.axis, angle, j.origin);
}
export const clips = Object.fromEntries(joints.map(j => [j.id, {
  label: j.id + ' ' + j.name + ' ' + j.limits.join('…') + '° / 3s', duration: 3, loop: true,
  update(t,m) { pose(t,m,j); }
}]));
clips.all_servos = {label: 'S01–S14 full range / 42s',duration:42,loop:true,
  update(t,m) {const i=Math.min(13,Math.floor(t/3)); pose(t%3,m,joints[i]);}};
'''
    TARGET.with_name(TARGET.name+'.js').write_text(js)
    report={'robot':str(TARGET),'step_deg':5,'joints':[],
            'method':'FCL leaf mesh candidates; not a BRep or load-bearing certification'}
    print('Meshing existing parts for collision testing',flush=True)
    meshes={r:_mesh(s,.3,.15) for r,s in leaves.items()}
    for joint in joints:
        moving=set(joint['refs'])
        managers=[trimesh.collision.CollisionManager(),trimesh.collision.CollisionManager()]
        for r,mesh in meshes.items(): managers[0 if r in moving else 1].add_object(r,mesh)
        def contacts():
            _,_,data=managers[0].in_collision_other(managers[1],return_names=True,return_data=True)
            result={}
            for c in data:
                pair=tuple(sorted(c.names)); depth=max(0,float(c.depth))
                result[pair]=max(result.get(pair,0),depth)
            return result
        baseline=contacts()
        candidates={}
        for angle in _angles(*joint['limits'],5):
            transform=trimesh.transformations.rotation_matrix(math.radians(angle),joint['axis'],joint['origin'])
            for r in moving: managers[0].set_transform(r,transform)
            for pair,depth in contacts().items():
                if depth-baseline.get(pair,0)>.05:
                    candidates.setdefault(pair,[]).append(float(angle))
        item={**joint,'candidates':[{'refs':list(pair),'labels':[leaves[r].label for r in pair],
                                    'angles_deg':angles} for pair,angles in candidates.items()],
              'neutral_candidates':[{'refs':list(pair),'depth_mm':d} for pair,d in baseline.items() if d>.05]}
        report['joints'].append(item)
        (ROOT/'cad/review/all_servos_motion_report.json').write_text(json.dumps(report,indent=2))
        print(joint['id'], 'candidate_pairs=',len(candidates),flush=True)
    print('Animation and full 14-servo sampled report written',flush=True)

if __name__=='__main__': main()
