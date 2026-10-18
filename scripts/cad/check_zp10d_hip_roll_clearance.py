import sys,json,itertools,math
from pathlib import Path
sys.path.insert(0,'/root/work/Open_Duck_Playground/scripts/cad')
if sys.argv[1:] == ['--knee']:
 import set_zp10d_knee_stance as review
else:
 import assess_zp10d_hip_roll as review
import mujoco,numpy as np,trimesh
from cadgen import build123d as bd,read_step
rpath=review.WORK/'assessment.json';report=json.loads(rpath.read_text());mass=json.loads(review.MASS.read_text());cad=json.loads((review.OUT/'LEG_LINK_SHORTENING_REPORT.json').read_text())
assert report['standing_passed'] and review.digest(review.SCENE)==report['scene_sha256']
model=mujoco.MjModel.from_xml_path(str(review.SCENE));data=mujoco.MjData(model);zero=mujoco.MjData(model);mujoco.mj_forward(model,zero)
parts=[p for p in mass['parts'] if not any(s in p['chain'] for s in ('S11','S12','S13','S14','A_plus_y','A_minus_y'))]
byid={p['index']:p for p in parts};moving={p['index'] for p in parts if 'S02' in p['chain'] or 'S07' in p['chain']}
geometry={};bounds={};gid={}
for p in parts:
 i=p['index'];g=model.geom(f'visual_{i:03d}').id;gid[i]=g
 mid=model.geom_dataid[g];a=model.mesh_vertadr[mid];b=a+model.mesh_vertnum[mid];f=model.mesh_faceadr[mid];e=f+model.mesh_facenum[mid]
 mesh=trimesh.Trimesh(model.mesh_vert[a:b]*1000,model.mesh_face[f:e],process=False)
 manager=trimesh.collision.CollisionManager();manager.add_object(str(i),mesh);geometry[i]=manager
 bounds[i]=(mesh.bounds.mean(0),(mesh.bounds[1]-mesh.bounds[0])/2)
pairs=[(a['index'],b['index']) for a,b in itertools.combinations(parts,2) if ({a['index'],b['index']}&moving) and a['body']!=b['body'] and not(a['servo'] and a['servo']==b['servo'])]
old=report['postures'][0];selected=report['postures'][-1]
samples=[('home',selected['initial_qpos']),('settled',selected['summary']['settled_qpos'])]
samples +=[(f'transition_{f["time_s"]:.2f}',f['qpos']) for f in report['transition']['frames'][::25]]
for case in selected['disturbances']:
 samples +=[(case['name'],max(case['frames'],key=lambda f:f['base_tilt_deg'])['qpos'])]
near=[];shape_cache={};minimum=math.inf

def exact_shape(i):
 if i not in shape_cache:
  p=cad['parts'][i];path=p.get('candidate',p['original']);shape_cache[i]=bd.Compound(read_step(path).solids())
 body=model.geom_bodyid[gid[i]];R=data.xmat[body].reshape(3,3)@zero.xmat[body].reshape(3,3).T
 t=(data.xpos[body]-R@zero.xpos[body]+R@zero.xpos[model.body('base').id])*1000
 plane=bd.Plane(origin=tuple(t),x_dir=tuple(R[:,0]),z_dir=tuple(R[:,2]))
 return shape_cache[i].moved(bd.Location(plane))

for label,qpos in samples:
 data.qpos[:]=qpos;mujoco.mj_forward(model,data);worldbounds={}
 for i in byid:
  g=gid[i];R=data.geom_xmat[g].reshape(3,3);t=data.geom_xpos[g]*1000
  T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;geometry[i].set_transform(str(i),T)
  c,e=bounds[i];center=R@c+t;extent=abs(R)@e;worldbounds[i]=(center-extent,center+extent)
 for i,j in pairs:
  a,b=worldbounds[i],worldbounds[j]
  if np.any(np.maximum(a[0],b[0])-np.minimum(a[1],b[1])>1.):continue
  gap=float(geometry[i].min_distance_other(geometry[j]))
  minimum=min(minimum,gap)
  if gap<1.:
   first,second=exact_shape(i),exact_shape(j);exact=float(first.distance_to(second));overlap=0.
   if exact<1e-5:
    common=first.intersect(second);overlap=sum(s.volume for s in common.solids()) if common else 0.
   near.append(dict(sample=label,parts=[i,j],labels=[byid[i]['label'],byid[j]['label']],mesh_gap_mm=gap,exact_gap_mm=exact,overlap_mm3=overlap))
 print('local pose',label,'near',len(near),flush=True)
result=dict(scope='Legs and directly adjoining trunk/hip interfaces only; different rigid bodies, vendor-servo internals excluded.',sample_count=len(samples),part_count=len(parts),pair_count=len(pairs),near_pairs=near,minimum_screened_mesh_gap_mm=minimum if math.isfinite(minimum) else None,screen_threshold_mm=1.,passed=all(p['exact_gap_mm']>=.2-1e-5 and p['overlap_mm3']<=1e-5 for p in near),not_continuous_motion_certificate=True)
report['local_pose_check']=result
report['local_check_script_sha256']=review.digest(__file__)
rpath.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('LOCAL',json.dumps(result,ensure_ascii=False),flush=True)
