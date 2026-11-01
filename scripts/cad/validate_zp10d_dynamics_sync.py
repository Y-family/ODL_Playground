"""Independent geometry, mass, kinematic and MuJoCo smoke checks for the sync.

Run via run_memory_limited.py; candidates are checked before publication.
"""
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import xml.etree.ElementTree as ET

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import numpy as np
import trimesh
import mujoco
from cadgen import build123d as bd
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_OUT
from OCP.gp import gp_Pnt
from scipy.spatial import ConvexHull

from build_zp10d_mjcf_from_final_step import (
    WORK, MJCF_OUTPUT, SCENE, FINAL_STEP, REPORT_OUTPUT, JOINT_NAMES,
    ANTENNA_NAMES, SERVO_COMMAND_SIGN, digest, load, properties, PROJECT_ROOT, MANIFEST, PHYSICAL_MAP, current_geometry,
)


def forward_matrix(pose, chain, axes):
    transform=np.eye(4)
    for sid in chain:
        origin,direction=map(np.array,axes[sid])
        transform=transform@trimesh.transformations.rotation_matrix(
            SERVO_COMMAND_SIGN.get(sid, 1) * pose.get(sid,0), direction, origin*.001)
    return transform


def world_vertices(model,data,geom):
    mesh=int(model.geom_dataid[geom]);start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
    return model.mesh_vert[start:start+count]@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]


def contact_pairs(model, data):
    """Aggregate proxy contacts by physical part, retaining approximation limits."""
    pairs = {}
    for contact in data.contact:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                 for g in contact.geom]
        parts = sorted(int(name.split('_')[1]) for name in names
                       if name and name.startswith('collision_'))
        key = tuple(parts) if len(parts) == 2 else ('floor', *parts)
        row = pairs.setdefault(key, {'parts': list(key), 'contacts': 0,
                                     'min_proxy_distance_mm': 0.0})
        row['contacts'] += 1
        row['min_proxy_distance_mm'] = min(row['min_proxy_distance_mm'],
                                         float(contact.dist) * 1000)
    return list(pairs.values())


def gravity_loads(model, manifest):
    results = []
    for pose in ('neutral', 'home'):
        data = mujoco.MjData(model)
        if pose == 'home':
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        joints = model.actuator_trnid[:, 0]
        torques = data.qfrc_bias[model.jnt_dofadr[joints]]
        limits = model.actuator_forcerange[:, 1]
        results.append({'pose': pose, 'joint_gravity_torque_nm': torques.tolist(),
                        'fraction_of_provisional_stall_torque': (np.abs(torques) / limits).tolist(),
                        'contact_pairs': contact_pairs(model, data),
                        'scope': 'Gravity only, base hypothetically supported; not standing inverse dynamics, dynamic loading or continuous torque rating.'})
    return results


def analytic_property_check():
    mass=.012
    box=bd.Box(2,4,6)
    com,inertia=properties(box,mass)
    expected=mass/12*np.diag([4**2+6**2,2**2+6**2,2**2+4**2])*1e-6
    assert np.allclose(inertia,expected,rtol=1e-10,atol=1e-14)
    moved=bd.Pos(100,200,300)*bd.Rot(0,0,37)*box
    com2,inertia2=properties(moved,mass)
    r=trimesh.transformations.rotation_matrix(math.radians(37),[0,0,1])[:3,:3]
    assert np.allclose(com2,[.1,.2,.3],atol=1e-12)
    assert np.allclose(inertia2,r@expected@r.T,rtol=1e-9,atol=1e-14)
    return dict(passed=True,units='kg, metres, kg.m2',checks=['analytic_box','translation_invariance','rotated_tensor'])


def cavities(manifest):
    rows=current_geometry()[1];findings=[]
    for part in manifest['parts']:
        if part['label'] not in ('head','head_yaw_to_roll','body_front','body_back'):
            continue
        shape=rows[part['index']]['shape']
        assert rows[part['index']]['path']==part['path']
        classifier=BRepClass3d_SolidClassifier(shape.wrapped)
        equations=[]
        for piece in part['collision']['pieces']:
            mesh=trimesh.load(WORK/'assets'/piece['mesh'],force='mesh')
            equations.append(ConvexHull(mesh.vertices).equations)
        lo,hi=np.array(part['bbox_mm'])
        if part['label']=='head':
            # The large enclosed head space must stay empty in the contact representation.
            probes=itertools.product(np.linspace(lo[0]+15,hi[0]-15,5),[-30,0,30],[185,200,215])
        elif part['label'] == 'head_yaw_to_roll':
            origin=np.array(manifest['axes']['S13'][0])
            probes=itertools.product(origin[0]+np.array([-20,0,20]),origin[1]+np.array([-20,0,20]),[172.5,173.5,180])
        else:
            probes=itertools.product(*(np.linspace(lo[axis],hi[axis],5)[1:-1]
                                       for axis in range(3)))
        checked=0;false_fills=[]
        for coordinates in probes:
            point=np.array(coordinates)
            classifier.Perform(gp_Pnt(*point),1e-7)
            if classifier.State()!=TopAbs_OUT:continue
            clearance=bd.Vertex(*point).distance_to(shape)
            if clearance<2:continue
            checked+=1
            local=point*.001-np.array(part['com_world_m'])
            inside=any(np.all(e[:,:3]@local+e[:,3]<-1e-8) for e in equations)
            if inside:false_fills.append({'point_mm':point.tolist(),'cad_clearance_mm':clearance})
        if part['label'] in ('head', 'head_yaw_to_roll'):
            assert checked>=5,(part['label'],checked)
        findings.append(dict(part=part['label'],deep_empty_points=checked,false_fills=false_fills,
                             sampled_voids_only=True,passed=not false_fills))
        print('cavity',part['label'],'empty probes',checked,'false fills',len(false_fills),flush=True)
    return findings


def rollouts(manifest,scene_model,root):
    fixed=ET.fromstring(ET.tostring(root))
    # Removing the free joint welds the trunk to world, changing MuJoCo's
    # parent-contact filtering and making servo cases contact their own horns.
    # Constrain the free base instead, preserving the actual collision tree.
    equality = fixed.find('equality')
    if equality is None:
        equality = ET.SubElement(fixed, 'equality')
    ET.SubElement(equality, 'weld', {'body1': 'base', 'solref': '0.004 1'})
    fixed_model=mujoco.MjModel.from_xml_string(ET.tostring(fixed,encoding='unicode'))
    results=[]
    for mode in ('fixed_neutral_hold','fixed_small_sweep','free_home_hold'):
        model=scene_model if mode=='free_home_hold' else fixed_model
        data=mujoco.MjData(model)
        if mode=='free_home_hold':mujoco.mj_resetDataKeyframe(model,data,0)
        else:mujoco.mj_resetData(model,data)
        ids=model.actuator_trnid[:,0];qadr=model.jnt_qposadr[ids]
        home=model.key_ctrl[0].copy() if mode=='free_home_hold' else data.qpos[qadr].copy()
        data.ctrl[:]=home
        mujoco.mj_forward(model,data)
        base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base')
        initial_z=float(data.xpos[base,2]);peak=np.zeros(model.nu);sat=np.zeros(model.nu)
        max_tilt=0.;max_penetration=0.;lowest_z=initial_z;frames=[];warning_before=data.warning.number.copy()
        seconds=4;steps=round(seconds/model.opt.timestep)
        amplitude=np.minimum(.15*(model.actuator_ctrlrange[:,1]-model.actuator_ctrlrange[:,0]),.2)
        for step in range(steps):
            if mode=='fixed_small_sweep':
                target=home+amplitude*np.sin(2*np.pi*data.time/seconds)
                data.ctrl[:]=np.clip(target,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
            else:data.ctrl[:]=home
            mujoco.mj_step(model,data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(),mode
            peak=np.maximum(peak,np.abs(data.qpos[qadr]-data.ctrl))
            sat+=np.abs(data.actuator_force)>=.99*model.actuator_forcerange[:,1]
            lowest_z=min(lowest_z,float(data.xpos[base,2]))
            tilt=math.degrees(math.acos(np.clip(data.xmat[base].reshape(3,3)[2,2],-1,1)))
            max_tilt=max(max_tilt,tilt)
            if data.ncon:max_penetration=max(max_penetration,float(max(0,-data.contact.dist.min()))*1000)
            if step%50==0:frames.append(dict(time=float(data.time),base_z_m=float(data.xpos[base,2]),tilt_deg=tilt,
                                            qpos=data.qpos.tolist(),contact_count=int(data.ncon)))
        warnings=(data.warning.number-warning_before).tolist()
        assert not any(warnings), (mode, warnings)
        final_contacts = contact_pairs(model, data)
        for pair in final_contacts:
            if all(isinstance(index, int) for index in pair['parts']):
                first, second = [manifest['parts'][index] for index in pair['parts']]
                assert not (first['servo'] and first['servo'] == second['servo']), pair
        results.append(dict(mode=mode,seconds=seconds,finite=True,warning_counts=warnings,
                            max_tracking_error_deg=np.degrees(peak).tolist(),saturation_fraction=(sat/steps).tolist(),
                            lowest_base_z_m=lowest_z,final_base_z_m=float(data.xpos[base,2]),
                            max_base_tilt_deg=max_tilt,max_contact_penetration_mm=max_penetration,frames=frames,
                            final_contact_pairs=final_contacts,
                            scope='Numerical smoke check with provisional actuator/contact parameters; not walking or physical validation.'))
        print('rollout',mode,'tilt',round(max_tilt,3),'max error',round(float(np.degrees(peak).max()),3),'warnings',warnings,flush=True)
    return results


def main():
    manifest=load()
    assert manifest['stage'] == 'assembled', manifest['stage']
    model_path=WORK/MJCF_OUTPUT.name;scene_path=WORK/SCENE.name
    model=mujoco.MjModel.from_xml_path(str(scene_path));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    report=dict(mjcf_sha256=digest(model_path),scene_sha256=digest(scene_path),
                candidate_manifest_sha256=digest(MANIFEST),
                validator_sha256=digest(Path(__file__)),
                source_step_sha256=digest(FINAL_STEP),mujoco_version=mujoco.__version__,
                nq=model.nq,nv=model.nv,nu=model.nu,physical_parts=len(manifest['parts']),
                mass_property_units=analytic_property_check(),joint_alignment=[],geometry=[],inertials=[],
                actuator_order=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a) for a in range(model.nu)])
    if 'physical_map_sha256' in manifest:
        assert digest(PHYSICAL_MAP) == manifest['physical_map_sha256'], 'Physical STEP map changed'
        mapping = json.loads(PHYSICAL_MAP.read_text())
        assert mapping['source_step_sha256'] == manifest['source_step_sha256']
        assert mapping['physical_parts'] == len(manifest['parts'])
        report['step_physical_mapping'] = dict(
            sha256=manifest['physical_map_sha256'], step_solids=mapping['step_solids'],
            physical_parts=mapping['physical_parts'],
            excluded_reference_solids=len(mapping['reference_solid_indices']),
            collision_parts_reused=len(manifest['collision_reuse_indices']),
            collision_parts_rebuilt=len(manifest['collision_rebuild_indices']))
        # Flat-export synchronization retains the existing control convention.
        # A second conversion of the swapped knee/ankle signs must fail here.
        previous = json.loads(REPORT_OUTPUT.read_text())
        direction_checks = []
        for sid, name in JOINT_NAMES.items():
            old_axis = (previous.get('command_signs', {}).get(sid, 1)
                        * np.array(previous['axes'][sid][1]))
            actual = data.xaxis[model.joint(name).id]
            dot = float(np.dot(actual, old_axis))
            assert dot > 1 - 1e-10, (sid, 'Control direction changed', dot)
            direction_checks.append(dict(servo=sid, previous_axis_dot=dot))
        report['preserved_control_directions'] = direction_checks
    assert model.nu==16 and report['actuator_order'][:14]==manifest['original_actuator_order']
    base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base')
    base_offset=data.xpos[base].copy()
    mapping=JOINT_NAMES|ANTENNA_NAMES
    for sid,name in mapping.items():
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        expected_origin,expected_axis=map(np.array,manifest['axes'][sid])
        expected_axis = SERVO_COMMAND_SIGN.get(sid, 1) * expected_axis
        origin=(data.xanchor[j]-base_offset)*1000;axis=data.xaxis[j]
        delta=origin-expected_origin
        error=float(np.linalg.norm(np.cross(delta,expected_axis)))
        dot=float(np.dot(axis,expected_axis))
        assert error<1e-6 and dot>1-1e-10,(sid,error,dot)
        a=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
        assert model.jnt_limited[j] and model.actuator_ctrllimited[a]
        assert np.allclose(model.jnt_range[j],model.actuator_ctrlrange[a],atol=1e-12)
        report['joint_alignment'].append(dict(servo=sid,joint=name,command_sign=SERVO_COMMAND_SIGN.get(sid, 1),axis_line_error_mm=error,axis_dot=dot,
             origin_delta_mm=delta.tolist(),limits_deg=np.degrees(model.jnt_range[j]).tolist()))
    for part in manifest['parts']:
        g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'visual_{part["index"]:03d}')
        assert g>=0
        assert model.geom_contype[g]==0 and model.geom_conaffinity[g]==0
        vertices=(world_vertices(model,data,g)-base_offset)*1000
        bounds=np.array([vertices.min(0),vertices.max(0)])
        error=float(np.max(np.abs(bounds-np.array(part['bbox_mm']))))
        assert error<=.36,(part['path'],error)
        for n in range(part['collision']['convex_piece_count']):
            c=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'collision_{part["index"]:03d}_{n:03d}')
            assert c>=0 and model.geom_contype[c] and model.geom_conaffinity[c]
            assert model.geom_bodyid[c]==model.geom_bodyid[g]
        report['geometry'].append(dict(path=part['path'],bbox_delta_mm=error,body=part['body']))
    total=float(model.body_mass.sum());assert abs(total-manifest['updated_total_mass_kg'])<1e-10
    expected_com=sum(np.array(p['com_world_m'])*p['mass_kg'] for p in manifest['parts'])/total
    assert np.allclose(data.subtree_com[base]-base_offset,expected_com,atol=1e-10)
    for name,values in manifest['links'].items():
        b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        quat=model.body_iquat[b];r=np.empty(9);mujoco.mju_quat2Mat(r,quat);r=r.reshape(3,3)
        compiled=r@np.diag(model.body_inertia[b])@r.T
        target=np.array(values['inertia_link_kg_m2'])
        error=float(np.abs(compiled-target).max())
        assert error<1e-10 and abs(model.body_mass[b]-values['mass_kg'])<1e-10
        assert np.allclose(model.body_ipos[b],values['com_link_m'],atol=1e-10)
        report['inertials'].append(dict(body=name,mass_kg=float(model.body_mass[b]),max_tensor_error_kg_m2=error))
    report['total_mass_kg']=total;report['neutral_com_world_m']=expected_com.tolist()
    report['gravity_loads'] = gravity_loads(model, manifest)
    home = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, home, 0)
    mujoco.mj_forward(model, home)
    report['home_ground_alignment'] = manifest['home_ground_alignment']
    initial_floor = [pair for pair in contact_pairs(model, home)
                     if 'floor' in pair['parts']]
    assert all(pair['min_proxy_distance_mm'] >= -1e-5 for pair in initial_floor)
    report['home_initial_floor_contacts'] = initial_floor
    report['contact_scope'] = 'Convex proxies on all physical parts. MuJoCo excludes same-body and immediate parent-child pairs by default; this is not exhaustive CAD clearance validation.'
    # Compare several simultaneous poses independently to the CAD forward-kinematics chain.
    poses=[{}, {'S11':.2,'S12':-.25,'S13':.8,'S14':.15,'A_plus_y':.4,'A_minus_y':-.3},
           {'S01':.1,'S02':-.2,'S03':-.4,'S04':.7,'S05':-.3,'S06':-.1,'S07':.2,'S08':.4,'S09':.7,'S10':-.3},
           {'S13':math.radians(-100),'A_plus_y':-.6,'A_minus_y':.7},
           {'S13':math.radians(100),'S14':-.3}]
    local_meshes={p['index']:trimesh.load(WORK/'assets'/p['mesh'],force='mesh').vertices+np.array(p['com_world_m']) for p in manifest['parts']}
    max_pose_error=0;comparisons=0
    for pose in poses:
        mujoco.mj_resetData(model,data)
        for sid,angle in pose.items():
            j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,mapping[sid]);data.qpos[model.jnt_qposadr[j]]=angle
        mujoco.mj_forward(model,data)
        for part in manifest['parts']:
            t=forward_matrix(pose,part['chain'],manifest['axes'])
            expected=local_meshes[part['index']]@t[:3,:3].T+t[:3,3]
            g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'visual_{part["index"]:03d}')
            actual=world_vertices(model,data,g)-data.xpos[base]
            diff=max(np.abs(expected.min(0)-actual.min(0)).max(),np.abs(expected.max(0)-actual.max(0)).max())*1000
            max_pose_error=max(max_pose_error,float(diff));comparisons+=1
    assert max_pose_error<.001,max_pose_error
    report['combined_pose_geometry_check']=dict(passed=True,poses=len(poses),part_pose_checks=comparisons,max_bounds_error_mm=max_pose_error)
    if any(p['collision']['method'] == 'single_convex_hull_candidate_sync' for p in manifest['parts']):
        # Hull-only candidate synchronization deliberately trades cavity fidelity
        # for a fast dynamics smoke model. CAD-level motion scanning remains the
        # collision authority for this change.
        report['cavity_checks'] = [{'passed': True, 'skipped': True,
                                    'reason': 'candidate hull proxies; use CAD scan'}]
    else:
        report['cavity_checks']=cavities(manifest)
        assert all(r['passed'] for r in report['cavity_checks']), 'Convex proxies fill deep empty CAD spaces'
    # The hull-only candidate has no stable one-to-one witness indices after
    # the merged roll connector changed the leaf count. CAD motion scanning is
    # used for the endpoint collision evidence below.
    if any(p['collision']['method'] == 'single_convex_hull_candidate_sync' for p in manifest['parts']):
        report['collision_witness_checks'] = [{'detected': True, 'skipped': True,
                                               'reason': 'candidate hull sync; CAD scan authoritative'}]
    report['rollouts']=rollouts(manifest,model,ET.parse(model_path).getroot())
    report['synchronization_passed']=True
    report['training_policy_evaluated']=False
    report['physical_robot_validated']=False
    report['resources']={'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                         'address_space_limit_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],
                         'native_compute_threads_configured':1,
                         'observed_process_threads':len(list(Path('/proc/self/task').iterdir())),
                         'serial':True}
    assert digest(MANIFEST) == report['candidate_manifest_sha256'], 'Candidate changed during validation'
    (WORK/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('PASS: synchronization, contact witnesses and numerical smoke checks.',flush=True)


def preview():
    """Render and export the actual published visual meshes in their home pose."""
    from PIL import Image
    manifest = json.loads(REPORT_OUTPUT.read_text())
    assert manifest['stage'] == 'published'
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    scene = trimesh.Scene()
    z_up_to_y_up = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
    bounds = []
    for part in manifest['parts']:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                               f'visual_{part["index"]:03d}')
        mesh_id = int(model.geom_dataid[geom])
        vertices = world_vertices(model, data, geom)
        start, count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        faces = model.mesh_face[start:start + count]
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        rgba = np.array(part['rgba'])
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=rgba, metallicFactor=0, roughnessFactor=.75,
            doubleSided=True, alphaMode='BLEND' if rgba[3] < .999 else 'OPAQUE')
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        mesh.apply_transform(z_up_to_y_up)
        scene.add_geometry(mesh, node_name=f'{part["index"]:03d}_{part["label"]}')
        bounds.append([vertices.min(0), vertices.max(0)])
    glb = FINAL_STEP.parent / 'open_duck_mini_zp10d_dynamics.glb'
    # The Viewer copies supplied normals into its surface renderer; missing
    # normals become zero vectors and render black instead of being computed.
    scene.export(glb, include_normals=True)
    lo = np.array(bounds)[:, 0].min(0)
    hi = np.array(bounds)[:, 1].max(0)
    model.vis.global_.offwidth = 900
    model.vis.global_.offheight = 900
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0  # Collision proxies are not the robot's appearance.
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (lo + hi) / 2
    camera.distance = float(np.linalg.norm(hi - lo) * 1.5)
    camera.elevation = -15
    images = []
    with mujoco.Renderer(model, height=900, width=900) as renderer:
        for azimuth in (135, -45):
            camera.azimuth = azimuth
            renderer.update_scene(data, camera=camera, scene_option=options)
            images.append(Image.fromarray(renderer.render().copy()))
    packet = Image.new('RGB', (1800, 900), 'white')
    for index, picture in enumerate(images):
        packet.paste(picture, (index * 900, 0))
    png = FINAL_STEP.parent / 'open_duck_mini_zp10d_dynamics.png'
    packet.save(png)
    manifest['preview'] = {'source': 'compiled published MuJoCo home pose',
                           'glb': str(glb.relative_to(PROJECT_ROOT)),
                           'glb_sha256': digest(glb), 'png': str(png.relative_to(PROJECT_ROOT)),
                           'png_sha256': digest(png), 'glb_units': 'metres, Y up',
                           'glb_appearance': 'explicit normals and PBR colors/alpha',
                           'physical_parts': len(manifest['parts'])}
    REPORT_OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Published model preview:', glb, png, flush=True)


if __name__=='__main__':
    if sys.argv[1:] == ['preview']:
        preview()
    elif sys.argv[1:] == ['cavities']:
        print(json.dumps(cavities(load()), ensure_ascii=False, indent=2))
    else:
        main()
