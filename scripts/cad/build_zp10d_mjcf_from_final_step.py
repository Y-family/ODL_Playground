#!/usr/bin/env python3
"""Synchronize the current STEP to MuJoCo. Run through run_memory_limited.py.

prepare -> collision INDEX (one process per part) -> assemble -> validate
Candidates stay in /tmp until the independent validation permits publish.
The CAD geometry is an input and is never rewritten.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import shutil
import sys
import xml.etree.ElementTree as ET

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import numpy as np
import trimesh
from cadgen import build123d as bd, read_step

from full_motion_current import JOINT_NAMES, LIMITS, _leaves, is_moving, measured_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_STEP = PROJECT_ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
ORIGINAL_STEP = PROJECT_ROOT / 'cad/source_exports/open_duck_original_isolated_branch.step'
ROBOT_PICKLE = PROJECT_ROOT / 'cad/zp10d_onshape_export/robot.pkl'
MODEL_DIR = PROJECT_ROOT / 'playground/open_duck_mini_zp10d/xmls'
MJCF_TEMPLATE = MODEL_DIR / 'open_duck_mini_zp10d_draft.xml'
MJCF_OUTPUT = MODEL_DIR / 'open_duck_mini_zp10d.xml'
SCENE = MODEL_DIR / 'scene_flat_terrain.xml'
ASSET_DIR = MODEL_DIR / 'assets'
REPORT_OUTPUT = FINAL_STEP.parent / 'open_duck_mini_zp10d_mass_properties.json'
WORK = Path('/tmp/open-duck-dynamics-sync')
MANIFEST = WORK / 'manifest.json'
ONSHAPE_TO_ROBOT_SITE_PACKAGES = Path('/root/.local/share/uv/tools/onshape-to-robot/lib/python3.12/site-packages')
ZP10D_MASS_KG = 0.056
MESH_DEFLECTION_MM = 0.35
COLLISION_THRESHOLD_MM = 1.0
FALLBACK_DENSITY_KG_MM3 = 1.25e-6
ANTENNA_NAMES = {'A_plus_y': 'left_antenna', 'A_minus_y': 'right_antenna'}
PRINTED = set('left_cache right_cache antenna left_antenna_holder right_antenna_holder head_pitch_to_yaw neck_left_sheet neck_right_sheet roll_motor_top roll_motor_bottom leg_spacer left_knee_to_ankle_left_sheet left_knee_to_ankle_right_sheet body_front body_middle_bottom body_back body_middle_top battery_pack_lid head_yaw_to_roll right_roll_to_pitch left_roll_to_pitch foot_side foot_top foot_bottom_tpu foot_bottom_pla head head_bot_sheet head_roll_mount trunk_top trunk_bottom speaker_stand speaker_interface flash_reflector_interface flash_light_module'.split())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    data = json.loads(MANIFEST.read_text())
    assert digest(FINAL_STEP) == data['source_step_sha256'], 'Source STEP changed'
    return data


def save(data):
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def _load_robot():
    import pickle
    sys.path.append(str(ONSHAPE_TO_ROBOT_SITE_PACKAGES))
    with ROBOT_PICKLE.open('rb') as stream:
        return pickle.load(stream)


def _format_vector(values):
    return ' '.join(f'{float(x):.14g}' for x in values)


def write_xml(tree, path):
    xml = ET.tostring(tree.getroot(), encoding='unicode', xml_declaration=True)
    Path(path).write_text('\n'.join(line.rstrip() for line in xml.splitlines()) + '\n')


def _normalize(label):
    return re.sub(r'_\d+$', '', label.lower())


def properties(shape, mass):
    center = bd.Shape.combined_center([shape], bd.CenterOf.MASS)
    inertia = np.asarray(shape.matrix_of_inertia) * (mass/shape.volume) * 1e-6
    return np.array(tuple(center))*1e-3, (inertia+inertia.T)*0.5


def visual_bounds(shape, mesh_bounds_mm):
    box = shape.bounding_box()
    conservative = np.array([tuple(box.min), tuple(box.max)])
    bounds = conservative.copy()
    method = 'cad_bounding_box'
    if np.abs(bounds - mesh_bounds_mm).max() > MESH_DEFLECTION_MM + .01:
        # OCCT can overestimate trimmed spline extrema. Measure the support
        # planes against the actual BRep before attributing the gap to meshing.
        for axis in range(3):
            for side in range(2):
                plane = bd.Face.make_plane(bd.Plane(origin=conservative[side],
                                                    z_dir=np.eye(3)[axis]))
                bounds[side, axis] += shape.distance_to(plane) * (1 if side == 0 else -1)
        method = 'brep_distance_to_outer_support_planes'
    assert np.abs(bounds - mesh_bounds_mm).max() <= MESH_DEFLECTION_MM + .01
    return {'bbox_mm': bounds.tolist(), 'bbox_method': method,
            'conservative_bbox_mm': conservative.tolist()}


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat *= -1
    return quat



def _set_pose(element, parent, world):
    local = np.linalg.inv(parent) @ world
    element.set('pos', _format_vector(local[:3,3]))
    element.set('quat', _format_vector(_matrix_to_quaternion(local[:3,:3])))


def _rgba(shape):
    color = getattr(shape, 'color', None)
    if color is not None:
        return list(color)
    return [.65,.68,.72,1]


def prepare():
    import mujoco
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    WORK.mkdir(exist_ok=True)
    (WORK/'assets').mkdir(exist_ok=True)
    source_sha = digest(FINAL_STEP)
    original = _leaves(read_step(ORIGINAL_STEP))
    current_root = read_step(FINAL_STEP)
    current = _leaves(current_root)
    assert len(current) == 115
    reference = defaultdict(list)
    for row in original:
        reference[(row['root'],row['label'])].append(row)
    robot = _load_robot()
    baseline_parts = [p for link in robot.links for p in link.parts
                      if not p.name.startswith(('wj_wk00_', 'drive_palonier', 'passive_palonier'))]
    baseline_by_label = defaultdict(list)
    for part in baseline_parts:
        baseline_by_label[_normalize(part.name)].append(part)
    baseline_by_label['head'] = baseline_by_label.pop('head', [])
    axes = measured_axes(current_root)
    body_for = {sid:robot.get_joint(name).child.name for sid,name in JOINT_NAMES.items()}
    frames = {'trunk_assembly':np.eye(4)}
    parents = {}
    for sid,name in JOINT_NAMES.items():
        joint = robot.get_joint(name)
        frame = joint.T_world_joint.copy()
        frame[:3,3] = np.array(axes[sid][0])*1e-3
        frames[joint.child.name] = frame
        parents[joint.child.name] = joint.parent.name
    for row in current:
        if row['label'] != 'sg90':
            continue
        side = 'A_plus_y' if row['shape'].center().Y > .1 else 'A_minus_y'
        cylinders = []
        for face in row['shape'].faces():
            surface = BRepAdaptor_Surface(face.wrapped, True)
            if surface.GetType()==GeomAbs_Cylinder and abs(surface.Cylinder().Radius()-2.425)<1e-6:
                cylinder=surface.Cylinder()
                cylinders.append([list(cylinder.Location().Coord()),list(cylinder.Axis().Direction().Coord())])
        assert cylinders, 'SG90 shaft cylinder missing'
        axes[side] = cylinders[0]
        name = ANTENNA_NAMES[side]
        frame=np.eye(4);frame[:3,3]=np.array(axes[side][0])*1e-3
        frames[name+'_assembly']=frame
        parents[name+'_assembly']='head_assembly'
    used_reference=defaultdict(int)
    servo_volumes=defaultdict(float)
    for row in current:
        if row['servo']:
            servo_volumes[row['servo']]+=row['shape'].volume
    parts=[]
    for index,row in enumerate(current):
        shape=row['shape']
        chain=[sid for sid in JOINT_NAMES if is_moving(row,sid)]
        body=body_for[chain[-1]] if chain else 'trunk_assembly'
        if row['label'] in ('antenna','left_antenna_holder','right_antenna_holder'):
            side='A_plus_y' if shape.center().Y>.1 else 'A_minus_y'
            chain.append(side);body=ANTENNA_NAMES[side]+'_assembly'
        volume=shape.volume
        assert volume>0 and len(shape.solids())==1, row['path']
        if row['servo']:
            mass=ZP10D_MASS_KG*volume/servo_volumes[row['servo']]
            provenance={'method':'vendor_group_mass_distributed_by_current_solid_volume',
                        'group':row['servo'],'group_mass_kg':ZP10D_MASS_KG,'measured':False}
        else:
            key=(row['root'],row['label']);ordinal=used_reference[key];used_reference[key]+=1
            candidates=reference[key]
            assert ordinal<len(candidates), f'No source BRep for {row["path"]}'
            before=candidates[ordinal]['shape']
            before_com=np.array(tuple(bd.Shape.combined_center([before],bd.CenterOf.MASS)))*1e-3
            label=_normalize(row['label'])
            if label in ('speaker_bottom','speaker_top'):
                nominal=baseline_by_label['full_speaker'][0]
                group_volume=sum(item['shape'].volume for item in original
                                 if item['root']==row['root'] and item['label'] in ('speaker_bottom','speaker_top'))
                old_mass=float(nominal.mass)*before.volume/group_volume
            else:
                choices=baseline_by_label[label]
                assert choices, f'No baseline mass for {row["path"]}'
                nominal=min(choices,key=lambda p:np.linalg.norm((p.T_world_part@[*p.com,1])[:3]-before_com))
                old_mass=float(nominal.mass)
            printed=row['label'] in PRINTED
            if old_mass<=0:
                mass=volume*FALLBACK_DENSITY_KG_MM3
                provenance={'method':'provisional_PLA_density_for_missing_legacy_mass',
                            'density_kg_m3':1250,'measured':False,'legacy_mass_kg':old_mass}
            else:
                mass=old_mass*volume/before.volume if printed else old_mass
                provenance={'method':'legacy_effective_density' if printed else 'legacy_component_mass',
                            'reference_part':nominal.name,'reference_mass_kg':old_mass,
                            'reference_volume_mm3':before.volume,'measured':False}
            provenance['reference_com_world_m']=before_com.tolist()
        com,inertia=properties(shape,mass)
        vertices,faces=shape.tessellate(MESH_DEFLECTION_MM,.3)
        mesh=trimesh.Trimesh(np.array([tuple(v) for v in vertices])*1e-3-com,
                             np.array(faces),process=True)
        bounds = visual_bounds(shape, (mesh.bounds + com) * 1000)
        stem=f'current_{index:03d}_{row["label"]}'
        output=WORK/'assets'/f'{stem}.stl'
        mesh.export(output)
        parts.append(dict(index=index,path=row['path'],label=row['label'],root=row['root'],
                          servo=row['servo'],chain=chain,body=body,mass_kg=mass,
                          volume_mm3=volume,com_world_m=com.tolist(),
                          inertia_world_kg_m2=inertia.tolist(),**bounds,
                          rgba=_rgba(shape),mesh=f'{stem}.stl',mesh_sha256=digest(output),
                          mass_provenance=provenance,mesh_vertices=len(mesh.vertices),mesh_faces=len(mesh.faces)))
        print('export',index,row['label'],f'{mass*1000:.3f} g',flush=True)
        del mesh,vertices,faces
        gc.collect()
    # Preserve sensor conventions; translate sensor datums with their actual CAD hosts.
    sites={}
    for link in robot.links:
        for name,frame in link.frames.items():
            sites[name]=frame.copy()
    host_labels={'imu':'BNO055','left_foot':'foot_bottom_tpu','right_foot':'foot_bottom_tpu','head':'head','trunk':'trunk_top'}
    for name,label in host_labels.items():
        matches=[p for p in parts if p['label']==label]
        if name in ('left_foot','right_foot'):
            expected='foot_assembly' if name=='left_foot' else 'foot_assembly_2'
            matches=[p for p in matches if p['body']==expected]
        assert len(matches)==1,(name,len(matches))
        part=matches[0]
        sites[name][:3,3]+=np.array(part['com_world_m'])-np.array(part['mass_provenance']['reference_com_world_m'])
    old_model=mujoco.MjModel.from_xml_path(str(SCENE))
    home={mujoco.mj_id2name(old_model,mujoco.mjtObj.mjOBJ_JOINT,j):float(old_model.key_qpos[0,old_model.jnt_qposadr[j]])
          for j in range(old_model.njnt) if old_model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE}
    home_ctrl={old_model.actuator(a).name:float(old_model.key_ctrl[0,a]) for a in range(old_model.nu)}
    data=dict(source_step=str(FINAL_STEP.relative_to(PROJECT_ROOT)),source_step_sha256=source_sha,
              original_step_sha256=digest(ORIGINAL_STEP),robot_pickle_sha256=digest(ROBOT_PICKLE),
              template_sha256=digest(MJCF_TEMPLATE),previous_mjcf_sha256=digest(MJCF_OUTPUT),
              axes=axes,limits_deg=LIMITS,body_for=body_for,
              frames={k:v.tolist() for k,v in frames.items()},parents=parents,
              sites={k:v.tolist() for k,v in sites.items()},parts=parts,
              home_joint_angles_rad=home,home_ctrl_rad=home_ctrl,home_free_qpos=old_model.key_qpos[0,:7].tolist(),
              original_actuator_order=[name for a in range(old_model.nu)
                  if (name:=mujoco.mj_id2name(old_model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)) in JOINT_NAMES.values()],
              baseline_total_mass_kg=sum(p.mass for l in robot.links for p in l.parts),
              updated_total_mass_kg=sum(p['mass_kg'] for p in parts),stage='prepared')
    assert digest(FINAL_STEP)==source_sha
    save(data)
    print('Prepared',len(parts),'physical leaves; mass',data['updated_total_mass_kg'],flush=True)


def collision(index):
    import coacd
    from scipy.spatial import QhullError
    data=load();part=data['parts'][index]
    mesh=trimesh.load(WORK/'assets'/part['mesh'],force='mesh',process=True)
    hull=mesh.convex_hull
    ratio=abs(hull.volume)/max(abs(mesh.volume),1e-12)
    threshold = .5 if part['label'] in ('head', 'head_yaw_to_roll','left_cache','right_cache') else COLLISION_THRESHOLD_MM
    if mesh.is_watertight and ratio<1.02:
        pieces=[(hull.vertices*1000,hull.faces)]
        method='single_convex_hull_under_2_percent_volume_difference'
    elif part['servo'] or (part['mass_provenance']['method'] == 'legacy_component_mass'
                          and part['label'] != 'roll_bearing'):
        # Small purchased-component holes need not become hundreds of contacts.
        # Preserve cavities in printed shells/brackets with decomposition below.
        pieces=[(hull.vertices*1000,hull.faces)]
        method=('conservative_servo_housing_or_horn_hull' if part['servo'] else
                'conservative_purchased_component_hull')
    else:
        coacd.set_log_level('warn')
        pieces=coacd.run_coacd(coacd.Mesh(mesh.vertices*1000,mesh.faces),
                              threshold=threshold,real_metric=True,
                              max_convex_hull=64,preprocess_mode='off',
                              resolution=1000,mcts_nodes=5,mcts_iterations=5,mcts_max_depth=2,
                              merge=True,seed=0)
        method='coacd_convex_decomposition'
    assert pieces, f'No collision geometry for {part["path"]}'
    outputs=[]; discarded=0
    for n,(vertices,faces) in enumerate(pieces):
        try:
            piece=trimesh.Trimesh(np.asarray(vertices)*1e-3,np.asarray(faces),process=True).convex_hull
        except QhullError:
            discarded+=1
            continue
        if piece.volume<=1e-15 or not piece.is_watertight:
            discarded+=1
            continue
        name=f'{Path(part["mesh"]).stem}_c{n:03d}.stl'
        path=WORK/'assets'/name;piece.export(path)
        outputs.append({'mesh':name,'sha256':digest(path),'volume_mm3':float(piece.volume*1e9),
                        'vertices':len(piece.vertices)})
    assert outputs, f'All collision pieces degenerate: {part["path"]}'
    part['collision']={'method':method,'threshold_mm':threshold if method=='coacd_convex_decomposition' else None,
                       'hull_limit':64 if method=='coacd_convex_decomposition' else 1,'preprocess_mode':'off',
                       'convex_piece_count':len(outputs),'discarded_degenerate_pieces':discarded,
                       'input_hull_volume_ratio':ratio,'pieces':outputs,
                       'approximate':True,'not_a_0_2_mm_clearance_certificate':True}
    save(data)
    print('collision',index,part['label'],'pieces',len(outputs),'hull ratio',round(ratio,3),flush=True)


def _link_properties(parts,frame):
    mass=sum(p['mass_kg'] for p in parts)
    assert mass>0
    com=sum(np.array(p['com_world_m'])*p['mass_kg'] for p in parts)/mass
    inertia=np.zeros((3,3))
    for p in parts:
        delta=np.array(p['com_world_m'])-com
        inertia+=np.array(p['inertia_world_kg_m2'])+p['mass_kg']*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
    rotation=frame[:3,:3]
    local_inertia=rotation.T@inertia@rotation
    eig=np.linalg.eigvalsh(local_inertia)
    assert eig[0]>0 and eig[-1]<=eig[:2].sum()+1e-12
    return dict(mass_kg=mass,com_link_m=(rotation.T@(com-frame[:3,3])).tolist(),
                com_world_m=com.tolist(),inertia_link_kg_m2=local_inertia.tolist(),
                inertia_eigenvalues_kg_m2=eig.tolist(),part_indices=[p['index'] for p in parts])


def assemble():
    import mujoco
    data=load()
    assert all('collision' in p for p in data['parts']), 'Collision export incomplete'
    assert digest(MJCF_TEMPLATE)==data['template_sha256']
    parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree=ET.parse(MJCF_TEMPLATE,parser=parser);root=tree.getroot()
    root.set('model','open_duck_mini_zp10d')
    root.insert(0,ET.Comment(f' Current STEP SHA256 {data["source_step_sha256"]}; estimated mass distribution and actuator response, not measured hardware. '))
    root.find('compiler').set('meshdir',str(WORK/'assets'))
    root.find('compiler').set('autolimits','true')
    option=root.find('option');option.set('iterations','50');option.set('ls_iterations','10');option.set('timestep','0.002')
    for node in root.iter():
        for attr in ('class','childclass'):
            if node.get(attr)=='zp10d_draft':node.set(attr,'zp10d')
            elif node.get(attr)=='open_duck_mini_v2':node.set(attr,'open_duck_mini_zp10d')
    defaults=root.find('default')
    antenna_default=ET.SubElement(defaults,'default',{'class':'sg90_provisional'})
    ET.SubElement(antenna_default,'joint',{'damping':'0.005','frictionloss':'0.001','armature':'0.00001'})
    ET.SubElement(antenna_default,'position',{'kp':'0.3','kv':'0.002','forcerange':'-0.1 0.1'})
    bodies={b.get('name'):b for b in root.iter('body')}
    frames={k:np.array(v) for k,v in data['frames'].items()}
    base=bodies['base'];base.set('pos','0 0 0.22');base.set('quat','1 0 0 0')
    for side,name in ANTENNA_NAMES.items():
        child=ET.SubElement(bodies['head_assembly'],'body',{'name':name+'_assembly'})
        ET.SubElement(child,'joint',{'name':name,'type':'hinge','class':'sg90_provisional',
                                    'range':_format_vector([-math.pi/2,math.pi/2])})
        bodies[name+'_assembly']=child
    for sid,name in JOINT_NAMES.items():
        joint=root.find(f'.//joint[@name="{name}"]')
        assert joint is not None
        frame=frames[data['body_for'][sid]]
        joint.set('axis',_format_vector(frame[:3,:3].T@np.array(data['axes'][sid][1])))
        joint.set('pos','0 0 0')
        joint.set('range',_format_vector(np.radians(data['limits_deg'][sid])))
    for side,name in ANTENNA_NAMES.items():
        joint=bodies[name+'_assembly'].find('joint')
        joint.set('axis',_format_vector(np.array(data['axes'][side][1])))
    asset=root.find('asset')
    for mesh in list(asset.findall('mesh')):asset.remove(mesh)
    used_assets=set();links={}
    for name,frame in frames.items():
        body=bodies[name]
        parent=np.eye(4) if name=='trunk_assembly' else frames[data['parents'][name]]
        _set_pose(body,parent,frame)
        for node in list(body):
            if node.tag in ('geom','inertial'):body.remove(node)
        parts=[p for p in data['parts'] if p['body']==name]
        links[name]=_link_properties(parts,frame)
        values=links[name];inertia=np.array(values['inertia_link_kg_m2'])
        ET.SubElement(body,'inertial',{'pos':_format_vector(values['com_link_m']),
                      'mass':f'{values["mass_kg"]:.14g}',
                      'fullinertia':_format_vector([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]])})
        for part in parts:
            world=np.eye(4);world[:3,3]=part['com_world_m']
            geom=ET.SubElement(body,'geom',{'name':f'visual_{part["index"]:03d}',
                        'type':'mesh','class':'visual','mesh':Path(part['mesh']).stem,
                        'rgba':_format_vector(part['rgba']),'contype':'0','conaffinity':'0','mass':'0'})
            _set_pose(geom,frame,world)
            filenames=[part['mesh']]+[p['mesh'] for p in part['collision']['pieces']]
            for filename in filenames:
                if filename not in used_assets:
                    ET.SubElement(asset,'mesh',{'name':Path(filename).stem,'file':filename,'inertia':'shell'})
                    used_assets.add(filename)
            for n,piece in enumerate(part['collision']['pieces']):
                geom=ET.SubElement(body,'geom',{'name':f'collision_{part["index"]:03d}_{n:03d}',
                        'type':'mesh','class':'collision','mesh':Path(piece['mesh']).stem,
                        'contype':'1','conaffinity':'1','mass':'0','friction':'0.6 0.01 0.001',
                        'solref':'0.01 1','solimp':'0.9 0.95 0.001'})
                _set_pose(geom,frame,world)
        for site in body.findall('site'):
            site_name=site.get('name')
            if site_name in data['sites']:_set_pose(site,frame,np.array(data['sites'][site_name]))
    for site in base.findall('site'):
        if site.get('name') in data['sites']:_set_pose(site,np.eye(4),np.array(data['sites'][site.get('name')]))
    actuators=root.find('actuator')
    for node in list(actuators):
        if node.get('joint') not in JOINT_NAMES.values():actuators.remove(node)
    for name in ANTENNA_NAMES.values():
        ET.SubElement(actuators,'position',{'class':'sg90_provisional','name':name,'joint':name,'inheritrange':'1'})
    candidate=WORK/MJCF_OUTPUT.name
    ET.indent(tree,space='  ');write_xml(tree, candidate)
    model=mujoco.MjModel.from_xml_path(str(candidate))
    # Joint traversal adds antennas within the head subtree; keyframes must be remapped by name.
    scene_tree=ET.parse(SCENE,parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    scene_root=scene_tree.getroot();scene_root.find('include').set('file',str(candidate))
    qpos=model.qpos0.copy();qpos[:7]=data['home_free_qpos']
    for name,value in data['home_joint_angles_rad'].items():
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        assert jid>=0;qpos[model.jnt_qposadr[jid]]=value
    home_ctrl=data.get('home_ctrl_rad',data['home_joint_angles_rad'])
    ctrl=[home_ctrl.get(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a),0.) for a in range(model.nu)]
    key=scene_root.find('keyframe/key[@name="home"]');key.set('qpos',_format_vector(qpos));key.set('ctrl',_format_vector(ctrl))
    ET.indent(scene_tree,space='  ');write_xml(scene_tree, WORK/SCENE.name)
    scene_model = mujoco.MjModel.from_xml_path(str(WORK/SCENE.name))
    state = mujoco.MjData(scene_model)
    mujoco.mj_resetDataKeyframe(scene_model, state, 0)
    mujoco.mj_forward(scene_model, state)
    floor = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    normal = state.geom_xmat[floor].reshape(3, 3)[:, 2]
    assert floor >= 0 and np.allclose(normal, [0, 0, 1])
    lowest = math.inf
    for geom in range(scene_model.ngeom):
        if not scene_model.geom_contype[geom] or scene_model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = scene_model.geom_dataid[geom]
        start, count = scene_model.mesh_vertadr[mesh], scene_model.mesh_vertnum[mesh]
        vertices = (scene_model.mesh_vert[start:start + count]
                    @ state.geom_xmat[geom].reshape(3, 3).T + state.geom_xpos[geom])
        lowest = min(lowest, float(((vertices - state.geom_xpos[floor]) @ normal).min()))
    # An old home height can bury enlarged feet in the floor. Translate the
    # free base only; keep every CAD-relative part pose and joint angle intact.
    shift = .0002 - lowest
    qpos[2] += shift
    data['home_ground_alignment'] = {'previous_lowest_collision_height_m': lowest,
                                     'base_height_adjustment_m': shift,
                                     'initial_ground_gap_m': .0002,
                                     'base_z_m': float(qpos[2])}
    data['home_free_qpos'] = qpos[:7].tolist()
    key.set('qpos', _format_vector(qpos))
    write_xml(scene_tree, WORK/SCENE.name)
    data.update(stage='assembled',links=links,candidate_mjcf_sha256=digest(candidate),
                candidate_scene_sha256=digest(WORK/SCENE.name),asset_count=len(used_assets),
                collision_piece_count=sum(p['collision']['convex_piece_count'] for p in data['parts']))
    save(data)
    print('Assembled current candidate:',len(links),'bodies,',model.nu,'actuators,',data['collision_piece_count'],'collision pieces',flush=True)


def publish():
    import mujoco
    data=load()
    validation=json.loads((WORK/'validation.json').read_text())
    assert validation['synchronization_passed'], 'Synchronization checks did not pass'
    assert validation['mjcf_sha256']==digest(WORK/MJCF_OUTPUT.name)
    assert validation['scene_sha256']==digest(WORK/SCENE.name)
    assert validation['candidate_manifest_sha256']==digest(MANIFEST)
    assert validation['source_step_sha256']==data['source_step_sha256']
    assert data['stage']=='assembled', data['stage']
    for part in data['parts']:
        for filename,expected in [(part['mesh'],part['mesh_sha256'])]+[(p['mesh'],p['sha256']) for p in part['collision']['pieces']]:
            source=WORK/'assets'/filename
            assert digest(source)==expected
            temp=ASSET_DIR/(filename+'.tmp');shutil.copyfile(source,temp);temp.replace(ASSET_DIR/filename)
            assert digest(ASSET_DIR/filename)==expected
    tree=ET.parse(WORK/MJCF_OUTPUT.name);tree.getroot().find('compiler').set('meshdir','assets')
    temp=MJCF_OUTPUT.with_suffix('.xml.tmp');write_xml(tree, temp)
    scene_tree=ET.parse(WORK/SCENE.name);scene_tree.getroot().find('include').set('file',MJCF_OUTPUT.name)
    scene_temp=SCENE.with_suffix('.xml.tmp');write_xml(scene_tree, scene_temp)
    mujoco.MjModel.from_xml_path(str(temp))
    temp.replace(MJCF_OUTPUT);scene_temp.replace(SCENE)
    compiled=mujoco.MjModel.from_xml_path(str(SCENE))
    candidate=mujoco.MjModel.from_xml_path(str(WORK/SCENE.name))
    for attribute in ('body_mass','body_ipos','body_inertia','body_pos','body_quat',
                      'jnt_pos','jnt_axis','jnt_range','geom_pos','geom_quat',
                      'actuator_ctrlrange','actuator_forcerange','key_qpos','key_ctrl'):
        assert np.array_equal(getattr(compiled,attribute),getattr(candidate,attribute)), attribute
    data.update(ok=True,stage='published',mjcf=str(MJCF_OUTPUT.relative_to(PROJECT_ROOT)),
                mjcf_sha256=digest(MJCF_OUTPUT),scene_sha256=digest(SCENE),validation=validation,
                published_verification={'compiled':True,'candidate_arrays_identical':True,
                                        'asset_hashes_verified':True,'nq':compiled.nq,
                                        'nv':compiled.nv,'nu':compiled.nu},
                assumptions={
                 'mass_status':'estimates; physical part weights, print settings and internal distributions are not measured',
                 'zp10d_mass_kg_each':ZP10D_MASS_KG,
                 'zp10d_mass_distribution':'56 g per current servo assembly, split over body and horns by current BRep volume; S01/S06 single-output actual mass remains unmeasured',
                 'printed_parts':'original robot-pickle mass divided by original BRep volume, applied to current BRep volume',
                 'other_components':'original component mass retained, current BRep used for centre and inertia with uniform effective density',
                 'legacy_zero_mass_parts':'flash reflector/light module assigned provisional 1250 kg/m3 instead of zero mass',
                 'sg90_actuation':'provisional kp=0.3, kv=0.002, +/-0.1 Nm; +/-90 degrees is a diagnostic range, not an established physical operating range',
                 'zp10d_response':'inherited kp=13.37, damping=0.56, frictionloss=0.068, armature=0.027; unmeasured',
                 'zp10d_torque':'published stall torque +/-0.980665 Nm; not continuous rated torque',
                 'collision':'printed shells/brackets use convex decomposition, with per-part target parameters recorded above; hull caps may exceed those targets. Servo housings/horns and purchased components use conservative hulls that fill holes. Original STEP self-intersection warnings remain; not a 0.2 mm clearance proof. Same-body and immediate parent-child contacts are excluded by MuJoCo.',
                 'structural_strength':'rigid-body model only; no FEA or printed-part strength certification',
                })
    REPORT_OUTPUT.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    save(data)
    print('Published current model;',data['updated_total_mass_kg'],'kg;',data['collision_piece_count'],'collision pieces')


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'prepare'
    if mode=='prepare':prepare()
    elif mode=='collision':collision(int(sys.argv[2]))
    elif mode=='assemble':assemble()
    elif mode=='publish':publish()
    else:raise SystemExit('usage: prepare | collision INDEX | assemble | publish')


if __name__=='__main__':
    main()
