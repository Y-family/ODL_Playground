"""Prepare a documented primitive contact approximation for MJX training.

The full CAD-derived scene is unchanged and remains the dynamics acceptance
model. This training model keeps every joint, inertia, actuator, sensor and home;
it uses floor-only contact, as in the original walking-training model, with one
box envelope per moving body. It cannot validate robot self-collision.
"""

import gc
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from reference import MODEL, SCENE, ROOT, source_hashes

SCRATCH = Path('/tmp/zp10d-training')
TRAINING_XML = SCRATCH / 'scene.xml'


def prepare():
    import mujoco
    full = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(full)
    mujoco.mj_resetDataKeyframe(full, data, full.key('home').id)
    mujoco.mj_forward(full, data)
    tree = ET.parse(MODEL)
    model = tree.getroot()
    # Explicit inertials own mass; removing visual/collision meshes cannot change it.
    for asset in model.findall('asset'):
        model.remove(asset)
    bodies = {b.get('name'): b for b in model.findall('.//body')}
    bounds = {}
    for name, body in bodies.items():
        for geom in body.findall('geom'):
            body.remove(geom)
        bid = full.body(name).id
        vertices = []
        for gid in np.where((full.geom_bodyid == bid) & (full.geom_contype != 0))[0]:
            mid = full.geom_dataid[gid]
            v = full.mesh_vert[full.mesh_vertadr[mid]:full.mesh_vertadr[mid] + full.mesh_vertnum[mid]]
            world = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            vertices.append((world - data.xpos[bid]) @ data.xmat[bid].reshape(3, 3))
        if not vertices:
            continue
        vertices = np.concatenate(vertices)
        low, high = vertices.min(0), vertices.max(0)
        # Foot body frames: local Y points up at accepted home. Keep the sole's
        # own outline and bottom height, instead of fitting the whole ankle envelope.
        if name in ('foot_assembly', 'foot_assembly_2'):
            sole = vertices[vertices[:, 1] < low[1] + .0005]
            low[[0, 2]], high[[0, 2]] = sole[:, [0, 2]].min(0), sole[:, [0, 2]].max(0)
            high[1] = low[1] + .003
        size = (high - low) / 2
        assert (size > 0).all()
        label = ('left_foot_contact' if name == 'foot_assembly' else
                 'right_foot_contact' if name == 'foot_assembly_2' else name + '_envelope')
        attrs = dict(name=label, type='box', pos=' '.join(map(str, (low + high) / 2)),
                     size=' '.join(map(str, size)), contype='0', conaffinity='1', mass='0',
                     friction='0.6 0.01 0.001', solref='0.01 1', solimp='0.9 0.95 0.001')
        ET.SubElement(body, 'geom', attrs)
        bounds[name] = dict(low=low.tolist(), high=high.tolist())
    compiler = model.find('compiler')
    compiler.attrib.pop('meshdir', None)
    scene = ET.parse(SCENE).getroot()
    model.find('worldbody').append(scene.find('worldbody/body'))
    for geom in model.findall('.//geom'):
        geom.attrib.pop('material', None)
    model.append(scene.find('keyframe'))
    # No graphical assets or giant mesh collision groups are needed on a learner.
    SCRATCH.mkdir(exist_ok=True)
    ET.indent(model, space='  ')
    ET.ElementTree(model).write(TRAINING_XML, encoding='utf-8', xml_declaration=True)
    small = mujoco.MjModel.from_xml_path(str(TRAINING_XML))
    for attr in ('nq', 'nv', 'nu'):
        assert getattr(small, attr) == getattr(full, attr)
    for attr in ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'body_pos', 'body_quat',
                 'jnt_pos', 'jnt_axis', 'jnt_range', 'dof_armature', 'dof_damping', 'dof_frictionloss',
                 'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'actuator_ctrlrange',
                 'key_qpos', 'key_ctrl'):
        np.testing.assert_allclose(getattr(small, attr), getattr(full, attr), atol=1e-12, rtol=0,
                                   err_msg=attr)
    metadata = dict(source_hashes=source_hashes(), full_geoms=full.ngeom, training_geoms=small.ngeom,
                    approximation='one box per body; floor contact only; self-contact omitted',
                    unchanged='joint tree, inertias, armature, damping, frictionloss, gains, force/position limits, home',
                    bounds=bounds)
    (SCRATCH / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'bounds'}, indent=2), flush=True)
    del full, data
    gc.collect()


if __name__ == '__main__':
    prepare()
