"""Prepare a documented primitive contact approximation for MJX training.

The full CAD-derived scene is unchanged and remains the dynamics acceptance
model. This training model keeps every joint, inertia, actuator, sensor and home;
it uses floor-only contact with box envelopes and the original CAD sole meshes.
Matching a foot's outer hull does not preserve its contact manifold. Higher foot
parts are omitted here; full-model validation remains necessary.
"""

import gc
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from reference import MODEL, SCENE, ROOT, source_hashes

SCRATCH = Path('/tmp/zp10d-training')
TRAINING_XML = SCRATCH / 'scene.xml'


def prepare(scratch=SCRATCH):
    import mujoco
    full = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(full)
    mujoco.mj_resetDataKeyframe(full, data, full.key('home').id)
    mujoco.mj_forward(full, data)
    tree = ET.parse(MODEL)
    model = tree.getroot()
    source_meshes = {mesh.get('name'): copy.deepcopy(mesh) for mesh in model.findall('asset/mesh')}
    # Explicit inertials own mass; removing visual/collision meshes cannot change it.
    for asset in model.findall('asset'):
        model.remove(asset)
    assets = ET.SubElement(model, 'asset')
    bodies = {b.get('name'): b for b in model.findall('.//body')}
    bounds = {}
    sole_sources = {}
    for name, body in bodies.items():
        sole = [copy.deepcopy(g) for g in body.findall('geom')
                if g.get('class') == 'collision' and 'foot_bottom_tpu' in g.get('mesh', '')]
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
        size = (high - low) / 2
        assert (size > 0).all()
        label = ('left_foot_contact' if name == 'foot_assembly' else
                 'right_foot_contact' if name == 'foot_assembly_2' else name + '_envelope')
        attrs = dict(name=label, type='box', pos=' '.join(map(str, (low + high) / 2)),
                     size=' '.join(map(str, size)), contype='0', conaffinity='1', mass='0',
                     friction='0.6 0.01 0.001', solref='0.01 1', solimp='0.9 0.95 0.001')
        if name in ('foot_assembly', 'foot_assembly_2'):
            if len(sole) != 1:
                raise ValueError(f'{name}: expected one original CAD sole collision mesh')
            # Preserve the real sole's faces and placement. A whole-foot hull
            # changes the floor contact manifold despite matching outer bounds.
            geom = sole[0]
            sole_sources[label] = geom.get('name')
            mesh = source_meshes[geom.get('mesh')]
            mesh.set('file', str((MODEL.parent/'assets'/mesh.get('file')).resolve()))
            assets.append(mesh)
            geom.set('name', label)
            geom.set('contype', '0')
            geom.set('conaffinity', '1')
            body.append(geom)
        else:
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
    scratch.mkdir(exist_ok=True, parents=True)
    training_xml = scratch / 'scene.xml'
    ET.indent(model, space='  ')
    ET.ElementTree(model).write(training_xml, encoding='utf-8', xml_declaration=True)
    small = mujoco.MjModel.from_xml_path(str(training_xml))
    for attr in ('nq', 'nv', 'nu'):
        assert getattr(small, attr) == getattr(full, attr)
    for attr in ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'body_pos', 'body_quat',
                 'jnt_pos', 'jnt_axis', 'jnt_range', 'dof_armature', 'dof_damping', 'dof_frictionloss',
                 'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'actuator_ctrlrange',
                 'key_qpos', 'key_ctrl'):
        np.testing.assert_allclose(getattr(small, attr), getattr(full, attr), atol=1e-12, rtol=0,
                                   err_msg=attr)
    # This fails for the old whole-foot hull even when its bounds are exact.
    for label, source_name in sole_sources.items():
        a, b = small.geom(label).id, full.geom(source_name).id
        for attr in ('geom_pos', 'geom_quat', 'geom_friction', 'geom_solref', 'geom_solimp'):
            np.testing.assert_array_equal(getattr(small, attr)[a], getattr(full, attr)[b])
        am, bm = small.geom_dataid[a], full.geom_dataid[b]
        for kind in ('vert', 'face'):
            start, count = getattr(small, f'mesh_{kind}adr')[am], getattr(small, f'mesh_{kind}num')[am]
            other, other_count = getattr(full, f'mesh_{kind}adr')[bm], getattr(full, f'mesh_{kind}num')[bm]
            assert count == other_count
            np.testing.assert_array_equal(getattr(small, f'mesh_{kind}')[start:start+count],
                                          getattr(full, f'mesh_{kind}')[other:other+count])
    metadata = dict(source_hashes=source_hashes(), full_geoms=full.ngeom, training_geoms=small.ngeom,
                    approximation='original CAD sole collision meshes; boxes on other bodies; floor-only; self-contact and upper-foot contacts omitted',
                    foot_contacts='one unmodified original foot_bottom_tpu collision mesh per foot; full model still required for acceptance',
                    sole_source_geoms=sole_sources,
                    training_xml_sha256=hashlib.sha256(training_xml.read_bytes()).hexdigest(),
                    unchanged='joint tree, inertias, armature, damping, frictionloss, gains, force/position limits, home',
                    bounds=bounds)
    (scratch / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'bounds'}, indent=2), flush=True)
    del full, data
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SCRATCH)
    prepare(parser.parse_args().output)
