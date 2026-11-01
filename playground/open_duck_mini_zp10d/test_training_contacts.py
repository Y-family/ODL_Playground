"""Check compiled foot contact support against the full CAD-derived vertices."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import mujoco
from mujoco import mjx
import numpy as np
from playground.open_duck_mini_zp10d.reference import ROOT, SCENE, source_hashes


def check_native_candidate(model, trajectory):
    import jax
    import jax.numpy as jp
    from playground.open_duck_mini_zp10d.contacts import sole_topology, plane_mesh_contacts
    from playground.open_duck_mini_zp10d.actuators import configure
    from playground.open_duck_mini_zp10d.reference import apply_reference_stance
    configure(model, 2.)
    apply_reference_stance(model, json.loads((ROOT / 'data/reference.json').read_text()))
    floor = model.geom('floor').id
    feet = [model.geom(side + '_foot_contact').id for side in ('left', 'right')]
    topologies = [sole_topology(model, g) for g in feet]
    rng = np.random.default_rng(3)
    poses = []
    for _ in range(256):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
        data.qpos[7:] += rng.uniform(-.1, .1, model.nq - 7)
        mujoco.mj_kinematics(model, data)
        height = min((v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min()
                     for g, (v, _) in zip(feet, topologies))
        data.qpos[2] -= height + rng.uniform(.00001, .001)
        poses.append(data.qpos.copy())
    if trajectory:
        with np.load(trajectory) as saved:
            poses.extend(saved['qpos'])
    cases = [[], []]
    for pose in poses:
        data = mujoco.MjData(model)
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for side, gid in enumerate(feet):
            native = [c for c in data.contact if set(c.geom) == {floor, gid}]
            cases[side].append((data.geom_xpos[floor].copy(), data.geom_xmat[floor].reshape(3, 3).copy(),
                data.geom_xpos[gid].copy(), data.geom_xmat[gid].reshape(3, 3).copy(),
                np.array([c.dist for c in native]), np.array([c.pos for c in native])))
    worst = 0.
    for gid, (vertices, adjacent), rows in zip(feet, topologies, cases):
        evaluate = jax.jit(jax.vmap(lambda p, r, c, q: plane_mesh_contacts(
            p, r, c, q, jp.array(vertices), jp.array(adjacent), model.geom_rbound[gid])))
        inputs = [jp.array(np.stack([r[k] for r in rows])) for k in range(4)]
        distances, points, frames = map(np.asarray, evaluate(*inputs))
        assert np.isfinite(points).all() and np.isfinite(frames).all()
        for row, dist, pos in zip(rows, distances, points):
            active = dist <= 0.
            assert active.sum() == len(row[4]), (dist, row[4])
            if active.any():
                np.testing.assert_allclose(dist[active], row[4], atol=2e-7, rtol=0.)
                np.testing.assert_allclose(pos[active], row[5], atol=2e-7, rtol=0.)
                worst = max(worst, float(np.max(abs(pos[active] - row[5]))))
    print(f'PASS: {jax.default_backend()} native sole candidate matches C MuJoCo contact counts/order '
          f'and points for {len(poses)*2} independent foot states; max position error {worst:g}m. '
          'Not trajectory or training acceptance.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=Path('/tmp/zp10d-training/scene.xml'))
    parser.add_argument('--native-candidate', action='store_true')
    parser.add_argument('--trajectory', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.scene.with_name('manifest.json').read_text())
    assert manifest['source_hashes'] == source_hashes()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    training_model = mjx.put_model(model)
    full = mujoco.MjModel.from_xml_path(str(SCENE))
    full_data = mujoco.MjData(full)
    mujoco.mj_resetDataKeyframe(full, full_data, full.key('home').id)
    mujoco.mj_forward(full, full_data)
    normals = np.random.default_rng(0).normal(size=(2048, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    errors = []
    for side, name in enumerate(('left', 'right')):
        gid = model.geom(name + '_foot_contact').id
        assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
        mid = model.geom_dataid[gid]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        site = model.site(name + '_foot').id
        local = (world - data.site_xpos[site]) @ data.site_xmat[site].reshape(3, 3)
        source = full.geom(manifest['sole_source_geoms'][name + '_foot_contact']).id
        source_mesh = full.geom_dataid[source]
        original = full.mesh_vert[full.mesh_vertadr[source_mesh]:
            full.mesh_vertadr[source_mesh] + full.mesh_vertnum[source_mesh]]
        original = original @ full_data.geom_xmat[source].reshape(3, 3).T + full_data.geom_xpos[source]
        source_site = full.site(name + '_foot').id
        original = (original - full_data.site_xpos[source_site]) @ full_data.site_xmat[source_site].reshape(3, 3)
        error = np.max(abs((original @ normals.T).min(0) - (local @ normals.T).min(0)))
        assert error < 2e-7, (name, error)
        training_vertices = np.asarray(training_model._impl.mesh_convex[mid].vert)
        training_world = training_vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        training_local = (training_world - data.site_xpos[site]) @ data.site_xmat[site].reshape(3, 3)
        training_error = np.max(abs((original @ normals.T).min(0) - (training_local @ normals.T).min(0)))
        assert training_error < 2e-7, (name, training_error)
        errors.append(float(error))
    print('PASS: native and MJX convex sole meshes preserve original CAD sole plane support '
          f'for 2048 directions per foot; maximum errors {errors} m. Not walking acceptance.')
    del full, full_data
    if args.native_candidate:
        check_native_candidate(model, args.trajectory)


if __name__ == '__main__':
    main()
