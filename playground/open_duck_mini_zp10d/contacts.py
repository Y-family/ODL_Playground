"""MJX sole contacts matching MuJoCo 3.12's mjc_PlaneConvex.

The native algorithm uses the deepest hull vertex and up to two sufficiently
distant neighbours. MJX instead chooses four points in a 1 mm skin. This
adapter preserves the CAD hull and pads the native contacts to four slots.
Reference: google-deepmind/mujoco, 3.12.0, engine_collision_convex.c
(Apache-2.0), mjc_PlaneConvex/addplanemesh. Long closed-loop trajectories
remain sensitive to floating-point/contact differences.
"""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco.mjx._src import collision_driver, math
from mujoco.mjx._src.types import GeomType


_original_plane_mesh = collision_driver._COLLISION_FUNC[(GeomType.PLANE, GeomType.MESH)]


def sole_topology(model, geom):
    mid = model.geom_dataid[geom]
    adr = model.mesh_graphadr[mid]
    if adr < 0:
        raise ValueError('Native sole contacts require a convex hull graph')
    count = model.mesh_graph[adr]
    offsets = model.mesh_graph[adr + 2:adr + 2 + count]
    ids = model.mesh_graph[adr + 2 + count:adr + 2 + 2 * count]
    edges = model.mesh_graph[adr + 2 + 2 * count:]
    vertices = model.mesh_vert[model.mesh_vertadr[mid]:
                               model.mesh_vertadr[mid] + model.mesh_vertnum[mid]][ids].copy()
    rows = []
    for offset in offsets:
        row = []
        for index in edges[offset:]:
            if index < 0:
                break
            row.append(index)
        rows.append(row)
    adjacent = np.full((count, max(map(len, rows))), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        adjacent[index, :len(row)] = row
    return vertices, adjacent


def plane_mesh_contacts(plane_pos, plane_mat, pos, mat, vertices, adjacent, rbound):
    normal = plane_mat[:, 2]
    world = vertices @ mat.T + pos
    distance = (world - plane_pos) @ normal
    best = jp.argmin(distance)
    first = world[best] - .5 * distance[best] * normal
    neighbours = adjacent[best]
    safe = jp.maximum(neighbours, 0)
    valid = ((neighbours >= 0) & (distance[safe] < 0.)
             & (jp.linalg.norm(world[safe] - first, axis=1) >= .3 * rbound))
    # Preserve native graph order, selecting at most two neighbours.
    order = jp.where(valid, jp.arange(len(neighbours)), len(neighbours))
    chosen = jax.lax.top_k(-order, 2)[1]
    indices = jp.concatenate((best[None], safe[chosen], best[None]))
    enabled = jp.concatenate((jp.ones(1, dtype=bool), valid[chosen], jp.zeros(1, dtype=bool)))
    enabled &= distance[best] <= 0.
    dist = jp.where(enabled, distance[indices], 1.)
    points = world[indices] - .5 * dist[:, None] * normal
    frame = jp.broadcast_to(math.make_frame(normal), (4, 3, 3))
    return dist, points, frame


def install_native_sole_contacts(model):
    """Install only for this named training scene and its two zero-margin soles."""
    if mujoco.__version__ != '3.12.0':
        raise ValueError('Revalidate native contact selection for this MuJoCo version')
    tables = {}
    for side in ('left', 'right'):
        gid = model.geom(side + '_foot_contact').id
        if model.geom_margin[gid] != 0 or model.geom_margin[model.geom('floor').id] != 0:
            raise ValueError('Native sole adapter requires zero contact margins')
        tables[int(model.geom_dataid[gid])] = sole_topology(model, gid)

    def collide(m, d, key, geom):
        if m.names != model.names or key.data_ids[1] not in tables:
            return _original_plane_mesh(m, d, key, geom)
        vertices, adjacent = map(jp.asarray, tables[key.data_ids[1]])
        g1, g2 = geom.T
        result = jax.vmap(lambda a, b: plane_mesh_contacts(
            d.geom_xpos[a], d.geom_xmat[a], d.geom_xpos[b], d.geom_xmat[b],
            vertices, adjacent, m.geom_rbound[b]))(g1, g2)
        return jax.tree.map(jp.concatenate, result)

    collide.ncon = 4
    collision_driver._COLLISION_FUNC[(GeomType.PLANE, GeomType.MESH)] = collide
