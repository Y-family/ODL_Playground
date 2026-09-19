"""Coarse rigid-placement search; all results require exact interface checks."""

import itertools
import json
import math
import sys

import compact_head_local as compact
from cadgen import build123d as bd, read_step
import numpy as np
import trimesh
from scan_s01_s14_motion_collisions import _mesh


def main():
    compact.rows()
    index = int(sys.argv[1])
    if index in (85, 92):
        source = bd.Compound([read_step(compact.WORK / f'leaf_{part}.step').moved(bd.Pos(-8, compact.SHORTEN_Y_SIDE_MM, 0))
                              for part in sorted(compact.SPEAKER if index == 85 else compact.SPEAKER - {85})])
    else:
        source = read_step(compact.WORK / f'candidate_{index}.step')
    mesh = _mesh(source, .25, .3)
    centre = mesh.bounds.mean(axis=0)
    fixed = trimesh.collision.CollisionManager()
    neighbours = ({82, 89, 81, 50, 53, 78, 90, 98, 86, 85, 80, 92, 93}
                  if index == 91 else {82, 89, 81, 50, 53, 78, 90, 98, 86, 91, 102})
    for other in sorted(neighbours):
        part = read_step(compact.WORK / f'candidate_{other}.step')
        fixed.add_object(str(other), _mesh(part, .25, .3))
        del part
    ring = trimesh.collision.CollisionManager()
    ring.add_object('ring', _mesh(read_step(compact.WORK / 'candidate_53.step'), .25, .3))
    poses = [trimesh.transformations.rotation_matrix(math.radians(angle), (-1, 0, 0), (75.95, .1, 196.85093881236))
             for angle in range(-30, 31, 5)]
    found = []
    for turn in (0, 90, -90, 180):
        rotation = trimesh.transformations.rotation_matrix(math.radians(turn), (0, 0, 1), centre)
        for horizontal, lateral, vertical in itertools.product(range(-20, 101, 5), range(-25, 26, 5), range(-5, 26, 5)):
            transform = trimesh.transformations.translation_matrix((horizontal, lateral, vertical)) @ rotation
            points = trimesh.transform_points(mesh.bounding_box.vertices, transform)
            if points[:, 0].min() < -63 or points[:, 0].max() > 105 or points[:, 1].min() < -85 or points[:, 1].max() > 85 or points[:, 2].max() > 234:
                continue
            if fixed.min_distance_single(mesh, transform=transform) < .3:
                continue
            if any(ring.min_distance_single(mesh, transform=pose @ transform) < .3 for pose in poses):
                continue
            found.append(dict(turn=turn, offset=[horizontal, lateral, vertical],
                              score=abs(horizontal) + abs(lateral) + abs(vertical) + abs(turn) / 10,
                              transform=transform.tolist()))
        print('finished turn', turn, 'feasible', len(found), flush=True)
        if found:
            break
    found.sort(key=lambda item: item['score'])
    (compact.WORK / f'layout_search_{index}.json').write_text(json.dumps(found, indent=2))
    print(json.dumps(found[:8], indent=2), flush=True)


if __name__ == '__main__':
    main()
