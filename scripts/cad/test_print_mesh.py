"""Run via run_memory_limited.py: seams close, real CAD openings stay open."""
import resource

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import numpy as np
import trimesh
from cadgen import build123d as bd
from print_mesh import conform_seams, recover_planar_faces, tessellate
from export_zp10d_print_parts import classification_catalog, classify_part


def main():
    catalog, _ = classification_catalog()
    for label in ('left_eye', 'right_eye', 'bulb', 'left_knee_to_ankle_left_sheet'):
        assert classify_part(label, catalog)['classification'] == 'print'
    assert classify_part('holder', catalog)['classification'] == 'purchased'
    for label in ('antenna', 'new_unknown_part'):
        assert classify_part(label, catalog)['classification'] == 'unresolved'

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0]]
    faces = [[0, 2, 4], [4, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    mesh = trimesh.Trimesh(vertices, faces)
    assert not mesh.is_watertight
    assert conform_seams(mesh)['split_vertices'] == 1
    assert mesh.is_volume and abs(mesh.volume - 1 / 6) < 1e-12

    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    conform_seams(mesh)
    assert not mesh.is_watertight, 'A real opening must not be capped'

    shape = bd.Box(10, 10, 2) - bd.Cylinder(1, 4)
    mesh, missing, _ = tessellate(shape, .08, .15)
    assert not missing and mesh.is_volume
    top = next(i for i, f in enumerate(shape.faces())
               if f.geom_type == bd.GeomType.PLANE and f.normal_at().Z > .99)
    mesh.update_faces(mesh.face_normals[:, 2] < .99)
    patches = recover_planar_faces(mesh, shape, np.eye(4), [top], .08)
    conform_seams(mesh)
    assert patches and mesh.is_volume
    assert not mesh.contains([[0, 0, 0]])[0], 'The through-hole must remain empty'
    assert abs(mesh.volume - shape.volume) / shape.volume < .01
    print('PASS: evidence-based classification/unknowns, seam subdivision, real CAD openings preserved')


if __name__ == '__main__':
    main()
