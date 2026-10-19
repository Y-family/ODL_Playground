"""Conforming print tessellation; recover only faces present in the source CAD."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Pnt

SEAM_TOLERANCE_MM = .005


def tessellate(shape, deflection, angle):
    # Read OCC's result directly: build123d.tessellate silently retries a failed
    # absolute mesh with relative deflection and then assumes every face exists.
    BRepTools.Clean_s(shape.wrapped)
    operation = BRepMesh_IncrementalMesh(shape.wrapped, deflection, False, angle, False)
    vertices, triangles, missing = [], [], []
    for index, face in enumerate(shape.faces()):
        location = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face.wrapped, location)
        if poly is None:
            missing.append(index)
            continue
        offset = len(vertices)
        transform = location.Transformation()
        vertices.extend(poly.Node(i).Transformed(transform).Coord()
                        for i in range(1, poly.NbNodes() + 1))
        for i in range(1, poly.NbTriangles() + 1):
            ids = list(poly.Triangle(i).Get())
            if face.wrapped.Orientation() == TopAbs_REVERSED:
                ids.reverse()
            triangles.append([offset + v - 1 for v in ids])
    return trimesh.Trimesh(vertices, triangles, process=True), missing, operation.GetStatusFlags()


def recover_planar_faces(mesh, shape, transform, missing, deflection):
    """Triangulate failed CAD planes using their existing neighbouring edges.

    This does not cap arbitrary holes: every patch has a named source face,
    lies on its plane, passes CAD containment, and matches its area within 1%.
    """
    import shapely
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    patches = []
    for index in missing:
        face = shape.faces()[index]
        surface = BRepAdaptor_Surface(face.wrapped, True)
        from OCP.GeomAbs import GeomAbs_Plane
        if surface.GetType() != GeomAbs_Plane:
            raise ValueError(f'Unmeshed nonplanar CAD face {index}')
        plane = surface.Plane().Position()
        rotation, shift = transform[:3, :3], transform[:3, 3]
        origin = np.array(plane.Location().Coord()) @ rotation.T + shift
        axes = np.array([plane.XDirection().Coord(), plane.YDirection().Coord()]) @ rotation.T
        normal = np.array(tuple(face.normal_at())) @ rotation.T
        counts = np.bincount(mesh.edges_unique_inverse)
        edges = mesh.edges_unique[counts == 1]
        points = mesh.vertices[edges]
        on_plane = (abs((points - origin) @ normal) <= 1e-4).all(axis=1)
        box = face.bounding_box()
        corners = trimesh.bounds.corners([tuple(box.min), tuple(box.max)])
        bounds = ((corners @ rotation.T + shift - origin) @ axes.T)
        xy = (points - origin) @ axes.T
        in_box = ((xy >= bounds.min(axis=0) - deflection)
                  & (xy <= bounds.max(axis=0) + deflection)).all(axis=(1, 2))
        edges = edges[on_plane & in_box]
        if not len(edges):
            raise ValueError(f'No neighbouring boundary for CAD face {index}')
        nodes = np.unique(edges)
        projected = (mesh.vertices - origin) @ axes.T
        polygons = polygonize(unary_union([LineString(projected[e]) for e in edges]))
        tree = cKDTree(projected[nodes])
        vertices, triangles, area = mesh.vertices.tolist(), [], 0.
        for polygon in polygons:
            # polygonize also returns the interior of each real hole as a cell.
            # Classify the whole cell strictly before tolerating chord errors.
            probe = np.array(polygon.representative_point().coords[0]) @ axes + origin
            world_probe = (probe - shift) @ rotation
            if BRepClass_FaceClassifier(face.wrapped, gp_Pnt(*world_probe), 1e-7).State() not in (TopAbs_IN, TopAbs_ON):
                continue
            for triangle in shapely.get_parts(shapely.constrained_delaunay_triangles(polygon)):
                xy = np.asarray(triangle.exterior.coords)[:3]
                xyz = xy @ axes + origin
                # Reject projected cells belonging to a real CAD opening.
                probes = np.vstack([xyz.mean(axis=0), (xyz + np.roll(xyz, 1, axis=0)) / 2])
                if any(BRepClass_FaceClassifier(face.wrapped, gp_Pnt(*p), deflection).State()
                       not in (TopAbs_IN, TopAbs_ON) for p in (probes - shift) @ rotation):
                    continue
                distance, ids = tree.query(xy)
                ids = nodes[ids]
                for k in range(3):
                    if distance[k] > 1e-8:
                        ids[k] = len(vertices)
                        vertices.append(xyz[k].tolist())
                if np.dot(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]), normal) < 0:
                    ids = ids[::-1]
                triangles.append(ids)
                area += triangle.area
        if not triangles or abs(area - face.area) / face.area >= .01:
            raise ValueError(f'CAD face {index} patch area mismatch: {area} versus {face.area}')
        mesh.vertices = np.array(vertices)
        mesh.faces = np.vstack([mesh.faces, triangles])
        patches.append(dict(face=index, cad_area_mm2=face.area, mesh_area_mm2=area,
                            triangles=len(triangles)))
    return patches


def conform_seams(mesh):
    """Split boundary triangles at existing near-collinear seam vertices.

    Max seam deviation is 5 microns, below the 80-micron mesh deflection.
    No general hole cap is used. Vertex welding is limited to 0.0001 mm bins.
    """
    old_vertices = mesh.vertices.copy()
    mesh.merge_vertices(digits_vertex=4)
    weld_distance = float(cKDTree(mesh.vertices).query(old_vertices)[0].max())
    assert weld_distance < .00018
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-7))
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    changes, max_distance = 0, 0.
    for _ in range(6):
        counts = np.bincount(mesh.edges_unique_inverse)
        ids = np.flatnonzero(counts == 1)
        edges = mesh.edges_unique[ids]
        nodes = np.unique(edges)
        if not len(nodes):
            break
        tree = cKDTree(mesh.vertices[nodes])
        new_faces, dropped = [], set()
        for eid, (a, b) in zip(ids, edges):
            start, end = mesh.vertices[[a, b]]
            delta = end - start
            length = np.linalg.norm(delta)
            candidates = nodes[tree.query_ball_point((start + end) / 2,
                                                     length / 2 + SEAM_TOLERANCE_MM)]
            candidates = candidates[(candidates != a) & (candidates != b)]
            if not len(candidates):
                continue
            t = (mesh.vertices[candidates] - start) @ delta / length ** 2
            distance = np.linalg.norm(mesh.vertices[candidates] - (start + t[:, None] * delta), axis=1)
            keep = (t > 1e-6) & (t < 1 - 1e-6) & (distance <= SEAM_TOLERANCE_MM)
            candidates, t = candidates[keep], t[keep]
            if not len(candidates):
                continue
            fi = int(np.where(mesh.faces_unique_edges == eid)[0][0])
            if fi in dropped:
                continue
            face = mesh.faces[fi]
            k = next(k for k in range(3) if set(face[[k, (k + 1) % 3]]) == {a, b})
            order = np.argsort(t) if face[k] == a else np.argsort(t)[::-1]
            chain = [face[k], *candidates[order], face[(k + 1) % 3]]
            new_faces.extend([[x, y, face[(k + 2) % 3]] for x, y in zip(chain, chain[1:])])
            dropped.add(fi)
            changes += len(candidates)
            max_distance = max(max_distance, float(distance[keep].max()))
        if not dropped:
            break
        mask = np.ones(len(mesh.faces), bool)
        mask[list(dropped)] = False
        mesh.faces = np.vstack([mesh.faces[mask], new_faces])
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-7))
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    if mesh.is_watertight:
        # Keep inward-facing cavity shells: do not invert each shell separately.
        mesh.fix_normals(multibody=False)
    return dict(split_vertices=changes, max_edge_deviation_mm=max_distance,
                max_weld_distance_mm=weld_distance)
