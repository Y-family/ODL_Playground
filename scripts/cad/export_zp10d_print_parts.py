"""Export current STEP print occurrences in mm; run via run_memory_limited.py.

No remodelling, scaling, arbitrary hole capping, or deduplication. Manufacturing
classification uses upstream print/BOM evidence; unknown parts remain unresolved.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import resource

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import numpy as np
import trimesh
from cadgen import build123d as bd, read_step
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.gp import gp_Trsf
from print_mesh import tessellate, recover_planar_faces, conform_seams
from build_zp10d_mjcf_from_final_step import FINAL_STEP, REPORT_OUTPUT
from full_motion_current import _leaves

ROOT = Path(__file__).resolve().parents[2]
SHELLS = {'head', 'left_cache', 'right_cache', 'body_front', 'body_back',
          'body_middle_bottom', 'body_middle_top', 'battery_pack_lid'}
DEFLECTION_MM = .08
ANGULAR_TOLERANCE_RAD = .15


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classification_catalog():
    sources = ROOT / 'cad/printing/sources'
    guide = sources / 'print_guide.md'
    bom = sources / 'bom.csv'
    entries = re.findall(r'^- (\S+)\.stl x(\d+)', guide.read_text(), re.M)
    if not entries or len(dict(entries)) != len(entries):
        raise ValueError('Missing or duplicate upstream print guide entries')
    catalog = {label: dict(classification='print', source='upstream_print_guide',
                           reference_file=f'print/{label}.stl', upstream_quantity=int(quantity))
               for label, quantity in entries}
    # Current shortened ZP10D leg plates retained their original part family.
    for side in ('left', 'right'):
        catalog[f'left_knee_to_ankle_{side}_sheet'] = dict(catalog[f'knee_to_ankle_{side}_sheet'])
    bom_rows = {row[1].strip(): i for i, row in enumerate(csv.reader(bom.open()), 1)
                if len(row) > 1 and row[1].strip()}
    purchased = {
        'usb_c_charger': 'usb-c charger', 'power_switch': 'Small power switch',
        'cell': '18650 cell', 'holder': '18650 cell holder', 'bms': '2S BMS',
        'RaspberryPiZeroW': 'Raspberry pi zero 2w', 'roll_bearing': 'Bearing',
        'glass': 'Eye diffusers', 'sg90': '9g servos',
        'speaker_bottom': 'Speaker', 'speaker_top': 'Speaker',
        'Cam': 'Raspberry pi camera', 'BNO055': 'IMU (BNO055)',
        'Board': 'Servo control board',
    }
    for label, entry in purchased.items():
        if label in catalog or entry not in bom_rows:
            raise ValueError(f'Conflicting or missing manufacturing evidence for {label}')
        catalog[label] = dict(classification='purchased', source='upstream_bom',
                              bom_part=entry, csv_row=bom_rows[entry])
    for component in ('body', 'passive_horn', 'active_horn'):
        catalog[f'ZP10D_vendor_{component}'] = dict(
            classification='purchased', source='current_vendor_servo_assembly',
            note='Supplied ZP10D servo/horns replace the original BOM STS3215 group; not printed')
    return catalog, dict(
        print_guide=dict(file=str(guide.relative_to(ROOT)), sha256=digest(guide),
                         upstream_commit='b23317a485b3cec7d8417f352478778b3475173c',
                         url='https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/docs/print_guide.md'),
        bom=dict(file=str(bom.relative_to(ROOT)), sha256=digest(bom),
                 url='https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit',
                 snapshot_date='2026-10-19'))


def classify_part(label, catalog):
    return catalog.get(label, dict(classification='unresolved',
                                   reason='No upstream print/BOM evidence; manufacturing method requires confirmation'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'cad/printing')
    parser.add_argument('--step', action='store_true', help='also export individual exact STEP solids')
    args = parser.parse_args()
    catalog, classification_sources = classification_catalog()
    source_sha = digest(FINAL_STEP)
    source = json.loads(REPORT_OUTPUT.read_text())
    if source_sha != source['source_step_sha256']:
        raise ValueError('STEP and dynamics occurrence manifest differ; synchronize first')
    leaves = _leaves(read_step(FINAL_STEP))
    assert len(leaves) == len(source['parts'])
    output = args.output / 'STL'
    quarantine = args.output / 'quarantine'
    output.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(exist_ok=True)
    records, excluded, unresolved = [], [], []
    for index, (row, original) in enumerate(zip(leaves, source['parts'])):
        assert (row['path'], row['label']) == (original['path'], original['label'])
        decision = classify_part(row['label'], catalog)
        if decision['classification'] != 'print':
            entry = dict(index=index, label=row['label'], path=row['path'], evidence=decision)
            (excluded if decision['classification'] == 'purchased' else unresolved).append(entry)
            # Retain former exports for review, outside the confirmed print folders.
            for folder, suffix in [('STL', '.stl'), ('STEP', '.step')]:
                previous = args.output / folder / f'{index:03d}_{row["label"]}{suffix}'
                if previous.exists():
                    saved = args.output / 'unclassified' / folder / previous.name
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    previous.replace(saved)
            continue
        shape = row['shape']
        cad_volume = float(shape.volume)
        valid_cad = bool(BRepCheck_Analyzer(shape.wrapped, True, False).IsValid())
        solid_count = len(shape.solids())
        assert cad_volume > 0 and solid_count == 1, row['path']
        np.testing.assert_allclose(cad_volume, original['volume_mm3'], rtol=1e-8)
        try:
            mesh, missing, mesh_status = tessellate(shape, DEFLECTION_MM, ANGULAR_TOLERANCE_RAD)
        except (AttributeError, RuntimeError) as error:
            records.append(dict(index=index, label=row['label'], occurrence_path=row['path'],
                                quantity=1, file=None, status='quarantined',
                                cad_valid=valid_cad, error=str(error)))
            print(index, row['label'], 'MESH FAILED', str(error), flush=True)
            continue
        # A repeatable initial orientation, not a slicer/support prescription.
        planes = [f for f in shape.faces() if f.geom_type == bd.GeomType.PLANE]
        transform = np.eye(4)
        if planes:
            face = max(planes, key=lambda f: f.area)
            transform = trimesh.geometry.align_vectors(tuple(face.normal_at()), [0, 0, -1])
            mesh.apply_transform(transform)
        shift = -(mesh.bounds[0] + mesh.bounds[1]) / 2
        shift[2] = -mesh.bounds[0, 2]
        mesh.apply_translation(shift)
        transform[:3, 3] += shift
        # Binary STL stores float32 coordinates. Remove only duplicate and
        # zero-area triangles after that quantization, without filling holes.
        mesh.vertices = mesh.vertices.astype(np.float32).astype(float)
        mesh.merge_vertices(digits_vertex=7)
        before_faces = len(mesh.faces)
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-7))
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        dropped_faces = before_faces - len(mesh.faces)
        patches = recover_planar_faces(mesh, shape, transform, missing, DEFLECTION_MM) if missing else []
        seam_repair = conform_seams(mesh) if not mesh.is_watertight else None
        # Repaired triangles must also survive the final binary STL quantization.
        mesh.vertices = mesh.vertices.astype(np.float32).astype(float)
        volume_error = abs(float(mesh.volume) - cad_volume) / cad_volume
        # An internal sealed cavity is a second, inward-facing boundary shell.
        # Require exactly one positive outer shell; never discard the cavity.
        boundary_volumes = [float(s.volume) for s in mesh.split(only_watertight=False)]
        one_outer = sum(v > 1e-6 for v in boundary_volumes) == 1
        mesh_valid = bool(mesh.is_volume and one_outer and volume_error < .01)
        accepted = valid_cad and mesh_valid
        filename = f'{index:03d}_{row["label"]}.stl'
        path = (output if accepted else quarantine) / filename
        # Never leave an older accepted mesh when a new export fails validation.
        other = (quarantine if accepted else output) / filename
        other.unlink(missing_ok=True)
        mesh.export(path)
        # Validate the actual handed-off file, including binary STL precision.
        actual = trimesh.load_mesh(path, process=True)
        np.testing.assert_allclose(actual.bounds, mesh.bounds, atol=2e-5)
        assert actual.is_watertight == mesh.is_watertight
        assert actual.is_winding_consistent == mesh.is_winding_consistent
        assert actual.body_count == mesh.body_count
        assert abs(actual.volume - mesh.volume) / cad_volume < 1e-5
        step_file = None
        if accepted and args.step:
            folder = args.output / 'STEP'
            folder.mkdir(exist_ok=True)
            rigid = gp_Trsf()
            rigid.SetValues(*transform[:3].ravel())
            placed = shape.moved(bd.Location(rigid))
            placed.parent = None
            np.testing.assert_allclose(placed.volume, cad_volume, rtol=1e-9)
            step_file = folder / filename.replace('.stl', '.step')
            bd.export_step(placed, step_file)
        fits = bool((actual.extents <= 180).all())
        records.append(dict(
            index=index, label=row['label'], occurrence_path=row['path'], root=row['root'],
            classification_evidence=decision,
            body=original['body'], quantity=1, file=str(path.relative_to(args.output)),
            sha256=digest(path), units='mm', source_to_print_transform=transform.tolist(),
            dimensions_mm=actual.extents.tolist(), cad_volume_mm3=cad_volume,
            mesh_volume_mm3=float(actual.volume), mesh_volume_relative_error=volume_error,
            cad_valid=valid_cad, cad_solid_count=solid_count,
            mesh_watertight=bool(actual.is_watertight),
            mesh_winding_consistent=bool(actual.is_winding_consistent),
            mesh_body_count=int(actual.body_count), faces=len(actual.faces),
            boundary_shell_volumes_mm3=boundary_volumes,
            duplicate_or_zero_area_faces_removed=dropped_faces,
            mesher_status=int(mesh_status), rebuilt_cad_faces=patches, seam_repair=seam_repair,
            step_file=str(step_file.relative_to(args.output)) if step_file else None,
            step_sha256=digest(step_file) if step_file else None,
            fits_180mm_cube=fits, status='validated_mesh' if accepted else 'quarantined',
            material='PETG', infill_percent=10 if row['label'] in SHELLS else 30,
            material_note=('legacy name denotes flexible sole; PETG grip/compliance must be tested'
                           if row['label'] == 'foot_bottom_tpu' else None)))
        print(index, row['label'], 'OK' if accepted else 'QUARANTINED',
              np.round(actual.extents, 2).tolist(),
              dict(cad_valid=valid_cad, mesh_valid=mesh_valid, volume_error=volume_error), flush=True)
    assert digest(FINAL_STEP) == source_sha
    report = dict(
        source_step=str(FINAL_STEP.relative_to(ROOT)), source_step_sha256=source_sha,
        source_manifest_sha256=digest(REPORT_OUTPUT), exporter_sha256=digest(Path(__file__)),
        print_mesh_sha256=digest(Path(__file__).with_name('print_mesh.py')),
        classification_sources=classification_sources,
        source_occurrence_count=len(leaves), unresolved_parts=unresolved,
        manufacturing_classification_complete=not unresolved,
        units='mm', occurrence_count=len(records),
        validated_count=sum(r['status'] == 'validated_mesh' for r in records),
        deflection_mm=DEFLECTION_MM, angular_tolerance_rad=ANGULAR_TOLERANCE_RAD,
        orientation='largest planar face down, centered XY, minimum Z=0; supports not sliced',
        acceptance='single valid positive-volume BRep and closed consistently wound mesh '
                   'with one positive outer shell (negative cavity shells retained); '
                   'mesh/CAD volume difference <1%; not a strength or assembly certificate',
        parts=records, excluded_purchased_parts=excluded)
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'Validated {report["validated_count"]}/{len(records)} print occurrences; '
          f'excluded {len(excluded)} purchased occurrences; {len(unresolved)} unresolved', flush=True)


if __name__ == '__main__':
    main()
