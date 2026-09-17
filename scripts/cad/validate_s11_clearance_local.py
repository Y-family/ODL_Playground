"""Check all S11 followers against a local shell or trunk at 2.5 degree intervals."""
import argparse
import hashlib
import json
import math
import resource
from pathlib import Path

from cadgen import read_step

import finish_s11_clearance_local as model
import refit_upper_servo_interfaces_local as refit


def check_seats(part, name):
    if name == 'body_front':
        seats = [(65, 36, 39, 1.75)]
        passages = [(65, 36, 46, 1.75), (65, 39, 46, 3.25)]
    elif name == 'body_middle_top':
        seats = [(65, 30.2, 36, 2)]
        passages = [(65, 29, 36, 1.5), (65, 30.2, 36, 2)]
    else:
        return []
    results = []
    for sign in (-1, 1):
        for height, start, end, radius in seats:
            centre = model.bd.Vector(0, sign * 45, height)
            direction = model.bd.Vector(1, 0, 0)
            outer = refit._rod(centre, direction, start, end, 3.5)
            bore = refit._rod(centre, direction, start, end, radius)
            ring = refit._cut(outer, [bore])
            volume = sum(item.volume for item in refit._items(part.intersect(ring)))
            ratio = volume / ring.volume
            if ratio < .99999:
                raise ValueError(f'{name}: incomplete screw support at Y={sign * 45}, Z={height}')
            results.append(dict(y=sign * 45, z=height, support_ratio=ratio))
        for height, start, end, radius in passages:
            bore = refit._rod(model.bd.Vector(0, sign * 45, height), model.bd.Vector(1, 0, 0),
                              start, end, radius)
            if sum(item.volume for item in refit._items(part.intersect(bore))) > 1e-5:
                raise ValueError(f'{name}: blocked screw passage')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fixed', choices=('body_front', 'body_back', 'body_middle_top',
                                         'body_middle_bottom', 'trunk_top'))
    parser.add_argument('--work', type=Path, default=Path('/tmp/s11_local'))
    parser.add_argument('--limit-degrees', type=float, default=30)
    parser.add_argument('--current', action='store_true')
    args = parser.parse_args()
    refit._start_memory_watchdog()
    rows = json.loads((args.work / 'leaves.json').read_text())
    suffix = '_candidate' if args.fixed in model.CHANGED and not args.current else ''
    fixed = read_step(args.work / f'{args.fixed}{suffix}.step')
    fixed_bounds = fixed.bounding_box()
    seats = check_seats(fixed, args.fixed)
    near_box = refit._world_box(fixed_bounds.min.X - 1, fixed_bounds.max.X + 1,
                                fixed_bounds.min.Y - 1, fixed_bounds.max.Y + 1,
                                fixed_bounds.min.Z - 1, fixed_bounds.max.Z + 1)
    candidate_suffix = '' if args.current else '_candidate'
    fingerprints = {name: hashlib.sha256((args.work / f'{name}{candidate_suffix}.step').read_bytes()).hexdigest()
                    for name in model.CHANGED}
    issues = []
    minimum_gap = None
    checks = 0
    for row in rows:
        if not row['moving']:
            continue
        candidate = args.work / f'{row["label"]}_candidate.step'
        moving = read_step(candidate if row['label'] in model.CHANGED and not args.current else row['file'])
        base = moving.bounding_box()
        steps = round(args.limit_degrees / 2.5)
        if steps < 1 or abs(steps * 2.5 - args.limit_degrees) > 1e-6:
            raise ValueError('Angle limit must be a positive multiple of 2.5 degrees')
        for index in range(-steps, steps + 1):
            angle = index * 2.5
            sine, cosine = math.sin(math.radians(angle)), math.cos(math.radians(angle))
            corners = [(9 + cosine * (horizontal - 9) + sine * (height - 90.000938812376),
                        90.000938812376 - sine * (horizontal - 9) + cosine * (height - 90.000938812376))
                       for horizontal in (base.min.X, base.max.X)
                       for height in (base.min.Z, base.max.Z)]
            low = (min(point[0] for point in corners), base.min.Y, min(point[1] for point in corners))
            high = (max(point[0] for point in corners), base.max.Y, max(point[1] for point in corners))
            if any(low[coordinate] > tuple(fixed_bounds.max)[coordinate] + 1 or
                   high[coordinate] < tuple(fixed_bounds.min)[coordinate] - 1 for coordinate in range(3)):
                continue
            posed = moving.rotate(model.AXIS, angle)
            bounds = posed.bounding_box()
            if any(tuple(bounds.min)[coordinate] > tuple(fixed_bounds.max)[coordinate] + 1 or
                   tuple(bounds.max)[coordinate] < tuple(fixed_bounds.min)[coordinate] - 1
                   for coordinate in range(3)):
                continue
            nearby = posed.intersect(near_box)
            if sum(item.volume for item in refit._items(nearby)) < 1e-8:
                continue
            local = refit._shape(nearby)
            overlap = sum(item.volume for item in refit._items(local.intersect(fixed)))
            gap = min(1.0, local.distance_to(fixed))
            checks += 1
            minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
            if overlap > 1e-4 or gap < .19999:
                issues.append(dict(path=row['path'], angle=angle, overlap=overlap, gap=gap))
        print(args.fixed, row['label'], 'checked', flush=True)
    if any(hashlib.sha256((args.work / f'{name}{candidate_suffix}.step').read_bytes()).hexdigest() != digest
           for name, digest in fingerprints.items()):
        raise ValueError('Candidate changed during validation')
    report = dict(fixed=args.fixed, checks=checks, min_gap=minimum_gap,
                  issues=issues, fingerprints=fingerprints, step_degrees=2.5,
                  angles_degrees=[-args.limit_degrees, args.limit_degrees], moving_leaves=sum(row['moving'] for row in rows),
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  screw_seats=seats)
    (args.work / f'{args.fixed}_motion.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
