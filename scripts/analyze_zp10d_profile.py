"""Summarize measured GPU events, recovering CUDA Graph labels from saved HLO.

Uses installed protobuf schemas without importing TensorFlow or initializing CUDA.
Run with the project Python and an optional profiling output directory.
"""

import ast
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS, (1_000_000_000, 1_000_000_000))

import json
import statistics

from google.protobuf import any_pb2, descriptor_pool, duration_pb2, message_factory, timestamp_pb2


PROJECT = Path(__file__).resolve().parents[1]


def load_schema(path):
    tree = ast.parse(path.read_text())
    serialized = next(ast.literal_eval(n.args[0]) for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == 'AddSerializedFile')
    descriptor_pool.Default().AddSerializedFile(serialized)


def category(label):
    for scope, name in (
        ('profile_solver_linesearch', 'physics/solver/line_search'),
        ('profile_solver_update_gradient', 'physics/solver/gradient'),
        ('profile_solver_update_constraint', 'physics/solver/constraint'),
        ('profile_solversolve', 'physics/solver/other'),
        ('profile_collision', 'physics/collision'),
        ('profile_physics', 'physics/other'),
        ('profile_reward', 'reward'),
        ('profile_observation', 'observation'),
        ('profile_reset', 'reset'),
        ('profile_ppo_loss', 'ppo_loss_and_gradient'),
        ('profile_sampling', 'sampling/other'),
    ):
        if scope in label:
            return name
    return 'other' if label else 'unknown'


def union_us(intervals):
    end = total = 0.
    for start, stop in sorted(intervals):
        total += max(0., stop - max(end, start))
        end = max(end, stop)
    return total


def event_stats(plane, event):
    result = {}
    for stat in event.stats:
        kind = stat.WhichOneof('value')
        result[plane.stat_metadata[stat.metadata_id].name] = (
            plane.stat_metadata[stat.ref_value].name if kind == 'ref_value' else getattr(stat, kind))
    return result


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / 'playground/open_duck_mini_zp10d/training_gpu/profile'
    protobuf = Path(sys.prefix) / 'lib/python3.12/site-packages/tensorflow'
    for relative in ('tsl/profiler/protobuf/xplane_pb2.py', 'compiler/xla/xla_data_pb2.py',
                     'compiler/xla/service/metrics_pb2.py', 'compiler/xla/service/hlo_pb2.py'):
        load_schema(protobuf / relative)
    pool = descriptor_pool.Default()
    xspace_type = message_factory.GetMessageClass(pool.FindMessageTypeByName('tensorflow.profiler.XSpace'))
    hlo_type = message_factory.GetMessageClass(pool.FindMessageTypeByName('xla.HloProto'))
    trace = next((output / 'trace').rglob('*.xplane.pb'))
    raw = trace.read_bytes()
    space = xspace_type.FromString(raw)
    report = dict(source=str(trace), source_sha256=hashlib.sha256(raw).hexdigest(),
                  errors=list(space.errors), warnings=list(space.warnings))
    del raw
    modules = {}
    for plane in space.planes:
        if plane.name != '/host:metadata':
            continue
        for metadata in plane.event_metadata.values():
            for stat in metadata.stats:
                if plane.stat_metadata[stat.metadata_id].name == 'Hlo Proto':
                    module = hlo_type.FromString(stat.bytes_value).hlo_module
                    modules[module.id] = module
    labels = {}
    fusion_categories = {}
    for program, module in modules.items():
        computations = {c.id: c for c in module.computations}
        for computation in module.computations:
            for instruction in computation.instructions:
                key = (program, re.sub('[^a-zA-Z0-9_]', '_', instruction.name))
                label = instruction.metadata.op_name
                members = [i.metadata.op_name for c in instruction.called_computation_ids
                           for i in computations[c].instructions if i.metadata.op_name] if instruction.opcode == 'fusion' else []
                groups = {category(x) for x in members}
                # Prefer the fused instruction's attribution; recover missing labels
                # only when every labelled member agrees. Retain mixed attribution.
                if not label and len(groups) == 1:
                    label = members[0]
                labels[key] = label
                fusion_categories[key] = sorted(groups)
    # Library kernels inside CUDA Graphs have no HLO instruction name. Match
    # their launch correlation ID to an enclosing, labelled host loop instead.
    host_labels = defaultdict(set)
    for (_, name), label in labels.items():
        if label:
            host_labels[name].add(label)
    correlations = {}
    solver_ranges = []
    iteration_starts = []
    for plane in space.planes:
        if plane.name != '/host:CPU':
            continue
        for line in plane.lines:
            stack = []
            for event in sorted(line.events, key=lambda e: (e.offset_ps, -e.duration_ps)):
                start = line.timestamp_ns * 1000 + event.offset_ps
                stop = start + event.duration_ps
                while stack and stack[-1][0] <= start:
                    stack.pop()
                name = re.sub('[^a-zA-Z0-9_]', '_', plane.event_metadata[event.metadata_id].name)
                candidates = host_labels[name]
                label = next(iter(candidates)) if len(candidates) == 1 else ''
                if label.endswith('profile_solversolve/while'):
                    solver_ranges.append((start, stop))
                # Each solver iteration invokes exactly one line-search scan.
                if label.endswith('profile_solver_linesearch/while'):
                    iteration_starts.append(start)
                if label and category(label) != 'other':
                    stack.append((stop, label))
                stats = event_stats(plane, event)
                if 'correlation_id' in stats and stack and stop <= stack[-1][0]:
                    correlations[(stats.get('context_id'), stats['correlation_id'])] = stack[-1][1]
    iteration_starts.sort()
    iterations = [bisect_left(iteration_starts, hi) - bisect_left(iteration_starts, lo)
                  for lo, hi in solver_ranges]
    kernel_us = Counter()
    kernel_counts = Counter()
    by_kernel = defaultdict(lambda: [0, 0.])
    transfers = defaultdict(lambda: [0, 0.])
    attribution = Counter()
    host = defaultdict(lambda: [0, 0.])
    intervals = []
    durations = []
    mixed_us = Counter()
    origin = min(line.timestamp_ns for plane in space.planes for line in plane.lines)
    for plane in space.planes:
        if not (plane.name.startswith('/device:GPU:') or plane.name == '/host:CPU'):
            continue
        for line in plane.lines:
            for event in line.events:
                name = plane.event_metadata[event.metadata_id].name
                duration = event.duration_ps / 1e6
                if plane.name == '/host:CPU':
                    host[name][0] += 1
                    host[name][1] += duration
                    continue
                stats = event_stats(plane, event)
                start = (line.timestamp_ns - origin) / 1000 + event.offset_ps / 1e6
                intervals.append((start, start + duration))
                if 'kernel_details' not in stats:
                    transfers[name][0] += 1
                    transfers[name][1] += duration
                    continue
                key = (stats.get('program_id'), re.sub('[^a-zA-Z0-9_]', '_', name))
                label = stats.get('name', '')
                source = 'event_metadata'
                if not label:
                    label = labels.get(key, '')
                    source = 'saved_hlo' if label else 'unclassified'
                if not label and not fusion_categories.get(key):
                    label = correlations.get((stats.get('context_id'), stats.get('correlation_id')), '')
                    source = 'host_launch_ancestor' if label else source
                group = category(label)
                if not label and len(fusion_categories.get(key, [])) > 1:
                    group = 'mixed_fusion'
                if len(fusion_categories.get(key, [])) > 1:
                    mixed_us[group] += duration
                attribution[source] += duration
                kernel_us[group] += duration
                kernel_counts[group] += 1
                by_kernel[(name, group)][0] += 1
                by_kernel[(name, group)][1] += duration
                durations.append(duration)
    total = sum(kernel_us.values())
    span = max(stop for _, stop in intervals) - min(start for start, _ in intervals)
    report.update(
        kernel_count=len(durations), kernel_seconds=total / 1e6,
        gpu_event_span_seconds=span / 1e6, gpu_event_union_seconds=union_us(intervals) / 1e6,
        median_kernel_us=statistics.median(durations),
        kernels_under_10us=sum(x < 10 for x in durations),
        kernels_under_10us_seconds=sum(x for x in durations if x < 10) / 1e6,
        batched_solver_iterations=dict(physics_batches=len(iterations), total=sum(iterations),
                                       mean=statistics.mean(iterations), min=min(iterations), max=max(iterations),
                                       histogram=dict(sorted(Counter(iterations).items()))),
        attribution_seconds={k: v / 1e6 for k, v in attribution.items()},
        categories=[dict(category=k, kernels=kernel_counts[k], seconds=v / 1e6,
                         percent_kernel_time=100 * v / total,
                         mixed_fusion_seconds=mixed_us[k] / 1e6)
                    for k, v in kernel_us.most_common()],
        top_kernels=[dict(name=k[0], category=k[1], calls=v[0], seconds=v[1] / 1e6)
                     for k, v in sorted(by_kernel.items(), key=lambda item: -item[1][1])[:30]],
        transfers={k: dict(calls=v[0], seconds=v[1] / 1e6) for k, v in transfers.items()},
        host_inclusive=[dict(name=k, calls=v[0], seconds=v[1] / 1e6)
                        for k, v in sorted(host.items(), key=lambda item: -item[1][1])[:40]],
        host_while=[dict(name=k, calls=v[0], seconds=v[1] / 1e6,
                         labels=sorted({label for (program, key), label in labels.items()
                                        if key == re.sub('[^a-zA-Z0-9_]', '_', k) and label}))
                    for k, v in host.items() if k.startswith('while.')],
        notes=['GPU durations are measured CUPTI events, not static HLO cost estimates.',
               'Each kernel is counted once; host inclusive durations overlap and must not be added.',
               'Fused kernel labels indicate attribution, not an exact arithmetic split.',
               'Library kernels without labels use the enclosing host launch scope, linked by correlation ID.',
               'Solver iteration counts describe the batch of 512 environments, not each environment.',
               'GPU span/union include callbacks and transfers; steady throughput is in profile.json.'])
    assert sum(kernel_counts.values()) == len(durations)
    assert abs(sum(x['percent_kernel_time'] for x in report['categories']) - 100) < 1e-8
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('kernel_count', 'kernel_seconds', 'median_kernel_us',
                                            'kernels_under_10us', 'categories', 'host_while')}, indent=2))


if __name__ == '__main__':
    main()
