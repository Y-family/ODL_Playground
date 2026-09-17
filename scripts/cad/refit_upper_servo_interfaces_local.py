#!/usr/bin/env python3
"""Refit the upper ZP10D servo interfaces on top of the current local model.

The covers-forward review STEP fixed the ZP10D fits by fusing free-standing
cylindrical pads (``_case_mount_tools`` / ``_horn_mount_tools`` /
``_outer_horn_bracket_tools`` in ``build_zp10d_left_leg_complete``) onto the
printed parts.  Those pads are forbidden, and in some places they replaced the
original connector structure outright.

This model keeps the current local STEP as the single input version and makes
local, dimensional-only edits:

* inside the interface region the added pad material is dropped and the
  original connector structure (ear plate / face plate) is put back,
* that structure is then thickened along the joint axis until its bearing face
  reaches the ZP10D horn face (horn hosts) or the servo case face (case hosts),
* the ZP10D screw pattern is re-cut (horn dia 3.2 through + dia 6.5 counterbore,
  case dia 2.0 through) and the rotating horn / vendor body envelope is cleared.

Nothing is added elsewhere, no cylinder pad is fused on, no spacer is
introduced and untouched parts keep their current geometry, colours and
transforms.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import threading
import time
from pathlib import Path

from cadgen import build123d as bd
from cadgen import read_step
from cadgen.color import linear_to_srgb
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.Geom2d import Geom2d_TrimmedCurve
from OCP.GeomAbs import GeomAbs_CurveType, GeomAbs_SurfaceType
from OCP.ShapeCustom import ShapeCustom
from OCP.ShapeFix import ShapeFix_Shape
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCP.gp import gp_Trsf, gp_Vec
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.TopoDS import TopoDS

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v


ROOT = m.PROJECT_ROOT
SOURCE = ROOT / "cad/local_fix/open_duck_mini_covers_forward_review.step"
ORIGINAL = ROOT / "cad/source_exports/open_duck_original_isolated_branch.step"
DEFAULT_WORK = Path("/tmp/refit")
# ``restore`` interfaces need the original branch geometry.  Reading the whole
# 47 MB branch next to the 47 MB current model blows the memory cap, so the few
# restored parts are pre-extracted into this folder and read one at a time.
ORIGINAL_CACHE = Path("/tmp/refit_orig")

PLATE_MM = 3.0
HORN_CLEAR_RADIUS_MM = m.HORN_CLEARANCE_RADIUS_MM
PLATE_EDGE_RADIUS_MM = m.HORN_BRACKET_PAD_RADIUS_MM
POCKET_DEPTH_MM = 12.0
SERVO_CLEARANCE_MM = 0.2
# Restored case walls stop this far short of the ZP10D case faces, and the
# case relief is cut a little tighter so it never touches the wall itself.
CASE_WALL_CLEARANCE_MM = 0.20
# Half window cast back around each case screw that falls in a wall opening.
CASE_SEAT_HALF_MM = 7.00
CASE_RELIEF_CLEARANCE_MM = 0.15
CASE_FACE_GAP_MM = 0.10
EXTEND_STEP_MM = 1.0
BAND_MM = 5.0
BLOCK_SPAN_MM = 400.0
THROUGH_MM = 80.0
HORN_LATERAL_MM = 13.0
HORN_PLUG_RADIUS_MM = 10.0
HORN_HOLE_RADIUS_MM = m.HORN_HOLE_DIAMETER_MM / 2.0
HORN_COUNTERBORE_RADIUS_MM = m.HORN_COUNTERBORE_DIAMETER_MM / 2.0
CASE_HOLE_RADIUS_MM = m.CASE_HOLE_DIAMETER_MM / 2.0
CASE_PAD_CUT_RADIUS_MM = m.ROLL_CASE_FAR_PAD_RADIUS_MM + 0.5
HORN_PAD_CUT_RADIUS_MM = m.HORN_PAD_RADIUS_MM + 0.5
HORN_PAD_CUT_HALF_MM = m.HORN_PAD_REACH_MM + 1.0
CASE_PAD_CUT_HALF_MM = m.CASE_PAD_EXTRA_REACH_MM + 0.5

# Radius of the horn mounting hub.  The original branch mounts every horn on a
# diameter-22 hub cast into the connector (measured r 11.00 faces on the
# original head_pitch_to_yaw), so the resized hubs keep that diameter.
HORN_HUB_RADIUS_MM = 11.0
HORN_BOSS_RADIUS_MM = 12.0
HORN_BOSS_BORE_RADIUS_MM = 10.3
HORN_BOSS_ROOT_MM = 1.0
# The disc is held by a plain cylindrical boss - no skirt, no cast ramp and no
# longer disc: the boss keeps its original depth and the void underneath the
# disc is closed by lifting the bore floor, i.e. by filling that cavity with a
# flat face instead of growing anything.
#
# The only opening left under the disc is the horn clearance pocket, and it
# only has to let the disc's own hub through: the ZP10D passive horn reaches no
# lower than its own lowest point, so the bore floor belongs one clearance
# below that.  Drilling further (the old ``low - 4.0``) leaves a blind dia 20.6
# cavity right where the disc meets f70/f77.  Bore axis, radius, screw pattern
# and counterbores are untouched.
HORN_BOSS_BORE_FLOOR_MM = SERVO_CLEARANCE_MM

# The plate the disc stands on has its own opening for the disc, and the disc
# sits off-centre in it: the two are 0.02 mm apart over most of the
# circumference but 1.2 mm apart on the -X side, so the junction reads as a
# slot that never got closed.  The fill is that empty crescent itself - flat
# bottom on the boss root plane, flat top on the plate's own top face - fused
# back in.  It adds nothing outside the plate's opening, never touches the disc
# wall's outside, and neither grows nor moves the disc.
HORN_BOSS_FILL_OUTER_MM = 13.35
HORN_BOSS_FILL_STEPS_MM = ()


# The S11 neck-pitch case sits in a pocket of ``trunk_top`` whose walls are 0.5
# and 1.0 mm off the ZP10D case, so the servo itself cannot move sideways; the
# S01/S06 horn bosses are trimmed instead so they stop 0.2 mm short of the S11
# horns at X=-9.0.
BOSS_TRIM_X_MAX_MM = -9.20

# The original branch carries the neck-pitch servo in one cast rectangular tub:
# floor top on the case bottom face, wall inner faces flush on the case faces,
# walls about 5 mm thick stopping just under the two horns, case screws going
# through the front and back walls, the case's right side open towards the
# outer shell.  The covers-forward model deleted that tub and hung the servo on
# four fused cylinders, so it is cast again as a single box at the ZP10D
# dimensions.
S11_TUB_WALL_MM = 5.00
S11_TUB_TOP_MM = 78.00
S11_POCKET_CLEARANCE_MM = 0.10
# The neck-pitch servo was coincident with the hip-yaw horns, so the whole servo
# is moved this far towards the robot's front (+X).  The neck hangs off S11's
# horns - the two neck sheets are bolted to them and carry the head pitch servo
# and the head - so everything S11 drives moves with it, otherwise those horn
# joints would open up.  The antenna parts ride on the head.
S11_SHIFT_X_MM = 8.00
HEAD_YAW_RING_INNER_RADIUS_MM = 38.5
HEAD_YAW_RING_OUTER_RADIUS_MM = 50.75
HEAD_FLOOR_OPENING_RADII_MM = (54.0, 60.0)
HEAD_FLOOR_DROP_MM = 177.50093881236 - 171.87687762474
NECK_PLATE_WIDTH_MM = 22.0
HEAD_ROLL_BEARING_RISE_MM = 3.0
S11_FOLLOW_PATHS = (
    ("trunk_assembly <1>", "S11_NECK_PITCH_ZP10D"),
    ("neck_pitch_assembly <1>",),
    ("neck_yaw_assembly <1>",),
    ("head_assembly <1>",),
    # printed connector between the head pitch and head yaw servos: it is a
    # sibling of the neck group in this document, not a child of it.
    ("head_pitch_to_yaw",),
)
S11_FOLLOW_LABELS = ("antenna", "left_antenna_holder", "right_antenna_holder")
# Refitted connectors that ride on the moved chain.  They are rebuilt in the
# model's own frame (the servos they wrap are read before the shift), so the
# shift is baked into the part when it is written.
S11_SHIFTED_PARTS = ("head_pitch_to_yaw", "head_yaw_to_roll", "head_roll_mount", "neck_left_sheet", "neck_right_sheet")
S11_HOLE_CUT_HALF_MM = 8.00
# The original branch drills the case screws with a diameter-2.5 through hole
# and a diameter-5 by 3 mm counterbore opened on the outer face of each wall
# (measured on the original trunk_top: r 1.25 faces and r 2.5, 3.00 mm deep
# counterbores on the four hole axes).  The same screw sizes are used here.
S11_CASE_HOLE_RADIUS_MM = 1.25
S11_CASE_COUNTERBORE_RADIUS_MM = 2.50
S11_CASE_COUNTERBORE_DEPTH_MM = 3.00
# The covers-forward model kept the neck pocket floor at the original seat level
# (Z 54.89); the ZP10D case bottom sits 4.61 mm higher.  The tub is one box and
# that difference is only its floor thickness - nothing is stacked on the floor
# and no second shape is blended in beside it.
S11_HOST_FLOOR_MM = 54.89
S11_ROOT_BITE_MM = 1.00
# The tub reaches past the two hip-yaw servos, so it is trimmed to their
# envelopes before it is cast in, which keeps the parts collision free.
S11_NEIGHBOUR_SERVOS = ("S01_RIGHT_HIP_YAW_ZP10D", "S06_LEFT_HIP_YAW_ZP10D")

# The restored upper connectors clamp the servo case between the end walls the
# original branch cast into the same part.  Each wall is cut off at the plane
# where its inner face belongs and cast out again by its own thickness, so the
# wall keeps the original section and only its length/position follows the ZP10D.
# The wall's outer face and thickness are measured on the original branch; the
# case face it bolts against is measured on the ZP10D case itself at each of its
# screw axes.  Entries are (axis, side sign, old outer face, wall thickness).
ORIGINAL_WALLS = {
    "head_assembly <1>__head_roll_mount": (
        ((1.0, 0.0, 0.0), -1.0, 41.00, 4.00),
        ((1.0, 0.0, 0.0), 1.0, 81.00, 4.00),
    ),
    "neck_yaw_assembly <1>__head_yaw_to_roll": (
        ((0.0, 0.0, 1.0), -1.0, 171.75, 6.00),
        ((0.0, 0.0, 1.0), 1.0, 213.95, 4.00),
    ),
}

# Hole sizes of the original branch: dia 2.5 through holes with dia 5.5
# counterbores 3 mm deep, opened on the outer face of each wall.
ORIGINAL_CASE_HOLE_RADIUS_MM = 1.25
ORIGINAL_CASE_COUNTERBORE_RADIUS_MM = 2.75
ORIGINAL_CASE_COUNTERBORE_DEPTH_MM = 3.00

# (path key, servo label) -> extra options for the bolted-horn boss refit.
INTERFACE_OPTIONS = {
    (("trunk_assembly <1>", "trunk_top"), "S01_RIGHT_HIP_YAW_ZP10D"): {"trim_x_max": BOSS_TRIM_X_MAX_MM},
    (("trunk_assembly <1>", "trunk_top"), "S06_LEFT_HIP_YAW_ZP10D"): {"trim_x_max": BOSS_TRIM_X_MAX_MM},
    (("head_assembly <1>", "head_roll_mount"), "S14_HEAD_ROLL_ZP10D"): {
        "walls": ORIGINAL_WALLS["head_assembly <1>__head_roll_mount"],
        "original_case_holes": True,
    },
    (("neck_yaw_assembly <1>", "head_yaw_to_roll"), "S13_HEAD_YAW_ZP10D"): {
        "walls": ORIGINAL_WALLS["neck_yaw_assembly <1>__head_yaw_to_roll"],
        "original_case_holes": True,
    },
}

SUPPORT_GAP_MM = 0.15
SUPPORT_LATERAL_MM = 6.0

# (assembly path, interface kind, servo label, horn side, along, lateral, restore)
# ``restore`` swaps the interface region back to the original connector
# structure before the resize; it is only used where the covers-forward model
# deleted that structure and replaced it with pads.  ``trunk_top`` keeps its
# current geometry and only gets the dimensional nudge.
# The ownership follows JointSpec in validate_s01_s10_interfaces_local and
# validate_s11_s14_interfaces_local.
INTERFACES = (
    (("trunk_assembly <1>", "trunk_top"), "support", "S01_RIGHT_HIP_YAW_ZP10D", "passive", 40.0, 16.0, "bolt"),
    (("trunk_assembly <1>", "trunk_top"), "support", "S06_LEFT_HIP_YAW_ZP10D", "passive", 40.0, 16.0, "bolt"),
    (("trunk_assembly <1>", "trunk_top"), "case", "S11_NECK_PITCH_ZP10D", None, 55.0, 22.0, "wrap"),
    (("neck_pitch_assembly <1>", "neck_left_sheet"), "horn", "S11_NECK_PITCH_ZP10D", "active", 0.0, 0.0, "neck"),
    (("neck_pitch_assembly <1>", "neck_right_sheet"), "horn", "S11_NECK_PITCH_ZP10D", "passive", 0.0, 0.0, "neck"),
    (("head_pitch_to_yaw",), "horn", "S12_HEAD_PITCH_ZP10D", "both", 46.0, 17.0, "ring"),
    (("head_pitch_to_yaw",), "horn", "S13_HEAD_YAW_ZP10D", "both", 46.0, 17.0, "restore"),
    (("neck_yaw_assembly <1>", "head_yaw_to_roll"), "case", "S13_HEAD_YAW_ZP10D", None, 55.0, 22.0, "restore"),
    (("neck_yaw_assembly <1>", "head_yaw_to_roll"), "horn", "S14_HEAD_ROLL_ZP10D", "active", 46.0, 18.0, "restore"),
    (("head_assembly <1>", "head_roll_mount"), "case", "S14_HEAD_ROLL_ZP10D", None, 55.0, 22.0, "restore"),
)


def _step_colour(color):
    """Encode a linear channel colour so a STEP write/read round trip is identity.

    cadgen treats a ``Color``'s channels as linear RGB and decodes the value in
    a STEP file with sRGB->linear, but the writer stores the channels verbatim.
    Writing them as-is therefore darkens every part on the way back in, so the
    channels are converted to sRGB just before export.
    """
    if color is None:
        return None
    channels = [float(channel) for channel in color]
    if len(channels) < 3:
        return None
    alpha = channels[3] if len(channels) > 3 else 1.0
    return (
        linear_to_srgb(channels[0]),
        linear_to_srgb(channels[1]),
        linear_to_srgb(channels[2]),
        alpha,
    )


def _encode_colours(node):
    encoded = _step_colour(getattr(node, "color", None))
    if encoded is not None:
        node.color = encoded
    for child in _children(node):
        _encode_colours(child)
    return node


def _children(shape):
    return list(getattr(shape, "children", None) or [])


MEMORY_LIMIT_MB = min(float(os.environ.get("REFIT_MEMORY_LIMIT_MB", "3500")), 4000.0)


def _rss_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def _start_memory_watchdog():
    """Abort the run before the host runs out of memory."""

    def watch():
        while True:
            rss = _rss_mb()
            memory = {}
            with open("/proc/meminfo") as handle:
                for line in handle:
                    key, value, *_ = line.split()
                    memory[key.rstrip(":")] = int(value)
            host_used = (memory["MemTotal"] - memory["MemAvailable"]) / 1024.0
            if rss > MEMORY_LIMIT_MB or host_used > 4500.0:
                print(f"ABORT: process RSS {rss:.0f} MiB, host used {host_used:.0f} MiB", flush=True)
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


def _walk(shape):
    yield shape
    for child in _children(shape):
        yield from _walk(child)


def _find(shape, path):
    node = shape
    for label in path:
        hits = [child for child in _walk(node) if child.label == label]
        if len(hits) != 1:
            raise ValueError(f"{' / '.join(path)}: {label} matched {len(hits)} nodes")
        node = hits[0]
    return node


def _shifted_servo(servo):
    """The neck-pitch servo placed ``S11_SHIFT_X_MM`` towards the robot's front.

    Only the tub around S11 is measured off the moved servo; every other part
    of the model is refitted in its own frame.
    """
    offset = bd.Pos(S11_SHIFT_X_MM, 0.0, 0.0)

    def move(node):
        children = _children(node)
        if children:
            return bd.Compound(
                children=[move(child) for child in children], label=node.label
            )
        if node.wrapped.ShapeType() != TopAbs_SOLID:
            raise ValueError(f"{node.label}: servo part is not a solid")
        return node.moved(offset)

    return move(servo)


def _baked_shift(part):
    """``part`` with the chain's shift baked into its geometry, placement default."""
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(S11_SHIFT_X_MM, 0.0, 0.0))
    moved = BRepBuilderAPI_Transform(part.wrapped, trsf, True).Shape()
    solids = []
    explorer = TopExp_Explorer(moved, TopAbs_SOLID)
    while explorer.More():
        solids.append(TopoDS.Solid_s(explorer.Current()))
        explorer.Next()
    if len(solids) != 1:
        raise ValueError(f"{part.label}: shifted part is not a single solid")
    baked = bd.Solid(solids[0])
    baked.label = part.label
    if getattr(part, "color", None) is not None:
        baked.color = part.color
    return baked


def _replace_leaf(root, old, new):
    for parent in _walk(root):
        children = list(_children(parent))
        for index, child in enumerate(children):
            if child is old:
                children[index] = new
                parent.children = tuple(children)
                return True
    return False


def _shift_s11_chain(robot):
    """Move the neck-pitch servo and everything it drives towards the front.

    S11 sat coincident with the two hip-yaw horns, so the servo is shifted
    ``S11_SHIFT_X_MM`` towards +X.  Its two horns are bolted to the neck sheets,
    which carry the head pitch servo and the head, so everything above it
    travels along - otherwise those horn joints would open up by the shift.
    Returns the labels that were moved, for the log.

    The placement rides on the leaves, and a leaf's placement is what both the
    measurement helpers (``_case_mount_lines`` and the bounding boxes) and the
    STEP writer read, so each leaf gets the offset in front of its own
    placement.  Group nodes keep theirs, which would otherwise apply twice.
    """
    offset = bd.Pos(S11_SHIFT_X_MM, 0.0, 0.0)
    targets = []
    seen = set()
    for path in S11_FOLLOW_PATHS:
        for node in _walk(_find(robot, path)):
            if not _children(node) and id(node) not in seen:
                seen.add(id(node))
                targets.append(node)
    for label in S11_FOLLOW_LABELS:
        for node in _walk(robot):
            if node.label == label and not _children(node) and id(node) not in seen:
                seen.add(id(node))
                targets.append(node)
    moved = []
    for node in targets:
        node.location = offset * node.location
        moved.append(node.label)
    return moved


def _items(result):
    if result is None:
        return []
    if hasattr(result, "volume"):
        return [result]
    return list(result)


def _shape(result):
    items = [item for item in _items(result) if item.volume > 0.0]
    if not items:
        raise ValueError("empty boolean result")
    return items[0] if len(items) == 1 else bd.Compound(children=items)


def _fuse(*shapes):
    return _shape(shapes[0].fuse(*shapes[1:]))


def _cut(shape, tools):
    result = shape
    for index, tool in enumerate(tools, start=1):
        try:
            result = _shape(result.cut(tool))
        except ValueError as error:
            raise ValueError(
                f"cut {index}/{len(tools)} failed on a {tool.volume:.1f} mm3 tool "
                f"(remaining part {result.volume:.1f} mm3)"
            ) from error
    return result


def _rod(point, direction, start, end, radius):
    direction = direction.normalized()
    middle = point + direction * ((start + end) / 2.0)
    cylinder = bd.Cylinder(radius, end - start)
    if direction.Z < -0.999999:
        cylinder = cylinder.rotate(bd.Axis.X, 180)
    elif direction.Z < 0.999999:
        rotation_axis = bd.Vector(0, 0, 1).cross(direction)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, direction.Z))))
        cylinder = cylinder.rotate(bd.Axis((0, 0, 0), rotation_axis.normalized()), angle)
    return cylinder.moved(bd.Pos(*middle))


def _cone(point, direction, start, end, radius_start, radius_end):
    """A truncated cone, axis ``direction``, from ``start`` up to ``end``."""
    direction = direction.normalized()
    middle = point + direction * ((start + end) / 2.0)
    cone = bd.Cone(radius_start, radius_end, end - start)
    if direction.Z < -0.999999:
        cone = cone.rotate(bd.Axis.X, 180)
    elif direction.Z < 0.999999:
        rotation_axis = bd.Vector(0, 0, 1).cross(direction)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, direction.Z))))
        cone = cone.rotate(bd.Axis((0, 0, 0), rotation_axis.normalized()), angle)
    return cone.moved(bd.Pos(*middle))


def _oriented_box(point, direction, lateral, start, end):
    box = bd.Box(2.0 * lateral, 2.0 * lateral, end - start)
    if direction.Z < -0.999999:
        box = box.rotate(bd.Axis.X, 180)
    elif direction.Z < 0.999999:
        rotation_axis = bd.Vector(0, 0, 1).cross(direction)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, direction.Z))))
        box = box.rotate(bd.Axis((0, 0, 0), rotation_axis.normalized()), angle)
    middle = point + direction * ((start + end) / 2.0)
    return box.moved(bd.Pos(*middle))


def _projection(shape, point, direction):
    box = shape.bounding_box()
    values = [
        (bd.Vector(x, y, z) - point).dot(direction)
        for x in (box.min.X, box.max.X)
        for y in (box.min.Y, box.max.Y)
        for z in (box.min.Z, box.max.Z)
    ]
    return min(values), max(values)


def _block(part, point, direction, lateral, start, end):
    result = part.intersect(_oriented_box(point, direction, lateral, start, end))
    items = [item for item in _items(result) if item.volume > 0.5]
    if not items:
        return None
    return items[0] if len(items) == 1 else bd.Compound(children=items)


def _plug(part, point, direction, radius, start, end):
    result = part.intersect(_rod(point, direction, start, end, radius))
    items = [item for item in _items(result) if item.volume > 0.5]
    if not items:
        return None
    return items[0] if len(items) == 1 else bd.Compound(children=items)


def _grow_hub(part, point, direction, radius, current, target):
    """Extend an existing hub face out to ``target`` in one clean union.

    Moving a material band step by step (``_extend``) fuses a new solid per
    millimetre and makes the face count explode on these cast covers, so the
    hub is lengthened with a single cylinder that bites 1 mm into the material
    that is already there.
    """
    if target <= current + 0.02:
        return part
    return _fuse(part, _rod(point, direction, current - 1.0, target, radius))


def _extend(part, point, direction, lateral, current, target):
    """Move the material band at ``current`` until the face reaches ``target``."""
    delta = target - current
    if abs(delta) <= 0.02:
        return part
    if delta > 0.0:
        band = _block(part, point, direction, lateral, current - BAND_MM, current + 0.01)
    else:
        band = _block(part, point, direction, lateral, current - 0.01, current + BAND_MM)
    if band is None:
        return part
    steps = max(1, int(math.ceil(abs(delta) / EXTEND_STEP_MM)))
    moved = [
        band.moved(bd.Pos(*(direction * (math.copysign(index * abs(delta) / steps, delta)))))
        for index in range(1, steps + 1)
    ]
    return _fuse(part, *moved)


def _body_relief(body, clearance=SERVO_CLEARANCE_MM, log=None):
    """The vendor body grown sideways by the rotational clearance.

    Offsetting the body once costs a fraction of what fusing five almost
    coincident copies of it costs (that union peaks over 2.5 GB and minutes of
    boolean time on these covers), and the result is trimmed back to the body's
    own Z range so the seat under the case keeps its contact.

    OCC's 3D offset drops the whole result ("Null TopoDS_Shape") for some
    body/amount pairs even though the request is a plain outward offset, and
    which pairs fail is not monotonic in the amount (the S14 case fails at 0.15
    but succeeds at 0.20, the S13 case the other way round).  Any amount within
    the clearance band gives the same fit, so the first amount that yields a
    solid is used.
    """
    grown = None
    used = clearance
    for amount in (
        clearance,
        clearance + 0.05,
        clearance + 0.10,
        clearance - 0.05,
        clearance * 2.0,
    ):
        if amount <= 0.0:
            continue
        try:
            candidate = bd.offset(body, amount=amount)
        except Exception:
            candidate = None
        if candidate is not None and getattr(candidate, "volume", 0.0) > 0.0:
            grown, used = candidate, amount
            break
    if grown is None:
        raise ValueError("body clearance offset failed for every amount")
    if log is not None and abs(used - clearance) > 1e-9:
        log.append(
            f"    body clearance {clearance:.2f} -> {used:.2f} (OCC offset retry)"
        )
    box = body.bounding_box()
    window = _world_box(
        box.min.X - 50.0,
        box.max.X + 50.0,
        box.min.Y - 50.0,
        box.max.Y + 50.0,
        box.min.Z,
        box.max.Z,
    )
    return _shape(grown.intersect(window))


# Every leaf that is not being refitted and is not a ZP10D itself.  A cast
# boss is trimmed against these envelopes so it can never grow into a
# neighbouring part.
NEIGHBOUR_LEAVES = []
# Only the neck-pitch tub part needs the neighbour sweep: it is the one whose
# cast disc sits among free-standing frame bars.  The head connectors are left
# on their own verified path.
CLEAR_NEIGHBOURS = False


def _clear_neighbours(shape, log=None, tag="", clearance=SERVO_CLEARANCE_MM):
    """Cut ``shape`` back out of every neighbour envelope it reaches into."""
    box = shape.bounding_box()
    hits = []
    for other, shift in NEIGHBOUR_LEAVES:
        b = other.bounding_box()
        # A part that rides on the S11 chain is tested (and cut) where it ends
        # up.  The located copy is built only for the parts that pass this box
        # filter - copying all of them up front costs hundreds of megabytes.
        bminx, bmaxx = b.min.X + shift, b.max.X + shift
        if (
            bmaxx < box.min.X
            or bminx > box.max.X
            or b.max.Y < box.min.Y
            or b.min.Y > box.max.Y
            or b.max.Z < box.min.Z
            or b.min.Z > box.max.Z
        ):
            continue
        if shift:
            other = other.moved(bd.Pos(shift, 0.0, 0.0))
        result = other.intersect(shape)
        volume = 0.0
        if result is not None:
            volume = sum(item.volume for item in _items(result))
        if volume <= 0.001:
            continue
        # Sliding the neighbour envelope a clearance towards the boss is far
        # cheaper than offsetting it (the offset of these covers peaks over the
        # memory cap) and leaves the same 0.2 mm on the face that meets the boss.
        ob = other.bounding_box()
        moved = bd.Vector(
            (box.min.X + box.max.X) / 2.0 - (ob.min.X + ob.max.X) / 2.0,
            (box.min.Y + box.max.Y) / 2.0 - (ob.min.Y + ob.max.Y) / 2.0,
            (box.min.Z + box.max.Z) / 2.0 - (ob.min.Z + ob.max.Z) / 2.0,
        )
        if moved.length > 1e-9:
            moved = moved.normalized() * clearance
        tool = other.moved(bd.Pos(moved.X, moved.Y, moved.Z))
        shape = _cut(shape, [tool])
        hits.append((other.label, volume))
    if hits and log is not None:
        log.append(
            f"{tag}: clearance {clearance:.2f} cast against "
            + ", ".join(f"{label} ({volume:.3f} mm3)" for label, volume in hits)
        )
    return shape


def _horn_sides(side, passive, active):
    if side == "active":
        return (active,)
    if side == "passive":
        return (passive,)
    if side == "both":
        return (passive, active)
    raise ValueError(side)


def _region(servo, kind, side, along, lateral, vendor_body):
    """Oriented box covering the servo interface on the host part."""
    body, passive, active = v._servo_parts(servo)
    axis, centre = v._axis(servo)
    if kind == "support":
        return _oriented_box(centre, axis, lateral, -along, along), None
    lines = (
        v._case_mount_lines(body, vendor_body)
        if kind == "case"
        else v._horn_pattern_lines(active, axis, centre)
    )
    direction = lines[0][1].normalized()
    if kind == "horn":
        return _oriented_box(centre, axis, lateral, -along, along), None
    return _oriented_box(centre, direction, lateral, -along, along), None


def _refit_horn(part, servo, side, vendor_body, log, cut_pads=False):
    """Thicken the existing ear/plate until it bears on the ZP10D horn face."""
    body, passive, active = v._servo_parts(servo)
    axis, centre = v._axis(servo)
    for horn in _horn_sides(side, passive, active):
        lines = v._horn_pattern_lines(horn, axis, centre)
        if cut_pads:
            part = _cut(
                part,
                [
                    _rod(point, line_direction, -HORN_PAD_CUT_HALF_MM, HORN_PAD_CUT_HALF_MM, HORN_PAD_CUT_RADIUS_MM)
                    for point, line_direction in lines
                ],
            )
            log.append(f"{servo.label} {horn.label}: removed {len(lines)} fused horn pads")
        direction = lines[0][1].normalized()
        outward = direction if (lines[0][0] - centre).dot(direction) > 0 else -direction
        low, high = _projection(horn, centre, outward)
        target = high + PLATE_MM
        block = _plug(
            part, centre, outward, HORN_PLUG_RADIUS_MM, low - 8.0, high + PLATE_MM + 6.0
        )
        if block is None:
            log.append(f"{servo.label} {horn.label}: no material on the horn axis")
            continue
        current = _projection(block, centre, outward)[1]
        if current < target - 0.02:
            log.append(
                f"{servo.label} {horn.label}: horn {low:.2f}..{high:.2f} face {current:.2f} "
                f"-> {target:.2f} (thicken {target - current:+.2f} mm)"
            )
            part = _grow_hub(part, centre, outward, HORN_HUB_RADIUS_MM, current, target)
        else:
            log.append(
                f"{servo.label} {horn.label}: horn {low:.2f}..{high:.2f} face {current:.2f} "
                f"already reaches {target:.2f}, no thickening"
            )
        tools = [_rod(centre, outward, low - 1.0, high, HORN_CLEAR_RADIUS_MM)]
        for point, line_direction in lines:
            offset = (point - centre).dot(outward)
            tools.append(
                _rod(
                    point,
                    outward,
                    target - m.HORN_COUNTERBORE_DEPTH_MM - offset,
                    target + 1.0 - offset,
                    HORN_COUNTERBORE_RADIUS_MM,
                )
            )
            tools.append(
                _rod(point, line_direction, -THROUGH_MM, THROUGH_MM, HORN_HOLE_RADIUS_MM)
            )
        tools.append(_body_relief(body, log=log))
        part = _cut(part, tools)
    return part


def _extrude_section(part, direction, at, delta, log=None):
    """Cast the part's own section at ``at`` by ``delta`` along ``direction``."""
    if abs(delta) <= 0.02:
        return part
    faces = []
    for face in part.faces():
        if "PLANE" not in str(face.geom_type):
            continue
        centre = face.center()
        try:
            normal = face.normal_at(centre)
        except Exception:
            continue
        if abs(normal.dot(direction)) < 0.999:
            continue
        if abs(centre.dot(direction) - at) > 0.6:
            continue
        faces.append(face)
    if not faces:
        raise ValueError(f"no section face at {at:.2f} along {tuple(direction)}")
    outward = direction if delta > 0.0 else -direction
    tools = [bd.extrude(face, amount=abs(delta), dir=outward) for face in faces]
    if log is not None:
        log.append(
            f"    section at {at:.2f}: {len(faces)} face(s) cast {delta:+.2f} mm outward"
        )
    return _fuse(part, *tools)


def _side_box(outward, at):
    """A big box covering the half space the unit vector ``outward`` points to.

    ``at`` is the plane's own coordinate along ``outward``.
    """
    return _oriented_box(bd.Vector(0, 0, 0), outward, 400.0, at, at + 1000.0)


def _stretch_wall(part, axis, sign, old_outer, thickness, new_inner, log):
    """Move one cast end wall along ``axis`` so it meets the ZP10D case face.

    The original branch bolts the servo case between two end walls cast into
    the same part.  The wall keeps its own section and its own thickness; only
    its position follows the bigger case, so the material outward of the wall's
    old inner face is cut off there, moved along the axis by the amount the
    case grew, and the gap that opens where the host keeps running (the cap
    plate and the side arms) is cast from the freshly cut section.  Nothing is
    added and nothing is reshaped: the wall only changes its length.
    """
    outward = axis * sign
    old_inner = old_outer - sign * thickness
    delta = new_inner - old_inner
    if abs(delta) <= 0.02:
        log.append(f"    end wall {old_outer:.2f}: already on the ZP10D case face")
        return part
    at = old_inner * sign
    slab = _cut(part, [_side_box(-outward, -at)])
    rest = _cut(part, [_side_box(outward, at)])
    moved = slab.moved(bd.Pos(axis * delta))
    bridge = _extrude_section(rest, axis, old_inner, delta)
    log.append(
        f"    end wall {old_outer:.2f} -> {old_outer + delta:.2f} along "
        f"({axis.X:.0f},{axis.Y:.0f},{axis.Z:.0f}): inner face {old_inner:.2f} -> "
        f"{new_inner:.2f} on the ZP10D case face, thickness {thickness:.2f}"
    )
    return _fuse(rest, moved, bridge)


def _wall_plane_faces(part, axis, at):
    """The part's own planar faces on the wall plane at ``at`` along ``axis``."""
    faces = []
    for face in part.faces():
        if "PLANE" not in str(face.geom_type):
            continue
        centre = face.center()
        try:
            normal = face.normal_at(centre)
        except Exception:
            continue
        if abs(normal.dot(axis)) < 0.999:
            continue
        if abs(centre.dot(axis) - at) > 0.6:
            continue
        faces.append(face)
    return faces


def _case_seat(part, wall_axis, sign, new_inner, thickness, body, lines, log):
    """Cast the wall's own section back around the case's end screws.

    The original wall is a ring whose opening is cut to the *original* case
    cross section.  On the bigger ZP10D the case's end screws can fall inside
    that opening, where the wall has nothing left to bolt and the case has no
    seat.  A window of the wall's own filled section is cast back around each
    screw, so the wall keeps its own silhouette, its own thickness and its own
    faces; where it already carries material nothing changes.
    """
    outward = wall_axis * sign
    faces = _wall_plane_faces(part, wall_axis, new_inner)
    if not faces:
        raise ValueError(f"no wall face at {new_inner:.2f} along {tuple(wall_axis)}")
    tools = []
    for face in faces:
        outline = bd.Face(face.outer_wire())
        if outline.area <= 1e-6:
            continue
        tools.append(bd.extrude(outline, amount=thickness, dir=outward))
    slab = _fuse(*tools)
    seats = []
    for point, _ in lines:
        window = _oriented_box(
            point, wall_axis, CASE_SEAT_HALF_MM, -thickness - 2.0, thickness + 2.0
        )
        piece = slab.intersect(window)
        items = [item for item in _items(piece) if item.volume > 0.5]
        if not items:
            continue
        seats.append(items[0] if len(items) == 1 else bd.Compound(children=items))
    if not seats:
        log.append(f"    end wall {new_inner:.2f}: no case seat needed")
        return part
    for index, seat in enumerate(seats, start=1):
        before = part.volume
        part = _fuse(part, seat)
        log.append(
            f"    end wall {new_inner:.2f}: case seat {index}/{len(seats)} cast "
            f"({before:.1f} -> {part.volume:.1f} mm3), thickness {thickness:.2f}"
        )
    return part


def _refit_case(part, servo, vendor_body, log, walls=(), original_holes=False):
    """Re-cut the ZP10D case interface on the host part.

    ``walls`` lists the end walls the original branch casts around the case as
    ``(axis, side sign, old outer face, wall thickness)``.  When it is given the
    host is the original structure and each wall is stretched along the axis
    until its inner face lands on the ZP10D case face, so the case stays bolted
    between the same two walls and only their length follows the bigger servo.
    Otherwise the covers-forward host is kept and only the fused cylindrical
    screw bosses are cut away.
    """
    body, _, _ = v._servo_parts(servo)
    axis, centre = v._axis(servo)
    lines = v._case_mount_lines(body, vendor_body)
    if walls:
        # Restored connector: the original branch already casts the case between
        # its own end walls, so there is no fused boss to cut away.
        log.append(f"{servo.label} case: original end walls kept")
    else:
        bosses = [
            _rod(point, direction, -CASE_PAD_CUT_HALF_MM, CASE_PAD_CUT_HALF_MM, CASE_PAD_CUT_RADIUS_MM)
            for point, direction in v._case_mount_lines(body, vendor_body)
        ]
        part = _cut(part, bosses)
        log.append(f"{servo.label} case: removed {len(bosses)} fused screw bosses")
    direction = lines[0][1].normalized()
    body_low, body_high = _projection(body, centre, direction)
    if walls:
        for wall_axis, sign, old_outer, thickness in walls:
            wall_axis = bd.Vector(*wall_axis).normalized()
            outward = wall_axis * sign
            own = [
                point
                for point, _ in lines
                if (point - centre).dot(outward) > 0.0
            ]
            if not own:
                raise ValueError(f"{servo.label} case: no screw axis on one end wall")
            seat = max((point - centre).dot(outward) for point in own)
            new_inner = centre.dot(wall_axis) + (seat + CASE_WALL_CLEARANCE_MM) * sign
            part = _stretch_wall(
                part, wall_axis, sign, old_outer, thickness, new_inner, log
            )
            own_lines = [
                (point, direction)
                for point, direction in lines
                if (point - centre).dot(outward) > 0.0
            ]
            part = _case_seat(
                part, wall_axis, sign, new_inner, thickness, body, own_lines, log
            )
    else:
        block = _block(part, centre, direction, 26.0, -BLOCK_SPAN_MM, BLOCK_SPAN_MM)
        if block is None:
            raise ValueError(f"{servo.label} case: no host material on the screw axis")
        part_low, part_high = _projection(block, centre, direction)
        if part_low > body_high:
            face, target = part_low, body_high + CASE_FACE_GAP_MM
        elif part_high < body_low:
            face, target = part_high, body_low - CASE_FACE_GAP_MM
        else:
            face = None
            log.append(f"{servo.label} case: host already meets the case envelope")
        if face is not None:
            log.append(
                f"{servo.label} case: body {body_low:.2f}..{body_high:.2f} host {part_low:.2f}..{part_high:.2f} "
                f"face {face:.2f} -> {target:.2f} (move {target - face:+.2f} mm)"
            )
            part = _extend(part, centre, direction, 26.0, face, target)
    hole_radius = ORIGINAL_CASE_HOLE_RADIUS_MM if original_holes else CASE_HOLE_RADIUS_MM
    tools = [
        _rod(point, hole_direction, -THROUGH_MM, THROUGH_MM, hole_radius)
        for point, hole_direction in lines
    ]
    if original_holes:
        for point, hole_direction in lines:
            outward = hole_direction.normalized()
            if (point - centre).dot(outward) < 0.0:
                outward = -outward
            outer = _projection(part, centre, outward)[1]
            offset = (point - centre).dot(outward)
            tools.append(
                _rod(
                    point,
                    outward,
                    outer - offset - ORIGINAL_CASE_COUNTERBORE_DEPTH_MM,
                    outer - offset + 1.0,
                    ORIGINAL_CASE_COUNTERBORE_RADIUS_MM,
                )
            )
    log.append(
        f"{servo.label} case: recut {len(lines)} holes dia {2 * hole_radius:.1f}"
        + (f" + dia {2 * ORIGINAL_CASE_COUNTERBORE_RADIUS_MM:.1f} counterbores"
           if original_holes else "")
    )
    tools.append(_body_relief(body, clearance=CASE_RELIEF_CLEARANCE_MM, log=log))
    return _cut(part, tools)


def _world_box(x0, x1, y0, y1, z0, z1):
    box = bd.Box(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
    return box.moved(bd.Pos((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0))


def _refit_s11_case(part, servo, vendor_body, log):
    """Cast the S11 neck-pitch case mount as one rectangular tub.

    The original branch carries the neck-pitch servo in a single cast tub of
    ``trunk_top``: a floor the case bottom rests on, wall inner faces flush on
    the case end faces, and walls about 5 mm thick stopping just under the
    horns, with the case screws passing through the wall thickness.  The
    covers-forward model deleted that tub and hung the ZP10D on four fused
    cylinders, so the same tub is cast again in one piece and only its
    dimensions follow the bigger servo:

    * floor top on the ZP10D case bottom face,
    * wall inner faces on the two ZP10D case end faces (Y +21.225 / -18.775),
    * 5 mm walls, top at Z 78.0 (under the horns at Z 80.0),
    * the four M2 case screws re-cut along their own axes.

    No cylinder, bracket or spacer is introduced, and the tub is a single box
    with a single pocket cut out of it.
    """
    # The covers-forward model hung the servo on four fused cylinders that sit on
    # the *old* case screw axes, so the removal cut has to run on the servo's own
    # (unshifted) frame - cutting on the shifted axes would only carve empty space
    # and leave the cylinders standing outside the new tub.
    pads = v._case_mount_lines(v._servo_parts(servo)[0], vendor_body)
    part = _cut(
        part,
        [
            _rod(point, direction, -CASE_PAD_CUT_HALF_MM, CASE_PAD_CUT_HALF_MM, CASE_PAD_CUT_RADIUS_MM)
            for point, direction in pads
        ],
    )
    log.append(
        f"{servo.label} case: removed {len(pads)} fused screw bosses at their own "
        f"X {pads[0][0].X:.2f}/{pads[-1][0].X:.2f}"
    )

    # The tub and its screw holes follow the servo to its new place; the rest of
    # this part keeps the model's own frame.
    servo = _shifted_servo(servo)
    body, _, _ = v._servo_parts(servo)
    axis, centre = v._axis(servo)
    if abs(abs(axis.Y) - 1.0) > 1e-6:
        raise ValueError(f"{servo.label}: expected a Y joint axis, got {tuple(axis)}")

    lines = v._case_mount_lines(body, vendor_body)

    footprint = body.bounding_box()
    face_ys = sorted({round(point.Y, 4) for point, _ in lines})
    back_y, front_y = face_ys[0], face_ys[-1]
    seat_z = footprint.min.Z
    floor_z = S11_HOST_FLOOR_MM - S11_ROOT_BITE_MM
    outer_back = back_y - S11_TUB_WALL_MM
    outer_front = front_y + S11_TUB_WALL_MM
    # ONE box, one layer.  The walls are the same thickness on all four sides of
    # the case, and the height the ZP10D case gained over the original seat is
    # carried by the floor's own thickness - no second plate is stacked on the
    # floor, no ramp is fused beside the walls.
    tub_shape = _world_box(
        footprint.min.X - S11_TUB_WALL_MM,
        footprint.max.X + S11_TUB_WALL_MM,
        outer_back,
        outer_front,
        floor_z,
        S11_TUB_TOP_MM,
    )
    log.append(
        f"{servo.label} case: one-piece tub, walls {S11_TUB_WALL_MM:.2f} on all four "
        f"sides, own floor {seat_z - floor_z:.2f} thick from Z {floor_z:.2f} up to the "
        f"case bottom Z {seat_z:.2f} (no stacked plate, no ramp)"
    )
    for neighbour_label in S11_NEIGHBOUR_SERVOS:
        neighbour = SERVOS.get(neighbour_label)
        if neighbour is None:
            continue
        for neighbour_part in v._servo_parts(neighbour):
            if neighbour_part is None:
                continue
            items = [
                item
                for item in _items(tub_shape.intersect(neighbour_part))
                if item.volume > 0.001
            ]
            if not items:
                continue
            boxes = [item.bounding_box() for item in items]
            corner = _world_box(
                min(box.min.X for box in boxes) - SERVO_CLEARANCE_MM,
                max(box.max.X for box in boxes) + SERVO_CLEARANCE_MM,
                min(box.min.Y for box in boxes) - SERVO_CLEARANCE_MM,
                max(box.max.Y for box in boxes) + SERVO_CLEARANCE_MM,
                min(box.min.Z for box in boxes) - SERVO_CLEARANCE_MM,
                max(box.max.Z for box in boxes) + SERVO_CLEARANCE_MM,
            )
            tub_shape = _shape(tub_shape.cut(corner))
            log.append(
                f"{servo.label} case: tub corner X {corner.bounding_box().min.X:.2f}.."
                f"{corner.bounding_box().max.X:.2f} Y {corner.bounding_box().min.Y:.2f}.."
                f"{corner.bounding_box().max.Y:.2f} cleared for {neighbour_label}"
            )
    part = _fuse(part, tub_shape)
    log.append(
        f"{servo.label} case: tub X {tub_shape.bounding_box().min.X:.2f}.."
        f"{tub_shape.bounding_box().max.X:.2f} "
        f"Y {outer_back:.2f}..{outer_front:.2f} "
        f"Z {S11_HOST_FLOOR_MM - S11_ROOT_BITE_MM:.2f}..{S11_TUB_TOP_MM:.2f} "
        f"(walls {S11_TUB_WALL_MM:.1f} on case faces Y {back_y:.3f}/{front_y:.3f})"
    )
    pocket = _world_box(
        footprint.min.X - S11_POCKET_CLEARANCE_MM,
        footprint.max.X + S11_POCKET_CLEARANCE_MM,
        back_y,
        front_y,
        seat_z,
        S11_TUB_TOP_MM + 1.0,
    )
    part = _cut(part, [pocket])
    log.append(
        f"{servo.label} case: pocket X {footprint.min.X - S11_POCKET_CLEARANCE_MM:.2f}.."
        f"{footprint.max.X + S11_POCKET_CLEARANCE_MM:.2f} Y {back_y:.3f}..{front_y:.3f} "
        f"floor top Z {seat_z:.2f} (case bottom), open above {S11_TUB_TOP_MM:.1f}"
    )
    hole_tools = []
    for point, _ in lines:
        hole_tools.append(
            _rod(
                point,
                bd.Vector(0, 1, 0),
                -S11_HOLE_CUT_HALF_MM,
                S11_HOLE_CUT_HALF_MM,
                S11_CASE_HOLE_RADIUS_MM,
            )
        )
        outward = 1.0 if point.Y > centre.Y else -1.0
        outer_face = outer_front if outward > 0 else outer_back
        hole_tools.append(
            _rod(
                bd.Vector(point.X, outer_face, point.Z),
                bd.Vector(0, outward, 0),
                -S11_CASE_COUNTERBORE_DEPTH_MM,
                0.5,
                S11_CASE_COUNTERBORE_RADIUS_MM,
            )
        )
    part = _cut(part, hole_tools)
    log.append(
        f"{servo.label} case: re-cut {len(lines)} case holes dia "
        f"{2 * S11_CASE_HOLE_RADIUS_MM:.1f} through the walls with dia "
        f"{2 * S11_CASE_COUNTERBORE_RADIUS_MM:.1f} x "
        f"{S11_CASE_COUNTERBORE_DEPTH_MM:.1f} counterbores on the outer faces"
    )

    relief = [body]
    for relief_axis in (bd.Vector(1, 0, 0), bd.Vector(0, 1, 0)):
        for sign in (1.0, -1.0):
            relief.append(body.moved(bd.Pos(*(relief_axis * (sign * SERVO_CLEARANCE_MM)))))
    part = _cut(part, [_fuse(*relief)])
    log.append(f"{servo.label} case: case clearance {SERVO_CLEARANCE_MM:.2f} cut, floor keeps contact")
    return part


def _refit_support(part, servo, side, log):
    """Advance the existing passive-horn bearing face until it makes contact.

    Only the material that already faces the horn is moved, and only by the
    millimetres the ZP10D envelope needs, so the face keeps its own shape.
    """
    _, passive, active = v._servo_parts(servo)
    axis, centre = v._axis(servo)
    horn = passive if side == "passive" else active
    gap = part.distance_to(horn)
    if gap <= SUPPORT_GAP_MM + 1e-6:
        log.append(f"{servo.label} support: gap {gap:.3f} already within target")
        return part
    lines = v._horn_pattern_lines(horn, axis, centre)
    direction = lines[0][1].normalized()
    outward = direction if (lines[0][0] - centre).dot(direction) > 0 else -direction
    low, high = _projection(horn, centre, outward)
    grow = gap - SUPPORT_GAP_MM
    for near, sign in ((low, 1.0), (high, -1.0)):
        window = (near - 8.0, near + 0.5) if sign > 0 else (near - 0.5, near + 8.0)
        block = _block(part, centre, outward, SUPPORT_LATERAL_MM, *window)
        if block is None:
            continue
        face = _projection(block, centre, outward)[1 if sign > 0 else 0]
        offset = abs(face - near)
        if abs(offset - gap) > 0.10:
            log.append(
                f"{servo.label} support: candidate near {near:.3f} face {face:.3f} "
                f"offset {offset:.3f} != gap {gap:.3f}, skipped"
            )
            continue
        target = face + sign * grow
        log.append(
            f"{servo.label} support: gap {gap:.3f} bearing face {face:.3f} -> {target:.3f} "
            f"(grow {grow:.3f} mm towards the horn)"
        )
        return _extend(part, centre, outward, SUPPORT_LATERAL_MM, face, target)
    log.append(f"{servo.label} support: bearing face not found, left alone")
    return part


def _close_horn_collar_slot(part, centre, log, tag):
    top = max(
        (face for face in part.faces()
         if str(face.geom_type) == "GeomType.PLANE"
         and abs(face.center().Z - centre.Z - 22.5) < 0.001),
        key=lambda face: face.area,
    ).center().Z
    bottom = centre.Z + 20.7
    for lateral in (-1.0, 1.0):
        fill = bd.Box(2.2, 8.4, top - bottom).moved(
            bd.Pos(centre.X - 12.0, centre.Y + lateral * 7.2, (top + bottom) / 2.0)
        )
        previous_volume = part.volume
        part = _fuse(part, fill).clean()
        log.append(
            f"{tag}: collar side {lateral:+.0f} filled flush at Z {top:.6f}, "
            f"added {part.volume - previous_volume:.3f} mm3"
        )
    return part




def _refit_bolt(part, servo, side, log, trim_x_max=None):
    """Grow the host's own material into a horn mounting boss and bolt pattern.

    The covers-forward model left the upper ZP10D horn with no host material at
    all: the dia 20 face plate and its four R7 screw axes float in free space.
    The host is thickened locally until its face reaches the horn face plus one
    plate thickness, then the horn clearance bore, the four dia 3.2 through
    holes and the dia 6.5 counterbores are cut.  Nothing is added away from the
    horn, and the boss is fused into the host's own wall.

    ``trim_x_max`` cuts the boss back at that world X so it cannot reach into
    the neighbouring S11 neck-pitch horns; the R7 pattern and its bore stay
    inside the remaining material.
    """
    body, passive, active = v._servo_parts(servo)
    axis, centre = v._axis(servo)
    for horn in _horn_sides(side, passive, active):
        pad_lines = v._horn_pattern_lines(horn, axis, centre)
        part = _cut(
            part,
            [
                _rod(point, line_direction, -HORN_PAD_CUT_HALF_MM, HORN_PAD_CUT_HALF_MM, HORN_PAD_CUT_RADIUS_MM)
                for point, line_direction in pad_lines
            ],
        )
        log.append(f"{servo.label} {horn.label}: removed {len(pad_lines)} fused horn pads")
        lines = v._horn_pattern_lines(horn, axis, centre)
        direction = lines[0][1].normalized()
        outward = direction if (lines[0][0] - centre).dot(direction) > 0 else -direction
        low, high = _projection(horn, centre, outward)
        target = high + PLATE_MM
        block = _plug(part, centre, outward, HORN_BOSS_RADIUS_MM, low - 14.0, high + 1.0)
        if block is None:
            raise ValueError(f"{servo.label} {horn.label}: no host material to grow")
        current = _projection(block, centre, outward)[1]
        boss = _rod(
            centre,
            outward,
            current - HORN_BOSS_ROOT_MM,
            target,
            HORN_BOSS_RADIUS_MM,
        )
        log.append(
            f"{servo.label} {horn.label}: plain dia {2 * HORN_BOSS_RADIUS_MM:.1f} boss, "
            f"rooted {HORN_BOSS_ROOT_MM:.2f} mm into the host face (no skirt), "
            f"{boss.volume:.0f} mm3 kept"
        )
        if CLEAR_NEIGHBOURS:
            boss = _clear_neighbours(
                boss, log=log, tag=f"{servo.label} {horn.label} disc"
            )
        if trim_x_max is not None:
            trim = bd.Box(120.0, 120.0, 120.0).moved(
                bd.Pos(trim_x_max + 60.0, centre.Y, centre.Z)
            )
            boss = _shape(boss.cut(trim))
            log.append(
                f"{servo.label} {horn.label}: boss trimmed at X={trim_x_max:.2f} "
                f"to clear the S11 horns ({boss.volume:.0f} mm3 kept)"
            )
        part = _fuse(part, boss)
        log.append(
            f"{servo.label} {horn.label}: host face grown {current:.3f} -> {target:.3f} "
            f"(+{target - current:.3f} mm), boss dia {2 * HORN_BOSS_RADIUS_MM:.1f}"
        )
        opening = _cut(
            _rod(centre, outward, current - HORN_BOSS_ROOT_MM, current,
                 HORN_BOSS_FILL_OUTER_MM),
            [
                _rod(centre, outward, current - HORN_BOSS_ROOT_MM - 1.0,
                     high - PLATE_MM + 1.0, HORN_BOSS_RADIUS_MM)
            ],
        )
        crescent = _cut(
            opening,
            [
                part,
                bd.Box(120.0, 120.0, 120.0).moved(
                    bd.Pos(centre.X + 0.5 + 60.0, centre.Y, centre.Z)
                ),
            ],
        )
        if any(item.volume > m.MAX_DISCONNECTED_TRIM_MM3 for item in _items(crescent)):
            log.append(
                f"{servo.label} {horn.label}: filled the empty crescent between the "
                f"disc wall and the plate opening ({crescent.volume:.1f} mm3, "
                f"flat top on the plate face)"
            )
            part = _fuse(part, crescent)
        part = _close_horn_collar_slot(
            part, centre, log, f"{servo.label} {horn.label}"
        )
        tools = [
            _rod(
                centre,
                outward,
                low - HORN_BOSS_BORE_FLOOR_MM,
                high + 0.001,
                HORN_BOSS_BORE_RADIUS_MM,
            )
        ]
        for point, _ in lines:
            tools.append(
                _rod(point, outward, -0.5, PLATE_MM + 1.0, HORN_HOLE_RADIUS_MM)
            )
            tools.append(
                _rod(
                    point,
                    outward,
                    PLATE_MM - m.HORN_COUNTERBORE_DEPTH_MM,
                    PLATE_MM + 1.0,
                    HORN_COUNTERBORE_RADIUS_MM,
                )
            )
        part = _cut(part, tools)
    return part


SERVOS = {}


# ---------------------------------------------------------------------------
# Export hygiene
# ---------------------------------------------------------------------------
# The booleans that resize these cast covers leave two kinds of artefact on the
# result even though the solid itself is sound:
#
# * an edge whose pcurve range sits outside the pcurve's own parameter domain,
#   so a reader that rebuilds the pcurve (``Geom2d_TrimmedCurve``) raises
#   "parameters out of range";
# * faces produced by extruding/offsetting a section that carry hyperbolic
#   edges, which cannot be re-based on Bezier form ("cosh" overflow).
#
# Both only affect parametric data, never dimensions, and both are repaired
# here before a part is written: the shape is re-parameterised consistently and
# the few extruded/offset faces are re-based on B-splines.  Only the refitted
# parts pass through this - untouched parts are never rebuilt.
UNREADABLE_CURVES = (
    GeomAbs_CurveType.GeomAbs_Hyperbola,
    GeomAbs_CurveType.GeomAbs_Parabola,
)
UNREADABLE_SURFACES = (
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceType.GeomAbs_OffsetSurface,
)
# Mode switches ``ShapeFix_Wire`` accepts; only the ones it actually exposes are
# turned on, the rest are skipped (OCCT renames them between versions).
_WIRE_FIX_MODES = (
    "FixReversed2d",
    "FixEdgeCurves",
    "FixSameParameter",
    "FixConnected",
    "FixGaps2d",
    "FixGaps3d",
    "FixClosed",
    "FixDegenerated",
    "FixLacking",
    "FixReorder",
    "FixAddCurve3d",
    "FixAddPCurve",
)


def _shape_fix(shape):
    """Re-parameterise every edge/pcurve of ``shape`` so they agree."""
    fixer = ShapeFix_Shape(shape)
    fixer.SetMaxTolerance(1e-3)
    wire = fixer.FixFaceTool().FixWireTool()
    for name in _WIRE_FIX_MODES:
        attribute = f"{name}Mode"
        if hasattr(wire, attribute):
            setattr(wire, attribute, 1)
    fixer.FixFaceTool().FixWireMode = 1
    fixer.FixSolidTool().FixShellMode = 1
    fixer.Perform()
    return fixer.Shape()


def _audit_export_shape(shape):
    """Count the defects that stop a parametric reader: (trims, curves, faces)."""
    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, faces)
    trims = 0
    curves = 0
    surfaces = 0
    for index in range(1, faces.Extent() + 1):
        face = TopoDS.Face_s(faces.FindKey(index))
        if BRepAdaptor_Surface(face).GetType() in UNREADABLE_SURFACES:
            surfaces += 1
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            edge = TopoDS.Edge_s(explorer.Current())
            explorer.Next()
            if BRep_Tool.Degenerated_s(edge):
                continue
            if BRepAdaptor_Curve(edge).GetType() in UNREADABLE_CURVES:
                curves += 1
            pcurve = BRep_Tool.CurveOnSurface_s(edge, face, 0.0, 0.0)
            if pcurve is None:
                trims += 1
                continue
            first, last = BRep_Tool.Range_s(edge, face)
            try:
                Geom2d_TrimmedCurve(pcurve, first, last)
            except Exception:
                trims += 1
    return trims, curves, surfaces


def _unlocated_solid(solid):
    """``solid`` with a default location, so it is written as a plain solid.

    A shape whose placement is numerically identity but not the *default*
    location still makes the STEP writer emit an assembly with a single
    component, and a reader then hands that back as a compound - the part would
    gain an extra level in the assembly tree.  Dropping such a placement does
    not move anything, so it is dropped; a real placement is left alone.
    """
    location = solid.Location()
    if location.IsIdentity():
        return solid
    transformation = location.Transformation()
    default = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    values = [
        transformation.Value(row, column)
        for row in (1, 2, 3)
        for column in (1, 2, 3, 4)
    ]
    if all(abs(value - reference) <= 1e-9 for value, reference in zip(values, default)):
        return solid.Located(TopLoc_Location())
    return solid


def _repair_export(shape, tag, log=None):
    """Make ``shape`` readable by a parametric extractor, without moving it."""
    shape = _shape_fix(shape)
    trims, curves, surfaces = _audit_export_shape(shape)
    if curves or surfaces:
        shape = ShapeCustom.ConvertToBSpline_s(shape, True, True, True, False)
        shape = _shape_fix(shape)
        trims, curves, surfaces2 = _audit_export_shape(shape)
        if log is not None:
            log.append(
                f"    export repair: {trims} pcurve(s) rebuilt, {surfaces} extruded/offset "
                f"face(s) and {curves} hyperbolic edge(s) re-based on B-splines "
                f"({surfaces2} left)"
            )
    if trims:
        raise ValueError(f"{tag}: {trims} edge pcurve(s) still unreadable after repair")
    # ``ShapeFix_Shape`` hands the solid back wrapped in a compound; unwrap it so
    # the part keeps its place in the assembly tree.
    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(_unlocated_solid(TopoDS.Solid_s(explorer.Current())))
        explorer.Next()
    return solids[0] if len(solids) == 1 else shape


def _refit_s12_ring(part, servo, log):
    _, passive, active = v._servo_parts(servo)
    axis, centre = v._axis(servo)
    original_top = centre.Z + 20.1
    bridge_top = centre.Z + 15.67593881237
    bridge_bottom = bridge_top - 4.0
    part = _cut(part, [
        _world_box(-100.0, 100.0, 15.0, 100.0, centre.Z - 20.0, original_top + 0.001),
        _world_box(-100.0, 100.0, -100.0, -15.0, centre.Z - 20.0, original_top + 0.001),
    ])
    back_extension = _world_box(centre.X - 21.0, centre.X - 17.0,
                                -10.9, 11.1, bridge_bottom, original_top)
    part = _fuse(part, back_extension)
    for horn, sign in ((passive, -1.0), (active, 1.0)):
        direction = bd.Vector(0.0, sign, 0.0)
        lines = v._horn_pattern_lines(horn, axis, centre)
        inner = lines[0][0].Y * sign
        outer = inner + 4.0
        left = bd.Vector(centre.X - 11.0, sign * outer, centre.Z)
        right = bd.Vector(centre.X + 11.0, sign * outer, centre.Z)
        top_left = bd.Vector(left.X, left.Y, bridge_top)
        top_right = bd.Vector(right.X, right.Y, bridge_top)
        outline = bd.Wire([
            bd.Edge.make_three_point_arc(left, bd.Vector(centre.X, sign * outer, centre.Z - 11.0), right),
            bd.Edge.make_line(right, top_right),
            bd.Edge.make_line(top_right, top_left),
            bd.Edge.make_line(top_left, left),
        ])
        wall = bd.extrude(bd.Face(outline), amount=4.0, dir=-direction)
        lower_y, upper_y = sorted((0.0, sign * outer))
        bridge = _world_box(centre.X - 21.0, centre.X + 9.975,
                            lower_y, upper_y, bridge_bottom, bridge_top)
        part = _fuse(part, wall, bridge).clean()
        tools = [_rod(centre, direction, inner - centre.dot(direction) - 0.01,
                      outer - centre.dot(direction) + 0.01, 3.8)]
        for position, _ in lines:
            tools.extend([
                _rod(position, direction, -0.01, 4.01, HORN_HOLE_RADIUS_MM),
                _rod(position, direction, 2.0, 4.01, HORN_COUNTERBORE_RADIUS_MM),
            ])
        part = _cut(part, tools).clean()
        log.append(f"S12 {horn.label}: continuous ring wall, inner Y {sign * inner:.3f}, 4 mm wall; full-width bridge extended")
    return part


def _refit_neck(original, servo, side, vendor_body, log):
    _, passive, active = v._servo_parts(servo)
    axis, centre = v._axis(servo)
    horn = active if side == "active" else passive
    sign = 1.0 if side == "active" else -1.0
    direction = bd.Vector(0.0, sign, 0.0)
    lines = v._horn_pattern_lines(horn, axis, centre)
    horn_face = lines[0][0].Y * sign
    upper_body = v._servo_parts(SERVOS["S12_HEAD_PITCH_ZP10D"])[0]
    case_lines = [
        line for line in v._case_mount_lines(upper_body, vendor_body)
        if line[0].Y * sign > 0.0
    ]
    case_face = case_lines[0][0].Y * sign
    outer = horn_face + 4.0
    bounds = original.bounding_box()
    radius = 11.0
    half_width = (bounds.max.X - bounds.min.X) / 2.0
    shoulder = centre.Z + 30.89
    top = max(bounds.max.Z, max(point.Z for point, _ in case_lines) + 4.0)
    height = shoulder - centre.Z
    length_squared = half_width**2 + height**2
    tangent_x = (radius**2 * half_width + radius * height * math.sqrt(length_squared - radius**2)) / length_squared
    tangent_z = (radius**2 * height - radius * half_width * math.sqrt(length_squared - radius**2)) / length_squared
    def point(horizontal, vertical):
        return bd.Vector(centre.X + horizontal, sign * outer, vertical)
    bottom_left = point(-tangent_x, centre.Z + tangent_z)
    bottom_right = point(tangent_x, centre.Z + tangent_z)
    outline = bd.Wire([
        bd.Edge.make_three_point_arc(bottom_left, point(0.0, centre.Z - radius), bottom_right),
        bd.Edge.make_line(bottom_right, point(half_width, shoulder)),
        bd.Edge.make_line(point(half_width, shoulder), point(half_width, top)),
        bd.Edge.make_line(point(half_width, top), point(-half_width, top)),
        bd.Edge.make_line(point(-half_width, top), point(-half_width, shoulder)),
        bd.Edge.make_line(point(-half_width, shoulder), bottom_left),
    ])
    part = bd.extrude(bd.Face(outline), amount=outer - case_face, dir=-direction)
    transition_height = centre.Z + 20.0
    pocket_points = [
        (-100.0, centre.Z - 20.0), (horn_face, centre.Z - 20.0),
        (horn_face, transition_height), (case_face, transition_height),
        (case_face, top + 1.0), (-100.0, top + 1.0),
    ]
    pocket_wire = bd.Wire.make_polygon([
        bd.Vector(centre.X - 40.0, sign * lateral, vertical)
        for lateral, vertical in pocket_points
    ], close=True)
    pocket = bd.extrude(bd.Face(pocket_wire), amount=80.0, dir=bd.Vector(1, 0, 0))
    part = _cut(part, [pocket, upper_body])
    tools = [_rod(centre, direction, horn_face - centre.dot(direction) - 0.01,
                  outer - centre.dot(direction) + 0.01, 3.8)]
    for position, _ in lines:
        tools.extend([
            _rod(position, direction, -0.01, 4.01, HORN_HOLE_RADIUS_MM),
            _rod(position, direction, 2.0, 4.01, HORN_COUNTERBORE_RADIUS_MM),
        ])
    for position, _ in case_lines:
        thickness = outer - case_face
        tools.extend([
            _rod(position, direction, -0.01, thickness + 0.01, 1.25),
            _rod(position, direction, thickness - 3.0, thickness + 0.01, 2.5),
        ])
    part = _cut(part, tools).clean()
    log.append(f"neck {side}: original tapered plate and R11 end; horn face {sign * horn_face:.3f}, 4 mm thick; case face {sign * case_face:.3f}; integral straight-wall step")
    return part


def _refit_part(part, original_part, entries, vendor_body, log, path=()):
    if original_part is not None:
        part = _shape(original_part)
        log.append(
            f"{path[-1] if path else 'part'}: base cast from the original branch "
            f"structure ({part.volume:.0f} mm3); only its dimensions follow the ZP10D"
        )
    for kind, servo_label, side, along, lateral, approach in entries:
        servo = SERVOS[servo_label]
        if approach == "neck":
            return _refit_neck(part, servo, side, vendor_body, log)
        if approach == "ring":
            part = _refit_s12_ring(part, servo, log)
            continue
        options = INTERFACE_OPTIONS.get((tuple(path), servo_label), {})
        if approach == "bolt":
            part = _refit_bolt(
                part, servo, side, log, trim_x_max=options.get("trim_x_max")
            )
        if approach == "restore":
            log.append(f"{servo.label} {kind}: original structure kept")
        if approach == "bolt":
            pass
        elif approach == "wrap":
            part = _refit_s11_case(part, servo, vendor_body, log)
        elif kind == "horn":
            part = _refit_horn(part, servo, side, vendor_body, log, cut_pads=approach == "cut")
        elif kind == "case":
            part = _refit_case(
                part,
                servo,
                vendor_body,
                log,
                walls=options.get("walls", ()),
                original_holes=options.get("original_case_holes", False),
            )
        else:
            part = _refit_support(part, servo, side, log)
        fragments = sorted(
            (solid.volume for solid in part.solids() if solid.volume > m.MAX_DISCONNECTED_TRIM_MM3),
            reverse=True,
        )
        log.append(f"    -> solids {[round(v, 1) for v in fragments]}")
    return part


def _refit_head_motion(part, s13_case_x, s14_hole_positions, head_floor=None):
    name = part.label
    if name == "body_front":
        colour = part.color
        axis = bd.Axis((9, .725, 90.000938812376), (0, 1, 0))
        tools = [_world_box(-2.2, 20.2, lower_y, upper_y, 90.000938812376, 138).rotate(axis, 45)
                 for lower_y, upper_y in ((-29.475, -18.575), (21.025, 30.925))]
        part = _cut(part, tools).clean()
        if not part.is_valid or len(part.solids()) != 1:
            raise ValueError("body_front: neck openings must remain one valid solid")
        part.label = name
        part.color = colour
        return part
    if name == "trunk_top":
        return _smooth_trunk_tub_join(part)
    if name in ("neck_left_sheet", "neck_right_sheet"):
        centre_x = 1.0 + S11_SHIFT_X_MM
        colour = part.color
        part = _cut(part, [
            _world_box(-200, centre_x - NECK_PLATE_WIDTH_MM / 2, -100, 100, 0, 300),
            _world_box(centre_x + NECK_PLATE_WIDTH_MM / 2, 200, -100, 100, 0, 300),
        ]).clean()
        if not part.is_valid or len(part.solids()) != 1:
            raise ValueError(f"{name}: width adjustment must remain one valid solid")
        part.label = name
        part.color = colour
        return part
    if name not in ("head_pitch_to_yaw", "head_yaw_to_roll", "head_bot_sheet", "head"):
        return part
    colour = part.color
    centre_x = 1.0 + S11_SHIFT_X_MM
    centre = bd.Vector(centre_x, 0.1, 0)
    vertical = bd.Vector(0, 0, 1)
    lower = 171.87687762474
    upper = 177.87687762474
    if name == "head_pitch_to_yaw":
        part = _stretch_wall(part, bd.Vector(1, 0, 0), -1, centre_x - 21, 4, centre_x - 33, [])
        part = _cut(part, [
            _world_box(centre_x - 33, centre_x + 100, -100, 100, 171.776, 178.276),
            _world_box(centre_x - 100, centre_x - 20, 4.1, 100, lower - 4.2, 178.276),
            _world_box(centre_x - 100, centre_x - 20, -100, -3.9, lower - 4.2, 178.276),
            _rod(centre, vertical, 178.276, 223.57593881241, 32.4),
        ])
        part = _fuse(part,
            _world_box(centre_x - 37, centre_x - 33, -10.9, 11.1, 213.00093881237, 226.57593881241),
            _world_box(centre_x - 37, centre_x - 10.99, -10.9, 11.1, 223.57593881241, 226.57593881241))
        rear_inner_x = centre_x - 14.1
        rear_outer_x = rear_inner_x - 3.0
        part = _cut(part, [
            _world_box(centre_x - 100, rear_outer_x, -100, 100, lower - 4.2, 300),
        ])
        part = _fuse(part, _world_box(
            rear_outer_x, rear_inner_x, -10.9, 11.1, lower - 4.2, 226.57593881241))
        part = _smooth_s13_arm_junctions(part, centre_x)
    elif name == "head_yaw_to_roll":
        shaft_region = _world_box(centre_x - 100, centre_x - 40, -20, 20, 180, 212)
        shaft = _shape(part.intersect(shaft_region))
        part = _fuse(_cut(part, [shaft_region]), shaft.moved(bd.Pos(0, 0, HEAD_ROLL_BEARING_RISE_MM)))
        annulus = _cut(_rod(centre, vertical, lower, upper - .226, HEAD_YAW_RING_OUTER_RADIUS_MM),
                       [_rod(centre, vertical, lower - 1, upper + 1, HEAD_YAW_RING_INNER_RADIUS_MM)])
        part = _fuse(part, annulus)
        seats = _world_box(centre_x + 15, centre_x + 51, -12.46, 12.66, lower - 1, upper + 1)
        opening = _cut(_rod(centre, vertical, lower - .01, upper + .01, HEAD_YAW_RING_INNER_RADIUS_MM), [seats])
        inner_wall = _shape(_rod(centre, vertical, upper, 207.1, HEAD_YAW_RING_INNER_RADIUS_MM).intersect(
            _world_box(centre_x - 100, centre_x, -100, 100, upper, 207.1)))
        part = _cut(part, [opening, inner_wall])
        case_x = s13_case_x
        faces = [face for face in part.faces() if face.geom_type == bd.GeomType.PLANE
                 and face.normal_at().X < -.999 and face.area > 100 and face.center().X > case_x]
        if len(faces) != 1:
            raise ValueError("S13: expected one front plate inner face")
        extension = faces[0].center().X - case_x
        part = _fuse(part, bd.extrude(faces[0], amount=extension, dir=bd.Vector(-1, 0, 0)))
        tools = []
        for position in s14_hole_positions:
            tools.extend([
                _rod(position, bd.Vector(-1, 0, 0), -.01, position.X - case_x + .01, HORN_HOLE_RADIUS_MM),
                _rod(position, bd.Vector(-1, 0, 0), PLATE_MM - m.HORN_COUNTERBORE_DEPTH_MM,
                     position.X - case_x + .01, HORN_COUNTERBORE_RADIUS_MM),
            ])
        part = _cut(part, tools)
    elif name == "head_bot_sheet":
        radius_x, radius_y = HEAD_FLOOR_OPENING_RADII_MM
        opening = _rod(centre, vertical, 177, 181, radius_y)
        opening = opening.scale((radius_x / radius_y, 1, 1), about=centre)
        part = _cut(part, [opening])
        part = part.moved(bd.Pos(0, 0, -HEAD_FLOOR_DROP_MM))
    else:
        part = _extend_head_lower_shell(part)
        if head_floor is None:
            raise ValueError("head: missing floor mounting clearance")
        part = _cut(part, [head_floor])
    part = part.clean()
    if not part.is_valid or len(part.solids()) != 1:
        raise ValueError(f"{name}: motion clearance must remain one valid solid")
    part.label = name
    part.color = colour
    return part


def _extend_head_lower_shell(part):
    split_z = 184.0
    lower_region = _world_box(-200, 200, -200, 200, 100, split_z)
    bearing_region = _world_box(-50, -20, -20, 20, 100, split_z + .01)
    lower_region = _cut(lower_region, [bearing_region])
    lower_shell = _shape(part.intersect(lower_region))
    upper_shell = _cut(part, [lower_region])
    section_faces = [face for face in lower_shell.faces()
                     if face.geom_type == bd.GeomType.PLANE
                     and face.normal_at().Z > .999
                     and abs(face.center().Z - split_z) < 1e-5]
    if not section_faces:
        raise ValueError("head: missing lower shell extension section")
    extensions = [bd.extrude(face, amount=HEAD_FLOOR_DROP_MM, dir=bd.Vector(0, 0, -1))
                  for face in section_faces]
    return _fuse(upper_shell, lower_shell.moved(bd.Pos(0, 0, -HEAD_FLOOR_DROP_MM)),
                 *extensions).clean()


def _smooth_s13_arm_junctions(part, centre_x):
    upper_seat_z = 223.57593881241
    lower_seat_z = 171.57593881241
    upper_arm = _world_box(centre_x - 17.1, centre_x, -10.9, 11.1,
                           upper_seat_z, upper_seat_z + 3.0)
    horn_region = _rod(bd.Vector(centre_x, .1, 0), bd.Vector(0, 0, 1),
                       upper_seat_z - .01, upper_seat_z + 3.01, 11.0)
    part = _fuse(part, _cut(upper_arm, [horn_region]))
    shoulder = _world_box(centre_x - 100, centre_x + 100, -100, 100,
                          lower_seat_z, 171.67687762474)
    rear_wall = _world_box(centre_x - 17.1, centre_x - 14.1, -10.9, 11.1,
                           lower_seat_z - .01, 171.68687762474)
    return _cut(part, [_cut(shoulder, [rear_wall])]).clean()


def _smooth_trunk_tub_join(part):
    fresh = _fuse(part,
                  _world_box(-21.36, -5.9999, -17.88, 17.88, 54.890938812376, 78),
                  _world_box(-6, 24, -23.775, 26.225, 49.991, 53.891)).clean()
    edges = []
    for edge in fresh.edges():
        bounds = edge.bounding_box()
        if (edge.geom_type == bd.GeomType.LINE and bounds.size.Z > 20
                and bounds.size.X < 1e-5 and bounds.size.Y < 1e-5
                and abs(bounds.min.X - 24) < 1e-4
                and (abs(bounds.min.Y + 23.775) < 1e-5 or abs(bounds.min.Y - 26.225) < 1e-5)):
            edges.append(edge)
    if len(edges) != 2:
        raise ValueError("trunk_top: expected two outer tub corners")
    fresh = fresh.fillet(2.0, edges).clean()
    for sign in (-1, 1):
        lower_y, upper_y = (17.88, 20.88) if sign > 0 else (-20.88, -17.88)
        corner = _world_box(-9, -5.9999, lower_y, upper_y, 54.890938812376, 78)
        relief = _rod(bd.Vector(-9, sign * 20.88, 54), bd.Vector(0, 0, 1), 0, 25, 3)
        fresh = _fuse(fresh, _cut(corner, [relief])).clean()
    unified = ShapeUpgrade_UnifySameDomain(fresh.wrapped, True, True, False)
    unified.SetLinearTolerance(1e-6)
    unified.Build()
    fixed = ShapeFix_Shape(unified.Shape())
    fixed.Perform()
    fresh = bd.Solid(fixed.Shape())
    if not fresh.is_valid or len(fresh.solids()) != 1:
        raise ValueError("trunk_top: smoothed tub must remain one valid solid")
    fresh.label = part.label
    fresh.color = part.color
    return fresh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "cad/local_fix/open_duck_mini_upper_refit.step",
    )
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    _start_memory_watchdog()
    print(f"watchdog armed at {MEMORY_LIMIT_MB:.0f} MB", flush=True)

    wanted = args.only or []
    grouped = {}
    for path, kind, servo_label, side, along, lateral, approach in INTERFACES:
        grouped.setdefault(path, []).append((kind, servo_label, side, along, lateral, approach))

    jobs = [
        (path, "__".join(path), entries)
        for path, entries in grouped.items()
        if not (wanted and "__".join(path) not in wanted)
        and not (not wanted and args.assemble)
    ]

    robot = read_step(SOURCE)
    vendor = read_step(m.VENDOR_SERVO)
    vendor_body = max(_children(vendor), key=lambda child: child.volume)
    for _, _, servo_label, _, _, _, _ in INTERFACES:
        SERVOS[servo_label] = _find(robot, (servo_label,))
    print(f"resolved {len(SERVOS)} servo references", flush=True)

    global CLEAR_NEIGHBOURS
    CLEAR_NEIGHBOURS = "__".join(wanted) == "trunk_assembly <1>__trunk_top"
    refit_labels = {path[-1] for path in grouped}
    offset = bd.Pos(S11_SHIFT_X_MM, 0.0, 0.0)

    def collect(node, path=()):
        kids = _children(node)
        if kids:
            for child in kids:
                collect(child, path + (child.label,))
            return
        if node.label.startswith("ZP10D_vendor") or node.label in refit_labels:
            return
        rides = node.label in S11_FOLLOW_LABELS or any(
            path[: len(follow)] == follow for follow in S11_FOLLOW_PATHS
        )
        # The parts that ride on the S11 chain are measured where they end up,
        # otherwise the disc would be trimmed against material that has moved
        # away by the time the model is assembled.
        NEIGHBOUR_LEAVES.append((node, S11_SHIFT_X_MM if rides else 0.0))

    collect(robot)
    print(f"collected {len(NEIGHBOUR_LEAVES)} neighbour leaves", flush=True)

    for path, key, entries in jobs:
        cached = args.work / f"{key}.step"
        if cached.exists():
            print(f"skip {key} (cached)", flush=True)
            continue
        log = []
        print(f"refit {key}", flush=True)
        current_part = _find(robot, path)
        try:
            original_part = None
            if any(approach in ("restore", "neck") for *_, approach in entries):
                small = ORIGINAL_CACHE / f"{key}.step"
                if not small.exists():
                    raise SystemExit(
                        f"{key}: run /tmp/probe/extract_orig_parts.py first, "
                        f"{small} is missing"
                    )
                original_part = read_step(str(small))
            part = _refit_part(current_part, original_part, entries, vendor_body, log, path=path)
        except Exception:
            for line in log:
                print(f"  {line}", flush=True)
            raise
        for line in log:
            print(f"  {line}", flush=True)
        solids = [solid for solid in part.solids() if solid.volume > m.MAX_DISCONNECTED_TRIM_MM3]
        if len(solids) != 1:
            raise ValueError(f"{key}: refit produced {len(solids)} solids "
                             f"{[round(s.volume, 1) for s in solids]}")
        fresh = bd.Solid(_repair_export(solids[0].wrapped, key, log))
        if path[-1] == "trunk_top":
            unified = ShapeUpgrade_UnifySameDomain(fresh.wrapped, True, True, False)
            unified.SetLinearTolerance(1e-6)
            unified.Build()
            fixed = ShapeFix_Shape(unified.Shape())
            fixed.Perform()
            fresh = bd.Solid(fixed.Shape())
            if not fresh.is_valid or len(fresh.solids()) != 1:
                raise ValueError("trunk_top: face unification produced invalid geometry")
        if path[-1] in S11_SHIFTED_PARTS:
            fresh = _baked_shift(fresh)
            log.append(
                f"    chain shift {S11_SHIFT_X_MM:+.2f} mm in X baked into the part"
            )
        fresh.label = path[-1]
        if getattr(current_part, "color", None) is not None:
            fresh.color = _step_colour(current_part.color)
        bd.export_step(fresh, str(cached))
        print(
            f"  {key}: current {current_part.volume:.0f} -> {fresh.volume:.0f} mm3",
            flush=True,
        )

    if args.assemble or not wanted:
        # The refits above are measured in the model's own frame; the chain is
        # moved only now, so a part is never cut against a servo that already
        # travelled.
        for name in _shift_s11_chain(robot):
            print(f"S11 chain shifted {S11_SHIFT_X_MM:+.2f} mm in X: {name}", flush=True)
        replacements = {}
        for path in grouped:
            cached = args.work / f"{'__'.join(path)}.step"
            if not cached.exists():
                print(f"keep current geometry for {'__'.join(path)} (not refitted yet)", flush=True)
                continue
            replacements[path] = read_step(str(cached))

        s13_body = v._servo_parts(_find(robot, ("S13_HEAD_YAW_ZP10D",)))[0]
        s13_case_x = round(s13_body.bounding_box().max.X, 6)
        s14_servo = _find(robot, ("S14_HEAD_ROLL_ZP10D",))
        s14_axis, s14_origin = v._axis(s14_servo)
        s14_horn = v._servo_parts(s14_servo)[2]
        s14_hole_positions = [position for position, _ in v._horn_pattern_lines(s14_horn, s14_axis, s14_origin)]
        head_floor = _refit_head_motion(_find(robot, ("head_bot_sheet",)), s13_case_x, s14_hole_positions)

        def rebuild(node, path):
            children = _children(node)
            if not children:
                return _refit_head_motion(replacements.get(path, node), s13_case_x, s14_hole_positions, head_floor)
            kids = [rebuild(child, path + (child.label,)) for child in children]
            compound = bd.Compound(children=kids, label=node.label)
            compound.location = node.location
            return compound

        result = _encode_colours(rebuild(robot, ()))
        bd.export_step(result, str(args.out))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
