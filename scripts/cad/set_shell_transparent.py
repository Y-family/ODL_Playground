"""Make the white printed shell of the local review STEP transparent.

The white PLA of the Open Duck Mini -- body_front, body_back, body_middle_top,
body_middle_bottom, head, left_cache, right_cache -- is COLOUR_RGB 0.917647
sRGB, the same value the MuJoCo materials carry. Transparency is a presentation
style, not geometry, so the edit is textual and surgical: every style whose
colour is that white and whose product is one of those parts gains the two
entities OCCT's own writer emits for a colour carrying alpha --

    #n = SURFACE_STYLE_RENDERING_WITH_PROPERTIES(.NORMAL_SHADING.,<colour>,(#m));
    #m = SURFACE_STYLE_TRANSPARENT(1 - alpha);

-- and nothing else in the document moves. The STEP's own reader (cadgen, any
OCC tool) comes back with alpha on exactly those seven parts. Re-running is a
no-op. Not touched: the 27.5 mm white joint-frame markers (Part 4), which are
export scaffolding rather than shell.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT/'cad/local_fix/open_duck_mini_covers_forward_review.step'
SHELL = {'body_front','body_back','body_middle_top','body_middle_bottom',
         'head','left_cache','right_cache'}
WHITE = '0.917647078693'
STYLE_CLASSES = ('STYLED_ITEM','OVER_RIDING_STYLED_ITEM')


def parse(text):
    body = text.split('DATA;', 1)[1].split('ENDSEC;', 1)[0]
    return {int(m.group(1)): ' '.join(m.group(2).split())
            for m in re.finditer(r'#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|\Z)', body, re.S)}


def refs(ent, e):
    return [int(x) for x in re.findall(r'#(\d+)', ent.get(e, ''))]


def typ(ent, e):
    m = re.match(r'[A-Za-z_0-9]+', ent.get(e, ''))
    return m.group(0) if m else ''


def arg1(ent, e):
    m = re.match(r"[A-Za-z_0-9]+\(\s*'([^']*)'", ent.get(e, ''))
    return m.group(1) if m else None


def product_of(ent, pds):
    """The PRODUCT name behind a PRODUCT_DEFINITION_SHAPE, a few hops down."""
    frontier = [pds]
    for _ in range(4):
        nxt = []
        for e in frontier:
            for c in refs(ent, e):
                if typ(ent, c) == 'PRODUCT': return arg1(ent, c)
                if typ(ent, c) in ('PRODUCT_DEFINITION','PRODUCT_DEFINITION_FORMATION',
                                   'PRODUCT_DEFINITION_SHAPE','PRODUCT_DEFINITION_CONTEXT'):
                    nxt.append(c)
        frontier = nxt
    return None


def solid_names(ent):
    """Solid entity id -> the product it is the shape of."""
    rep_product = {}
    for e in ent:
        if typ(ent, e) == 'SHAPE_DEFINITION_REPRESENTATION':
            r = refs(ent, e)
            if len(r) >= 2: rep_product[r[1]] = product_of(ent, r[0])
    names = {}
    for e in ent:
        t = typ(ent, e)
        if t.endswith('SHAPE_REPRESENTATION') and 'CONTEXT' not in t and rep_product.get(e):
            for item in refs(ent, e): names.setdefault(item, rep_product[e])
    return names


def colour_of(ent, e, depth=0, seen=None):
    """The COLOUR_RGB id at the bottom of a presentation style chain."""
    seen = seen or set()
    if e in seen or depth > 8: return None
    seen = seen | {e}
    if typ(ent, e) == 'COLOUR_RGB': return e
    if typ(ent, e) in ('FILL_AREA_STYLE_COLOUR','FILL_AREA_STYLE','SURFACE_STYLE_FILL_AREA',
                       'SURFACE_STYLE_USAGE','PRESENTATION_STYLE_ASSIGNMENT','SURFACE_SIDE_STYLE'):
        for r in refs(ent, e):
            c = colour_of(ent, r, depth + 1, seen)
            if c: return c
    return None


def side_style_of(ent, styled):
    for style in refs(ent, refs(ent, styled)[0]):
        for side in refs(ent, style):
            if typ(ent, side) == 'SURFACE_SIDE_STYLE': return side
    return None


def shell_styles(ent):
    """SURFACE_SIDE_STYLE id -> (product, COLOUR_RGB id), one per white shell style."""
    names = solid_names(ent)
    found = {}
    for e in ent:
        if typ(ent, e) not in STYLE_CLASSES: continue
        product = names.get(refs(ent, e)[1])
        if product not in SHELL: continue
        side = side_style_of(ent, e)
        colour = colour_of(ent, side) if side else None
        if colour and WHITE in ent[colour]: found[side] = (product, colour)
    return found


def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else .35
    text = TARGET.read_text()
    ent = parse(text)
    targets = {s: v for s, v in shell_styles(ent).items()          # a side style
               if not any(typ(ent, c).startswith('SURFACE_STYLE_RENDERING')   # already
                          for c in refs(ent, s))}                             # done -> no-op
    if not targets: return print('already transparent:', TARGET)
    next_id, inserts = max(ent) + 1, {}
    for side, (product, colour) in sorted(targets.items()):
        render, transparent = next_id, next_id + 1
        next_id += 2
        inserts[side] = (render,
            f'#{render} = SURFACE_STYLE_RENDERING_WITH_PROPERTIES('
            f'.NORMAL_SHADING.,#{colour},(#{transparent}));\n'
            f'#{transparent} = SURFACE_STYLE_TRANSPARENT({1. - alpha:.10f});')
        print(f'{product:20s} side_style=#{side} colour=#{colour} -> #{render}')
    text = re.sub(r'#(\d+)\s*=\s*SURFACE_SIDE_STYLE\((.*?)\);',
                  lambda m: (f'#{m.group(1)} = SURFACE_SIDE_STYLE({m.group(2)[:-1]},'
                             f'#{inserts[int(m.group(1))][0]}));\n{inserts[int(m.group(1))][1]}')
                  if int(m.group(1)) in inserts else m.group(0), text, flags=re.S)
    TARGET.write_text(text)
    print(f'{len(targets)} style(s) transparent (alpha {alpha}) in {TARGET}')


if __name__ == '__main__': main()
