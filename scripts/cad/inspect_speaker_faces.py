import json
from pathlib import Path
from cadgen import read_step
import refit_upper_servo_interfaces_local as refit
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'cad/local_fix/open_duck_mini_upper_refit.step'

def walk(n):
    ch=refit._children(n)
    if not ch: yield n
    else:
        for c in ch: yield from walk(c)

def main():
    r=read_step(TARGET); s=next(x for x in walk(r) if x.label=='speaker_stand')
    rows=[]
    for i,f in enumerate(s.faces()):
        ad=BRepAdaptor_Surface(f.wrapped,True)
        if ad.GetType()!=GeomAbs_Cylinder: continue
        cy=ad.Cylinder();loc=cy.Location().Coord();axis=cy.Axis().Direction();
        rows.append(dict(i=i,radius=cy.Radius(),loc=list(loc),axis=list(axis.Coord()),bounds=list(f.bounding_box().min)+list(f.bounding_box().max),area=f.area))
    rows.sort(key=lambda q:(q['radius'],q['loc'][2],q['loc'][0],q['loc'][1]))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
