import json
from pathlib import Path
from cadgen import read_step
import refit_upper_servo_interfaces_local as refit
ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'cad/local_fix/open_duck_mini_upper_refit.step'
def walk(n):
    ch=refit._children(n)
    if not ch: yield n
    else:
        for c in ch: yield from walk(c)
def main():
 r=read_step(TARGET);s=next(x for x in walk(r) if x.label=='speaker_stand')
 for x0 in [-100,0,20,35,45,50,55,60,65,70,75,80,85]:
  b=refit._world_box(x0,200,-100,50,223.5,240)
  q=refit._items(s.intersect(b)); q=[x for x in q if x.volume>1e-6]
  if q:
   c=refit._shape(s.intersect(b));bb=c.bounding_box(); print(json.dumps(dict(x0=x0,vol=c.volume,solids=len(c.solids()),bounds=list(bb.min)+list(bb.max))))
if __name__=='__main__':main()
