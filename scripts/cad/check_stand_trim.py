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
 r=read_step(TARGET); leaves=list(walk(r)); by={}
 for x in leaves: by.setdefault(x.label,[]).append(x)
 stand=by['speaker_stand'][0]
 for x0 in [35,45,50,55,60,65,70,75,80,85]:
  q=refit._shape(stand.intersect(refit._world_box(x0,200,-100,50,223.5,240)))
  print(json.dumps(dict(x0=x0,vol=q.volume,solids=len(q.solids()),head=q.distance_to(by['head'][0]),mount=q.distance_to(by['head_roll_mount'][0]))))
if __name__=='__main__':main()
