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
def overlap(a,b):
    return sum(x.volume for x in refit._items(a.intersect(b)))
def main():
  r=read_step(TARGET); leaves=list(walk(r)); by={}
  for x in leaves: by.setdefault(x.label,[]).append(x)
  speakers=by['speaker_interface']+by['speaker_bottom']+by['speaker_top']
  fixed=by['head']+by['head_pitch_to_yaw']+by['head_yaw_to_roll']+by['head_roll_mount']+by['head_bot_sheet']
  for dx,dy in [(-15,5),(0,5),(15,5),(-10,5),(-20,5),(-15,10)]:
    rows=[]
    for s in speakers:
      m=s.moved(refit.bd.Pos(float(dx),float(dy),0))
      gaps=[(m.distance_to(f),overlap(m,f),f.label) for f in fixed]
      g=min(gaps,key=lambda q:q[0]); rows.append(dict(label=s.label,min_gap=g[0],overlap=g[1],against=g[2]))
    print(json.dumps(dict(dx=dx,dy=dy,rows=rows),default=float))
if __name__=='__main__':main()
