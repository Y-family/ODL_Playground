"""Measured occupancy section of the current right yellow end plate."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cadgen import read_step,build123d as bd
import build_zp10d_left_leg_complete as m

root=read_step('cad/local_fix/open_duck_mini_trunk_inward_candidate.step')
branch=m._one_child(root,'right_roll_to_pitch_assembly <1>')
yellow=m._one_child(branch,'right_roll_to_pitch')
baseline=read_step('cad/local_fix/open_duck_mini_both_legs_trunk_s01_s10.step')
original=m._one_child(m._one_child(baseline,'right_roll_to_pitch_assembly <1>'),'right_roll_to_pitch')
servo=next(p for p in branch.children if p.label.startswith('S03_'))
z=-2.0
xs=np.arange(-27,14.01,.25)
ys=np.arange(-112,-100.99,.125)
data=np.zeros((len(xs),len(ys)),dtype=int)
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN,TopAbs_ON
yc=BRepClass3d_SolidClassifier(yellow.wrapped)
oc=BRepClass3d_SolidClassifier(original.wrapped)
sc=[BRepClass3d_SolidClassifier(s.wrapped) for p in servo.children for s in p.solids()]
def inside(c,p):
    c.Perform(p,1e-7)
    return c.State() in (TopAbs_IN,TopAbs_ON)
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        point=gp_Pnt(float(x),float(y),z)
        if inside(yc,point):data[i,j]=1 if inside(oc,point) else 2
        if any(inside(c,point) for c in sc):data[i,j]=3
fig,ax=plt.subplots(figsize=(12,7))
ax.imshow(data,origin='lower',extent=[ys[0]-.0625,ys[-1]+.0625,xs[0]-.125,xs[-1]+.125],
          cmap=ListedColormap(['#f7f8fa','#efb537','#e05a27','#454c57']),vmin=0,vmax=3,aspect='auto',interpolation='nearest')
for y in [-110.425,-107.425,-104.425]:ax.axvline(y,color='#303844',ls='--',lw=.8)
ax.set_title('RIGHT YELLOW END PLATE / S03 — actual STEP section at Z = -2 mm',pad=20)
ax.set_xlabel('World Y (mm)   → toward robot centre (Y = 0)')
ax.set_ylabel('World X (mm)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#efb537',label='Original yellow material'),Patch(color='#e05a27',label='Added inward material'),Patch(color='#454c57',label='S03 servo')],loc='upper left',bbox_to_anchor=(1.01,1))
ax.annotate('',xy=(-107.425,-25),xytext=(-110.425,-25),arrowprops={'arrowstyle':'<->','lw':1.5})
ax.text(-108.925,-24,'3 mm original',ha='center',fontsize=10)
ax.annotate('',xy=(-104.425,12),xytext=(-110.425,12),arrowprops={'arrowstyle':'<->','lw':1.5})
ax.text(-107.425,10.4,'6 mm total where space permits',ha='center',fontsize=10)
fig.text(.09,.015,'Section sampled from current STEP (0.25 × 0.125 mm grid). Orange-red = actual added material; grey blocks inward growth.',fontsize=9)
fig.subplots_adjust(right=.76,bottom=.13,top=.88)
out=Path('cad/local_fix/yellow_endplate_section.png')
fig.savefig(out,dpi=180)
print(out,flush=True)
