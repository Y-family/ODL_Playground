import gc
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/speaker-rework/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import numpy as np
import trimesh
from cadgen import read_step, build123d as bd
from repair_speaker_layout_local import seat_plane, FRONT_CENTER, FRONT_NORMAL
import refit_upper_servo_interfaces_local as refit
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools

ROOT=Path(__file__).resolve().parents[2]
TARGET=Path('/tmp/speaker-rework/review.step')
OUT=ROOT/'cad/local_fix/SPEAKER_LAYOUT_REVIEW.png'
MESH_BOUNDS={}
MESH_TRIANGLES={}

def walk(n):
    ch=refit._children(n)
    if not ch: yield n
    else:
        for c in ch: yield from walk(c)

def tri(shape, chord=0.7):
    if shape.label in MESH_TRIANGLES:
        return MESH_TRIANGLES[shape.label]
    # STEP exports do not carry a display triangulation for newly fused faces.
    # Mesh the individual leaf before asking build123d for its triangles.
    BRepTools.Clean_s(shape.wrapped)
    # A few narrow support faces need a finer deflection than the display
    # chord; 0.3 mm also makes the result deterministic across OCC builds.
    BRepMesh_IncrementalMesh(shape.wrapped, min(chord, 0.3), False, 0.3, False)
    v,f=shape.tessellate(min(chord, 0.3),0.3)
    p=np.array([[q.X,q.Y,q.Z] for q in v])
    MESH_BOUNDS[shape.label]=[p.min(axis=0).tolist(),p.max(axis=0).tolist()]
    triangles=p[np.array(f)]
    MESH_TRIANGLES[shape.label]=triangles
    return triangles

def draw(ax, layers, axes, title):
    for shape,color,alpha in layers:
        triangles=tri(shape)
        ax.add_collection(PolyCollection(triangles[:,:,axes],facecolors=color,
                                         edgecolors='none',alpha=alpha))
    ax.autoscale_view();ax.set_aspect('equal');ax.axis('off');ax.set_title(title)

def main():
    robot=read_step(TARGET); by={}
    for x in walk(robot): by.setdefault(x.label,[]).append(x)
    def one(label): return by[label][0]
    shell=[(one('head'),'#c8c8c8',.42),(one('head_bot_sheet'),'#555555',.3)]
    fixed=[(one('head_roll_mount'),'#666666',.7),(one('head_yaw_to_roll'),'#888888',.24)]
    stand=[(one('speaker_stand'),'#e88b17',1.0)]
    speakers=[(one(x),'#1d64b8',1.0) for x in ('speaker_interface','speaker_bottom','speaker_top')]
    fig=plt.figure(figsize=(18,12))
    ax=fig.add_subplot(2,3,1);draw(ax,shell+fixed+stand+speakers,[0,2],'Side cutaway; eye direction +X (right)')
    ax=fig.add_subplot(2,3,2);draw(ax,shell+fixed+stand+speakers,[1,2],'Front cutaway; upper speaker hanger')
    ax=fig.add_subplot(2,3,3);draw(ax,shell+fixed+stand+speakers,[0,1],'Top cutaway; speaker-to-shell fit')
    plane=seat_plane()
    patch=refit._shape(one('head').intersect(plane*refit._world_box(-24,24,-24,24,.01,25)))
    patch.label='grille_wall_region'
    triangles=tri(patch,.025)
    origin=np.array(FRONT_CENTER);normal=np.array(FRONT_NORMAL)
    u=np.array([1.,0.,0.]);v=np.cross(normal,u)
    coords=triangles-origin
    projected=np.stack((coords@u,coords@v),axis=-1)
    ax=fig.add_subplot(2,3,4)
    ax.add_collection(PolyCollection(projected,facecolors='#c8c8c8',edgecolors='none'))
    edge=one('speaker_top').faces()[0].outer_wire().edges()[0]
    rim=np.array([list(edge.position_at(i/200)) for i in range(201)])-origin
    ax.plot(rim@u,rim@v,color='#1d64b8',lw=2,label='Actual speaker front rim')
    ax.set_xlim(-24,24);ax.set_ylim(-24,24);ax.set_aspect('equal')
    ax.set_title('Exterior grille along acoustic axis; 91 circular holes')
    ax.set_xlabel('Across speaker (mm)');ax.set_ylabel('Up speaker (mm)');ax.legend(loc='lower right',fontsize=8)
    segments=[]
    for shape,color in [(patch,'#777777')]+[(one(x),'#1d64b8') for x in ('speaker_top','speaker_bottom','speaker_interface')]:
        triangles=tri(shape)
        mesh=trimesh.Trimesh(vertices=triangles.reshape(-1,3),faces=np.arange(triangles.size//3).reshape(-1,3),process=False)
        lines=trimesh.intersections.mesh_plane(mesh,plane_normal=v,plane_origin=origin)-origin
        lines=np.stack((lines@u,lines@normal),axis=-1)
        segments.append((lines,color))
    ax=fig.add_subplot(2,3,(5,6))
    for lines,color in segments: ax.add_collection(LineCollection(lines,colors=color,linewidths=1.2))
    ax.set_xlim(-23,23);ax.set_ylim(-19,6);ax.set_aspect('equal')
    ax.set_title('Actual centre section; speaker front at 0, head seating face at +0.25 mm')
    ax.set_xlabel('Across speaker (mm)');ax.set_ylabel('Distance along acoustic normal (mm)')
    inset=ax.inset_axes([.78,.22,.2,.6])
    for lines,color in segments: inset.add_collection(LineCollection(lines,colors=color,linewidths=1.5))
    inset.set_xlim(11.05,11.75);inset.set_ylim(-.3,3.8)
    inset.annotate('',xy=(11.4,.25),xytext=(11.4,0),arrowprops={'arrowstyle':'<->','color':'#e88b17'})
    inset.set_title('0.25 mm',fontsize=10);inset.set_xticks([])
    plt.tight_layout();OUT.parent.mkdir(exist_ok=True);fig.savefig(OUT,dpi=160,bbox_inches='tight');plt.close(fig)
    metadata=dict(source_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                  snapshot_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
                  mesh_bounds_mm=MESH_BOUNDS)
    (TARGET.parent/'render-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(OUT)

if __name__=='__main__':main()
