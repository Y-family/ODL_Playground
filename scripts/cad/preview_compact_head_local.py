"""Local diagnostic view when the browser snapshot backend is unavailable."""

import os
import sys
os.environ['MPLCONFIGDIR'] = '/tmp/mpl-config'

import compact_head_local as compact
from cadgen import read_step
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def main():
    if '--antenna' in sys.argv:
        figure = plt.figure(figsize=(12, 6))
        plots = [figure.add_subplot(1, 2, position, projection='3d') for position in (1, 2)]
        for index, colour in ((22, '#d4a833'), (91, '#168dcc'), (10, '#666666')):
            part = read_step(compact.WORK / f'candidate_{index}.step')
            vertices, faces = part.tessellate(.15, .2)
            polygons = np.array([list(vertex) for vertex in vertices])[np.array(faces)]
            for plot in plots:
                plot.add_collection3d(Poly3DCollection(polygons, facecolor=colour, edgecolor='#444444', linewidth=.05, alpha=.8))
        for plot, azimuth in zip(plots, (-60, 140)):
            plot.set(xlim=(-70, 0), ylim=(10, 80), zlim=(175, 240))
            plot.set_box_aspect((70, 70, 65))
            plot.view_init(elev=25, azim=azimuth)
            plot.set_proj_type('ortho')
        figure.savefig(compact.WORK / 'antenna.png', dpi=150)
        return
    if '--rear' in sys.argv:
        figure = plt.figure(figsize=(12, 6))
        for position, prefix in enumerate(('leaf', 'candidate'), 1):
            plot = figure.add_subplot(1, 2, position, projection='3d')
            part = read_step(compact.WORK / f'{prefix}_82.step')
            vertices, faces = part.tessellate(.25, .3)
            vertices = np.array([list(vertex) for vertex in vertices])
            plot.add_collection3d(Poly3DCollection(vertices[np.array(faces)], facecolor='#bcc6ce', edgecolor='#526574', linewidth=.06))
            plot.set(xlim=(-100, 110), ylim=(-100, 100), zlim=(170, 240), title=prefix)
            plot.set_box_aspect((210, 200, 70))
            plot.view_init(elev=15, azim=180)
            plot.set_proj_type('ortho')
        figure.savefig(compact.WORK / 'rear.png', dpi=150)
        return
    figure = plt.figure(figsize=(15, 6))
    plots = [figure.add_subplot(1, 3, position + 1, projection='3d') for position in range(3)]
    for index, colour in ((82, '#bbc1c5'), (53, '#edb13b'), (12, '#5a656b'),
                          (91, '#2791d0'), (102, '#2791d0'), (85, '#d85757')):
        part = read_step(compact.WORK / f'candidate_{index}.step')
        vertices, faces = part.tessellate(.25, .3)
        vertices = np.array([list(vertex) for vertex in vertices])
        polygons = vertices[np.array(faces)]
        for plot in plots:
            plot.add_collection3d(Poly3DCollection(polygons, facecolor=colour,
                                                   alpha=.06 if index == 82 else .8,
                                                   linewidth=0))
        del part, vertices, polygons
    for plot, elevation, azimuth, title in zip(plots, (90, 0, 25), (-90, -90, -55),
                                             ('Top (XY)', 'Side (XZ)', 'Isometric')):
        plot.set(xlim=(-90, 115), ylim=(-100, 100), zlim=(140, 240),
                 xlabel='X / mm', ylabel='Y / mm', zlabel='Z / mm', title=title)
        plot.set_box_aspect((205, 200, 100))
        plot.view_init(elev=elevation, azim=azimuth)
        plot.set_proj_type('ortho')
    figure.suptitle('UNVALIDATED compact layout: shell / yaw ring / antenna servos / speaker stand')
    output = compact.TARGET.parent / 'HEAD_COMPACT_LAYOUT.png'
    figure.savefig(output, dpi=130)
    print(output, flush=True)


if __name__ == '__main__':
    main()
