#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:10:16 2022

@author: jbernier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:18:20 2022

@author: jbernier
"""
import os

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib.text import Annotation
from matplotlib.patches import FancyArrowPatch

import matplotlib.pyplot as plt

import yaml

from hexrd import instrument

# %%
hexrd_dir = '/home/jbernier/Documents/GitHub/hexrd'
icfg = yaml.safe_load(
    open(os.path.join(hexrd_dir,
                      'hexrd/resources/pxrdip_reference_config.yml'),
         'r')
    )

instr = instrument.HEDMInstrument(icfg)
instr.source_distance = 22.5

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# %%
class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


setattr(Axes3D, 'annotate3D', _annotate3D)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

# %%
fig = plt.figure()
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
ax = fig.add_subplot(111, projection='3d')

rho = instr.source_distance


def draw_cs(ax, origin=np.zeros(3), scale=10, mscale=10, rmat=None):
    dirs = np.hstack([np.tile(origin.flatten(), (3, 1)),
                      [i for i in scale*np.eye(3)]])
    if rmat is not None:
        for i in range(3):
            dirs[i, -3:] = np.dot(dirs[i, -3:], rmat.T)

    colors = ['r', 'g', 'b']
    for i, qargs in enumerate(dirs):
        # ax.quiver3D(*qargs, color=colors[i], linewidth=2)
        ax.arrow3D(*qargs,
                   mutation_scale=mscale,
                   ec=colors[i],
                   fc=colors[i])



for i, (det_name, panel) in enumerate(instr.detectors.items()):
    verts = np.dot(
        np.hstack(
            [np.vstack([panel.corner_ll,
                        panel.corner_lr,
                        panel.corner_ur,
                        panel.corner_ul,
                        panel.corner_ll]),
             np.zeros((5, 1))]
        ),
        panel.rmat.T) + panel.tvec
    ax.add_collection3d(
        Poly3DCollection(
            [[tuple(i) for i in verts]],
            facecolor=colors[i], edgecolor='k', alpha=0.5
        ), zdir='y'
    )
    #ax.quiver3D(0, 0, 0, *panel.tvec, color='m')
    ax.arrow3D(0, 0, 0, *panel.tvec,
               mutation_scale=10,
               ec ='m',
               fc='c')
    ax.annotate3D(det_name, panel.tvec,
                  xytext=(3, 3), textcoords='offset points', color=colors[i])
    draw_cs(ax, panel.tvec, rmat=panel.rmat, mscale=5, scale=15)

draw_cs(ax, origin=np.zeros(3), mscale=20, scale=50)


# XRS plotting
xrs_vec = -rho*instr.beam_vector
# ax.quiver3D(0, 0, 0, *xrs_vec, color='k')
ax.arrow3D(0, 0, 0, *xrs_vec,
           mutation_scale=10,
           ec ='m',
           fc='k')
ax.scatter3D(*xrs_vec, marker='o', color='k', s=48)

# scene
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)
ax.set_xlabel(r'$\mathbf{X}_l$', fontsize=24, color='r')
ax.set_ylabel(r'$\mathbf{Y}_l$', fontsize=24, color='g')
ax.set_zlabel(r'$\mathbf{Z}_l$', fontsize=24, color='b')

fig.tight_layout()

plt.draw()
plt.show()
