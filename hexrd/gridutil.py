# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# =============================================================================
import numpy as np
from numpy.linalg import det

from hexrd.constants import USE_NUMBA, sqrt_epsf
if USE_NUMBA:
    import numba


def cellIndices(edges, points_1d):
    """
    get indices in a 1-d regular grid.

    edges are just that:

    point:            x (2.5)
                      |
    edges:   |1    |2    |3    |4    |5
             -------------------------
    indices: |  0  |  1  |  2  |  3  |
             -------------------------

    above the deltas are + and the index for the point is 1

    point:                  x (2.5)
                            |
    edges:   |5    |4    |3    |2    |1
             -------------------------
    indices: |  0  |  1  |  2  |  3  |
             -------------------------

    here the deltas are - and the index for the point is 2

    * can handle grids with +/- deltas
    * be careful when using with a cyclical angular array!  edges and points
      must be mapped to the same branch cut, and
      abs(edges[0] - edges[-1]) = 2*pi
    """
    ztol = sqrt_epsf

    assert len(edges) >= 2, "must have at least 2 edges"

    points_1d = np.r_[points_1d].flatten()
    delta = float(edges[1] - edges[0])

    if delta > 0:
        on_last_rhs = points_1d >= edges[-1] - ztol
        points_1d[on_last_rhs] = points_1d[on_last_rhs] - ztol
        idx = np.floor((points_1d - edges[0]) / delta)
    elif delta < 0:
        on_last_rhs = points_1d <= edges[-1] + ztol
        points_1d[on_last_rhs] = points_1d[on_last_rhs] + ztol
        idx = np.ceil((points_1d - edges[0]) / delta) - 1
    else:
        raise RuntimeError("edges array gives delta of 0")

    # # mark points outside range with NaN
    # off_lo = idx < 0
    # off_hi = idx >= len(edges) - 1
    # if np.any(off_lo):
    #     idx[off_lo] = np.nan
    # if np.any(off_hi):
    #     idx[off_hi] = np.nan
    return np.array(idx, dtype=int)


def _fill_connectivity(out, m, n, p):
    i_con = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                extra = k*(n+1)*(m+1)
                out[i_con, 0] = i + j*(n + 1) + 1 + extra
                out[i_con, 1] = i + j*(n + 1) + extra
                out[i_con, 2] = i + j + n*(j+1) + 1 + extra
                out[i_con, 3] = i + j + n*(j+1) + 2 + extra
                i_con += 1


if USE_NUMBA:
    _fill_connectivity = numba.njit(nogil=True, cache=True)(_fill_connectivity)


def cellConnectivity(m, n, p=1, origin='ul'):
    """
    p x m x n (layers x rows x cols)

    origin can be upper left -- 'ul' <default> or lower left -- 'll'

    choice will affect handedness (cw or ccw)
    """
    nele = p*m*n
    con = np.empty((nele, 4), dtype=int)

    _fill_connectivity(con, m, n, p)

    if p > 1:
        nele = m*n*(p-1)
        tmp_con3 = con.reshape(p, m*n, 4)
        hex_con = []
        for layer in range(p - 1):
            hex_con.append(np.hstack([tmp_con3[layer], tmp_con3[layer + 1]]))
        con = np.vstack(hex_con)
        pass
    if origin.lower().strip() == 'll':
        con = con[:, ::-1]
    return con


if USE_NUMBA:
    @numba.njit(nogil=True, cache=True)  # relies on loop extraction
    def cellCentroids(crd, con):
        nele, conn_count = con.shape
        dim = crd.shape[1]
        out = np.empty((nele, dim))
        inv_conn = 1.0/conn_count
        for i in range(nele):
            for j in range(dim):
                acc = 0.0
                for k in range(conn_count):
                    acc += crd[con[i, k], j]
                out[i, j] = acc * inv_conn
        return out

    @numba.njit(nogil=True, cache=True)
    def compute_areas(xy_eval_vtx, conn):
        areas = np.empty(len(conn))
        for i in range(len(conn)):
            c0, c1, c2, c3 = conn[i]
            vtx0x, vtx0y = xy_eval_vtx[conn[i, 0]]
            vtx1x, vtx1y = xy_eval_vtx[conn[i, 1]]
            v0x, v0y = vtx1x-vtx0x, vtx1y-vtx0y
            acc = 0
            for j in range(2, 4):
                vtx_x, vtx_y = xy_eval_vtx[conn[i, j]]
                v1x = vtx_x - vtx0x
                v1y = vtx_y - vtx0y
                acc += v0x*v1y - v1x*v0y

            areas[i] = 0.5 * acc
        return areas
else:
    def cellCentroids(crd, con):
        """
        con.shape = (nele, 4)
        crd.shape = (ncrd, 2)

        con.shape = (nele, 8)
        crd.shape = (ncrd, 3)
        """
        nele = con.shape[0]
        dim = crd.shape[1]
        centroid_xy = np.zeros((nele, dim))
        for i in range(len(con)):
            el_crds = crd[con[i, :], :]  # (4, 2)
            centroid_xy[i, :] = (el_crds).mean(axis=0)
        return centroid_xy

    def compute_areas(xy_eval_vtx, conn):
        areas = np.zeros(len(conn))
        for i in range(len(conn)):
            polygon = [[xy_eval_vtx[conn[i, j], 0],
                        xy_eval_vtx[conn[i, j], 1]] for j in range(4)]
            areas[i] = computeArea(polygon)
        return areas


def computeArea(polygon):
    """
    must be ORDERED and CONVEX!
    """
    n_vertices = len(polygon)
    polygon = np.array(polygon)

    triv = np.array([[[0, i - 1], [0, i]] for i in range(2, n_vertices)])

    area = 0
    for i in range(len(triv)):
        tvp = np.diff(
            np.hstack([polygon[triv[i][0], :],
                       polygon[triv[i][1], :]]), axis=0).flatten()
        area += 0.5 * np.cross(tvp[:2], tvp[2:])
    return area


def make_tolerance_grid(bin_width, window_width, num_subdivisions,
                        adjust_window=False, one_sided=False):
    if bin_width > window_width:
        bin_width = window_width
    if adjust_window:
        window_width = np.ceil(window_width/bin_width)*bin_width
    if one_sided:
        ndiv = abs(int(window_width/bin_width))
        grid = (np.arange(0, 2*ndiv+1) - ndiv)*bin_width
        ndiv = 2*ndiv
    else:
        ndiv = int(num_subdivisions*np.ceil(window_width/float(bin_width)))
        grid = np.arange(0, ndiv+1)*window_width/float(ndiv) - 0.5*window_width
    return ndiv, grid


def computeIntersection(line1, line2):
    """
    compute intersection of two-dimensional line intersection

    this is an implementation of two lines:

    line1 = [ [x0, y0], [x1, y1] ]
    line1 = [ [x3, y3], [x4, y4] ]


     <http://en.wikipedia.org/wiki/Line-line_intersection>
    """
    intersection = np.zeros(2)

    l1 = np.array(line1)
    l2 = np.array(line2)

    det_l1 = det(l1)
    det_l2 = det(l2)

    det_l1_x = det(np.vstack([l1[:, 0], np.ones(2)]).T)
    det_l1_y = det(np.vstack([l1[:, 1], np.ones(2)]).T)

    det_l2_x = det(np.vstack([l2[:, 0], np.ones(2)]).T)
    det_l2_y = det(np.vstack([l2[:, 1], np.ones(2)]).T)

    denominator = det(
        np.vstack([[det_l1_x, det_l1_y], [det_l2_x, det_l2_y]])
    )

    if denominator == 0:
        intersection = []
    else:
        intersection[0] = det(
            np.vstack(
                [[det_l1, det_l1_x], [det_l2, det_l2_x]]
            )
        ) / denominator
        intersection[1] = det(
            np.vstack(
                [[det_l1, det_l1_y], [det_l2, det_l2_y]]
            )
        ) / denominator
    return intersection


def isinside(point, boundary, ccw=True):
    """
    Assumes CCW boundary ordering
    """
    pointPositionVector = np.hstack([point - boundary[0, :], 0.])
    boundaryVector = np.hstack([boundary[1, :] - boundary[0, :], 0.])

    crossVector = np.cross(pointPositionVector, boundaryVector)

    inside = False
    if crossVector[2] > 0:
        if ccw:
            inside = True
    elif crossVector[2] < 0:
        if not ccw:
            inside = True
    else:
        inside = True

    return inside


def sutherlandHodgman(subjectPolygon, clipPolygon):
    """
    """
    subjectPolygon = np.array(subjectPolygon)
    clipPolygon = np.array(clipPolygon)

    numClipEdges = len(clipPolygon)

    prev_clipVertex = clipPolygon[-1, :]

    # loop over clipping edges
    outputList = np.array(subjectPolygon)
    for iClip in range(numClipEdges):

        curr_clipVertex = clipPolygon[iClip, :]

        clipBoundary = np.vstack(
            [curr_clipVertex,
             prev_clipVertex]
        )

        inputList = np.array(outputList)
        if len(inputList) > 0:
            prev_subjectVertex = inputList[-1, :]

        outputList = []

        for iInput in range(len(inputList)):

            curr_subjectVertex = inputList[iInput, :]

            if isinside(curr_subjectVertex, clipBoundary):
                if not isinside(prev_subjectVertex, clipBoundary):
                    subjectLineSegment = np.vstack(
                        [curr_subjectVertex, prev_subjectVertex]
                    )
                    outputList.append(
                        computeIntersection(subjectLineSegment, clipBoundary)
                    )
                    pass
                outputList.append(curr_subjectVertex)
            elif isinside(prev_subjectVertex, clipBoundary):
                subjectLineSegment = np.vstack(
                    [curr_subjectVertex, prev_subjectVertex]
                )
                outputList.append(
                    computeIntersection(subjectLineSegment, clipBoundary)
                )
                pass
            prev_subjectVertex = curr_subjectVertex
            prev_clipVertex = curr_clipVertex
            pass
        pass
    return outputList
