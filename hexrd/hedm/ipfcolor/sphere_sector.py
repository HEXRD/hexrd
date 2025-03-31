# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Saransh Singh <saransh1@llnl.gov> and others.
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
from hexrd.core import constants
import numpy as np
from hexrd.hedm.ipfcolor import colorspace

eps = constants.sqrt_epsf

'''
In this section we will list the vertices of the spherical triangles
for each of the point group. this will be in the form of a dictionary
conforming to the symbols in unitcell.py

the entries are organized as follows:

key : point group symmetry

entry 0 in list: number of SST triangles needed to specify the 
fundamental region. this could have values 0, 1 or 2. If it is 0,
then the whole sphere is valid (either upper or ower or both depending
on symmetry). If this is 1, then only 1 triangle is needed. If two, then
two triangles are needed i.e. 4 coordinates

entry 1 in list: specify coordinates of the triangle(s). this will be 
empty array if size is 0 in previous entry. If size in previous entry is 1,
then this will 3x3. Finally if size in previous entry is 2, this will be 3x4.

entry 2 in list is the connectivity of the triangles. This is only useful for
the cases of T, Th, O and symmetry groups.

entry 3 in list: if both upper anf lower hemispheres are to be considered. For
the case of x-rays only upper hemisphere is considered because of Friedel's law,
which imposes and artificial inversion symmetry such that hkl and -hkl can't be
distinguished. However, for the formulation is general and can handle all symmetry
groups. This will say 'upper' or 'both' depending on what hemisphere is considered.

there are no triangles for the triclininc cases and needs to be handles differently

'''
pg2vertex = {
    'c1': [3, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [-0.5, np.sqrt(3.)/2., 0.],
                        [-0.5, -np.sqrt(3.)/2., 0.]]).T,
          np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]]).T, 'both'],

    # supergroup 00 in our convention
    'ci': [3, np.array([[0., 0., 1.],
                        [1.0, 0., 0.],
                        [-0.5, np.sqrt(3.)/2., 0.],
                        [-0.5, -np.sqrt(3.)/2., 0.]]).T,
          np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]]).T, 'upper'],

    'c2': [2, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [-1., 0., 0.]]).T,
           np.array([[0, 1, 2], [0, 2, 3]]).T,
           'both'],

    # supergroup 1 in our convention
    'cs': [3, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [-0.5, np.sqrt(3.)/2., 0.],
                        [-0.5, -np.sqrt(3.)/2., 0.]]).T,
          np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]]).T,
           'upper'],

    'c2h': [2, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [0., 1., 0.],
                         [-1., 0., 0.]]).T,
            np.array([[0, 1, 2], [0, 2, 3]]).T,
            'upper'],

    'd2': [2, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [-1., 0., 0.]]).T,
           np.array([[0, 1, 2], [0, 2, 3]]).T,
           'upper'],

    # supergroup 2 in our convention
    'c2v': [2, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., -1.]]).T,
            np.array([[0, 1, 2], [3, 1, 2]]).T,
            'both'],

    # supergroup 3 in our convention
    'd2h': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [0., 1., 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'upper'],

    'c4': [1, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.]]).T,
           np.atleast_2d(np.array([0, 1, 2])).T,
           'both'],

    # supergroup 01 in our convention
    's4': [2, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [-1., 0., 0.]]).T,
           np.array([[0, 1, 2], [0, 2, 3]]).T,
           'upper'],

    'c4h': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [0., 1., 0.]]),
            np.atleast_2d(np.array([0, 1, 2])).T,
            'upper'],

    'd4': [1, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.0]]),
           np.atleast_2d(np.array([0, 1, 2])).T,
           'upper'],

    # supergroup 4 in our convention
    'c4v': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [1./np.sqrt(2.), 1./np.sqrt(2.), 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'both'],

    'd2d': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [0., 1., 0.]]),
            np.atleast_2d(np.array([0, 1, 2])).T,
            'upper'],

    # supergroup 5 in our convention
    'd4h': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [1./np.sqrt(2.), 1./np.sqrt(2.), 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'upper'],

    'c3': [1, np.array([[0., 0., 1.],
                        [np.sqrt(3.)/2., -0.5, 0.],
                        [0., 1., 0.]]).T,
           np.atleast_2d(np.array([0, 1, 2])).T,
           'both'],

    # supergroup 02 in our convention
    's6': [1, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [-0.5, np.sqrt(3.)/2., 0.]]).T,
           np.atleast_2d(np.array([0, 1, 2])).T,
           'upper'],

    'd3': [1, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [-0.5, np.sqrt(3.)/2., 0.]]).T,
           np.atleast_2d(np.array([0, 1, 2])).T,
           'upper'],

    # supergroup 6 in our convention
    'c3v': [1, np.array([[0., 0., 1.],
                         [np.sqrt(3.)/2., -0.5, 0.],
                         [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'both'],

    'd3d': [1, np.array([[0., 0., 1.],
                         [np.sqrt(3.)/2., -0.5, 0.],
                         [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'upper'],

    'c6': [1, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [0.5, np.sqrt(3.)/2., 0.]]).T,
           np.atleast_2d(np.array([0, 1, 2])).T,
           'both'],

    'c3h': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [-0.5, np.sqrt(3.)/2., 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'upper'],

    'c6h': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [0.5, np.sqrt(3.)/2., 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'upper'],

    'd6': [1, np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [0.5, np.sqrt(3.)/2., 0.]]).T,
           np.atleast_2d(np.array([0, 1, 2])).T,
           'both'],

    # supergroup 7 in our convention
    'c6v': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'both'],

    # supergroup 8 in our convention
    'd3h': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'upper'],

    # supergroup 9 in our convention
    'd6h': [1, np.array([[0., 0., 1.],
                         [1., 0., 0.],
                         [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.atleast_2d(np.array([0, 1, 2])).T,
            'upper'],

    # Special case with 2 triangles
    't': [2, np.array([[0., 0., 1.],
                       [1./np.sqrt(3.), -1./np.sqrt(3.), 1./np.sqrt(3.)],
                       [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)],
                       [1., 0., 0.]]).T,
          np.array([[0, 1, 2], [1, 3, 2]]).T,
          'upper'],

    # Special case with two triangles
    'th': [2, np.array([[0., 0., 1.],
                        [1./np.sqrt(2.), 0., 1./np.sqrt(2.)],
                        [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)],
                        [0., 1./np.sqrt(2.), 1./np.sqrt(2.)]]).T,
           np.array([[0, 1, 2], [0, 2, 3]]).T,
           'upper'],

    # Special case with two triangles, same as Th
    'o': [2, np.array([[0., 0., 1.],
                       [1./np.sqrt(2.), 0., 1./np.sqrt(2.)],
                       [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)],
                       [0., 1./np.sqrt(2.), 1./np.sqrt(2.)]]).T,
          np.array([[0, 1, 2], [0, 2, 3]]).T,
          'upper'],

    # supergroup 10 in our convention
    'td': [1, np.array([[0., 0., 1.],
                        [1./np.sqrt(3.), -1./np.sqrt(3.), 1./np.sqrt(3.)],
                        [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)]]).T,
           np.atleast_2d(np.array([0, 1, 2])).T,
           'upper'],

    # supergroup 11 in our convention
    'oh': [1, np.array([[0., 0., 1.],
                        [1./np.sqrt(2.), 0., 1./np.sqrt(2.)],
                        [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)]]).T,
           np.atleast_2d(np.array([0, 1, 2])).T,
           'upper']
}


class sector:
    '''
    @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    @DATE    10/28/2020 SS 1.0 original
    @DETAIL  this class is used to store spherical patch for a given point group.
             the class also has methods to compute the color of a direction by
             computing the hue, saturation and lightness values in [0,1]. these
             values can be converted to rgb for display with the well known 
             conversion formula.


    All the methodology and equations have been taken from the paper:
    Orientations – perfectly colored, G. Nolze and R. Hielscher, J. Appl. Cryst. (2016). 49, 1786–1802

    '''

    def __init__(self, pgsym, lauesym, supergroupsym, supergrouplauesym):
        '''
        AUTHOR: Saransh Singh, Lawrence Livermore national Lab, saransh1@llnl.gov
        DATE:   11/11/2020 SS 1.0 original
                11/12/2020 SS 1.1 added lauesym as additional input parameter
                11/23/2020 SS 1.2 added supergroupsym as additional parameter

        @detail: this routine initializes the data needed for reducing a 
        direction to the stereographic fundamental zone (standard
        stereographic triangle) for the pointgroup/lauegroup symmetry
        of the crystal.
        '''
        self.vertices = {}
        self.ntriangle = {}
        self.barycenter = {}
        self.connectivity = {}
        self.hemisphere = {}

        data = pg2vertex[pgsym]
        self.ntriangle['pg'] = data[0]
        self.vertices['pg'] = data[1]
        self.connectivity['pg'] = data[2]
        self.hemisphere['pg'] = data[3]

        data = pg2vertex[lauesym]
        self.ntriangle['laue'] = data[0]
        self.vertices['laue'] = data[1]
        self.connectivity['laue'] = data[2]
        self.hemisphere['laue'] = data[3]

        data = pg2vertex[supergroupsym]
        self.ntriangle['super'] = data[0]
        self.vertices['super'] = data[1]
        self.connectivity['super'] = data[2]
        self.hemisphere['super'] = data[3]

        data = pg2vertex[supergrouplauesym]
        self.ntriangle['superlaue'] = data[0]
        self.vertices['superlaue'] = data[1]
        self.connectivity['superlaue'] = data[2]
        self.hemisphere['superlaue'] = data[3]

        if(self.ntriangle['pg'] != 0):
            # compute the barycenter or the centroid of point group
            b = np.mean(self.vertices['pg'], axis=1)
            b = b/np.linalg.norm(b)
            self.barycenter['pg'] = b
        else:
            self.barycenter['pg'] = np.array([0., 0., 1.])

        if(self.ntriangle['laue'] != 0):
            # compute the barycenter or the centroid of the laue group triangle
            b = np.mean(self.vertices['laue'], axis=1)
            b = b/np.linalg.norm(b)
            self.barycenter['laue'] = b
        else:
            self.barycenter['laue'] = np.array([0., 0., 1.])

        if(self.ntriangle['super'] != 0):
            # compute the barycenter or the centroid of the supergroup group triangle
            b = np.mean(self.vertices['super'], axis=1)
            b = b/np.linalg.norm(b)
            self.barycenter['super'] = b
        else:
            self.barycenter['super'] = np.array([0., 0., 1.])

        if(self.ntriangle['superlaue'] != 0):
            # compute the barycenter or the centroid of the supergroup group triangle
            b = np.mean(self.vertices['superlaue'], axis=1)
            b = b/np.linalg.norm(b)
            self.barycenter['superlaue'] = b
        else:
            self.barycenter['superlaue'] = np.array([0., 0., 1.])

    def check_norm(self, dir3):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/29/2020 SS 1.0 original
        @PARAM   dir3 direction in fundamental sector. size is nx3
        @DETAIL  this function is used to make sure the directions are all unit norm

        '''
        n = np.linalg.norm(dir3, axis=1)
        mask = n > eps
        n = n[mask]
        dir3[mask, :] = dir3[mask, :]/np.tile(n, [3, 1]).T

    def check_hemisphere(self):

        zcoord = np.array([self.vx[2], self.vy[2], self.vz[2]])
        if(np.logical_or(np.all(zcoord >= 0.),  np.all(zcoord <= 0.))):
            pass
        else:
            raise RuntimeError("sphere_sector: the vertices of the stereographic \
                triangle are not in the same hemisphere")

    def inside_sphericalpatch(self, vertex, dir3):
        '''
            @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            @DATE    12/09/2020 SS 1.0 original
            @PARAM   vertex vertices of the spherical triangle
                     dir3 normalized direction vectors
                     switch which group to check. acceptable arguments are 'pg', 'laue', 'supergroup'
                     and 'supergroup_laue'
            @DETAIL  check if direction is inside a spherical patch
                     the logic used as follows:
                     if determinant of [x A B], [x B C] and [x C A] are 
                     all same sign, then the sphere is inside the traingle
                     formed by A, B and C
                     returns a mask with inside as True and outside as False
        '''
        nn = vertex.shape[1]

        mask = []
        d = np.zeros([nn, ])

        for x in dir3:
            x2 = np.atleast_2d(x).T

            for ii in range(nn):
                A = np.atleast_2d(vertex[:, np.mod(ii, nn)]).T
                B = np.atleast_2d(vertex[:, np.mod(ii+1, nn)]).T
                d[ii] = np.linalg.det(np.hstack((x2, A, B)))

                '''
                catching cases very close to FZ boundary when the
                determinant can be very small positive or negative
                number
                '''
                if(np.abs(d[ii]) < eps):
                    d[ii] = 0.

            ss = np.unique(np.sign(d))
            if(np.all(ss >= 0.)):
                mask.append(True)
            else:
                mask.append(False)

        mask = np.array(mask)
        return mask

    def fillet_region(self, dir3, switch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    12/09/2020 SS 1.0 original
        @PARAM   vertex vertices of the spherical triangle
                 dir3 normalized direction vectors
                 switch which group to check. acceptable arguments are 'super', 'superlaue'
                'super' ----> point group coloring
                'superlaue' ----> laue group coloring

        this function will check which fillet the point lies in
        returns 0 if its barycenter, vertex 0 and vertex 1
        returns 1 if its barycenter, vertex 1 and vertex 2
        returns 2 if its barycenter, vertex 2 and vertex 3

        it is implicitly assumed that the point lies inside the 
        spherical triangle. behavior is unknown if it is not the
        case

        first make the vertices of the three fillets
        '''

        vertex = np.copy(self.vertices[switch])
        fregion = -np.ones([dir3.shape[0], ]).astype(np.int32)

        bar_cen = self.barycenter[switch]

        # if barycenter matches one of the vertices, then remove that vertex
        mask = np.all(bar_cen == vertex.T,axis=1)
        vertex = vertex[:,~mask]

        nn = vertex.shape[1]
        f = np.zeros([nn, 3, 3])

        for i in range(nn):
            idx1 = np.mod(i, nn)
            idx2 = np.mod(i+1, nn)
            A = np.atleast_2d(vertex[:, idx1]).T
            B = np.atleast_2d(vertex[:, idx2]).T
            f[i, :, :] = np.hstack((np.atleast_2d(bar_cen).T, A, B))

        for i in range(nn):
            inside = np.logical_and(self.inside_sphericalpatch(
                                    np.squeeze(f[i, :, :]), dir3),
                                    fregion == -1)
            fregion[inside] = i

        return fregion

    def point_on_boundary(self, dir3, switch):
        '''
            @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            @DATE    12/09/2020 SS 1.0 original
            @PARAM   dir3 direction in fundamental sector. size is nx3
                     switch color using pg or laue group
            @DETAIL  this function figures out the equivalent point on the boundary 
            given that the point is inside the spherical triangle
        '''
        vertex = self.vertices[switch]
        fregion = self.fillet_region(dir3, switch)
        dir3_b = np.zeros(dir3.shape)
        nn = vertex.shape[1]

        for i in range(fregion.shape[0]):

            f = fregion[i]
            d = dir3[i, :]

            A = vertex[:, np.mod(f, nn)]
            B = vertex[:, np.mod(f+1, nn)]

            nhat = np.cross(B, A)
            nhat = nhat/np.linalg.norm(nhat)

            lam = np.dot(nhat, d)
            deldir = lam*nhat

            dp = d - deldir
            ndp = np.linalg.norm(dp)
            if(ndp > 0.):
                dp = dp/ndp
            else:
                dp = d

            dir3_b[i, :] = dp

        return dir3_b, fregion

    def calculate_rho(self, dir3, switch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    12/09/2020 SS 1.0 original
        @PARAM   dir3 direction in fundamental sector. size is nx3
                 switch color using pg or laue group
        @DETAIL  this function is used to calculate the azimuthal angle
                 of a bunch of directions. it is assumed all directions 
                 are indide the SST
        '''
        vertex = self.vertices[switch]
        bar_cen = self.barycenter[switch]
        rho = np.zeros([dir3.shape[0], ])

        # handle triclinic and monoclinic cases a little differently
        if(np.all(bar_cen == np.array([0., 0., 1.]))):
            rho = np.arctan2(dir3[:,1], dir3[:,0]) + np.pi

        else:
            dir3_b, fregion = self.point_on_boundary(dir3, switch)
            nn = vertex.shape[1]

            for i in range(fregion.shape[0]):
                f = fregion[i]
                d = dir3_b[i, :]

                A = vertex[:, np.mod(f, nn)]
                B = vertex[:, np.mod(f+1, nn)]

                # angle between A and B
                omega = np.dot(A, B)
                if(np.abs(omega) > 1.):
                    omega = np.sign(omega)

                # angle between point and A
                omegap = np.dot(A, d)
                if(np.abs(omegap) > 1.):
                    omegap = np.sign(omega)

                omega = np.arccos(omega)
                omegap = np.arccos(omegap)

                if(omegap != 0.):
                    rho[i] = 2*np.pi*omegap/omega/nn + f*2.*np.pi/nn
                else:
                    rho[i] = f*2.*np.pi/nn

        return rho

    def calculate_theta(self, dir3, switch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    12/09/2020 SS 1.0 original
        @PARAM   dir3 direction in fundamental sector. size is nx3
                 switch color using pg or laue group
        @DETAIL  this function is used to calculate the polar angle 
        of direction vectors. it is assumed that the direction vector
        lies inside the SST
        '''
        vertex = self.vertices[switch]
        dir3_b, fregion = self.point_on_boundary(dir3, switch)
        theta = np.zeros([dir3.shape[0], ])

        bar_cen = self.barycenter[switch]

        # handle triclinic and monoclinic cases a little differently
        if(np.all(bar_cen == np.array([0., 0., 1.]))):
            dp = np.dot(np.array([0., 0., 1.]), dir3.T)
            # catch some cases where dot product is 1+/-epsilon
            mask = np.abs(dp) > 1.
            dp[mask] = np.sign(dp[mask])
            theta = np.arccos(dp)

        else:
        # first calculate the angle the point makes with the barycenter
            omega = np.dot(bar_cen, dir3.T)
            mask = np.abs(omega) > 1.0
            omega[mask] = np.sign(omega[mask])

            # calculate the angle the boundary point makes with the barycenter
            omegap = np.dot(bar_cen, dir3_b.T)
            mask = np.abs(omegap) > 1.0
            omegap[mask] = np.sign(omegap[mask])

            omega = np.arccos(omega)
            omegap = np.arccos(omegap)

            zmask = omegap == 0.

            theta[~zmask] = np.pi*omega[~zmask]/omegap[~zmask]/2.0
            theta[zmask] = 0.0
        return theta

    def hue_speed(self, rho):
        '''
            @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            @DATE    12/09/2020 SS 1.0 original
            @PARAM   rho azimuthal angle
            @DETAIL  calculate the hue speed for a vector of azimuthal angles
                    this is utilized in increasing the area of the red, blue and 
                    green regions
        '''
        rho = rho - np.pi
        v = 0.5 + np.exp(-(4./7.)*rho**2) + \
            np.exp(-(4./7.)*(rho - 2.*np.pi/3.)**2) + \
            np.exp(-(4./7.)*(rho + 2.*np.pi/3.)**2)

        return v

    def hue_speed_normalization_factor(self):
        pass

    def calc_hue(self, dir3, switch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
                 11/23/2020 SS 1.2 added mask argument which tell the directions
                 for which the supergroup reductions dont match the point or laue
                 group reductions. mask has size dir3.shape[0]
                 12/09/2020 SS 2.0 completely rewrite the way in which computation
                 is performed. all the routines have been rewritten

        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 direction is outside the fundamental sector
                 switch color by laue group or point group? acceptable values are 'super', 'superlaue'
                'super' ----> point group coloring
                'superlaue' ----> laue group coloring
        @DETAIL  calculate hue. this is aggigned based on the azimuthal angle in the
        stereographic triangle. the laueswitch controls which fundamental sector to use.

        '''
        rho = self.calculate_rho(dir3, switch)
        r = np.linspace(0., 2*np.pi, 1000)
        v = self.hue_speed(r)
        cons = np.trapz(v, r)

        h = np.zeros(rho.shape)

        for i in range(rho.shape[0]):
            r = np.linspace(0., rho[i], 1000)
            v = self.hue_speed(r)
            h[i] = np.trapz(v, r)/cons

        return h

    def calc_saturation(self, l):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
                 11/12/2020 SS 1.1 added laueswitch as argument
                 11/16/2020 SS 2.0 more balanced saturation from JAC papaer
                 the factor lambda_s is hard coded to 1/4. Since lighness is needed,
                 the function now uses L as an input instead of dir3 to avoid recomputing L
                 11/23/2020 SS 1.2 added mask argument which tell the directions
                 for which the supergroup reductions dont match the point or laue
                 group reductions. mask has size dir3.shape[0]

        @PARAM   L lightness values
        @DETAIL  calculate saturation. this is always set to 1.

        '''
        s = 1. - 2.*0.25*np.abs(l - 0.5)
        return s

    def calc_lightness(self, dir3, mask, switch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
                 11/12/2020 SS 1.1 added laueswitch as argument
                 11/16/2020 SS 1.2 more balanced lightness key from the JAC paper
                 the factor lambda is hard coded to 1/4
                 11/23/2020 SS 1.2 added mask argument which tell the directions
                 for which the supergroup reductions dont match the point or laue
                 group reductions. mask has size dir3.shape[0]
                 12/09/2020 SS 2.0 completely rewrite the way in which computation
                 is performed. all the routines have been rewritten

        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 laueswitch get colors in laue group or pg
                 mask boolean mask for points where the symmetry reductions donot match the
                 supergroup symmetry reductions
        @DETAIL  this function is used to calculate the hsl color for direction vectors
                in dir3. if laueswitch is True, then color is assigned based on laue group

        '''
        theta = np.pi - self.calculate_theta(dir3, switch)
        f1 = theta/np.pi
        f2 = np.sin(theta/2.)**2
        l = 0.35*f1 + 0.65*f2
        l[~mask] = 1. - l[~mask]

        return l

    def get_color(self, dir3, mask, switch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
                 11/12/2020 SS 1.1 added laueswitch as argument
                 11/23/2020 SS 1.2 added mask argument which tell the directions
                 for which the supergroup reductions dont match the point or laue
                 group reductions. mask has size dir3.shape[0]
                
        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 mask True if symmetry reduction of dir3 using point group does not
                 match the super group and False otherwise
                 switch get colors in laue group or pg
        @DETAIL  this function is used to calculate the hsl color for direction vectors
                in dir3. if laueswitch is True, then color is assigned based on laue group

        '''
        hsl = np.zeros(dir3.shape)

        hsl[:, 0] = self.calc_hue(dir3, switch)
        hsl[:, 2] = self.calc_lightness(dir3, mask, switch)
        hsl[:, 1] = self.calc_saturation(hsl[:, 2])

        return hsl
