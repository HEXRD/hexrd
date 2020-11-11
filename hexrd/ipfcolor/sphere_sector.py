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
from hexrd import constants
import numpy as np
from hexrd.ipfcolor import colorspace

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

'''
pg2vertex = {
    'c1' : [0, [], [], 'both'],

    'ci' : [0, [], [], 'upper'],

    'c2' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [-1., 0., 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    'cs' : [0, [], [], 'upper'],

    'c2h' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [-1., 0., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'd2' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [-1., 0., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'c2v' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    'd2h' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'c4' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    's4' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [-1., 0., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'c4h' : [1, np.array([[0.,0.,1.],
                      [1., 0., 0.],
                      [0., 1., 0.0]]),
            np.array([0, 1, 2]),
            'upper'],

    'd4' : [1, np.array([[0.,0.,1.],
                      [1., 0., 0.],
                      [0., 1., 0.0]]),
            np.array([0, 1, 2]),
            'upper'],

    'c4v' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [1./np.sqrt(2.), 1./np.sqrt(2.), 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    'd2d' : [1, np.array([[0.,0.,1.],
                      [1., 0., 0.],
                      [0., 1., 0.]]),
            np.array([0, 1, 2]),
            'upper'],

    'd4h' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [1./np.sqrt(2.), 1./np.sqrt(2.), 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'c3' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [-0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    's6' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [-0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'd3' : [1, np.array([[0., 0., 1.],
                    [np.sqrt(3.)/2., -0.5, 0.],
                    [0., 1., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'c3v' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    'd3d' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'c6' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    'c3h' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [-0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'c6h' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'd6' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    'c6v' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    'd3h' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'd6h' : [1, np.array([[0.,0.,1.],
                    [1., 0., 0.],
                    [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    # Special case with 2 triangles
    't' : [2, np.array([[0.,0.,1.],
                    [1./np.sqrt(3.), -1./np.sqrt(3.), 1./np.sqrt(3.)],
                    [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)],
                    [1., 0., 0.]]).T,
            np.array([[0, 1, 2],[1, 3, 2]]).T,
           'upper'],

    # Special case with two triangles
    'th' : [2, np.array([[0.,0.,1.],
                    [1./np.sqrt(2.), 0., 1./np.sqrt(2.)],
                    [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)],
                    [0., 1./np.sqrt(2.), 1./np.sqrt(2.)]]).T,
          np.array([[0, 1, 2],[0, 2, 3]]).T,
          'upper'],

    # Special case with two triangles, same as Th
    'o' : [2, np.array([[0.,0.,1.],
                    [1./np.sqrt(2.), 0., 1./np.sqrt(2.)],
                    [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)],
                    [0., 1./np.sqrt(2.), 1./np.sqrt(2.)]]).T,
          np.array([[0, 1, 2],[0, 2, 3]]).T,
          'upper'],

    'td' : [1, np.array([[0.,0.,1.],
                    [1./np.sqrt(3.), -1./np.sqrt(3.), 1./np.sqrt(3.)],
                    [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'oh' : [1, np.array([[0.,0.,1.],
                    [1./np.sqrt(2.),0., 1./np.sqrt(2.)],
                    [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)]]).T,
            np.array([0, 1, 2]),
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
    def __init__(self, pgsym):
        '''
        AUTHOR: Saransh Singh, Lawrence Livermore national Lab, saransh1@llnl.gov
        DATE:   11/11/2020 SS 1.0 original

        @detail: this routine initializes the data needed for reducing a 
        direction to the stereographic fundamental zone (standard
        stereographic triangle) for the pointgroup/lauegroup symmetry
        of the crystal.
        '''
        data = pg2vertex[pgsym]
        
        self.ntriangle = data[0]
        self.vertices = data[1]
        self.connectivity = data[2]
        self.hemisphere = data[3]

        # vertex = pg2vertex[pgsym]

        # self.vx = vertex[:,0]
        # self.vy = vertex[:,1]
        # self.vz = vertex[:,2]

        '''
        make sure there are unit norm
        '''
        # nvx = np.linalg.norm(self.vx)
        # nvy = np.linalg.norm(self.vy)
        # nvz = np.linalg.norm(self.vz)
        
        # if(np.abs(nvx) > eps):
        #     self.vx /= nvx
        # else:
        #     raise RuntimeError("one of the spherical vertex is null.")

        # if(np.abs(nvy) > eps):
        #     self.vy /= nvy
        # else:
        #     raise RuntimeError("one of the spherical vertex is null.")

        # if(np.abs(nvz) > eps):
        #     self.vz /= nvz
        # else:
        #     raise RuntimeError("one of the spherical vertex is null.")

        # '''
        # check if all the vertices are in the same hemisphere
        # get the z-components of the vertices
        # '''
        # self.check_hemisphere()

        # # compute the barycenter or the centroid
        # self.barycenter = (self.vx + self.vy + self.vz) / 3.
        # self.barycenter /= np.linalg.norm(self.barycenter)

        # # this is the vector about which the azimuthal angle is 
        # # computed (vx - barycenter)
        # self.rx = self.vx - self.barycenter
        # self.rx /= np.linalg.norm(self.rx)

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
        dir3[mask,:] = dir3[mask,:]/np.tile(n,[3,1]).T

    def check_hemisphere(self):

        zcoord = np.array([self.vx[2], self.vy[2], self.vz[2]])
        if(np.logical_or( np.all(zcoord >= 0.),  np.all(zcoord <= 0.) ) ):
            pass
        else:
            raise RuntimeError("sphere_sector: the vertices of the stereographic \
                triangle are not in the same hemisphere")

    def calc_azi_rho(self, dir3):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 direction is outside the fundamental sector. size is nx3
        @DETAIL  this function is used to calculate the azimuthal angle of the
                 direction inside a spherical patch. this is computed as the angle
                 with the vector defined by the first vertex and barycenter and the
                 vector defined by direction and barycenter.

        '''
        '''
        first make sure all the directions are unit normal
        if not make them unit normals
        the directions with zero norms are ignored
        '''
        self.check_norm(dir3)
        c = np.dot(self.rx, dir3.T)
        s = np.cross(np.tile(self.rx,[dir3.shape[0],1]), dir3)
        s = np.arcsin(np.linalg.norm(s,axis=1))
        rho = np.arctan2(s, c) + np.pi

        return rho

    def calc_pol_theta(self, dir3):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 direction is outside the fundamental sector
        @DETAIL  this function is used to calculate the polar angle of the
                 direction inside a spherical patch. this is computed as the scaled
                 angular distance between direction and barycenter. the scaling is such
                 that the boundary is 90 degrees from barycenter.

        '''
        self.check_norm(dir3)
        pol = np.arccos(np.dot(self.barycenter, dir3.T))
        return pol

    def distance_boundary(self, rho):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/29/2020 SS 1.0 original
        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 direction is outside the fundamental sector
        @DETAIL  this function is used to calculate the distance from the boundary
                 point specified by the azimuthal angle rho and the barycenter

        '''
        pass

    def hue_speed(self, rho):
        v = 0.5 + np.exp(-(4./7.)*rho**2) + \
            np.exp(-(4./7.)*(rho - 2.*np.pi/3.)**2) + \
            np.exp(-(4./7.)*(rho + 2.*np.pi/3.)**2)

        return v

    def hue_speed_normalization_factor(self):
        pass

    def calc_hue(self, dir3):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 direction is outside the fundamental sector
        @DETAIL  this function is used to calculate the hue based on angular coordinates
                 of direction within the spherical patch. the velocity

        '''
        rho = self.calc_azi_rho(dir3)
        H = rho/2./np.pi
        return H

    def calc_saturation(self, dir3):
        S = np.ones([dir3.shape[0],])
        return S

    def calc_lightness(self, dir3):
        theta = self.calc_pol_theta(dir3)
        theta /= np.pi*theta.max()
        theta = np.pi - theta
        L = theta/np.pi
        return L

    def get_color(self, dir3):

        hsl = np.zeros(dir3.shape)

        hsl[:,0] = self.calc_hue(dir3)
        hsl[:,1] = self.calc_saturation(dir3)
        hsl[:,2] = self.calc_lightness(dir3)

        rgb = colorspace.hsl2rgb(hsl)

        return rgb
