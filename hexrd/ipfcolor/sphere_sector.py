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
'''
pg2vertex = {
    'c1' : [],
    'ci' : [],
    'c2' : [],
    'cs' : [],
    'c2h' : [],
    'd2' : [],
    'c2v' : [],
    'd2h' : [],
    'c4' : [],
    's4' : [],
    'c4h' : [],
    'd4' : [],
    'c4v' : [],
    'd2d' : [],
    'd4h' : [],
    'c3' : [],
    's6' : [],
    'd3' : [],
    'c3v' : [],
    'd3d' : [],
    'c6' : [],
    'c3h' : [],
    'c6h' : [],
    'd6' : [],
    'c6v' : [],
    'd3h' : [],
    'd6h' : [],
    't' : [],
    'th' : [],
    'o' : [],
    'td' : [],
    'oh' : np.array([[0.,0.,1.],
                    [1./np.sqrt(2.),0., 1./np.sqrt(2.)
                    ],
                    [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)]]).T
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
        initialize the class with a vertex and the barycenter
        the vertes are organized as columns
        '''
        vertex = pg2vertex[pgsym]

        self.vx = vertex[:,0]
        self.vy = vertex[:,1]
        self.vz = vertex[:,2]

        '''
        make sure there are unit norm
        '''
        nvx = np.linalg.norm(self.vx)
        nvy = np.linalg.norm(self.vy)
        nvz = np.linalg.norm(self.vz)
        
        if(np.abs(nvx) > eps):
            self.vx /= nvx
        else:
            raise RuntimeError("one of the spherical vertex is null.")

        if(np.abs(nvy) > eps):
            self.vy /= nvy
        else:
            raise RuntimeError("one of the spherical vertex is null.")

        if(np.abs(nvz) > eps):
            self.vz /= nvz
        else:
            raise RuntimeError("one of the spherical vertex is null.")

        '''
        check if all the vertices are in the same hemisphere
        get the z-components of the vertices
        '''
        self.check_hemisphere()

        # compute the barycenter or the centroid
        self.barycenter = (self.vx + self.vy + self.vz) / 3.
        self.barycenter /= np.linalg.norm(self.barycenter)

        # this is the vector about which the azimuthal angle is 
        # computed (vx - barycenter)
        self.rx = self.vx - self.barycenter
        self.rx /= np.linalg.norm(self.rx)

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
