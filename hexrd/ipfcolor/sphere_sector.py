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

there are no triangles for the triclininc cases and needs to be handles differently

'''
pg2vertex = {
    'c1' : [0, [], [], 'both'],

    # supergroup 00 in our convention
    'ci' : [0, [], [], 'upper'],

    'c2' : [2, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [-1., 0., 0.]]).T,
            np.array([[0, 1, 2],[0, 2, 3]]).T,
            'both'],

    # supergroup 1 in our convention
    'cs' : [4, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [-1., 0., 0.],
                    [0., -1., 0.]]).T,
            np.array([[0, 1, 2],[0, 2, 3],[0, 3, 4],[0, 4, 1]]).T,
            'upper'],

    'c2h' : [2, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [-1., 0., 0.]]).T,
            np.array([[0, 1, 2],[0, 2, 3]]).T,
            'upper'],

    'd2' : [2, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [-1., 0., 0.]]).T, 
           np.array([[0, 1, 2],[0, 2, 3]]).T,
            'upper'],

    # supergroup 2 in our convention
    'c2v' : [2, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [0.,0.,-1.]]).T,
            np.array([[0, 1, 2], [1, 3, 2]]).T,
            'both'],

    # supergroup 3 in our convention
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

    # supergroup 01 in our convention
    's4' : [2, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [-1., 0., 0.]]).T, 
            np.array([[0, 1, 2],[0, 2, 3]]).T,
            'upper'],

    'c4h' : [1, np.array([[0.,0.,1.],
                      [1., 0., 0.],
                      [0., 1., 0.]]),
            np.array([0, 1, 2]),
            'upper'],

    'd4' : [1, np.array([[0.,0.,1.],
                      [1., 0., 0.],
                      [0., 1., 0.0]]),
            np.array([0, 1, 2]),
            'upper'],

    # supergroup 4 in our convention
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

    # supergroup 5 in our convention
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

    # supergroup 02 in our convention
    's6' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [-0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    'd3' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [-0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    # supergroup 6 in our convention
    'c3v' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0.5, np.sqrt(3.)/2., 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    'd3d' : [1, np.array([[0., 0., 1.],
                    [np.sqrt(3.)/2., 0.5, 0.],
                    [np.sqrt(3.)/2., -0.5, 0.]]).T,
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

    # supergroup 7 in our convention
    'c6v' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.array([0, 1, 2]),
            'both'],

    # supergroup 8 in our convention
    'd3h' : [1, np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [np.sqrt(3.)/2., 0.5, 0.]]).T,
            np.array([0, 1, 2]),
            'upper'],

    # supergroup 9 in our convention
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

    # supergroup 10 in our convention
    'td' : [1, np.array([[0.,0.,1.],
                    [1./np.sqrt(3.), -1./np.sqrt(3.), 1./np.sqrt(3.)],
                    [1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)]]).T,
            np.array([0, 1, 2]),
            'upper'],

    # supergroup 11 in our convention
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
        data = pg2vertex[pgsym]
        self.ntriangle = data[0]
        self.vertices = data[1]
        self.connectivity = data[2]
        self.hemisphere = data[3]

        data = pg2vertex[lauesym]
        self.ntriangle_laue = data[0]
        self.vertices_laue = data[1]
        self.connectivity_laue = data[2]
        self.hemisphere_laue = data[3]

        data = pg2vertex[supergroupsym]
        self.ntriangle_supergroup = data[0]
        self.vertices_supergroup = data[1]
        self.connectivity_supergroup = data[2]
        self.hemisphere_supergroup = data[3]

        data = pg2vertex[supergrouplauesym]
        self.ntriangle_supergroup_laue = data[0]
        self.vertices_supergroup_laue = data[1]
        self.connectivity_supergroup_laue = data[2]
        self.hemisphere_supergroup_laue = data[3]

        if(self.ntriangle!= 0):
            # compute the barycenter or the centroid of point group
            self.barycenter = np.mean(self.vertices, axis=1)
            self.barycenter /= np.linalg.norm(self.barycenter)

            # compute the barycenter or the centroid of the laue group triangle
            self.barycenter_laue = np.mean(self.vertices_laue, axis=1)
            self.barycenter_laue /= np.linalg.norm(self.barycenter_laue)

            # compute the barycenter or the centroid of the supergroup group triangle
            self.barycenter_supergroup = np.mean(self.vertices_supergroup, axis=1)
            self.barycenter_supergroup /= np.linalg.norm(self.barycenter_supergroup)

            # compute the barycenter or the centroid of the supergroup group triangle
            self.barycenter_supergroup_laue = np.mean(self.vertices_supergroup_laue, axis=1)
            self.barycenter_supergroup_laue /= np.linalg.norm(self.barycenter_supergroup_laue)

        else:
            self.barycenter = np.array([0., 0., 1.])
            self.barycenter_laue = np.array([0., 0., 1.])
            self.barycenter_supergroup = np.array([0., 0., 1.])
            self.barycenter_supergroup_laue = np.array([0., 0., 1.])

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

    def calc_azi_rho(self, dir3, mask, laueswitch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
                 11/23/2020 SS 1.2 added mask argument which tell the directions
                 for which the supergroup reductions dont match the point or laue
                 group reductions. mask has size dir3.shape[0]. True means its the
                 same, otherwise different

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
        rho = np.zeros(dir3.shape[0])

        d1 = dir3[mask,:]
        if(laueswitch == False):
            if(self.ntriangle != 0):
                rx = self.vertices[:,0]
            else:
                rx = np.array([1., 0., 0.])
            b = self.barycenter_supergroup
        else:
            if(self.ntriangle != 0):
                rx = self.vertices_laue[:,0]
            else:
                rx = np.array([1., 0., 0.])
            b = self.barycenter_supergroup_laue

        '''
        we are splitting the handling of triclinic and other symmetries
        here. this is probably not the most efficient way to do things, but 
        easiest to implement for now. improvements will be made later
        '''
        if(np.any(mask == True)):
            if(self.ntriangle != 0):
                n1 = np.cross(b, rx)
                n1 = n1/np.linalg.norm(n1)

                n2 = np.cross(np.tile(b,[d1.shape[0],1]), d1)
                zmask = np.linalg.norm(n2, axis=1) > eps
                n2[zmask,:] = n2[zmask,:]/np.tile(np.linalg.norm(n2[zmask,:],axis=1),[n2[zmask,:].shape[1],1]).T

                dp = np.zeros([d1.shape[0],])
                dp[zmask] = np.dot(n1, n2[zmask,:].T)
                dp[~zmask] = 0.
                nmask = dp < 0.
                dp[dp > 1.] = 1.
                dp[dp < -1.] = -1.
                r = np.arccos(dp)
                r[nmask] += np.pi
            else:
                y = dir3[:,1]
                x = dir3[:,0]
                r = np.arctan2(y, x) + np.pi

            rho[mask] = r

        if(np.any(mask == False)):
            d2 = dir3[~mask,:]
            # b[2] = -b[2]
            if(self.ntriangle != 0):

                n1 = np.cross(b, rx)
                n1 = n1/np.linalg.norm(n1)
                n2 = np.cross(np.tile(b,[d2.shape[0],1]), d2)
                zmask = np.linalg.norm(n2, axis=1) > eps
                n2[zmask,:] = n2[zmask,:]/np.tile(np.linalg.norm(n2[zmask,:],axis=1),[n2[zmask,:].shape[1],1]).T
                dp = np.zeros([d2.shape[0],])
                dp[zmask] = np.dot(n1, n2[zmask,:].T)
                dp[~zmask] = 0.
                nmask = dp < 0.
                dp[dp > 1.] = 1.
                dp[dp < -1.] = -1.
                r = np.arccos(dp)
                r[nmask] += np.pi

            else:
                y = dir3[:,1]
                x = dir3[:,0]
                r = np.arctan2(y, x) + np.pi

            rho[~mask] = r

        return rho

    def calc_pol_theta(self, dir3, mask, laueswitch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
                 11/23/2020 SS 1.2 added mask argument which tell the directions
                 for which the supergroup reductions dont match the point or laue
                 group reductions. mask has size dir3.shape[0]

        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 direction is outside the fundamental sector

        @DETAIL  this function is used to calculate the polar angle of the
                 direction inside a spherical patch. this is computed as the scaled
                 angular distance between direction and barycenter. the scaling is such
                 that the boundary is 90 degrees from barycenter.

        '''
        self.check_norm(dir3)
        pol = np.zeros(dir3.shape[0])
        d1 = dir3[mask,:]
        if(laueswitch == False):
            b = self.barycenter_supergroup
        else:
            b = self.barycenter_supergroup_laue

        #pol[mask] = np.arccos(np.dot(b, d1.T))
        pol = np.arccos(np.dot(b, dir3.T))
        # if(np.any(mask == False)):
        #     pol[~mask] = pol[~mask]

        # if(np.any(mask == False)):
        #     d2 = dir3[~mask,:]
        #     b[2] = -b[2]
        #     pol[~mask] = np.arccos(np.dot(b, d2.T))
        #pol = pol*np.pi/pol.max()
        m = (pol <= np.pi/2.)
        if(np.sum(m) > 0):
            if(pol[m].max() > pol[m].min()):
                pol[m] = (np.pi/2.) * (pol[m] - pol[m].min())/(pol[m].max() - pol[m].min())
        if(np.sum(~m) > 0):
            if(pol[~m].max() > pol[~m].min()):
                pol[~m] = (np.pi/2.) * (pol[~m] - pol[~m].min())/(pol[~m].max() - pol[~m].min()) + np.pi/2.

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
        rho = rho - np.pi
        v = 0.5 + np.exp(-(4./7.)*rho**2) + \
            np.exp(-(4./7.)*(rho - 2.*np.pi/3.)**2) + \
            np.exp(-(4./7.)*(rho + 2.*np.pi/3.)**2)

        return v

    def hue_speed_normalization_factor(self):
        pass

    def calc_hue(self, dir3, mask, laueswitch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
                 11/23/2020 SS 1.2 added mask argument which tell the directions
                 for which the supergroup reductions dont match the point or laue
                 group reductions. mask has size dir3.shape[0]

        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 direction is outside the fundamental sector
        @DETAIL  calculate hue. this is aggigned based on the azimuthal angle in the
        stereographic triangle. the laueswitch controls which fundamental sector to use.

        '''
        H = np.zeros(dir3.shape[0])
        rho = self.calc_azi_rho(dir3, mask, laueswitch)
        # vel = self.hue_speed(rho)
        # den = np.trapz(vel, np.linspace(0., np.pi*2., dir3.shape[0]))

        # for i,r in enumerate(rho):
        #     npts = int(np.floor(dir3.shape[0]*r/2./np.pi))
        #     ang = np.linspace(0., r, npts)
        #     v = self.hue_speed(ang)
        #     num = np.trapz(v, ang)

        #     H[i] = num/den

        '''
        catching some edge cases
        '''
        # H[H > 1.] = 1.
        # H[H < 0.] = 0.
        H = rho/2./np.pi
        return H

    def calc_saturation(self, L, mask, laueswitch):
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
        S = 1. - 2.*0.25*np.abs(L - 0.5)
        return S

    def calc_lightness(self, dir3, mask, laueswitch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
                 11/12/2020 SS 1.1 added laueswitch as argument
                 11/16/2020 SS 1.2 more balanced lightness key from the JAC paper
                 the factor lambda is hard coded to 1/4
                 11/23/2020 SS 1.2 added mask argument which tell the directions
                 for which the supergroup reductions dont match the point or laue
                 group reductions. mask has size dir3.shape[0]

        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 laueswitch get colors in laue group or pg
        @DETAIL  this function is used to calculate the hsl color for direction vectors
                in dir3. if laueswitch is True, then color is assigned based on laue group

        '''
        theta = self.calc_pol_theta(dir3, mask, laueswitch)
        f1 = theta/np.pi
        f2 = np.sin(theta/2.)**2
        L = 0.25*f1 + 0.75*f2
        return 1. - L

    def get_color(self, dir3, mask, laueswitch):
        '''
        @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        @DATE    10/28/2020 SS 1.0 original
                 11/12/2020 SS 1.1 added laueswitch as argument
                 11/23/2020 SS 1.2 added mask argument which tell the directions
                 for which the supergroup reductions dont match the point or laue
                 group reductions. mask has size dir3.shape[0]

        @PARAM   dir3 direction in fundamental sector. behavior is undefined if
                 laueswitch get colors in laue group or pg
        @DETAIL  this function is used to calculate the hsl color for direction vectors
                in dir3. if laueswitch is True, then color is assigned based on laue group

        '''
        hsl = np.zeros(dir3.shape)

        hsl[:,0] = self.calc_hue(dir3, mask, laueswitch)
        hsl[:,2] = self.calc_lightness(dir3, mask, laueswitch)
        hsl[:,1] = self.calc_saturation(hsl[:,2], mask, laueswitch)
        return hsl
