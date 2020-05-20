#! /usr/bin/env python
# ============================================================
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
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free Software
# Foundation) version 2.1 dated February 1999.
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
# ============================================================
from math import sqrt, log

import numpy as num

try:
    import multiprocessing
    haveMultiProc = True
except:
    haveMultiProc = False

if haveMultiProc:
    dfltNCPU = int(multiprocessing.cpu_count()/2)
    if dfltNCPU > 3: 
        dfltNCPU = dfltNCPU/2
else:
    #
    dfltNCPU = 1


def dataToFrame(data, sumImg=True):
    """
    utility function to allow flexibility in input
    
    data can be:
    (*) an instance of ReadGE or the like, which is already set up, in which
        case all frames are used and flattened
    (*) a frame
    """
    if hasattr(data, 'getNFrames'):
        reader = data.makeNew()
        nFrames = reader.getNFrames()
        frame = reader.read(nframes=nFrames, sumImg=sumImg)
    elif hasattr(data, 'shape') and len(data.shape) == 2:
        'assume the data is a frame'
        frame = data
    else:
        raise RuntimeError('do not know what to do with data : '+str(type(data)))
    return frame

def getGaussNDParams(xList, w=None, v=None):
    
    nDim = len(xList)
    
    xVec = num.empty(nDim+nDim+2)
    
    if w is not None:
         assert xList[0].shape == w.shape,\
             'w not the same shape as other arguments; %s != %s' % (str(x.shape), str(w.shape))
    
    if v is None:
        bg   = 0.0e0
        vNbg = num.ones(x.shape)
    else:
        bg   = num.min(v)
        vNbg = v - bg
    if w is None:
        vwNbg = vNbg
    else:
        if len(w.shape) == 2:
            nQP = w.shape[1]
            vwNbg = num.tile(vNbg, (nQP,1)).T * w
        elif len(w.shape) == 1:
            vwNbg = vNbg * w
        else:
            raise NotImplementedError('shape of w has length %d' % (len(shape(w))))
    vwSum = float(num.sum(vwNbg))
    
    if vwSum <= 0:
        'just swag something'
        # raise RuntimeError, 'vwSum <= 0'
        vwNbg = num.ones_like( vwNbg )
        vwSum = float(num.sum(vwNbg))

    com = []
    for iDim, xThis in enumerate(xList):
        mu = num.sum(xThis * vwNbg) / vwSum
        com.append(mu)
    #
    for iDim, xThis in enumerate(xList):
        diffs = xThis - com[iDim]
        sigma = sqrt(num.sum(vwNbg*diffs*diffs)/vwSum)
        xVec[iDim]      = com[iDim]
        xVec[iDim+nDim] = sqrt(8.0 * log(2.0)) * sigma # FWHM
    xVec[nDim+nDim]   = vwNbg.max()
    xVec[nDim+nDim+1] = bg
    
    return xVec

