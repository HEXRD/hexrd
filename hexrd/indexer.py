#! /usr/bin/env python
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
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
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
import sys
import os
import copy
import ctypes
import tempfile
import glob
import logging
import time
import pdb

import numpy as num
#num.seterr(invalid='ignore')

import hexrd.matrixutil as mUtil

from hexrd.grain     import Grain, makeMeasuredScatteringVectors
from hexrd.rotations import \
     discreteFiber, mapAngle, \
     quatOfRotMat, quatProductMatrix, \
     rotMatOfExpMap, rotMatOfQuat
from hexrd.symmetry  import toFundamentalRegion
from hexrd           import xrdbase

from hexrd.transforms import xf
from hexrd.transforms import xfcapi
from hexrd.constants import USE_NUMBA

# FIXME: numba implementation of paintGridThis is broken
if USE_NUMBA:
    import numba

if xrdbase.haveMultiProc:
    multiprocessing = xrdbase.multiprocessing # formerly import


logger = logging.getLogger(__name__)


# module vars
piby2 = num.pi * 0.5
r2d = 180. / num.pi
d2r = num.pi / 180.

Xl = num.c_[1, 0, 0].T
Yl = num.c_[0, 1, 0].T
Zl = num.c_[0, 0, 1].T

fableSampCOB = num.dot(rotMatOfExpMap(piby2*Zl), rotMatOfExpMap(piby2*Yl))

class GrainSpotter:
    """
    Interface to grain spotter, which must be in the user's path
    """
    __execName = 'grainspotter'
    def __init__(self):
        self.__tempFNameList = []

        if (os.system('which '+self.__execName) != 0):
            print("need %s to be in the path" % (self.__execName), file=sys.stderr)
            raise RuntimeError("unrecoverable error")

        return

    def __call__(self, spotsArray, **kwargs):
        """
        writes gve and ini files to system, calls grainspotter, parses results.

        A word on spacegroup numbers: it appears that grainspotter is using the
        'VolA' tag for calls to SgInfo
        """
        location = self.__class__.__name__
        tic = time.time()

        phaseID   = None
        gVecFName = 'tmp'

        kwarglen = len(kwargs)
        if kwarglen > 0:
            argkeys = list(kwargs.keys())
            for i in range(kwarglen):
                if argkeys[i] == 'phaseID':
                    phaseID = kwargs[argkeys[i]]
                elif argkeys[i] == 'filename':
                    gVecFName = kwargs[argkeys[i]]

        planeData = spotsArray.getPlaneData(phaseID=phaseID)
        U0        = planeData.latVecOps['U0']
        symTag    = planeData.getLaueGroup()

        writeGVE(spotsArray, gVecFName, **kwargs)

        toc = time.time()
        print('in %s, setup took %g' % (location, toc-tic))
        tic = time.time()

        # tempFNameStdout = tempfile.mktemp()
        # self.__tempFNameList.append(tempFNameStdout)
        # tempFNameStdout = 'tmp.out'
        # grainSpotterCmd = (
        #     '%s %s > %s' % (self.__execName, gVecFName, tempFNameStdout)
        #     )
        grainSpotterCmd = '%s %s' % (self.__execName, gVecFName+'.ini')
        os.system(grainSpotterCmd)
        toc = time.time()
        print('in %s, execution took %g' % (location, toc-tic))
        tic = time.time()

        # add output files to cleanup list
        # self.__tempFNameList += glob.glob(gVecFName+'.*')

        # collect data from gff file'
        gffFile = gVecFName+'.gff'
        gffData = num.loadtxt(gffFile)
        if gffData.ndim == 1:
            gffData = gffData.reshape(1, len(gffData))
        gffData_U = gffData[:, 6:6+9]

        # process for output
        retval = convertUToRotMat(gffData_U, U0, symTag=symTag)

        toc = time.time()
        print('in %s, post-processing took %g' % (location, toc-tic))
        tic = time.time()

        return retval

    def __del__(self):
        self.cleanup()
        return

    def cleanup(self):
        for fname in self.__tempFNameList:
            os.remove(fname)
        return


def convertUToRotMat(Urows, U0, symTag='Oh', display=False):
    """
    Takes GrainSpotter gff ouput in rows

    U11 U12 U13 U21 U22 U23 U13 U23 U33

    and takes it into the hexrd/APS frame of reference

    Urows comes from grainspotter's gff output
    U0 comes from xrd.crystallography.latticeVectors.U0
    """

    numU, testDim = Urows.shape
    if testDim != 9:
        raise RuntimeError(
            "input must have 9 columns; received %d" % (testDim)
            )

    qin = quatOfRotMat(Urows.reshape(numU, 3, 3))
    # what the hell is happening here?:
    qout = num.dot(
        quatProductMatrix(quatOfRotMat(fableSampCOB), mult='left'),
        num.dot(
            quatProductMatrix(quatOfRotMat(U0.T), mult='right'),
            qin
            ).squeeze()
        ).squeeze()
    if qout.ndim == 1:
        qout = toFundamentalRegion(
            qout.reshape(4, 1), crysSym=symTag, sampSym=None
            )
    else:
        qout = toFundamentalRegion(qout, crysSym=symTag, sampSym=None)
    if display:
        print("quaternions in (Fable convention):")
        print(qin.T)
        print("quaternions out (hexrd convention, symmetrically reduced)")
        print(qout.T)
    Uout = rotMatOfQuat(qout)
    return Uout


def convertRotMatToFableU(rMats, U0=num.eye(3), symTag='Oh', display=False):
    """
    Makes GrainSpotter gff ouput

    U11 U12 U13 U21 U22 U23 U13 U23 U33

    and takes it into the hexrd/APS frame of reference

    Urows comes from grainspotter's gff output
    U0 comes from xrd.crystallography.latticeVectors.U0
    """
    qin = quatOfRotMat(num.atleast_3d(rMats))
    # what the hell is this?:
    qout = num.dot(
        quatProductMatrix(quatOfRotMat(fableSampCOB.T), mult='left'),
        num.dot(
            quatProductMatrix(quatOfRotMat(U0), mult='right'),
            qin).squeeze()
        ).squeeze()
    if qout.ndim == 1:
        qout = toFundamentalRegion(
            qout.reshape(4, 1), crysSym=symTag, sampSym=None
            )
    else:
        qout = toFundamentalRegion(qout, crysSym=symTag, sampSym=None)
    if display:
        print("quaternions in (hexrd convention):")
        print(qin.T)
        print("quaternions out (Fable convention, symmetrically reduced)")
        print(qout.T)
    Uout = rotMatOfQuat(qout)
    return Uout

######################################################################

"""
things for doing fiberSearch with multiprocessing;
multiprocessing has a hard time pickling a function defined in the local scope
of another function, so stuck putting the function out here;
"""
debugMultiproc = 0
if xrdbase.haveMultiProc:
    foundFlagShared = multiprocessing.Value(ctypes.c_bool)
    foundFlagShared.value = False
multiProcMode_MP   = None
spotsArray_MP      = None
candidate_MP       = None
dspTol_MP          = None
minCompleteness_MP = None
doRefinement_MP    = None
nStdDev_MP         = None
def testThisQ(thisQ):
    """
    NOTES:
    (*) doFit is not done here -- in multiprocessing, that would end
        up happening on a remote process and then different processes
        would have different data, unless spotsArray were made to be
        fancier

    (*) kludge stuff so that this function is outside of fiberSearch
    """
    global multiProcMode_MP
    global spotsArray_MP
    global candidate_MP
    global dspTol_MP
    global minCompleteness_MP
    global doRefinement_MP
    global nStdDev_MP
    # assign locals
    multiProcMode   = multiProcMode_MP
    spotsArray      = spotsArray_MP
    candidate       = candidate_MP
    dspTol          = dspTol_MP
    minCompleteness = minCompleteness_MP
    doRefinement    = doRefinement_MP
    nStdDev         = nStdDev_MP
    nSigmas = 2                         # ... make this a settable option?
    if multiProcMode:
        global foundFlagShared

    foundGrainData = None
    #print "testing %d of %d"% (iR+1, numTrials)
    thisRMat = rotMatOfQuat(thisQ)

    ppfx = ''
    if multiProcMode:
        proc = multiprocessing.current_process()
        ppfx = str(proc.name)+' : '
        if multiProcMode and foundFlagShared.value:
            """
            map causes this function to be applied to all trial orientations,
            but skip evaluations after an acceptable grain has been found
            """
            if debugMultiproc > 1:
                print(ppfx+'skipping on '+str(thisQ))
            return foundGrainData
        else:
            if debugMultiproc > 1:
                print(ppfx+'working on  '+str(thisQ))
    candidate.findMatches(rMat=thisRMat,
                          strainMag=dspTol,
                          claimingSpots=False,
                          testClaims=True,
                          updateSelf=True)
    if debugMultiproc > 1:
        print(ppfx+' for '+str(thisQ)+' got completeness : '\
              +str(candidate.completeness))
    if candidate.completeness >= minCompleteness:
        ## attempt to filter out 'junk' spots here by performing full
        ## refinement before claiming
        fineEtaTol = candidate.etaTol
        fineOmeTol = candidate.omeTol
        if doRefinement:
            if multiProcMode and foundFlagShared.value:
                'some other process beat this one to it'
                return foundGrainData
            print(ppfx+"testing candidate q = [%1.2e, %1.2e, %1.2e, %1.2e]"\
                  %tuple(thisQ))
            # not needed # candidate.fitPrecession(display=False)
            ## first fit
            candidate.fit(display=False)
            ## auto-tolerace based on statistics of current matches
            validRefls = candidate.grainSpots['iRefl'] > 0
            fineEtaTol = nStdDev * num.std(
                candidate.grainSpots['diffAngles'][validRefls, 1]
                )
            fineOmeTol = nStdDev * num.std(
                candidate.grainSpots['diffAngles'][validRefls, 2]
                )
            ## next fits with finer tolerances
            for iLoop in range(3):
                candidate.findMatches(etaTol=fineEtaTol,
                                      omeTol=fineOmeTol,
                                      claimingSpots=False,
                                      testClaims=True,
                                      updateSelf=True)
                # not needed # candidate.fitPrecession(display=False)
                candidate.fit(display=False)
            if candidate.completeness < minCompleteness:
                print(ppfx+"candidate failed")
                return foundGrainData
            if multiProcMode and foundFlagShared.value:
                'some other process beat this one to it'
                return foundGrainData
            # not needed # candidate.fitPrecession(display=False)
            # not needed? # candidate.fit(display=False)
            # not needed? # candidate.findMatches(etaTol=fineEtaTol,
            # not needed? #                       omeTol=fineOmeTol,
            # not needed? #                       claimingSpots=False,
            # not needed? #                       testClaims=True,
            # not needed? #                       updateSelf=True)
        else:
            ## at least do precession correction
            candidate.fitPrecession(display=False)
            candidate.findMatches(rMat=thisRMat,
                                  strainMag=dspTol,
                                  claimingSpots=False,
                                  testClaims=True,
                                  updateSelf=True)
            fineEtaTol = candidate.etaTol
            fineOmeTol = candidate.omeTol
            if candidate.completeness < minCompleteness:
                print(ppfx+"candidate failed")
                return foundGrainData
            if multiProcMode and foundFlagShared.value:
                'some other process beat this one to it'
                return foundGrainData
        if multiProcMode:
            foundFlagShared.value = True
        # # newGrain uses current candidate.rMat
        # # do not do claims here -- those handled outside of this call
        # foundGrain = candidate.newGrain(
        #     spotsArray, claimingSpots=False,
        #     omeTol=fineOmeTol,
        #     etaTol=fineEtaTol)
        # if multiProcMode:
        #     foundGrain.strip()
        cInfo = quatOfRotMat(candidate.rMat).flatten().tolist()
        cInfo.append(candidate.completeness)
        print(ppfx+"Grain found at q = [%1.2e, %1.2e, %1.2e, %1.2e] "\
              "with completeness %g" % tuple(cInfo))
        foundGrainData = candidate.getGrainData()
        'tolerances not actually set in candidate, so set them manually'
        foundGrainData['omeTol'] = fineOmeTol
        foundGrainData['etaTol'] = fineEtaTol

    return foundGrainData


def fiberSearch(spotsArray, hklList,
                iPhase=0,
                nsteps=120,
                minCompleteness=0.60,
                minPctClaimed=0.95,
                preserveClaims=False,
                friedelOnly=True,
                dspTol=None,
                etaTol=0.025,
                omeTol=0.025,
                etaTolF=0.00225,
                omeTolF=0.00875,
                nStdDev=2,
                quitAfter=None,
                doRefinement=True,
                debug=True,
                doMultiProc=True,
                nCPUs=None,
                outputGrainList=False
                ):
    """
    This indexer finds grains by performing 1-d searches along the fibers under
    the valid spots associated with each reflection order specified in hklList.
    The set of spots used to generate the candidate orientations may be
    restricted to Friedel pairs only.

    hklList *must* have length > 0;
    Dach hkl entry in hklList *must* be a tuple, not a list

    the output is a concatenated list of orientation matrices ((n, 3, 3)
    numpy.ndarray).
    """

    if not hasattr(hklList, '__len__'):
        raise RuntimeError(
            "the HKL list must have length, and len(hklList) > 0."
            )

    nHKLs = len(hklList)
    grainList = []
    nGrains = 0
    planeData = spotsArray.getPlaneData(iPhase)
    csym = planeData.getLaueGroup()
    bMat = planeData.latVecOps['B']
    if dspTol is None:
        dspTol = planeData.strainMag

    centroSymRefl = planeData.getCentroSymHKLs()

    candidate = Grain(spotsArray, rMat=None,
                      etaTol=etaTol, omeTol=omeTol)
    multiProcMode = xrdbase.haveMultiProc and doMultiProc
    #
    global foundFlagShared
    global multiProcMode_MP
    global spotsArray_MP
    global candidate_MP
    global dspTol_MP
    global minCompleteness_MP
    global doRefinement_MP
    global nStdDev_MP
    multiProcMode_MP   = multiProcMode
    spotsArray_MP      = spotsArray
    candidate_MP       = candidate
    dspTol_MP          = dspTol
    minCompleteness_MP = minCompleteness
    doRefinement_MP    = doRefinement
    nStdDev_MP         = nStdDev
    """
    set up for shared memory multiprocessing
    """
    if multiProcMode:
        nCPUs = nCPUs or xrdbase.dfltNCPU
        spotsArray.multiprocMode = True
        pool = multiprocessing.Pool(nCPUs)

    """
    HKL ITERATOR
    """
    if isinstance(quitAfter, dict):
        n_hkls_to_search = quitAfter['nHKLs']
    else:
        n_hkls_to_search = nHKLs

    if isinstance(quitAfter, int):
        quit_after_ngrains = quitAfter
    else:
        quit_after_ngrains = 0

    numTotal = len(spotsArray)
    pctClaimed = 0.
    time_to_quit = False
    tic = time.time()

    for iHKL in range(n_hkls_to_search):
        print("\n#####################\nProcessing hkl %d of %d\n" \
              % (iHKL+1, nHKLs))
        thisHKLID = planeData.getHKLID(hklList[iHKL])
        thisRingSpots0   = spotsArray.getHKLSpots(thisHKLID)
        thisRingSpots0W  = num.where(thisRingSpots0)[0]
        unclaimedOfThese = -spotsArray.checkClaims(indices=thisRingSpots0W)
        thisRingSpots    = copy.deepcopy(thisRingSpots0)
        thisRingSpots[thisRingSpots0W] = unclaimedOfThese
        if friedelOnly:
            # first, find Friedel Pairs
            spotsArray.findFriedelPairsHKL(hklList[iHKL],
                                           etaTol=etaTolF,
                                           omeTol=omeTolF)
            spotsIteratorI = spotsArray.getIterHKL(
                hklList[iHKL], unclaimedOnly=True, friedelOnly=True
                )
            # make some stuff for counters
            maxSpots = 0.5*(
                sum(thisRingSpots) \
                - sum(spotsArray.friedelPair[thisRingSpots] == -1)
                )
        else:
            spotsIteratorI = spotsArray.getIterHKL(
                hklList[iHKL], unclaimedOnly=True, friedelOnly=False
                )
            maxSpots = sum(thisRingSpots)
        """
        SPOT ITERATOR
          - this is where we iterate over all 'valid' spots for the current HKL
            as subject to the conditions of claims and ID as a friedel pair
            (when requested)
        """
        for iRefl, stuff in enumerate(spotsIteratorI):
            unclaimedOfThese = -spotsArray.checkClaims(indices=thisRingSpots0W)
            thisRingSpots    = copy.deepcopy(thisRingSpots0)
            thisRingSpots[thisRingSpots0W] = unclaimedOfThese
            if friedelOnly:
                iSpot, jSpot, angs_I, angs_J = stuff

                Gplus  = makeMeasuredScatteringVectors(*angs_I)
                Gminus = makeMeasuredScatteringVectors(*angs_J)

                Gvec = 0.5*(Gplus - Gminus)
                maxSpots = 0.5*(
                    sum(thisRingSpots) \
                    - sum(spotsArray.friedelPair[thisRingSpots] == -1)
                    )
            else:
                iSpot, angs_I = stuff
                Gvec  = makeMeasuredScatteringVectors(*angs_I)
                maxSpots = sum(thisRingSpots)
            print("\nProcessing reflection %d (spot %d), %d remain "\
                  "unclaimed\n" % (iRefl+1, iSpot, maxSpots))
            if multiProcMode and debugMultiproc > 1:
                marks = spotsArray._Spots__marks[:]
                print('marks : '+str(marks))
            # make the fiber;
            qfib = discreteFiber(hklList[iHKL], Gvec,
                                 B=bMat,
                                 ndiv=nsteps,
                                 invert=False,
                                 csym=csym, ssym=None)[0]
            # if +/- hkl aren't in the symmetry group, need '-' fiber
            if not centroSymRefl[thisHKLID]:
                minusHKL = -num.r_[hklList[iHKL]]
                qfibM = discreteFiber(minusHKL, Gvec,
                                      B=bMat,
                                      ndiv=nsteps,
                                      invert=False,
                                      csym=csym, ssym=None)[0]
                qfib = num.hstack([qfib, qfibM])
            # cull out duplicate orientations
            qfib = mUtil.uniqueVectors(qfib, tol=1e-4)
            numTrials = qfib.shape[1]
            """
            THIS IS THE BIGGIE; THE LOOP OVER THE DISCRETE ORIENTATIONS IN THE
            CURRENT FIBER
            """
            if multiProcMode:
                foundFlagShared.value = False
                qfibList = list(map(num.array, qfib.T.tolist()))
                #if debugMultiproc:
                #    print 'qfibList : '+str(qfibList)
                results = num.array(pool.map(testThisQ, qfibList, chunksize=1))
                trialGrains = results[num.where(num.array(results, dtype=bool))]
                # for trialGrain in trialGrains:
                #     trialGrain.restore(candidate)
            else:
                trialGrains = []
                for iR in range(numTrials):
                    foundGrainData = testThisQ(qfib[:, iR])
                    if foundGrainData is not None:
                        trialGrains.append(foundGrainData)
                        break
            'end of if multiProcMode'

            if len(trialGrains) == 0:
                print("No grain found containing spot %d\n" % (iSpot))
                # import pdb;pdb.set_trace()
            else:
                asMaster = multiProcMode
                'sort based on completeness'
                trialGrainCompletenesses = [
                    tgd['completeness'] for tgd in trialGrains
                    ]
                order = num.argsort(trialGrainCompletenesses)[-1::-1]
                for iTrialGrain in order:
                    foundGrainData = trialGrains[iTrialGrain]
                    foundGrain = Grain(
                        spotsArray,
                        grainData=foundGrainData,
                        claimingSpots=False
                        )
                    #check completeness before accepting
                    #especially important for multiproc
                    foundGrain.checkClaims() # updates completeness
                    if debugMultiproc:
                        print('final completeness of candidate is %g' \
                              % (foundGrain.completeness))
                    if foundGrain.completeness >= minCompleteness:
                        conflicts = foundGrain.claimSpots(asMaster=asMaster)
                        numConfl = num.sum(conflicts)
                        if numConfl > 0:
                            print('tried to claim %d spots that are already '\
                                  'claimed' % (numConfl))
                        grainList.append(foundGrain)
                        nGrains += 1
                numUnClaimed = num.sum(-spotsArray.checkClaims())
                numClaimed = numTotal - numUnClaimed
                pctClaimed = num.float(numClaimed) / numTotal
                print("Found %d grains so far, %f%% claimed" \
                      % (nGrains,100*pctClaimed))

                time_to_quit = (pctClaimed > minPctClaimed) or\
                  ((quit_after_ngrains > 0) and (nGrains >= quit_after_ngrains))
                if time_to_quit:
                    break
        'end of iRefl loop'

        if time_to_quit:
            break

    'end of iHKL loop'
    rMats = num.empty((len(grainList), 3, 3))
    for i in range(len(grainList)):
        rMats[i, :, :] = grainList[i].rMat

    if outputGrainList:
        retval = (rMats, grainList)
    else:
        retval = rMats

    if not preserveClaims:
        spotsArray.resetClaims()
    toc = time.time()
    print('fiberSearch execution took %g seconds' % (toc-tic))

    if multiProcMode:
        pool.close()
        spotsArray.multiprocMode = False
        foundFlagShared.value = False
    # global foundFlagShared
    # global multiProcMode_MP
    # global spotsArray_MP
    # global candidate_MP
    # global dspTol_MP
    # global minCompleteness_MP
    # global doRefinement_MP
    multiProcMode_MP = None
    spotsArray_MP = None
    candidate_MP = None
    dspTol_MP = None
    minCompleteness_MP = None
    doRefinement_MP = None

    return retval

def pgRefine(x, etaOmeMaps, omegaRange, threshold):
    phi = sum(x*x)
    if phi < 1e-7:
        q = [num.r_[1.,0.,0.,0.],]
    else:
        phi = num.sqrt(phi)
        n = (1. / phi) * x.flatten()
        cphi2 = num.cos(0.5*phi)
        sphi2 = num.sin(0.5*phi)
        q = [num.r_[cphi2, sphi2*n[0], sphi2*n[1], sphi2*n[2]],]
    c = paintGrid(
        q, etaOmeMaps, threshold=threshold, bMat=None, omegaRange=omegaRange,
        etaRange=None, debug=False
        )
    f = abs(1. - c)
    return f

paramMP = None
def paintGrid(quats, etaOmeMaps,
              threshold=None, bMat=None,
              omegaRange=None, etaRange=None,
              omeTol=d2r, etaTol=d2r,
              omePeriod=(-num.pi, num.pi),
              doMultiProc=False,
              nCPUs=None, debug=False):
    """
    do a direct search of omega-eta maps to paint each orientation in
    quats with a completeness

    bMat is in CRYSTAL frame

    etaOmeMaps is instance of xrd.xrdutil.CollapseOmeEta

    omegaRange=([-num.pi/3., num.pi/3.],) for example

    *) lifted mainly from grain.py

    *) self.etaGrid, self.omeGrid = num.meshgrid(self.etaEdges, self.omeEdges)
       this means that ETA VARIES FASTEST!

    ...make a new function that gets called by grain to do the g-vec angle
    computation?
    """

    quats = num.atleast_2d(quats)
    if quats.size == 4:
        quats = quats.reshape(4, 1)

    planeData = etaOmeMaps.planeData

    hklIDs    = num.r_[etaOmeMaps.iHKLList]
    hklList   = num.atleast_2d(planeData.hkls[:, hklIDs].T).tolist()
    nHKLS     = len(hklIDs)

    numEtas   = len(etaOmeMaps.etaEdges) - 1
    numOmes   = len(etaOmeMaps.omeEdges) - 1

    if threshold is None:
        threshold = num.zeros(nHKLS)
        for i in range(nHKLS):
            threshold[i] = num.mean(
                num.r_[
                    num.mean(etaOmeMaps.dataStore[i]),
                    num.median(etaOmeMaps.dataStore[i])
                    ]
                )
    elif threshold is not None and not hasattr(threshold, '__len__'):
        threshold = threshold * num.ones(nHKLS)
    elif hasattr(threshold, '__len__'):
        if len(threshold) != nHKLS:
            raise RuntimeError("threshold list is wrong length!")
        else:
            print("INFO: using list of threshold values")
    else:
        raise RuntimeError(
            "unknown threshold option. should be a list of numbers or None"
            )
    if bMat is None:
        bMat = planeData.latVecOps['B']

    """
    index munging here -- look away

    order of ome-eta map arrays is (i, j) --> (ome, eta)
    i.e. eta varies fastest.
    """
    # mapIndices = num.indices([numEtas, numOmes])
    # etaIndices = mapIndices[0].flatten()
    # omeIndices = mapIndices[1].T.flatten()
    # etaIndices = num.tile(range(numEtas), (numOmes))
    # omeIndices = num.tile(range(numOmes), (numEtas))
    # j_eta, i_ome = num.meshgrid(range(numEtas), range(numOmes))
    # etaIndices = j_eta.flatten()
    # omeIndices = i_ome.flatten()
    etaIndices = num.r_[list(range(numEtas))]
    omeIndices = num.r_[list(range(numOmes))]

    omeMin = None
    omeMax = None
    if omegaRange is None:              # this NEEDS TO BE FIXED!
        omeMin = [num.min(etaOmeMaps.omeEdges),]
        omeMax = [num.max(etaOmeMaps.omeEdges),]
    else:
        omeMin = [omegaRange[i][0] for i in range(len(omegaRange))]
        omeMax = [omegaRange[i][1] for i in range(len(omegaRange))]
    if omeMin is None:
        omeMin = [-num.pi, ]
        omeMax = [ num.pi, ]
    omeMin = num.asarray(omeMin)
    omeMax = num.asarray(omeMax)

    etaMin = None
    etaMax = None
    if etaRange is not None:
        etaMin = [etaRange[i][0] for i in range(len(etaRange))]
        etaMax = [etaRange[i][1] for i in range(len(etaRange))]
    if etaMin is None:
        etaMin = [-num.pi, ]
        etaMax = [ num.pi, ]
    etaMin = num.asarray(etaMin)
    etaMax = num.asarray(etaMax)

    multiProcMode = xrdbase.haveMultiProc and doMultiProc

    if multiProcMode:
        nCPUs = nCPUs or xrdbase.dfltNCPU
        chunksize = min(quats.shape[1] // nCPUs, 10)
        logger.info(
            "using multiprocessing with %d processes and a chunk size of %d",
            nCPUs, chunksize
            )
    else:
        logger.info("running in serial mode")
        nCPUs = 1

    # Get the symHKLs for the selected hklIDs
    symHKLs = planeData.getSymHKLs()
    symHKLs = [symHKLs[id] for id in hklIDs]
    # Restructure symHKLs into a flat NumPy HKL array with
    # each HKL stored contiguously (C-order instead of F-order)
    # symHKLs_ix provides the start/end index for each subarray
    # of symHKLs.
    symHKLs_ix = num.add.accumulate([0] + [s.shape[1] for s in symHKLs])
    symHKLs = num.vstack(s.T for s in symHKLs)

    # Pack together the common parameters for processing
    params = {
        'symHKLs': symHKLs,
        'symHKLs_ix': symHKLs_ix,
        'wavelength': planeData.wavelength,
        'hklList': hklList,
        'omeMin': omeMin,
        'omeMax': omeMax,
        'omeTol': omeTol,
        'omeIndices': omeIndices,
        'omePeriod': omePeriod,
        'omeEdges': etaOmeMaps.omeEdges,
        'etaMin': etaMin,
        'etaMax': etaMax,
        'etaTol': etaTol,
        'etaIndices': etaIndices,
        'etaEdges': etaOmeMaps.etaEdges,
        'etaOmeMaps': num.stack(etaOmeMaps.dataStore),
        'bMat': bMat,
        'threshold': threshold
        }

    # do the mapping
    start = time.time()
    retval = None
    if multiProcMode:
        # multiple process version
        pool = multiprocessing.Pool(nCPUs, paintgrid_init, (params, ))
        retval = pool.map(paintGridThis, quats.T, chunksize=chunksize)
        pool.close()
    else:
        # single process version.
        global paramMP
        paintgrid_init(params)    # sets paramMP
        retval = list(map(paintGridThis, quats.T))
        paramMP = None    # clear paramMP
    elapsed = (time.time() - start)
    logger.info("paintGrid took %.3f seconds", elapsed)

    return retval


def _meshgrid2d(x, y):
    """
    A special-cased implementation of num.meshgrid, for just
    two arguments. Found to be about 3x faster on some simple
    test arguments.
    """
    x, y = (num.asarray(x), num.asarray(y))
    shape = (len(y), len(x))
    dt = num.result_type(x, y)
    r1, r2 = (num.empty(shape, dt), num.empty(shape, dt))
    r1[...] = x[num.newaxis, :]
    r2[...] = y[:, num.newaxis]
    return (r1, r2)



def _normalize_ranges(starts, stops, offset, ccw=False):
    """normalize in the range [offset, 2*pi+offset[ the ranges defined
    by starts and stops.

    Checking if an angle lies inside a range can be done in a way that
    is more efficient than using validateAngleRanges.

    Note this function assumes that ranges don't overlap.
    """
    if ccw:
        starts, stops = stops, starts

    # results are in the range of [0, 2*num.pi]
    if not num.all(starts < stops):
        raise ValueError('Invalid angle ranges')


    # If there is a range that spans more than 2*pi,
    # return the full range
    two_pi = 2 * num.pi
    if num.any((starts + two_pi) < stops + 1e-8):
        return num.array([offset, two_pi+offset])

    starts = num.mod(starts - offset, two_pi) + offset
    stops = num.mod(stops - offset, two_pi) + offset

    order = num.argsort(starts)
    result = num.hstack((starts[order, num.newaxis],
                        stops[order, num.newaxis])).ravel()
    # at this point, result is in its final form unless there
    # is wrap-around in the last segment. Handle this case:
    if result[-1] < result[-2]:
        new_result = num.empty((len(result)+2,), dtype=result.dtype)
        new_result[0] = offset
        new_result[1] = result[-1]
        new_result[2:-1] = result[0:-1]
        new_result[-1] = offset + two_pi
        result = new_result

    if not num.all(starts[1:] > stops[0:-2]):
        raise ValueError('Angle ranges overlap')

    return result


def paintgrid_init(params):
    global paramMP
    paramMP = params

    # create valid_eta_spans, valid_ome_spans from etaMin/Max and omeMin/Max
    # this allows using faster checks in the code.
    # TODO: build valid_eta_spans and valid_ome_spans directly in paintGrid
    #       instead of building etaMin/etaMax and omeMin/omeMax. It may also
    #       be worth handling range overlap and maybe "optimize" ranges if
    #       there happens to be contiguous spans.
    paramMP['valid_eta_spans'] = _normalize_ranges(paramMP['etaMin'],
                                                   paramMP['etaMax'],
                                                   -num.pi)

    paramMP['valid_ome_spans'] = _normalize_ranges(paramMP['omeMin'],
                                                   paramMP['omeMax'],
                                                   min(paramMP['omePeriod']))
    return


###############################################################################
#
# paintGridThis contains the bulk of the process to perform for paintGrid for a
# given quaternion. This is also used as the basis for multiprocessing, as the
# work is split in a per-quaternion basis among different processes.
# The remainding arguments are marshalled into the module variable "paramMP".
#
# There is a version of PaintGridThis using numba, and another version used
# when numba is not available. The numba version should be noticeably faster.

def _check_dilated(eta, ome, dpix_eta, dpix_ome, etaOmeMap, threshold):
    """This is part of paintGridThis:

    check if there exists a sample over the given threshold in the etaOmeMap
    at (eta, ome), with a tolerance of (dpix_eta, dpix_ome) samples.

    Note this function is "numba friendly" and will be jitted when using numba.

    TODO: currently behaves like "num.any" call for values above threshold.
    There is some ambigutiy if there are NaNs in the dilation range, but it
    hits a value above threshold first.  Is that ok???

    FIXME: works in non-numba implementation of paintGridThis only
    <JVB 2017-04-27>
    """
    i_max, j_max = etaOmeMap.shape
    ome_start, ome_stop = (
        max(ome - dpix_ome, 0),
        min(ome + dpix_ome + 1, i_max)
    )
    eta_start, eta_stop = (
        max(eta - dpix_eta, 0),
        min(eta + dpix_eta + 1, j_max)
    )

    for i in range(ome_start, ome_stop):
        for j in range(eta_start, eta_stop):
            if etaOmeMap[i, j] > threshold:
                return 1
            if num.isnan(etaOmeMap[i, j]):
                return -1
    return 0


if USE_NUMBA:
    def paintGridThis(quat):
        # Note that this version does not use omeMin/omeMax to specify the valid
        # angles. It uses "valid_eta_spans" and "valid_ome_spans". These are
        # precomputed and make for a faster check of ranges than
        # "validateAngleRanges"
        symHKLs = paramMP['symHKLs'] # the HKLs
        symHKLs_ix = paramMP['symHKLs_ix'] # index partitioning of symHKLs
        bMat = paramMP['bMat']
        wavelength = paramMP['wavelength']
        omeEdges = paramMP['omeEdges']
        omeTol = paramMP['omeTol']
        omePeriod = paramMP['omePeriod']
        valid_eta_spans = paramMP['valid_eta_spans']
        valid_ome_spans = paramMP['valid_ome_spans']
        omeIndices = paramMP['omeIndices']
        etaEdges = paramMP['etaEdges']
        etaTol = paramMP['etaTol']
        etaIndices = paramMP['etaIndices']
        etaOmeMaps = paramMP['etaOmeMaps']
        threshold = paramMP['threshold']

        # dpix_ome and dpix_eta are the number of pixels for the tolerance in
        # ome/eta. Maybe we should compute this per run instead of per
        # quaternion
        del_ome = abs(omeEdges[1] - omeEdges[0])
        del_eta = abs(etaEdges[1] - etaEdges[0])
        dpix_ome = int(round(omeTol / del_ome))
        dpix_eta = int(round(etaTol / del_eta))

        debug = False
        if debug:
            print(( "using ome, eta dilitations of (%d, %d) pixels" \
                  % (dpix_ome, dpix_eta)))

        # get the equivalent rotation of the quaternion in matrix form (as
        # expected by oscillAnglesOfHKLs

        rMat = xfcapi.makeRotMatOfQuat(quat)

        # Compute the oscillation angles of all the symHKLs at once
        oangs_pair = xfcapi.oscillAnglesOfHKLs(symHKLs, 0., rMat, bMat,
                                               wavelength)
        # pdb.set_trace()
        return _filter_and_count_hits(oangs_pair[0], oangs_pair[1], symHKLs_ix,
                                      etaEdges, valid_eta_spans,
                                      valid_ome_spans, omeEdges, omePeriod,
                                      etaOmeMaps, etaIndices, omeIndices,
                                      dpix_eta, dpix_ome, threshold)


    @numba.jit
    def _find_in_range(value, spans):
        """find the index in spans where value >= spans[i] and value < spans[i].

        spans is an ordered array where spans[i] <= spans[i+1] (most often <
        will hold).

        If value is not in the range [spans[0], spans[-1][, then -2 is returned.

        This is equivalent to "bisect_right" in the bisect package, in which
        code it is based, and it is somewhat similar to NumPy's searchsorted,
        but non-vectorized

        """

        if value < spans[0] or value >= spans[-1]:
            return -2

        # from the previous check, we know 0 is not a possible result
        li = 0
        ri = len(spans)

        while li < ri:
            mi = (li + ri) // 2
            if value < spans[mi]:
                ri = mi
            else:
                li = mi+1

        return li

    @numba.njit
    def _angle_is_hit(ang, eta_offset, ome_offset, hkl, valid_eta_spans,
                      valid_ome_spans, etaEdges, omeEdges, etaOmeMaps,
                      etaIndices, omeIndices, dpix_eta, dpix_ome, threshold):
        """perform work on one of the angles.

        This includes:

        - filtering nan values

        - filtering out angles not in the specified spans

        - checking that the discretized angle fits into the sensor range (maybe
          this could be merged with the previous test somehow, for extra speed)

        - actual check for a hit, using dilation for the tolerance.

        Note the function returns both, if it was a hit and if it passed the
        filtering, as we'll want to discard the filtered values when computing
        the hit percentage.

        CAVEAT: added map-based nan filtering to _check_dilated; this may not
        be the best option.  Perhaps filter here? <JVB 2017-04-27>

        """
        tth, eta, ome = ang

        if num.isnan(tth):
            return 0, 0

        eta = _map_angle(eta, eta_offset)
        if _find_in_range(eta, valid_eta_spans) & 1 == 0:
            # index is even: out of valid eta spans
            return 0, 0

        ome = _map_angle(ome, ome_offset)
        if _find_in_range(ome, valid_ome_spans) & 1 == 0:
            # index is even: out of valid ome spans
            return 0, 0

        # discretize the angles
        eta_idx = _find_in_range(eta, etaEdges) - 1
        if eta_idx < 0:
            # out of range
            return 0, 0

        ome_idx = _find_in_range(ome, omeEdges) - 1
        if ome_idx < 0:
            # out of range
            return 0, 0

        eta = etaIndices[eta_idx]
        ome = omeIndices[ome_idx]
        isHit = _check_dilated(eta, ome, dpix_eta, dpix_ome,
                               etaOmeMaps[hkl], threshold[hkl])
        if isHit == -1:
            return 0, 0
        else:
            return isHit, 1

    @numba.njit
    def _filter_and_count_hits(angs_0, angs_1, symHKLs_ix, etaEdges,
                               valid_eta_spans, valid_ome_spans, omeEdges,
                               omePeriod, etaOmeMaps, etaIndices, omeIndices,
                               dpix_eta, dpix_ome, threshold):
        """assumes:
        we want etas in -pi -> pi range
        we want omes in ome_offset -> ome_offset + 2*pi range

        Instead of creating an array with the angles of angs_0 and angs_1
        interleaved, in this numba version calls for both arrays are performed
        getting the angles from angs_0 and angs_1. this is done in this way to
        reuse hkl computation. This may not be that important, though.

        """
        eta_offset = -num.pi
        ome_offset = num.min(omePeriod)
        hits = 0
        total = 0
        curr_hkl_idx = 0
        end_curr = symHKLs_ix[1]
        count = len(angs_0)

        for i in range(count):
            if i >= end_curr:
                curr_hkl_idx += 1
                end_curr = symHKLs_ix[curr_hkl_idx+1]

            # first solution
            hit, not_filtered = _angle_is_hit(
                angs_0[i], eta_offset, ome_offset,
                curr_hkl_idx, valid_eta_spans,
                valid_ome_spans, etaEdges,
                omeEdges, etaOmeMaps, etaIndices,
                omeIndices, dpix_eta, dpix_ome,
                threshold)
            hits += hit
            total += not_filtered

            # second solution
            hit, not_filtered = _angle_is_hit(
                angs_1[i], eta_offset, ome_offset,
                curr_hkl_idx, valid_eta_spans,
                valid_ome_spans, etaEdges,
                omeEdges, etaOmeMaps, etaIndices,
                omeIndices, dpix_eta, dpix_ome,
                threshold)
            hits += hit
            total += not_filtered

        return float(hits)/float(total) if total != 0 else 0.0

    @numba.njit
    def _map_angle(angle, offset):
        """Equivalent to xf.mapAngle in this context, and 'numba friendly'

        """
        return num.mod(angle-offset, 2*num.pi)+offset

    # use a jitted version of _check_dilated
    _check_dilated = numba.njit(_check_dilated)
else:
    def paintGridThis(quat):
        # unmarshall parameters into local variables
        symHKLs = paramMP['symHKLs'] # the HKLs
        symHKLs_ix = paramMP['symHKLs_ix'] # index partitioning of symHKLs
        bMat = paramMP['bMat']
        wavelength = paramMP['wavelength']
        omeEdges = paramMP['omeEdges']
        omeTol = paramMP['omeTol']
        omePeriod = paramMP['omePeriod']
        valid_eta_spans = paramMP['valid_eta_spans']
        valid_ome_spans = paramMP['valid_ome_spans']
        omeIndices = paramMP['omeIndices']
        etaEdges = paramMP['etaEdges']
        etaTol = paramMP['etaTol']
        etaIndices = paramMP['etaIndices']
        etaOmeMaps = paramMP['etaOmeMaps']
        threshold = paramMP['threshold']

        # dpix_ome and dpix_eta are the number of pixels for the tolerance in
        # ome/eta. Maybe we should compute this per run instead of
        # per-quaternion
        del_ome = abs(omeEdges[1] - omeEdges[0])
        del_eta = abs(etaEdges[1] - etaEdges[0])
        dpix_ome = int(round(omeTol / del_ome))
        dpix_eta = int(round(etaTol / del_eta))

        debug = False
        if debug:
            print(( "using ome, eta dilitations of (%d, %d) pixels" \
                  % (dpix_ome, dpix_eta)))

        # get the equivalent rotation of the quaternion in matrix form (as
        # expected by oscillAnglesOfHKLs

        rMat = xfcapi.makeRotMatOfQuat(quat)

        # Compute the oscillation angles of all the symHKLs at once
        oangs_pair = xfcapi.oscillAnglesOfHKLs(symHKLs, 0., rMat, bMat,
                                               wavelength)
        hkl_idx, eta_idx, ome_idx = _filter_angs(oangs_pair[0], oangs_pair[1],
                                                 symHKLs_ix, etaEdges,
                                                 valid_eta_spans, omeEdges,
                                                 valid_ome_spans, omePeriod)

        if len(hkl_idx > 0):
            hits, predicted = _count_hits(eta_idx, ome_idx, hkl_idx, etaOmeMaps,
                               etaIndices, omeIndices, dpix_eta, dpix_ome,
                               threshold)
            retval = float(hits) / float(predicted)
            if retval > 1:
                import pdb; pdb.set_trace()
        return retval

    def _normalize_angs_hkls(angs_0, angs_1, omePeriod, symHKLs_ix):
        # Interleave the two produced oang solutions to simplify later
        # processing
        oangs = num.empty((len(angs_0)*2, 3), dtype=angs_0.dtype)
        oangs[0::2] = angs_0
        oangs[1::2] = angs_1

        # Map all of the angles at once
        oangs[:, 1] = xf.mapAngle(oangs[:, 1])
        oangs[:, 2] = xf.mapAngle(oangs[:, 2], omePeriod)

        # generate array of symHKLs indices
        symHKLs_ix = symHKLs_ix*2
        hkl_idx = num.empty((symHKLs_ix[-1],), dtype=int)
        start = symHKLs_ix[0]
        idx=0
        for end in symHKLs_ix[1:]:
            hkl_idx[start:end] = idx
            start = end
            idx+=1

        return oangs, hkl_idx


    def _filter_angs(angs_0, angs_1, symHKLs_ix, etaEdges, valid_eta_spans,
                     omeEdges, valid_ome_spans, omePeriod):
        """
        This is part of paintGridThis:

        bakes data in a way that invalid (nan or out-of-bound) is discarded.
        returns:
          - hkl_idx, array of associated hkl indices
          - eta_idx, array of associated eta indices of predicted
          - ome_idx, array of associated ome indices of predicted
        """
        oangs, hkl_idx = _normalize_angs_hkls(angs_0, angs_1, omePeriod,
                                              symHKLs_ix)
        # using "right" side to make sure we always get an index *past* the value
        # if it happens to be equal. That is... we search the index of the first
        # value that is "greater than" rather than "greater or equal"
        culled_eta_indices = num.searchsorted(etaEdges, oangs[:, 1],
                                              side='right')
        culled_ome_indices = num.searchsorted(omeEdges, oangs[:, 2],
                                              side='right')
        # this check is equivalent to validateAngleRanges:
        #
        # The spans contains an ordered sucession of start and end angles which
        # form the valid angle spans. So knowing if an angle is valid is
        # equivalent to finding the insertion point in the spans array and
        # checking if the resulting insertion index is odd or even. An odd value
        # means that it falls between a start and a end point of the "valid
        # span", meaning it is a hit. An even value will result in either being
        # out of the range (0 or the last index, as length is even by
        # construction) or that it falls between a "end" point from one span and
        # the "start" point of the next one.
        valid_eta = num.searchsorted(valid_eta_spans, oangs[:, 1], side='right')
        valid_ome = num.searchsorted(valid_ome_spans, oangs[:, 2], side='right')
        # fast odd/even check
        valid_eta = valid_eta & 1
        valid_ome = valid_ome & 1
        # Create a mask of the good ones
        valid = ~num.isnan(oangs[:, 0]) # tth not NaN
        valid = num.logical_and(valid, valid_eta)
        valid = num.logical_and(valid, valid_ome)
        valid = num.logical_and(valid, culled_eta_indices > 0)
        valid = num.logical_and(valid, culled_eta_indices < len(etaEdges))
        valid = num.logical_and(valid, culled_ome_indices > 0)
        valid = num.logical_and(valid, culled_ome_indices < len(omeEdges))

        hkl_idx = hkl_idx[valid]
        eta_idx = culled_eta_indices[valid] - 1
        ome_idx = culled_ome_indices[valid] - 1

        return hkl_idx, eta_idx, ome_idx


    def _count_hits(eta_idx, ome_idx, hkl_idx, etaOmeMaps,
                    etaIndices, omeIndices, dpix_eta, dpix_ome, threshold):
        """
        This is part of paintGridThis:

        for every eta, ome, hkl check if there is a sample that surpasses the
        threshold in the eta ome map.
        """
        predicted = len(hkl_idx)
        hits = 0

        for curr_ang in range(predicted):
            culledEtaIdx = eta_idx[curr_ang]
            culledOmeIdx = ome_idx[curr_ang]
            iHKL = hkl_idx[curr_ang]
            # got a result
            eta = etaIndices[culledEtaIdx]
            ome = omeIndices[culledOmeIdx]
            isHit = _check_dilated(eta, ome, dpix_eta, dpix_ome,
                                   etaOmeMaps[iHKL], threshold[iHKL])

            if isHit > 0:
                hits += 1
            if isHit == -1:
                predicted -= 1

        return hits, predicted


def writeGVE(spotsArray, fileroot, **kwargs):
    """
    write Fable gve file from Spots object

    fileroot is the root string used to write the gve and ini files

    Outputs:

    No return value, but writes the following files:

    <fileroot>.gve
    <fileroot>_grainSpotter.ini (points to --> <fileroot>_grainSpotter.log)

    Keyword arguments:

    Mainly for GrainSpotter .ini file, but some are needed for gve files

    keyword        default              definitions
    ----------------------------------------------------------------------------
    'sgNum':       <225>
    'phaseID':     <None>
    'cellString':  <F>
    'omeRange':    <-60, 60, 120, 240>  the oscillation range(s)
                                        **currently pulls from spots
    'deltaOme':    <0.25, 0.25>         the oscillation delta(s)
                                        **currently pulls from spots
    'minMeas':     <24>
    'minCompl':    <0.7>
    'minUniqn':    <0.5>
    'uncertainty': <[0.10, 0.25, .50]>  the min [tTh, eta, ome] uncertainties
                                        in degrees
    'eulStep':     <2>
    'nSigmas':     <2>
    'minFracG':    <0.90>
    'nTrials':     <100000>
    'positionfit': <True>

    Notes:

    *) The omeRange is currently pulled from the spotsArray input; the kwarg
       has no effect as of now.  Will change this to 'override' the spots info
       if the user, say, wants to pare down the range.

    *) There is no etaRange argument yet, but presumably GrainSpotter knows
       how to deal with this.  Pending feature...
    """
    # check on fileroot
    assert isinstance(fileroot, str)

    # keyword argument processing
    phaseID     = None
    sgNum       = 225
    cellString  = 'P'
    omeRange    = num.r_[-60, 60]   # in DEGREES
    deltaOme    = 0.25              # in DEGREES
    minMeas     = 24
    minCompl    = 0.7
    minUniqn    = 0.5
    uncertainty = [0.10, 0.25, .50] # in DEGREES
    eulStep     = 2                 # in DEGREES
    nSigmas     = 2
    minFracG    = 0.90
    numTrials   = 100000
    positionFit = True

    kwarglen = len(kwargs)
    if kwarglen > 0:
        argkeys = list(kwargs.keys())
        # TODO: NO
        for i in range(kwarglen):
            if argkeys[i] == 'sgNum':
                sgNum = kwargs[argkeys[i]]
            elif argkeys[i] == 'phaseID':
                phaseID = kwargs[argkeys[i]]
            elif argkeys[i] == 'cellString':
                cellString = kwargs[argkeys[i]]
            elif argkeys[i] == 'omeRange':
                omeRange = kwargs[argkeys[i]]
            elif argkeys[i] == 'deltaOme':
                deltaOme = kwargs[argkeys[i]]
            elif argkeys[i] == 'minMeas':
                minMeas = kwargs[argkeys[i]]
            elif argkeys[i] == 'minCompl':
                minCompl = kwargs[argkeys[i]]
            elif argkeys[i] == 'minUniqn':
                minUniqn = kwargs[argkeys[i]]
            elif argkeys[i] == 'uncertainty':
                uncertainty = kwargs[argkeys[i]]
            elif argkeys[i] == 'eulStep':
                eulStep = kwargs[argkeys[i]]
            elif argkeys[i] == 'nSigmas':
                nSigmas = kwargs[argkeys[i]]
            elif argkeys[i] == 'minFracG':
                minFracG = kwargs[argkeys[i]]
            elif argkeys[i] == 'nTrials':
                numTrials = kwargs[argkeys[i]]
            elif argkeys[i] == 'positionfit':
                positionFit = kwargs[argkeys[i]]
            else:
                raise RuntimeError(
                    "Unrecognized keyword argument '%s'" % (argkeys[i])
                    )

    # grab some detector geometry parameters for gve file header
    # ...these are still hard-coded to be square
    mmPerPixel = float(spotsArray.detectorGeom.pixelPitch)
    nrows_p = spotsArray.detectorGeom.nrows - 1
    ncols_p = spotsArray.detectorGeom.ncols - 1

    row_p, col_p = spotsArray.detectorGeom.pixelIndicesOfCartesianCoords(
        spotsArray.detectorGeom.xc, spotsArray.detectorGeom.yc
        )
    yc_p = ncols_p - col_p
    zc_p = nrows_p - row_p

    wd_mu = spotsArray.detectorGeom.workDist * 1e3 # in microns (Soeren)

    osc_axis = num.dot(fableSampCOB.T, Yl).flatten()

    # start grabbing stuff from planeData
    planeData = spotsArray.getPlaneData(phaseID=phaseID)
    cellp   = planeData.latVecOps['dparms']
    U0      = planeData.latVecOps['U0']
    wlen    = planeData.wavelength
    dsp     = planeData.getPlaneSpacings()
    fHKLs   = planeData.getSymHKLs()
    tThRng  = planeData.getTThRanges()
    symTag  = planeData.getLaueGroup()

    # single range should be ok since entering hkls
    tThMin, tThMax = (r2d*tThRng.min(), r2d*tThRng.max())
    # not sure when this will ever *NOT* be the case, so setting it
    etaMin, etaMax = (0, 360)

    omeMin = spotsArray.getOmegaMins()
    omeMax = spotsArray.getOmegaMaxs()

    omeRangeString = ''
    for iOme in range(len(omeMin)):
        if hasattr(omeMin[iOme], 'getVal'):
            omeRangeString += 'omegarange %g %g\n' % (
                omeMin[iOme].getVal('degrees'), omeMax[iOme].getVal('degrees')
                )
        else:
            omeRangeString += 'omegarange %g %g\n' % (
                omeMin[iOme] * r2d, omeMax[iOme] * r2d
                )

    # convert angles
    cellp[3:] = r2d*cellp[3:]

    # make the theoretical hkls string
    gvecHKLString = ''
    for i in range(len(dsp)):
        for j in range(fHKLs[i].shape[1]):
            gvecHKLString += '%1.8f %d %d %d\n' % (
                1/dsp[i], fHKLs[i][0, j], fHKLs[i][1, j], fHKLs[i][2, j]
                )

    # now for the measured data section
    # xr yr zr xc yc ds eta omega
    gvecString = ''
    spotsIter = spotsArray.getIterPhase(phaseID, returnBothCoordTypes=True)
    for iSpot, angCOM, xyoCOM in spotsIter:
        sR, sC, sOme     = xyoCOM # detector coords
        sTTh, sEta, sOme = angCOM # angular coords (radians)
        sDsp = wlen / 2. / num.sin(0.5*sTTh) # dspacing

        # get raw y, z (Fable frame)
        yraw = ncols_p - sC
        zraw = nrows_p - sR

        # convert eta to fable frame
        rEta = mapAngle(90. - r2d*sEta, [0, 360], units='degrees')

        # make mesaured G vector components in fable frame
        mGvec = makeMeasuredScatteringVectors(
            sTTh, sEta, sOme, convention='fable', frame='sample'
            )

        # full Gvec components in fable lab frame
        # (for grainspotter position fit)
        gveXYZ = spotsArray.detectorGeom.angToXYO(
            sTTh, sEta, sOme, outputGve=True
            )

        # no 4*pi
        mGvec = mGvec / sDsp

        # make contribution
        gvecString += ('%1.8f ' * 8) + '%d %1.8f %1.8f %1.8f\n' % (
            mGvec[0], mGvec[1], mGvec[2], sR, sC, 1/sDsp, rEta, r2d*sOme,
            iSpot, gveXYZ[0, :], gveXYZ[1, :], gveXYZ[2, :]
            )

    # write gve file for grainspotter
    fid = open(fileroot+'.gve', 'w')
    print('%1.8f %1.8f %1.8f %1.8f %1.8f %1.8f ' % tuple(cellp) \
          +  cellString + '\n' \
          + '# wavelength = %1.8f\n' % (wlen) \
          + '# wedge = 0.000000\n' \
          + '# axis = %d %d %d\n' % tuple(osc_axis) \
          + '# cell__a %1.4f\n' %(cellp[0]) \
          + '# cell__b %1.4f\n' %(cellp[1]) \
          + '# cell__c %1.4f\n' %(cellp[2]) \
          + '# cell_alpha %1.4f\n' %(cellp[3]) \
          + '# cell_beta  %1.4f\n' %(cellp[4]) \
          + '# cell_gamma %1.4f\n' %(cellp[5]) \
          + '# cell_lattice_[P,A,B,C,I,F,R] %s\n' %(cellString) \
          + '# chi 0.0\n' \
          + '# distance %.4f\n' %(wd_mu) \
          + '# fit_tolerance 0.5\n' \
          + '# o11  1\n' \
          + '# o12  0\n' \
          + '# o21  0\n' \
          + '# o22 -1\n' \
          + '# omegasign %1.1f\n' %(num.sign(deltaOme)) \
          + '# t_x 0\n' \
          + '# t_y 0\n' \
          + '# t_z 0\n' \
          + '# tilt_x 0.000000\n' \
          + '# tilt_y 0.000000\n' \
          + '# tilt_z 0.000000\n' \
          + '# y_center %.6f\n' %(yc_p) \
          + '# y_size %.6f\n' %(mmPerPixel*1.e3) \
          + '# z_center %.6f\n' %(zc_p) \
          + '# z_size %.6f\n' %(mmPerPixel*1.e3) \
          + '# ds h k l\n' \
          + gvecHKLString \
          + '# xr yr zr xc yc ds eta omega\n' \
          + gvecString, file=fid)
    fid.close()

    ###############################################################
    # GrainSpotter ini parameters
    #
    # fileroot = tempfile.mktemp()
    if positionFit:
        positionString = 'positionfit'
    else:
        positionString = '!positionfit'

    if numTrials == 0:
        randomString = '!random\n'
    else:
        randomString = 'random %g\n' % (numTrials)

    fid = open(fileroot+'_grainSpotter.ini', 'w')
    # self.__tempFNameList.append(fileroot)
    print('spacegroup %d\n' % (sgNum) \
          + 'tthrange %g %g\n' % (tThMin, tThMax) \
          + 'etarange %g %g\n' % (etaMin, etaMax) \
          + 'domega %g\n' % (deltaOme) \
          + omeRangeString + \
          + 'filespecs %s.gve %s_grainSpotter.log\n' % (fileroot, fileroot) \
          + 'cuts %d %g %g\n' % (minMeas, minCompl, minUniqn) \
          + 'eulerstep %g\n' % (eulStep) \
          + 'uncertainties %g %g %g\n' \
               % (uncertainty[0], uncertainty[1], uncertainty[2]) \
          + 'nsigmas %d\n' % (nSigmas) \
          + 'minfracg %g\n' % (minFracG) \
          + randomString \
          + positionString + '\n', file=fid)
    fid.close()
    return
