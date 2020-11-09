import importlib.resources
import numpy as np
from hexrd import constants
from hexrd import symmetry, symbols
import hexrd.resources
import warnings
import h5py
from pathlib import Path
from scipy.interpolate import interp1d
import time

class unitcell:

    '''
    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       10/09/2018 SS 1.0 original
       @DATE:       10/15/2018 SS 1.1 added space group handling
    >> @DETAILS:    this is the unitcell class

    '''

    # initialize the unitcell class
    # need lattice parameters and space group data from HDF5 file
    def __init__(self, lp, sgnum, atomtypes, atominfo, U, dmin, beamenergy, sgsetting=0):

        self._tstart = time.time()
        self.pref  = 0.4178214

        self.atom_ntype = atomtypes.shape[0]
        self.atom_type  = atomtypes
        self.atom_pos   = atominfo
        self.U          = U

        self._dmin      = dmin

        # set some default values for
        # a,b,c,alpha,beta,gamma
        self._a = 1.
        self._b = 1.
        self._c = 1.
        self._alpha = 90.
        self._beta = 90.
        self._gamma = 90.

        if(lp[0].unit == 'angstrom'):
            self.a = lp[0].value * 0.1
        elif(lp[0].unit == 'nm'):
            self.a = lp[0].value
        else:
            raise ValueError('unknown unit in lattice parameter')

        if(lp[1].unit == 'angstrom'):
            self.b = lp[1].value * 0.1
        elif(lp[1].unit == 'nm'):
            self.b = lp[1].value
        else:
            raise ValueError('unknown unit in lattice parameter')

        if(lp[2].unit == 'angstrom'):
            self.c = lp[2].value * 0.1
        elif(lp[2].unit == 'nm'):
            self.c = lp[2].value
        else:
            raise ValueError('unknown unit in lattice parameter')

        if(lp[3].unit == 'degrees'):
            self.alpha = lp[3].value
        elif(lp[3].unit == 'radians'):
            self.alpha = np.degrees(lp[3].value)

        if(lp[4].unit == 'degrees'):
            self.beta = lp[4].value
        elif(lp[4].unit == 'radians'):
            self.beta = np.degrees(lp[4].value)

        if(lp[5].unit == 'degrees'):
            self.gamma = lp[5].value
        elif(lp[5].unit == 'radians'):
            self.gamma = np.degrees(lp[5].value)

        self.aniU       = False
        if(self.U.ndim > 1):
            self.aniU = True
            self.calcBetaij()

        '''
        initialize interpolation from table for anomalous scattering
        '''
        self.InitializeInterpTable()

        '''
        sets x-ray energy
        calculate wavelength
        also calculates anomalous form factors for xray scattering
        '''
        self.voltage = beamenergy * 1000.0
        '''
        calculate symmetry
        '''
        self.sgsetting = sgsetting
        self.sgnum = sgnum

        self._tstop = time.time()
        self.tinit = self._tstop - self._tstart

    def GetPgLg(self):
        '''
        simple subroutine to get point and laue groups
        to maintain consistency for planedata initialization
        in the materials class
        '''
        for k in list(_pgDict.keys()):
            if self.sgnum in k:
                pglg = _pgDict[k]
                self._pointGroup = pglg[0]
                self._laueGroup = pglg[1]

    def CalcWavelength(self):
        # wavelength in nm
        self.wavelength =       constants.cPlanck * \
                            constants.cLight /  \
                            constants.cCharge / \
                            self.voltage
        self.wavelength *= 1e9
        self.CalcAnomalous()


    def calcBetaij(self):

        self.betaij = np.zeros([self.atom_ntype,3,3])
        for i in range(self.U.shape[0]):
            U = self.U[i,:]
            self.betaij[i,:,:] = np.array([[U[0], U[3], U[4]],\
                                         [U[3], U[1], U[5]],\
                                         [U[4], U[5], U[2]]])

            self.betaij[i,:,:] *= 2. * np.pi**2 * self._aij

    def calcmatrices(self):

        a = self.a
        b = self.b
        c = self.c

        alpha = np.radians(self.alpha)
        beta  = np.radians(self.beta)
        gamma = np.radians(self.gamma)

        ca = np.cos(alpha);
        cb = np.cos(beta);
        cg = np.cos(gamma);
        sa = np.sin(alpha);
        sb = np.sin(beta);
        sg = np.sin(gamma);
        tg = np.tan(gamma);

        '''
            direct metric tensor
        '''
        self._dmt = np.array([[a**2, a*b*cg, a*c*cb],\
                             [a*b*cg, b**2, b*c*ca],\
                             [a*c*cb, b*c*ca, c**2]])
        self._vol = np.sqrt(np.linalg.det(self.dmt))

        if(self.vol < 1e-5):
            warnings.warn('unitcell volume is suspiciously small')

        '''
            reciprocal metric tensor
        '''
        self._rmt = np.linalg.inv(self.dmt)

        '''
            direct structure matrix
        '''
        self._dsm = np.array([[a, b*cg, c*cb],\
                             [0., b*sg, -c*(cb*cg - ca)/sg],
                             [0., 0., self.vol/(a*b*sg)]])

        '''
            reciprocal structure matrix
        '''
        self._rsm = np.array([[1./a, 0., 0.],\
                             [-1./(a*tg), 1./(b*sg), 0.],
                             [b*c*(cg*ca - cb)/(self.vol*sg), a*c*(cb*cg - ca)/(self.vol*sg), a*b*sg/self.vol]])

        ast = self.CalcLength([1,0,0],'r')
        bst = self.CalcLength([0,1,0],'r')
        cst = self.CalcLength([0,0,1],'r')

        self._aij = np.array([[ast**2, ast*bst, ast*cst],\
                              [bst*ast, bst**2, bst*cst],\
                              [cst*ast, cst*bst, cst**2]])

    ''' transform between any crystal space to any other space.
        choices are 'd' (direct), 'r' (reciprocal) and 'c' (cartesian)'''
    def TransSpace(self, v_in, inspace, outspace):
        if(inspace == 'd'):
            if(outspace == 'r'):
                v_out = np.dot(v_in, self.dmt)
            elif(outspace == 'c'):
                v_out = np.dot(self.dsm, v_in)
            else:
                raise ValueError('inspace in ''d'' but outspace can''t be identified')

        elif(inspace == 'r'):
            if(outspace == 'd'):
                v_out = np.dot(v_in, self.rmt)
            elif(outspace == 'c'):
                v_out = np.dot(self.rsm, v_in)
            else:
                raise ValueError('inspace in ''r'' but outspace can''t be identified')

        elif(inspace == 'c'):
            if(outspace == 'r'):
                v_out = np.dot(v_in, self.rsm)
            elif(outspace == 'd'):
                v_out = np.dot(v_in, self.dsm)
            else:
                raise ValueError('inspace in ''c'' but outspace can''t be identified')

        else:
            raise ValueError('incorrect inspace argument')

        return v_out

    ''' calculate dot product of two vectors in any space 'd' 'r' or 'c' '''
    def CalcDot(self, u, v, space):

        if(space == 'd'):
            dot = np.dot(u,np.dot(self.dmt,v))
        elif(space == 'r'):
            dot = np.dot(u,np.dot(self.rmt,v))
        elif(space == 'c'):
            dot = np.dot(u,v)
        else:
            raise ValueError('space is unidentified')

        return dot

    ''' calculate dot product of two vectors in any space 'd' 'r' or 'c' '''
    def CalcLength(self, u, space):

        if(space =='d'):
            vlen = np.sqrt(np.dot(u, np.dot(self.dmt, u)))
        elif(space =='r'):
            vlen = np.sqrt(np.dot(u, np.dot(self.rmt, u)))
        elif(spec =='c'):
            vlen = np.linalg.norm(u)
        else:
            raise ValueError('incorrect space argument')

        return vlen

    ''' normalize vector in any space 'd' 'r' or 'c' '''
    def NormVec(self, u, space):
        ulen = self.CalcLength(u, space)
        return u/ulen

    ''' calculate angle between two vectors in any space'''
    def CalcAngle(self, u, v, space):

        ulen = self.CalcLength(u, space)
        vlen = self.CalcLength(v, space)

        dot  = self.CalcDot(u, v, space)/ulen/vlen
        angle = np.arccos(dot)

        return angle

    ''' calculate cross product between two vectors in any space.

    cross product of two vectors in direct space is a vector in
    reciprocal space

    cross product of two vectors in reciprocal space is a vector in
    direct space

    the outspace specifies if a conversion needs to be made

     @NOTE: iv is the switch (0/1) which will either turn division
     by volume of the unit cell on or off.'''
    def CalcCross(self, p, q, inspace, outspace, vol_divide=False):
        iv = 0
        if(vol_divide):
            vol = self.vol
        else:
            vol = 1.0

        pxq = np.array([p[1]*q[2]-p[2]*q[1],\
                        p[2]*q[0]-p[0]*q[2],\
                        p[0]*q[1]-p[1]*q[0]])

        if(inspace == 'd'):
            '''
            cross product vector is in reciprocal space
            and can be converted to direct or cartesian space
            '''
            pxq *= vol

            if(outspace == 'r'):
                pass
            elif(outspace == 'd'):
                pxq = self.TransSpace(pxq, 'r', 'd')
            elif(outspace == 'c'):
                pxq = self.TransSpace(pxq, 'r', 'c')
            else:
                raise ValueError('inspace is ''d'' but outspace is unidentified')

        elif(inspace == 'r'):
            '''
            cross product vector is in direct space and
            can be converted to any other space
            '''
            pxq /= vol
            if(outspace == 'r'):
                pxq = self.TransSpace(pxq, 'd', 'r')
            elif(outspace == 'd'):
                pass
            elif(outspace == 'c'):
                pxq = self.TransSpace(pxq, 'd', 'c')
            else:
                raise ValueError('inspace is ''r'' but outspace is unidentified')

        elif(inspace == 'c'):
            '''
            cross product is already in cartesian space so no
            volume factor is involved. can be converted to any
            other space too
            '''
            if(outspace == 'r'):
                pxq = self.TransSpace(pxq, 'c', 'r')
            elif(outspace == 'd'):
                pxq = self.TransSpace(pxq, 'c', 'd')
            elif(outspace == 'c'):
                pass
            else:
                raise ValueError('inspace is ''c'' but outspace is unidentified')

        else:
            raise ValueError('inspace is unidentified')

        return pxq

    def GenerateRecipPGSym(self):


        self.SYM_PG_r = self.SYM_PG_d[0,:,:]
        self.SYM_PG_r = np.broadcast_to(self.SYM_PG_r,[1,3,3])

        self.SYM_PG_r_laue = self.SYM_PG_d[0,:,:]
        self.SYM_PG_r_laue = np.broadcast_to(self.SYM_PG_r_laue,[1,3,3])

        for i in range(1,self.npgsym):
            g = self.SYM_PG_d[i,:,:]
            g = np.dot(self.dmt, np.dot(g, self.rmt))
            g = np.round(np.broadcast_to(g,[1,3,3]))
            self.SYM_PG_r = np.concatenate((self.SYM_PG_r,g))

        for i in range(1,self.SYM_PG_d_laue.shape[0]):
            g = self.SYM_PG_d_laue[i,:,:]
            g = np.dot(self.dmt, np.dot(g, self.rmt))
            g = np.round(np.broadcast_to(g,[1,3,3]))
            self.SYM_PG_r_laue = np.concatenate((self.SYM_PG_r_laue,g))

        self.SYM_PG_r = self.SYM_PG_r.astype(np.int32)
        self.SYM_PG_r_laue = self.SYM_PG_r_laue.astype(np.int32)

    def CalcOrbit(self):
        '''
        calculate the equivalent position for the space group
        symmetry
        '''
        pass

    def CalcStar(self,v,space,applyLaue=False):
        '''
        this function calculates the symmetrically equivalent hkls (or uvws)
        for the reciprocal (or direct) point group symmetry.
        '''
        if(space == 'd'):
            if(applyLaue):
                sym = self.SYM_PG_d_laue
            else:
                sym = self.SYM_PG_d
        elif(space == 'r'):
            if(applyLaue):
                sym = self.SYM_PG_r_laue
            else:
                sym = self.SYM_PG_r
        else:
            raise ValueError('CalcStar: unrecognized space.')

        vsym = np.atleast_2d(v)
        for s in sym:
            vp = np.dot(s,v)
            # check if this is new
            isnew = True
            for vec in vsym:

                if(np.sum(np.abs(vp - vec)) < 1E-4):
                    isnew = False
                    break
            if(isnew):
                vsym = np.vstack((vsym, vp))

        return vsym

    def CalcPositions(self):
        '''
        calculate the asymmetric positions in the fundamental unitcell
        used for structure factor calculations
        '''
        numat = []
        asym_pos = []

        # using the wigner-seitz notation
        for i in range(self.atom_ntype):

            n = 1
            r = self.atom_pos[i,0:3]
            r = np.hstack((r, 1.))

            asym_pos.append(np.broadcast_to(r[0:3],[1,3]))

            for symmat in self.SYM_SG:
                # get new position
                rnew = np.dot(symmat, r)

                # reduce to fundamental unitcell with fractional
                # coordinates between 0-1
                rr = rnew[0:3]
                rr = np.modf(rr)[0]
                rr[rr < 0.] += 1.
                rr[np.abs(rr) < 1.0E-6] = 0.

                # check if this is new
                isnew = True
                for j in range(n):
                    if(np.sum(np.abs(rr - asym_pos[i][j,:])) < 1E-4):
                        isnew = False
                        break

                # if its new add this to the list
                if(isnew):
                    asym_pos[i] = np.vstack((asym_pos[i],rr))
                    n += 1

            numat.append(n)

        self.numat = np.array(numat)
        self.asym_pos = asym_pos

    def CalcDensity(self):
        '''
        calculate density, average atomic weight (avA)
        and average atomic number(avZ)
        '''
        self.avA = 0.0
        self.avZ = 0.0

        for i in range(self.atom_ntype):
            '''
            atype is atom type i.e. atomic number
            numat is the number of atoms of atype
            atom_pos(i,3) has the occupation factor
            '''
            atype = self.atom_type[i]
            numat = self.numat[i]
            occ   = self.atom_pos[i,3]
            avA  += numat * constants.atom_weights[atype-1] * occ # -1 due to 0 indexing in python
            avZ  += numat * atype


        self.density = avA / (self.vol * 1.0E-21 * constants.cAvogadro)

        av_natom = np.dot(self.numat, self.atom_pos[:,3])

        self.avA = avA / av_natom
        self.avZ = avZ / np.sum(self.numat)

    ''' calculate the maximum index of diffraction vector along each of the three reciprocal
     basis vectors '''
    def CalcMaxGIndex(self):
        while (1.0 / self.CalcLength(np.array([self.ih, 0, 0], dtype=np.float64), 'r') > self.dmin):
            self.ih = self.ih + 1
        while (1.0 / self.CalcLength(np.array([0, self.ik, 0], dtype=np.float64), 'r') > self.dmin):
            self.ik = self.ik + 1
        while (1.0 / self.CalcLength(np.array([0, 0, self.il], dtype=np.float64),'r') > self.dmin):
            self.il = self.il + 1

    def InitializeInterpTable(self):

        self.f1 = {}
        self.f2 = {}
        self.f_anam = {}

        data = importlib.resources.open_binary(hexrd.resources, 'Anomalous.h5')
        with h5py.File(data, 'r') as fid:
            for i in range(0,self.atom_ntype):

                Z    = self.atom_type[i]
                elem = constants.ptableinverse[Z]
                gid = fid.get('/'+elem)
                data = gid.get('data')

                self.f1[elem] = interp1d(data[:,7], data[:,1])
                self.f2[elem] = interp1d(data[:,7], data[:,2])

    def CalcAnomalous(self):

        for i in range(self.atom_ntype):

            Z = self.atom_type[i]
            elem = constants.ptableinverse[Z]
            f1 = self.f1[elem](self.wavelength)
            f2 = self.f2[elem](self.wavelength)
            frel = constants.frel[elem]
            Z = constants.ptable[elem]
            self.f_anam[elem] = np.complex(f1+frel-Z, f2)

    def CalcXRFormFactor(self, Z, s):

        '''
        we are using the following form factors for x-aray scattering:
        1. coherent x-ray scattering, f0 tabulated in Acta Cryst. (1995). A51,416-431
        2. Anomalous x-ray scattering (complex (f'+if")) tabulated in J. Phys. Chem. Ref. Data, 24, 71 (1995)
        and J. Phys. Chem. Ref. Data, 29, 597 (2000).
        3. Thompson nuclear scattering, fNT tabulated in Phys. Lett. B, 69, 281 (1977).

        the anomalous scattering is a complex number (f' + if"), where the two terms are given by
        f' = f1 + frel - Z
        f" = f2

        f1 and f2 have been tabulated as a function of energy in Anomalous.h5 in hexrd folder

        overall f = (f0 + f' + if" +fNT)
        '''
        elem = constants.ptableinverse[Z]
        sfact = constants.scatfac[elem]
        fe = sfact[5]
        fNT = constants.fNT[elem]
        frel = constants.frel[elem]
        f_anomalous = self.f_anam[elem]

        for i in range(5):
            fe += sfact[i] * np.exp(-sfact[i+6]*s)

        return (fe+fNT+f_anomalous)

    def CalcXRSF(self, hkl):

        '''
        the 1E-2 is to convert to A^-2
        since the fitting is done in those units
        '''
        s =  0.25 * self.CalcLength(hkl, 'r')**2 * 1E-2
        sf = np.complex(0.,0.)
        for i in range(0,self.atom_ntype):

            Z   = self.atom_type[i]
            ff  = self.CalcXRFormFactor(Z,s)

            if(self.aniU):
                T = np.exp(-np.dot(hkl,np.dot(self.betaij[i,:,:],hkl)))
            else:
                T = np.exp(-8.0*np.pi**2 * self.U[i]*s)

            ff *= self.atom_pos[i,3] * T

            for j in range(self.asym_pos[i].shape[0]):
                arg =  2.0 * np.pi * np.sum( hkl * self.asym_pos[i][j,:] )
                sf  = sf + ff * np.complex(np.cos(arg),-np.sin(arg))

        return np.abs(sf)**2

    ''' calculate bragg angle for a reflection. returns Nan if
        the reflections is not possible for the voltage/wavelength
    '''
    def CalcBraggAngle(self, hkl):
        glen = self.CalcLength(hkl, 'r')
        sth  = self.wavelength * glen * 0.5
        return np.arcsin(sth)

    def Allowed_HKLs(self, hkllist):
        '''
        this function checks if a particular g vector is allowed
        by lattice centering, screw axis or glide plane
        '''
        hkllist = np.atleast_2d(hkllist)

        centering = self.sg_hmsymbol[0]

        if(centering == 'P'):
            # all reflections are allowed
            mask = np.ones([hkllist.shape[0],],dtype=np.bool)

        elif(centering == 'F'):
            # same parity

            seo = np.sum(np.mod(hkllist+100,2),axis=1)
            mask = np.logical_not(np.logical_or(seo==1, seo==2))

        elif(centering == 'I'):
            # sum is even
            seo = np.mod(np.sum(hkllist,axis=1)+100,2)
            mask = (seo == 0)

        elif(centering == 'A'):
            # k+l is even
            seo = np.mod(np.sum(hkllist[:,1:3],axis=1)+100,2)
            mask = seo == 0

        elif(centering == 'B'):
            # h+l is even
            seo = np.mod(hkllist[:,0]+hkllist[:,2]+100,2)
            mask = seo == 0

        elif(centering == 'C'):
            # h+k is even
            seo = np.mod(hkllist[:,0]+hkllist[:,1]+100,2)
            mask = seo == 0

        elif(centering == 'R'):
            # -h+k+l is divisible by 3
            seo = np.mod(-hkllist[:,0]+hkllist[:,1]+hkllist[:,2]+90,3)
            mask = seo == 0
        else:
            raise RuntimeError('IsGAllowed: unknown lattice centering encountered.')

        hkls = hkllist[mask,:]

        if(not self.symmorphic):
            hkls = self.NonSymmorphicAbsences(hkls)

        return hkls.astype(np.int32)

    def omitscrewaxisabsences(self, hkllist, ax, iax):

        '''
        this function encodes the table on pg 48 of
        international table of crystallography vol A
        the systematic absences due to different screw
        axis is encoded here.
        iax encodes the primary, secondary or tertiary axis
        iax == 0 : primary
        iax == 1 : secondary
        iax == 2 : tertiary

        @NOTE: only unique b axis in monoclinic systems
        implemented as thats the standard setting
        '''
        if(self.latticeType == 'triclinic'):
            '''
                no systematic absences for the triclinic crystals
            '''
            pass

        elif(self.latticeType == 'monoclinic'):

            if(ax != '2_1'):
                raise RuntimeError('omitscrewaxisabsences: monoclinic systems can only have 2_1 screw axis.')
            '''
                only unique b-axis will be encoded
                it is the users responsibility to input
                lattice parameters in the standard setting
                with b-axis having the 2-fold symmetry
            '''
            if(iax == 1):
                mask1 = np.logical_and(hkllist[:,0] == 0,hkllist[:,2] == 0)
                mask2 = np.mod(hkllist[:,1]+100,2) != 0
                mask  = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]
            else:
                raise RuntimeError('omitscrewaxisabsences: only b-axis can have 2_1 screw axis')

        elif(self.latticeType == 'orthorhombic'):

            if(ax != '2_1'):
                raise RuntimeError('omitscrewaxisabsences: orthorhombic systems can only have 2_1 screw axis.')
            '''
            2_1 screw on primary axis
            h00 ; h = 2n
            '''
            if(iax == 0):
                mask1 = np.logical_and(hkllist[:,1] == 0,hkllist[:,2] == 0)
                mask2 = np.mod(hkllist[:,0]+100,2) != 0
                mask  = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]

            elif(iax == 1):
                mask1 = np.logical_and(hkllist[:,0] == 0,hkllist[:,2] == 0)
                mask2 = np.mod(hkllist[:,1]+100,2) != 0
                mask  = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]

            elif(iax == 2):
                mask1 = np.logical_and(hkllist[:,0] == 0,hkllist[:,1] == 0)
                mask2 = np.mod(hkllist[:,2]+100,2) != 0
                mask  = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]

        elif(self.latticeType == 'tetragonal'):

            if(iax == 0):
                mask1 = np.logical_and(hkllist[:,0] == 0, hkllist[:,1] == 0)
                if(ax == '4_2'):
                    mask2 = np.mod(hkllist[:,2]+100,2) != 0
                elif(ax in ['4_1','4_3']):
                    mask2 = np.mod(hkllist[:,2]+100,4) != 0
                mask = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]

            elif(iax == 1):
                mask1 = np.logical_and(hkllist[:,1] == 0, hkllist[:,2] == 0)
                mask2 = np.logical_and(hkllist[:,0] == 0, hkllist[:,2] == 0)
                if(ax == '2_1'):
                    mask3 = np.mod(hkllist[:,0]+100,2) != 0
                    mask4 = np.mod(hkllist[:,1]+100,2) != 0

                mask1 = np.logical_not(np.logical_and(mask1,mask3))
                mask2 = np.logical_not(np.logical_and(mask2,mask4))

                mask = ~np.logical_or(~mask1,~mask2)
                hkllist = hkllist[mask,:]

        elif(self.latticeType == 'trigonal'):
            mask1 = np.logical_and(hkllist[:,0] == 0,hkllist[:,1] == 0)

            if(iax == 0):
                if(ax in ['3_1', '3_2']):
                    mask2 = np.mod(hkllist[:,2]+90,3) != 0
            else:
                raise RuntimeError('omitscrewaxisabsences: trigonal systems can only have screw axis on primary axis ')

            mask  = np.logical_not(np.logical_and(mask1,mask2))
            hkllist = hkllist[mask,:]

        elif(self.latticeType == 'hexagonal'):

            mask1 = np.logical_and(hkllist[:,0] == 0,hkllist[:,1] == 0)

            if(iax == 0):
                if(ax == '6_3'):
                    mask2 = np.mod(hkllist[:,2]+100,2) != 0

                elif(ax in['3_1','3_2','6_2','6_4']):
                    mask2 = np.mod(hkllist[:,2]+90,3) != 0

                elif(ax in ['6_1','6_5']):
                    mask2 = np.mod(hkllist[:,2]+120,6) != 0
            else:
                raise RuntimeError('omitscrewaxisabsences: hexagonal systems can only have screw axis on primary axis ')

            mask  = np.logical_not(np.logical_and(mask1,mask2))
            hkllist = hkllist[mask,:]

        elif(self.latticeType == 'cubic'):
            mask1 = np.logical_and(hkllist[:,0] == 0,hkllist[:,1] == 0)
            mask2 = np.logical_and(hkllist[:,0] == 0,hkllist[:,2] == 0)
            mask3 = np.logical_and(hkllist[:,1] == 0,hkllist[:,2] == 0)

            if(ax in ['2_1','4_2']):
                mask4 = np.mod(hkllist[:,2]+100,2) != 0
                mask5 = np.mod(hkllist[:,1]+100,2) != 0
                mask6 = np.mod(hkllist[:,0]+100,2) != 0

            elif(ax in ['4_1','4_3']):
                mask4 = np.mod(hkllist[:,2]+100,4) != 0
                mask5 = np.mod(hkllist[:,1]+100,4) != 0
                mask6 = np.mod(hkllist[:,0]+100,4) != 0

            mask1 = np.logical_not(np.logical_and(mask1,mask4))
            mask2 = np.logical_not(np.logical_and(mask2,mask5))
            mask3 = np.logical_not(np.logical_and(mask3,mask6))

            mask = ~np.logical_or(~mask1,np.logical_or(~mask2,~mask3))

            hkllist = hkllist[mask,:]

        return hkllist.astype(np.int32)

    def omitglideplaneabsences(self, hkllist, plane, ip):
        '''
        this function encodes the table on pg 47 of
        international table of crystallography vol A
        the systematic absences due to different glide
        planes is encoded here.
        ip encodes the primary, secondary or tertiary plane normal
        ip == 0 : primary
        ip == 1 : secondary
        ip == 2 : tertiary


        @NOTE: only unique b axis in monoclinic systems
        implemented as thats the standard setting
        '''
        if(self.latticeType == 'triclinic'):
            pass

        elif(self.latticeType == 'monoclinic'):
            if(ip == 1):
                mask1 = hkllist[:,1] == 0

                if(plane == 'c'):
                    mask2 = np.mod(hkllist[:,2]+100,2) != 0
                elif(plane == 'a'):
                    mask2 = np.mod(hkllist[:,0]+100,2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:,0]+hkllist[:,2]+100,2) != 0

                mask = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]

        elif(self.latticeType == 'orthorhombic'):

            if(ip == 0):
                mask1 = hkllist[:,0] == 0

                if(plane == 'b'):
                    mask2 = np.mod(hkllist[:,1]+100,2) != 0
                elif(plane == 'c'):
                    mask2 = np.mod(hkllist[:,2]+100,2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:,1]+hkllist[:,2]+100,2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:,1]+hkllist[:,2]+100,4) != 0

                mask = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]

            elif(ip == 1):
                mask1 = hkllist[:,1] == 0

                if(plane == 'c'):
                    mask2 = np.mod(hkllist[:,2]+100,2) != 0
                elif(plane == 'a'):
                    mask2 = np.mod(hkllist[:,0]+100,2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:,0]+hkllist[:,2]+100,2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:,0]+hkllist[:,2]+100,4) != 0

                mask = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]

            elif(ip == 2):
                mask1 = hkllist[:,2] == 0

                if(plane == 'a'):
                    mask2 = np.mod(hkllist[:,0]+100,2) != 0
                elif(plane == 'b'):
                    mask2 = np.mod(hkllist[:,1]+100,2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:,0]+hkllist[:,1]+100,2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:,0]+hkllist[:,1]+100,4) != 0

                mask = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]

        elif(self.latticeType == 'tetragonal'):

            if(ip == 0):

                mask1 = hkllist[:,2] == 0

                if(plane == 'a'):
                    mask2 = np.mod(hkllist[:,0]+100,2) != 0

                elif(plane == 'b'):
                    mask2 = np.mod(hkllist[:,1]+100,2) != 0

                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:,0]+hkllist[:,1]+100,2) != 0

                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:,0]+hkllist[:,1]+100,4) != 0

                mask = np.logical_not(np.logical_and(mask1,mask2))
                hkllist = hkllist[mask,:]

            elif(ip == 1):

                mask1 = hkllist[:,0] == 0
                mask2 = hkllist[:,1] == 0

                if(plane in ['a','b']):
                    mask3 = np.mod(hkllist[:,1]+100,2) != 0
                    mask4 = np.mod(hkllist[:,0]+100,2) != 0

                elif(plane == 'c'):
                    mask3 = np.mod(hkllist[:,2]+100,2) != 0
                    mask4 = mask3

                elif(plane == 'n'):
                    mask3 = np.mod(hkllist[:,1]+hkllist[:,2]+100,2) != 0
                    mask4 = np.mod(hkllist[:,0]+hkllist[:,2]+100,2) != 0

                elif(plane == 'd'):
                    mask3 = np.mod(hkllist[:,1]+hkllist[:,2]+100,4) != 0
                    mask4 = np.mod(hkllist[:,0]+hkllist[:,2]+100,4) != 0

                mask1 = np.logical_not(np.logical_and(mask1,mask3))
                mask2 = np.logical_not(np.logical_and(mask2,mask4))

                mask = ~np.logical_or(~mask1,~mask2)
                hkllist = hkllist[mask,:]

            elif(ip == 2):

                mask1 = np.abs(hkllist[:,0]) == np.abs(hkllist[:,1])

                if(plane in ['c','n']):
                    mask2 = np.mod(hkllist[:,2]+100,2) != 0

                elif(plane == 'd'):
                    mask2 = np.mod(2*hkllist[:,0]+hkllist[:,2]+100,4) != 0

                mask = np.logical_not(np.logical_and(mask1,mask2))

                hkllist = hkllist[mask,:]


        elif(self.latticeType == 'trigonal'):

            if(plane != 'c'):
                raise RuntimeError('omitglideplaneabsences: only c-glide allowed for trigonal systems.')

            if(ip == 1):

                mask1 = hkllist[:,0] == 0
                mask2 = hkllist[:,1] == 0
                mask3 = hkllist[:,0] == -hkllist[:,1]
                if(plane == 'c'):
                    mask4 = np.mod(hkllist[:,2]+100,2) != 0
                else:
                    raise RuntimeError('omitglideplaneabsences: only c-glide allowed for trigonal systems.')

            elif(ip == 2):
                mask1 = hkllist[:,1] == hkllist[:,0]
                mask2 = hkllist[:,0] == -2*hkllist[:,1]
                mask3 = -2*hkllist[:,0] == hkllist[:,1]
                if(plane == 'c'):
                    mask4 = np.mod(hkllist[:,2]+100,2) != 0
                else:
                    raise RuntimeError('omitglideplaneabsences: only c-glide allowed for trigonal systems.')

            mask1 = np.logical_and(mask1,mask4)
            mask2 = np.logical_and(mask2,mask4)
            mask3 = np.logical_and(mask3,mask4)

            mask = np.logical_not(np.logical_or(mask1,np.logical_or(mask2,mask3)))
            hkllist = hkllist[mask,:]

        elif(self.latticeType == 'hexagonal'):

            if(plane != 'c'):
                raise RuntimeError('omitglideplaneabsences: only c-glide allowed for hexagonal systems.')

            if(ip == 2):
                mask1 = hkllist[:,0] == hkllist[:,1]
                mask2 = hkllist[:,0] == -2*hkllist[:,1]
                mask3 = -2*hkllist[:,0] == hkllist[:,1]

                mask4 = np.mod(hkllist[:,2]+100,2) != 0

                mask1 = np.logical_and(mask1,mask4)
                mask2 = np.logical_and(mask2,mask4)
                mask3 = np.logical_and(mask3,mask4)

                mask = np.logical_not(np.logical_or(mask1,np.logical_or(mask2,mask3)))

            elif(ip == 1):
                mask1 = hkllist[:,1] == 0
                mask2 = hkllist[:,0] == 0
                mask3 = hkllist[:,1] == -hkllist[:,0]

                mask4 = np.mod(hkllist[:,2]+100,2) != 0

            mask1 = np.logical_and(mask1,mask4)
            mask2 = np.logical_and(mask2,mask4)
            mask3 = np.logical_and(mask3,mask4)

            mask = np.logical_not(np.logical_or(mask1,np.logical_or(mask2,mask3)))
            hkllist = hkllist[mask,:]

        elif(self.latticeType == 'cubic'):

            if(ip == 0):
                mask1 = hkllist[:,0] == 0
                mask2 = hkllist[:,1] == 0
                mask3 = hkllist[:,2] == 0

                mask4 = np.mod(hkllist[:,0]+100,2) != 0
                mask5 = np.mod(hkllist[:,1]+100,2) != 0
                mask6 = np.mod(hkllist[:,2]+100,2) != 0

                if(plane  == 'a'):
                    mask1 = np.logical_or(np.logical_and(mask1,mask5),np.logical_and(mask1,mask6))
                    mask2 = np.logical_or(np.logical_and(mask2,mask4),np.logical_and(mask2,mask6))
                    mask3 = np.logical_and(mask3,mask4)

                    mask = np.logical_not(np.logical_or(mask1,np.logical_or(mask2,mask3)))

                elif(plane == 'b'):
                    mask1 = np.logical_and(mask1,mask5)
                    mask3 = np.logical_and(mask3,mask5)
                    mask = np.logical_not(np.logical_or(mask1,mask3))

                elif(plane == 'c'):
                    mask1 = np.logical_and(mask1,mask6)
                    mask2 = np.logical_and(mask2,mask6)
                    mask = np.logical_not(np.logical_or(mask1,mask2))

                elif(plane == 'n'):
                    mask4 = np.mod(hkllist[:,1]+hkllist[:,2]+100,2) != 0
                    mask5 = np.mod(hkllist[:,0]+hkllist[:,2]+100,2) != 0
                    mask6 = np.mod(hkllist[:,0]+hkllist[:,1]+100,2) != 0

                    mask1 = np.logical_not(np.logical_and(mask1,mask4))
                    mask2 = np.logical_not(np.logical_and(mask2,mask5))
                    mask3 = np.logical_not(np.logical_and(mask3,mask6))

                    mask = ~np.logical_or(~mask1,np.logical_or(~mask2,~mask3))

                elif(plane == 'd'):
                    mask4 = np.mod(hkllist[:,1]+hkllist[:,2]+100,4) != 0
                    mask5 = np.mod(hkllist[:,0]+hkllist[:,2]+100,4) != 0
                    mask6 = np.mod(hkllist[:,0]+hkllist[:,1]+100,4) != 0

                    mask1 = np.logical_not(np.logical_and(mask1,mask4))
                    mask2 = np.logical_not(np.logical_and(mask2,mask5))
                    mask3 = np.logical_not(np.logical_and(mask3,mask6))

                    mask = ~np.logical_or(~mask1,np.logical_or(~mask2,~mask3))
                else:
                    raise RuntimeError('omitglideplaneabsences: unknown glide plane encountered.')

                hkllist = hkllist[mask,:]

            if(ip == 2):

                mask1 = np.abs(hkllist[:,0]) == np.abs(hkllist[:,1])
                mask2 = np.abs(hkllist[:,1]) == np.abs(hkllist[:,2])
                mask3 = np.abs(hkllist[:,0]) == np.abs(hkllist[:,2])

                if(plane in ['a','b','c','n']):
                    mask4 = np.mod(hkllist[:,2]+100,2) != 0
                    mask5 = np.mod(hkllist[:,0]+100,2) != 0
                    mask6 = np.mod(hkllist[:,1]+100,2) != 0

                elif(plane == 'd'):
                    mask4 = np.mod(2*hkllist[:,0]+hkllist[:,2]+100,4) != 0
                    mask5 = np.mod(hkllist[:,0]+2*hkllist[:,1]+100,4) != 0
                    mask6 = np.mod(2*hkllist[:,0]+hkllist[:,1]+100,4) != 0

                else:
                    raise RuntimeError('omitglideplaneabsences: unknown glide plane encountered.')

                mask1 = np.logical_not(np.logical_and(mask1,mask4))
                mask2 = np.logical_not(np.logical_and(mask2,mask5))
                mask3 = np.logical_not(np.logical_and(mask3,mask6))

                mask = ~np.logical_or(~mask1,np.logical_or(~mask2,~mask3))

                hkllist = hkllist[mask,:]

        return hkllist

    def NonSymmorphicAbsences(self, hkllist):
        '''
        this function prunes hkl list for the screw axis and glide
        plane absences
        '''
        planes = constants.SYS_AB[self.sgnum][0]

        for ip,p in enumerate(planes):
            if(p != ''):
                hkllist = self.omitglideplaneabsences(hkllist, p, ip)

        axes = constants.SYS_AB[self.sgnum][1]
        for iax,ax in enumerate(axes):
            if(ax != ''):
                hkllist = self.omitscrewaxisabsences(hkllist, ax, iax)

        return hkllist

    def ChooseSymmetric(self, hkllist, InversionSymmetry=True):
        '''
        this function takes a list of hkl vectors and
        picks out a subset of the list picking only one
        of the symmetrically equivalent one. The convention
        is to choose the hkl with the most positive components.
        '''
        mask = np.ones(hkllist.shape[0],dtype=np.bool)
        laue = InversionSymmetry

        for i,g in enumerate(hkllist):
            if(mask[i]):

                geqv = self.CalcStar(g,'r',applyLaue=laue)

                for r in geqv[1:,]:
                    rid = np.where(np.all(r==hkllist,axis=1))
                    mask[rid] = False

        hkl = hkllist[mask,:].astype(np.int32)

        hkl_max = []

        for g in hkl:
            geqv = self.CalcStar(g,'r',applyLaue=laue)
            loc = np.argmax(np.sum(geqv,axis=1))
            gmax = geqv[loc,:]
            hkl_max.append(gmax)

        return np.array(hkl_max).astype(np.int32)

    def SortHKL(self, hkllist):
        '''
        this function sorts the hkllist by increasing |g|
        i.e. decreasing d-spacing. If two vectors are same
        length, then they are ordered with increasing
        priority to l, k and h
        '''
        glen = []
        for g in hkllist:
            glen.append(np.round(self.CalcLength(g,'r'),8))

        # glen = np.atleast_2d(np.array(glen,dtype=np.float)).T
        dtype = [('glen', float), ('max', int), ('sum', int), ('h', int), ('k', int), ('l', int)]

        a = []
        for i,gl in enumerate(glen):
            g = hkllist[i,:]
            a.append((gl, np.max(g), np.sum(g), g[0], g[1], g[2]))
        a = np.array(a, dtype=dtype)

        isort = np.argsort(a, order=['glen','max','sum','l','k','h'])
        return hkllist[isort,:]

    def getHKLs(self, dmin):
        '''
        this function generates the symetrically unique set of
        hkls up to a given dmin.
        dmin is in nm
        '''
        '''
        always have the centrosymmetric condition because of
        Friedels law for xrays so only 4 of the 8 octants
        are sampled for unique hkls. By convention we will
        ignore all l < 0
        '''

        hmin = -self.ih-1
        hmax = self.ih
        kmin = -self.ik-1
        kmax = self.ik
        lmin = -1
        lmax = self.il

        hkllist = np.array([[ih, ik, il] for ih in np.arange(hmax,hmin,-1) \
                  for ik in np.arange(kmax,kmin,-1) \
                  for il in np.arange(lmax,lmin,-1)])

        hkl_allowed = self.Allowed_HKLs(hkllist)

        hkl = []
        dsp = []

        hkl_dsp = []

        for g in hkl_allowed:

            # ignore [0 0 0] as it is the direct beam
            if(np.sum(np.abs(g)) != 0):

                dspace = 1./self.CalcLength(g,'r')

                if(dspace >= dmin):
                    hkl_dsp.append(g)

        '''
        we now have a list of g vectors which are all within dmin range
        plus the systematic absences due to lattice centering and glide
        planes/screw axis has been taken care of

        the next order of business is to go through the list and only pick
        out one of the symetrically equivalent hkls from the list.
        '''
        hkl_dsp = np.array(hkl_dsp).astype(np.int32)
        '''
        the inversionsymmetry switch enforces the application of the inversion
        symmetry regradless of whether the crystal has the symmetry or not
        this is necessary in the case of xrays due to friedel's law
        '''
        hkl = self.ChooseSymmetric(hkl_dsp, InversionSymmetry=True)

        '''
        finally sort in order of decreasing dspacing
        '''
        self.hkls = self.SortHKL(hkl)

        return self.hkls
    '''
        set some properties for the unitcell class. only the lattice
        parameters, space group and asymmetric positions can change,
        but all the dependent parameters will be automatically updated
    '''

    def Required_lp(self, p):
        return _rqpDict[self.latticeType][1](p)

    def Required_C(self,C):
        return np.array([C[x] for x in _StiffnessDict[self._laueGroup][0]])

    def MakeStiffnessMatrix(self, inp_Cvals):
        if(len(inp_Cvals) != len(_StiffnessDict[self._laueGroup][0])):
            raise IOError('number of constants entered is not correct. need a total of ',len(_StiffnessDict[self._laueGroup][0]),'independent constants')

        # initialize all zeros and fill the supplied values
        C = np.zeros([6,6])
        for i,x in enumerate(_StiffnessDict[self._laueGroup][0]):
            C[x] = inp_Cvals[i]

        # enforce the equality constraints
        C = _StiffnessDict[self._laueGroup][1](C)

        # finally fill the lower triangular matrix
        for i in range(6):
            for j in range(i):
                C[i,j] = C[j,i]

        self.stiffness    = C

    @property
    def compliance(self):
        if not hasattr(self, 'stiffness'):
            raise AttributeError('Stiffness not set on unit cell')

        return np.linalg.inv(self.stiffness)

    @compliance.setter
    def compliance(self, v):
        self.stiffness = np.linalg.inv(v)

    # lattice constants as properties
    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, val):
        self._a = val
        self.calcmatrices()
        self.ih = 1
        self.ik = 1
        self.il = 1
        self.CalcMaxGIndex()

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, val):
        self._b = val
        self.calcmatrices()
        self.ih = 1
        self.ik = 1
        self.il = 1
        self.CalcMaxGIndex()

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, val):
        self._c = val
        self.calcmatrices()
        self.ih = 1
        self.ik = 1
        self.il = 1
        self.CalcMaxGIndex()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val
        self.calcmatrices()
        self.ih = 1
        self.ik = 1
        self.il = 1
        self.CalcMaxGIndex()

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, val):
        self._beta = val
        self.calcmatrices()
        self.ih = 1
        self.ik = 1
        self.il = 1
        self.CalcMaxGIndex()

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val
        self.calcmatrices()
        self.ih = 1
        self.ik = 1
        self.il = 1
        self.CalcMaxGIndex()

    @property
    def dmin(self):
        return self._dmin

    @dmin.setter
    def dmin(self, v):
        if self._dmin == v:
            return

        self._dmin = v

        # Update the Max G Index
        self.ih = 1
        self.ik = 1
        self.il = 1
        self.CalcMaxGIndex()

    @property
    def U(self):
        return self._U

    @U.setter
    def U(self,Uarr):
        self._U = Uarr

    @property
    def voltage(self):
        return self._voltage

    @voltage.setter
    def voltage(self,v):
        self._voltage = v
        self.CalcWavelength()

    @property
    def wavelength(self):
        return self._mlambda

    @wavelength.setter
    def wavelength(self,mlambda):
        self._mlambda = mlambda

    # space group number
    @property
    def sgnum(self):
        return self._sym_sgnum

    @sgnum.setter
    def sgnum(self, val):
        if(not(isinstance(val, int))):
            raise ValueError('space group should be integer')
        if(not( (val >= 1) and (val <= 230) ) ):
            raise ValueError('space group number should be between 1 and 230.')

        self._sym_sgnum = val
        self.sg_hmsymbol = symbols.pstr_spacegroup[val-1].strip()

        self.SYM_SG, self.SYM_PG_d, self.SYM_PG_d_laue, \
        self.centrosymmetric, self.symmorphic = \
        symmetry.GenerateSGSym(self.sgnum, self.sgsetting)

        self.latticeType = symmetry.latticeType(self.sgnum)

        self.nsgsym = self.SYM_SG.shape[0]
        self.npgsym = self.SYM_PG_d.shape[0]

        self.GenerateRecipPGSym()

        '''
        asymmetric positions due to space group symmetry
        used for structure factor calculations
        '''
        self.CalcPositions()
        self.GetPgLg()

    @property
    def atom_pos(self):
        return self._atom_pos

    @atom_pos.setter
    def atom_pos(self, val):
        self._atom_pos = val

    @property
    def B_factor(self):
        return self._atom_pos[:,4]

    @B_factor.setter
    def B_factor(self, val):
        if (val.shape[0] != self.atom_ntype):
            raise ValueError('Incorrect shape for B factor')
        self._atom_pos[:,4] = val

    # asymmetric positions in unit cell
    @property
    def asym_pos(self):
        return self._asym_pos

    @asym_pos.setter
    def asym_pos(self, val):
        assert(type(val) == list), 'input type to asymmetric positions should be list'
        self._asym_pos = val

    @property
    def numat(self):
        return self._numat

    @numat.setter
    def numat(self, val):
        assert(val.shape[0] == self.atom_ntype),'shape of numat is not consistent'
        self._numat = val

    # different atom types; read only
    @property
    def Z(self):
        sz = self.atom_ntype
        return self.atom_type[0:atom_ntype]

    # direct metric tensor is read only
    @property
    def dmt(self):
        return self._dmt

    # reciprocal metric tensor is read only
    @property
    def rmt(self):
        return self._rmt

    # direct structure matrix is read only
    @property
    def dsm(self):
        return self._dsm

    # reciprocal structure matrix is read only
    @property
    def rsm(self):
        return self._rsm

    @property
    def vol(self):
        return self._vol

_rqpDict = {
    'triclinic': (tuple(range(6)), lambda p: p),  # all 6
    'monoclinic': ((0,1,2,4), lambda p: (p[0], p[1], p[2], 90, p[3], 90)), # note beta
    'orthorhombic': ((0,1,2),   lambda p: (p[0], p[1], p[2], 90, 90,   90)),
    'tetragonal': ((0,2),     lambda p: (p[0], p[0], p[1], 90, 90,   90)),
    'trigonal': ((0,2),     lambda p: (p[0], p[0], p[1], 90, 90,  120)),
    'hexagonal': ((0,2),     lambda p: (p[0], p[0], p[1], 90, 90,  120)),
    'cubic': ((0,),      lambda p: (p[0], p[0], p[0], 90, 90,   90)),
    }

_lpname = np.array(['a','b','c','alpha','beta','gamma'])

laue_1 = 'ci'
laue_2 = 'c2h'
laue_3 = 'd2h'
laue_4 = 'c4h'
laue_5 = 'd4h'
laue_6 = 's6'
laue_7 = 'd3d'
laue_8 = 'c6h'
laue_9 = 'd6h'
laue_10 = 'th'
laue_11 = 'oh'

_sgrange = lambda min, max: tuple(range(min, max + 1)) # inclusive range
_pgDict = {
    _sgrange(  1,   1): ('c1', laue_1),  # Triclinic
    _sgrange(  2,   2): ('ci', laue_1),  #                    laue 1
    _sgrange(  3,   5): ('c2', laue_2),  # Monoclinic
    _sgrange(  6,   9): ('cs', laue_2),
    _sgrange( 10,  15): ('c2h',laue_2),  #                    laue 2
    _sgrange( 16,  24): ('d2', laue_3),  # Orthorhombic
    _sgrange( 25,  46): ('c2v',laue_3),
    _sgrange( 47,  74): ('d2h',laue_3),  #                    laue 3
    _sgrange( 75,  80): ('c4', laue_4),  # Tetragonal
    _sgrange( 81,  82): ('s4', laue_4),
    _sgrange( 83,  88): ('c4h',laue_4),  #                    laue 4
    _sgrange( 89,  98): ('d4', laue_5),
    _sgrange( 99, 110): ('c4v',laue_5),
    _sgrange(111, 122): ('d2d',laue_5),
    _sgrange(123, 142): ('d4h',laue_5),  #                    laue 5
    _sgrange(143, 146): ('c3', laue_6),  # Trigonal
    _sgrange(147, 148): ('s6', laue_6),  #                    laue 6 [also c3i]
    _sgrange(149, 155): ('d3', laue_7),
    _sgrange(156, 161): ('c3v',laue_7),
    _sgrange(162, 167): ('d3d',laue_7),  #                    laue 7
    _sgrange(168, 173): ('c6', laue_8),  # Hexagonal
    _sgrange(174, 174): ('c3h',laue_8),
    _sgrange(175, 176): ('c6h',laue_8),  #                    laue 8
    _sgrange(177, 182): ('d6', laue_9),
    _sgrange(183, 186): ('c6v',laue_9),
    _sgrange(187, 190): ('d3h',laue_9),
    _sgrange(191, 194): ('d6h',laue_9),   #                    laue 9
    _sgrange(195, 199): ('t',  laue_10),  # Cubic
    _sgrange(200, 206): ('th', laue_10),  #                    laue 10
    _sgrange(207, 214): ('o',  laue_11),
    _sgrange(215, 220): ('td', laue_11),
    _sgrange(221, 230): ('oh', laue_11),  #                    laue 11
    }

'''
this dictionary has the mapping from laue group to number of elastic
constants needed in the voight 6x6 stiffness matrix. the compliance
matrix is just the inverse of the stiffness matrix
taken from International Tables for Crystallography Volume H
Powder diffraction
Edited by C. J. Gilmore, J. A. Kaduk and H. Schenk
'''
# independent components for the triclinic laue group
type1 = []
for i in range(6):
    for j in range(i,6):
        type1.append((i,j))
type1 = tuple(type1)

# independent components for the monoclinic laue group
# C14 = C15 = C24 = C25 = C34 = C35 = C46 = C56 = 0
type2 = list(type1)
type2.remove((0,3)); type2.remove((0,4)); type2.remove((1,3)); type2.remove((1,4))
type2.remove((2,3)); type2.remove((2,4)); type2.remove((3,5)); type2.remove((4,5))
type2 = tuple(type2)

# independent components for the orthorhombic laue group
# Above, plus C16 = C26 = C36 = C45 = 0
type3 = list(type2)
type3.remove((0,5));type3.remove((1,5));type3.remove((2,5));type3.remove((3,4))
type3 = tuple(type3)

# independent components for the cyclic tetragonal laue group
# monoclinic, plus C36 = C45 = 0, C22 = C11, C23 = C13, C26 = −C16, C55 = C44
type4 = list(type2)
type4.remove((2,5));type4.remove((3,4));type4.remove((1,1))
type4.remove((1,2));type4.remove((1,5));type4.remove((4,4))
type4 = tuple(type4)

# independent components for the dihedral tetragonal laue group
# Above,  plus C16 = 0
type5 = list(type4)
type5.remove((0,5))
type5 = tuple(type5)

# independent components for the trigonal laue group
# C16 = C26 = C34 = C35 = C36 = C45 = 0, C22 = C11, C23 = C13, C24 = −C14,
# C25 = −C15, C46 = −C15, C55 = C44, C56 = C14, C66 = (C11 − C12)/2
type6 = list(type1)
type6.remove((0,5));type6.remove((1,5));type6.remove((2,3))
type6.remove((2,4));type6.remove((2,5));type6.remove((3,4))
type6.remove((1,1));type6.remove((1,2));type6.remove((1,3))
type6.remove((1,4));type6.remove((3,5));type6.remove((4,4))
type6.remove((4,5));type6.remove((5,5))
type6 = tuple(type6)

# independent components for the rhombohedral laue group
# Above, plus C15 = 0
type7 = list(type6)
type7.remove((0,4))
type7 = tuple(type7)

# independent components for the hexagonal laue group
# Above, plus C14 = 0
type8 = list(type7)
type8.remove((0,3))
type8 = tuple(type8)

# independent components for the cubic laue group
# As for dihedral tetragonal, plus C13 = C12, C33 = C11, C66 = C44
type9 = list(type5)
type9.remove((0,2));type9.remove((2,2));type9.remove((5,5));

'''
these lambda functions take care of the equality constrains in the
matrices. if there are no equality constraints, then the identity
function is used
C22 = C11, C23 = C13, C24 = −C14,
# C25 = −C15, C46 = −C15, C55 = C44, C56 = C14, C66 = (C11 − C12)/2
'''
identity = lambda x: x

def C_cyclictet_eq(x):
    x[1,1] = x[0,0]
    x[1,2] = x[0,2]
    x[1,5] = -x[0,5]
    x[4,4] = x[3,3]
    return x

def C_trigonal_eq(x):
    x[1,1]=x[0,0]
    x[1,2]=x[0,2]
    x[1,3]=-x[0,3]
    x[1,4]=-x[0,4]
    x[3,5]=-x[0,4]
    x[4,4]=x[3,3]
    x[4,5]=x[0,3]
    x[5,5]=0.5*(x[0,0]-x[0,1])
    return x

def C_cubic_eq(x):
    x[0,2] = x[0,1]
    x[2,2] = x[0,0]
    x[5,5] = x[3,3]
    x[1,1] = x[0,0]
    x[1,2] = x[0,2]
    x[1,5] = -x[0,5]
    x[4,4] = x[3,3]
    return x

_StiffnessDict = {
    laue_1: [type1,identity], # triclinic, all 21 components in upper triangular matrix needed
    laue_2: [type2,identity], # monoclinic, 13 components needed
    laue_3: [type3,identity], # orthorhombic, 9 components needed
    laue_4: [type4,C_cyclictet_eq], # cyclic tetragonal, 7 components needed
    laue_5: [type5,C_cyclictet_eq], # dihedral tetragonal, 6 components needed
    laue_6: [type6,C_trigonal_eq], # trigonal I, 7 components
    laue_7: [type7,C_trigonal_eq], # rhombohedral, 6 components
    laue_8: [type8,C_trigonal_eq], # cyclic hexagonal, 5 components needed
    laue_9: [type8,C_trigonal_eq], # dihedral hexagonal, 5 components
    laue_10:[type9,C_cubic_eq],# cubic, 3 components
    laue_11:[type9,C_cubic_eq] # cubic, 3 components
}
