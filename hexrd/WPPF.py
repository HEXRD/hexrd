import importlib.resources
import numpy as np
import warnings
from hexrd.imageutil import snip1d
from hexrd.crystallography import PlaneData
from hexrd.material import Material
from hexrd.valunits import valWUnit
from hexrd.utils.multiprocess_generic import GenericMultiprocessing
from hexrd import spacegroup as SG
from hexrd import symmetry, symbols, constants
import hexrd.resources
import lmfit
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy import signal
from hexrd.valunits import valWUnit
import yaml
from os import path
import pickle
import time
import h5py
from pathlib import Path
from pylab import plot, ginput, show, axis, close, title


class Parameters:
    """
    ==================================================================================
    ==================================================================================

    >> @AUTHOR:     Saransh Singh, Lanwrence Livermore National Lab,
                    saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
    >> @DETAILS:    this is the parameter class which handles all refinement parameters
        for both the Rietveld and the LeBail refimentment problems

        ===============================================================================
        ===============================================================================
    """

    def __init__(self,
                 name=None,
                 vary=False,
                 value=0.0,
                 lb=-np.Inf,
                 ub=np.Inf):

        self.param_dict = {}

        if(name is not None):
            self.add(name=name,
                     vary=vary,
                     value=value,
                     lb=min,
                     ub=max)

    def add(self,
            name,
            vary=False,
            value=0.0,
            lb=-np.Inf,
            ub=np.Inf):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       05/18/2020 SS 1.0 original
            >> @DETAILS:    add a single named parameter
        '''
        self[name] = Parameter(name=name, vary=vary, value=value, lb=lb, ub=ub)

    def add_many(self,
                 names,
                 varies,
                 values,
                 lbs,
                 ubs):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       05/18/2020 SS 1.0 original
            >> @DETAILS:    load a list of named parameters
        '''
        assert len(names) == len(varies), "lengths of tuples not consistent"
        assert len(names) == len(values), "lengths of tuples not consistent"
        assert len(names) == len(lbs), "lengths of tuples not consistent"
        assert len(names) == len(ubs), "lengths of tuples not consistent"

        for i, n in enumerate(names):
            self.add(n, vary=varies[i], value=values[i], lb=lbs[i], ub=ubs[i])

    def load(self, fname):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       05/18/2020 SS 1.0 original
            >> @DETAILS:    load parameters from yaml file
        '''
        with open(fname) as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        for k in dic.keys():
            v = dic[k]
            self.add(k, value=np.float(v[0]), lb=np.float(v[1]),
                     ub=np.float(v[2]), vary=np.bool(v[3]))

    def dump(self, fname):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       05/18/2020 SS 1.0 original
            >> @DETAILS:    dump the class to a yaml looking file. name is the key and the list
                            has [value, lb, ub, vary] in that order
        '''
        dic = {}
        for k in self.param_dict.keys():
            dic[k] = [self[k].value, self[k].lb, self[k].ub, self[k].vary]

        with open(fname, 'w') as f:
            data = yaml.dump(dic, f, sort_keys=False)

    def dump_hdf5(self, file):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       01/15/2021 SS 1.0 original
            >> @DETAILS:    dump the class to a hdf5 file. the file argument could either be a 
                            string or a h5.File instance. If it is a filename, then HDF5 file
                            is created, a parameter group is created and data is written out
                            with data names being the parameter name. Else data written to Parameter
                            group in existing file object
        '''
        if(isinstance(file, str)):
            fexist = path.isfile(file)
            if(fexist):
                fid = h5py.File(file, 'r+')
            else:
                fid = h5py.File(file, 'x')

        elif(isinstance(file, h5py.File)):
            fid = file

        else:
            raise RuntimeError(
                'Parameters: dump_hdf5 Pass in a \
                 filename string or h5py.File object')

        if("/Parameters" in fid):
            del(fid["Parameters"])
        gid_top = fid.create_group("Parameters")

        for p in self:
            param = self[p]
            gid = gid_top.create_group(p)

            # write the value, lower and upper bounds and vary status
            did = gid.create_dataset("value", (1, ), dtype=np.float64)
            did.write_direct(np.array(param.value, dtype=np.float64))

            did = gid.create_dataset("lb", (1, ), dtype=np.float64)
            did.write_direct(np.array(param.lb, dtype=np.float64))

            did = gid.create_dataset("ub", (1, ), dtype=np.float64)
            did.write_direct(np.array(param.ub, dtype=np.float64))

            did = gid.create_dataset("vary", (1, ), dtype=np.bool)
            did.write_direct(np.array(param.vary, dtype=np.bool))

    def __getitem__(self, key):
        if(key in self.param_dict.keys()):
            return self.param_dict[key]
        else:
            raise ValueError('variable with name not found')

    def __setitem__(self, key, parm_cls):

        if(key in self.param_dict.keys()):
            warnings.warn(
                'variable already in parameter list. overwriting ...')
        if(isinstance(parm_cls, Parameter)):
            self.param_dict[key] = parm_cls
        else:
            raise ValueError('input not a Parameter class')

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if(self.n < len(self.param_dict.keys())):
            res = list(self.param_dict.keys())[self.n]
            self.n += 1
            return res
        else:
            raise StopIteration

    def __str__(self):
        retstr = 'Parameters{\n'
        for k in self.param_dict.keys():
            retstr += self[k].__str__()+'\n'

        retstr += '}'
        return retstr


class Parameter:
    """
    ===================================================================================
    ===================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
    >> @DETAILS:    the parameters class (previous one) is a collection of this
                    parameter class indexed by the name of each variable

    ================================================================================
    =================================================================================
    """

    def __init__(self,
                 name=None,
                 vary=False,
                 value=0.0,
                 lb=-np.Inf,
                 ub=np.Inf):

        self.name = name
        self.vary = vary
        self.value = value
        self.lb = lb
        self.ub = ub

    def __str__(self):
        retstr = '< Parameter \''+self.name+'\'; value : ' + \
            str(self.value)+'; bounds : ['+str(self.lb)+',' + \
            str(self.ub)+' ]; vary :'+str(self.vary)+' >'

        return retstr

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if(isinstance(name, str)):
            self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, minval):
        self._min = minval

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, maxval):
        self._max = maxval

    @property
    def vary(self):
        return self._vary

    @vary.setter
    def vary(self, vary):
        if(isinstance(vary, bool)):
            self._vary = vary


class Spectrum:
    """
    ==================================================================================
    ==================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
    >> @DETAILS:    spectrum class holds the a pair of x,y data, in this case, would be
                    2theta-intensity values

    ==================================================================================
    ==================================================================================
    """

    def __init__(self, x=None, y=None, name=''):
        if x is None:
            self._x = np.linspace(10., 100., 500)
        else:
            self._x = x
        if y is None:
            self._y = np.log(self._x ** 2) - (self._x * 0.2) ** 2
        else:
            self._y = y
        self.name = name
        self.offset = 0
        self._scaling = 1
        self.smoothing = 0
        self.bkg_Spectrum = None

    @staticmethod
    def from_file(filename, skip_rows=0):
        try:
            if filename.endswith('.chi'):
                skip_rows = 4
            data = np.loadtxt(filename, skiprows=skip_rows)
            x = data.T[0]
            y = data.T[1]
            name = path.basename(filename).split('.')[:-1][0]
            return Spectrum(x, y, name)

        except ValueError:
            print('Wrong data format for spectrum file! - ' + filename)
            return -1

    def save(self, filename, header=''):
        data = np.dstack((self._x, self._y))
        np.savetxt(filename, data[0], header=header)

    def set_background(self, Spectrum):
        self.bkg_spectrum = Spectrum

    def reset_background(self):
        self.bkg_spectrum = None

    def set_smoothing(self, amount):
        self.smoothing = amount

    def rebin(self, bin_size):
        """
        Returns a new Spectrum which is a rebinned version of the current one.
        """
        x, y = self.data
        x_min = np.round(np.min(x) / bin_size) * bin_size
        x_max = np.round(np.max(x) / bin_size) * bin_size
        new_x = np.arange(x_min, x_max + 0.1 * bin_size, bin_size)

        bins = np.hstack((x_min - bin_size * 0.5, new_x + bin_size * 0.5))
        new_y = (np.histogram(x, bins, weights=y)
                 [0] / np.histogram(x, bins)[0])

        return Spectrum(new_x, new_y)

    def dump_hdf5(self, file, name):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       01/15/2021 SS 1.0 original
            >> @DETAILS:    dump the class to a hdf5 file. the file argument could either be a 
                            string or a h5.File instance. If it is a filename, then HDF5 file
                            is created, a Spectrum group is created and data is written out.
                            Else data written to Spectrum group in existing file object
            >> @PARAMS file file name string or h5py.File object
                       name name ID of the spectrum e.g. experimental or simulated or background
        '''

        if(isinstance(file, str)):
            fexist = path.isfile(file)
            if(fexist):
                fid = h5py.File(file, 'r+')
            else:
                fid = h5py.File(file, 'x')

        elif(isinstance(file, h5py.File)):
            fid = file

        else:
            raise RuntimeError(
                'Parameters: dump_hdf5 Pass in a filename \
                 string or h5py.File object')

        name_spectrum = 'Spectrum/'+name
        if(name_spectrum in fid):
            del(fid[name_spectrum])
        gid = fid.create_group(name_spectrum)

        tth, I = self.data

        # make sure these arrays are not zero sized
        if(tth.shape[0] > 0):
            did = gid.create_dataset("tth", tth.shape, dtype=np.float64)
            did.write_direct(tth.astype(np.float64))

        if(I.shape[0] > 0):
            did = gid.create_dataset("intensity", I.shape, dtype=np.float64)
            did.write_direct(I.astype(np.float64))

    @property
    def data(self):
        if self.bkg_Spectrum is not None:
            # create background function
            x_bkg, y_bkg = self.bkg_Spectrum.data

            if not np.array_equal(x_bkg, self._x):
                # the background will be interpolated
                f_bkg = interp1d(x_bkg, y_bkg, kind='linear')

                # find overlapping x and y values:
                ind = np.where((self._x <= np.max(x_bkg)) &
                               (self._x >= np.min(x_bkg)))
                x = self._x[ind]
                y = self._y[ind]

                if len(x) == 0:
                    """ if there is no overlapping between background
                     and Spectrum, raise an error """
                    raise BkgNotInRangeError(self.name)

                y = y * self._scaling + self.offset - f_bkg(x)
            else:
                """ if Spectrum and bkg have the same
                 x basis we just delete y-y_bkg"""
                x, y = self._x, self._y * \
                    self._scaling + self.offset - y_bkg
        else:
            x, y = self.original_data

        if self.smoothing > 0:
            y = gaussian_filter1d(y, self.smoothing)
        return x, y

    @data.setter
    def data(self, data):
        (x, y) = data
        self._x = x
        self._y = y
        self.scaling = 1
        self.offset = 0

    @property
    def original_data(self):
        return self._x, self._y * self._scaling +\
            self.offset

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, new_value):
        self._x = new_value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, new_y):
        self._y = new_y

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        if value < 0:
            self._scaling = 0
        else:
            self._scaling = value

    def limit(self, x_min, x_max):
        x, y = self.data
        return Spectrum(x[np.where((x_min < x) & (x < x_max))],
                        y[np.where((x_min < x) & (x < x_max))])

    def extend_to(self, x_value, y_value):
        """
        Extends the current Spectrum to a specific x_value by filling it 
        with the y_value. Does not modify inplace but returns a new filled 
        Spectrum
        :param x_value: Point to which extend the Spectrum should be smaller
        than the lowest x-value in the Spectrum or vice versa
        :param y_value: number to fill the Spectrum with
        :return: extended Spectrum
        """
        x_step = np.mean(np.diff(self.x))
        x_min = np.min(self.x)
        x_max = np.max(self.x)
        if x_value < x_min:
            x_fill = np.arange(x_min - x_step, x_value -
                               x_step*0.5, -x_step)[::-1]
            y_fill = np.zeros(x_fill.shape)
            y_fill.fill(y_value)

            new_x = np.concatenate((x_fill, self.x))
            new_y = np.concatenate((y_fill, self.y))
        elif x_value > x_max:
            x_fill = np.arange(x_max + x_step, x_value+x_step*0.5, x_step)
            y_fill = np.zeros(x_fill.shape)
            y_fill.fill(y_value)

            new_x = np.concatenate((self.x, x_fill))
            new_y = np.concatenate((self.y, y_fill))
        else:
            return self

        return Spectrum(new_x, new_y)

    def plot(self, show=False, *args, **kwargs):
        plt.plot(self.x, self.y, *args, **kwargs)
        if show:
            plt.show()

    def nan_to_zero(self):
        """
        set the nan in spectrum to zero
        sometimes integrated spectrum in data can
        have some nans, so need to catch those
        """
        self._y = np.nan_to_num(self._y, copy=False, nan=0.0)

    # Operators:
    def __sub__(self, other):
        orig_x, orig_y = self.data
        other_x, other_y = other.data

        if orig_x.shape != other_x.shape:
            # todo different shape subtraction of spectra
            # seems the fail somehow...
            # the background will be interpolated
            other_fcn = interp1d(other_x, other_x, kind='linear')

            # find overlapping x and y values:
            ind = np.where((orig_x <= np.max(other_x)) &
                           (orig_x >= np.min(other_x)))
            x = orig_x[ind]
            y = orig_y[ind]

            if len(x) == 0:
                # if there is no overlapping between
                # background and Spectrum, raise an error
                raise BkgNotInRangeError(self.name)
            return Spectrum(x, y - other_fcn(x))
        else:
            return Spectrum(orig_x, orig_y - other_y)

    def __add__(self, other):
        orig_x, orig_y = self.data
        other_x, other_y = other.data

        if orig_x.shape != other_x.shape:
            # the background will be interpolated
            other_fcn = interp1d(other_x, other_x, kind='linear')

            # find overlapping x and y values:
            ind = np.where((orig_x <= np.max(other_x)) &
                           (orig_x >= np.min(other_x)))
            x = orig_x[ind]
            y = orig_y[ind]

            if len(x) == 0:
                # if there is no overlapping between
                # background and Spectrum, raise an error
                raise BkgNotInRangeError(self.name)
            return Spectrum(x, y + other_fcn(x))
        else:
            return Spectrum(orig_x, orig_y + other_y)

    def __rmul__(self, other):
        orig_x, orig_y = self.data
        return Spectrum(np.copy(orig_x), np.copy(orig_y) * other)

    def __eq__(self, other):
        if not isinstance(other, Spectrum):
            return False
        if np.array_equal(self.data, other.data):
            return True
        return False


class Material_LeBail:
    ''' 
    ========================================================================================
    ========================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
                    09/14/2020 SS 1.1 class can now be initialized using
                    a material.Material class instance
    >> @DETAILS:    Material_LeBail class is a stripped down version of the 
                    materials.Material class.this is done to keep the class lightweight 
                    and make sure only the information necessary for the lebail fit is kept

    =========================================================================================
    =========================================================================================
    '''

    def __init__(self,
                 fhdf=None,
                 xtal=None,
                 dmin=None,
                 material_obj=None):

        if(material_obj is None):
            self.dmin = dmin.value
            self._readHDF(fhdf, xtal)
            self._calcrmt()

            _, self.SYM_PG_d, self.SYM_PG_d_laue, \
                self.centrosymmetric, self.symmorphic = \
                symmetry.GenerateSGSym(self.sgnum, self.sgsetting)
            self.latticeType = symmetry.latticeType(self.sgnum)
            self.sg_hmsymbol = symbols.pstr_spacegroup[self.sgnum-1].strip()
            self.GenerateRecipPGSym()
            self.CalcMaxGIndex()
            self._calchkls()
        else:
            if(isinstance(material_obj, Material)):
                self._init_from_materials(material_obj)
            else:
                raise ValueError(
                    "Invalid material_obj argument. \
                    only Material class can be passed here.")

    def _readHDF(self, fhdf, xtal):

        # fexist = path.exists(fhdf)
        # if(fexist):
        fid = h5py.File(fhdf, 'r')
        xtal = "/"+xtal
        if xtal not in fid:
            raise IOError('crystal doesn''t exist in material file.')
        # else:
        #   raise IOError('material file does not exist.')

        gid = fid.get(xtal)

        self.sgnum = np.asscalar(np.array(gid.get('SpaceGroupNumber'),
                                          dtype=np.int32))
        self.sgsetting = np.asscalar(np.array(gid.get('SpaceGroupSetting'),
                                              dtype=np.int32))
        """
            IMPORTANT NOTE:
            note that the latice parameters is nm by default
            hexrd on the other hand uses A as the default units, so we
            need to be careful and convert it right here, so there is no
            confusion later on
        """
        self.lparms = list(gid.get('LatticeParameters'))
        fid.close()

    def _init_from_materials(self, material_obj):
        '''
        this function is used to initialize the materials_lebail class
        from an instance of the material.Material class. this option is
        provided for easy integration of the hexrdgui with WPPF.
        '''
        self.dmin = material_obj.dmin.getVal('nm')
        self.sgnum = material_obj.unitcell.sgnum
        self.sgsetting = material_obj.sgsetting

        if(material_obj.latticeParameters[0].unit == 'nm'):
            self.lparms = [x.value for x in material_obj.latticeParameters]
        elif(material_obj.latticeParameters[0].unit == 'angstrom'):
            lparms = [x.value for x in material_obj.latticeParameters]
            for i in range(3):
                lparms[i] /= 10.0
            self.lparms = lparms

        self.dmt = material_obj.unitcell.dmt
        self.rmt = material_obj.unitcell.rmt
        self.vol = material_obj.unitcell.vol

        self.centrosymmetric = material_obj.unitcell.centrosymmetric
        self.symmorphic = material_obj.unitcell.symmorphic

        self.latticeType = material_obj.unitcell.latticeType
        self.sg_hmsymbol = material_obj.unitcell.sg_hmsymbol

        self.ih = material_obj.unitcell.ih
        self.ik = material_obj.unitcell.ik
        self.il = material_obj.unitcell.il

        self.SYM_PG_d = material_obj.unitcell.SYM_PG_d
        self.SYM_PG_d_laue = material_obj.unitcell.SYM_PG_d_laue
        self.SYM_PG_r = material_obj.unitcell.SYM_PG_r
        self.SYM_PG_r_laue = material_obj.unitcell.SYM_PG_r_laue

        self.hkls = material_obj.planeData.getHKLs()

    def _calcrmt(self):

        a = self.lparms[0]
        b = self.lparms[1]
        c = self.lparms[2]

        alpha = np.radians(self.lparms[3])
        beta = np.radians(self.lparms[4])
        gamma = np.radians(self.lparms[5])

        ca = np.cos(alpha)
        cb = np.cos(beta)
        cg = np.cos(gamma)
        sa = np.sin(alpha)
        sb = np.sin(beta)
        sg = np.sin(gamma)
        tg = np.tan(gamma)

        '''
            direct metric tensor
        '''
        self.dmt = np.array([[a**2, a*b*cg, a*c*cb],
                             [a*b*cg, b**2, b*c*ca],
                             [a*c*cb, b*c*ca, c**2]])
        self.vol = np.sqrt(np.linalg.det(self.dmt))

        if(self.vol < 1e-5):
            warnings.warn('unitcell volume is suspiciously small')

        '''
            reciprocal metric tensor
        '''
        self.rmt = np.linalg.inv(self.dmt)

    def _calchkls(self):
        self.hkls = self.getHKLs(self.dmin)

    ''' calculate dot product of two vectors in any space 'd' 'r' or 'c' '''

    def CalcLength(self, u, space):

        if(space == 'd'):
            vlen = np.sqrt(np.dot(u, np.dot(self.dmt, u)))
        elif(space == 'r'):
            vlen = np.sqrt(np.dot(u, np.dot(self.rmt, u)))
        elif(spec == 'c'):
            vlen = np.linalg.norm(u)
        else:
            raise ValueError('incorrect space argument')

        return vlen

    def getTTh(self, wavelength):

        tth = []
        self.wavelength_allowed_hkls = []
        for g in self.hkls:
            glen = self.CalcLength(g, 'r')
            sth = glen*wavelength/2.
            if(np.abs(sth) <= 1.0):
                t = 2. * np.degrees(np.arcsin(sth))
                tth.append(t)
                self.wavelength_allowed_hkls.append(True)
            else:
                self.wavelength_allowed_hkls.append(False)
        tth = np.array(tth)
        return tth

    def GenerateRecipPGSym(self):

        self.SYM_PG_r = self.SYM_PG_d[0, :, :]
        self.SYM_PG_r = np.broadcast_to(self.SYM_PG_r, [1, 3, 3])
        self.SYM_PG_r_laue = self.SYM_PG_d[0, :, :]
        self.SYM_PG_r_laue = np.broadcast_to(self.SYM_PG_r_laue, [1, 3, 3])

        for i in range(1, self.SYM_PG_d.shape[0]):
            g = self.SYM_PG_d[i, :, :]
            g = np.dot(self.dmt, np.dot(g, self.rmt))
            g = np.round(np.broadcast_to(g, [1, 3, 3]))
            self.SYM_PG_r = np.concatenate((self.SYM_PG_r, g))

        for i in range(1, self.SYM_PG_d_laue.shape[0]):
            g = self.SYM_PG_d_laue[i, :, :]
            g = np.dot(self.dmt, np.dot(g, self.rmt))
            g = np.round(np.broadcast_to(g, [1, 3, 3]))
            self.SYM_PG_r_laue = np.concatenate((self.SYM_PG_r_laue, g))

        self.SYM_PG_r = self.SYM_PG_r.astype(np.int32)
        self.SYM_PG_r_laue = self.SYM_PG_r_laue.astype(np.int32)

    def CalcMaxGIndex(self):
        self.ih = 1
        while (1.0 / self.CalcLength(
                np.array([self.ih, 0, 0], dtype=np.float64), 'r')
                > self.dmin):
            self.ih = self.ih + 1
        self.ik = 1
        while (1.0 / self.CalcLength(
                np.array([0, self.ik, 0], dtype=np.float64), 'r')
                > self.dmin):
            self.ik = self.ik + 1
        self.il = 1
        while (1.0 / self.CalcLength(
                np.array([0, 0, self.il], dtype=np.float64), 'r')
                > self.dmin):
            self.il = self.il + 1

    def CalcStar(self, v, space, applyLaue=False):
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
            vp = np.dot(s, v)
            # check if this is new
            isnew = True
            for vec in vsym:
                if(np.sum(np.abs(vp - vec)) < 1E-4):
                    isnew = False
                    break
            if(isnew):
                vsym = np.vstack((vsym, vp))
        return vsym

    def Allowed_HKLs(self, hkllist):
        '''
        this function checks if a particular g vector is allowed
        by lattice centering, screw axis or glide plane
        '''
        hkllist = np.atleast_2d(hkllist)
        centering = self.sg_hmsymbol[0]
        if(centering == 'P'):
            # all reflections are allowed
            mask = np.ones([hkllist.shape[0], ], dtype=np.bool)
        elif(centering == 'F'):
            # same parity
            seo = np.sum(np.mod(hkllist+100, 2), axis=1)
            mask = np.logical_not(np.logical_or(seo == 1, seo == 2))
        elif(centering == 'I'):
            # sum is even
            seo = np.mod(np.sum(hkllist, axis=1)+100, 2)
            mask = (seo == 0)

        elif(centering == 'A'):
            # k+l is even
            seo = np.mod(np.sum(hkllist[:, 1:3], axis=1)+100, 2)
            mask = seo == 0
        elif(centering == 'B'):
            # h+l is even
            seo = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2)
            mask = seo == 0
        elif(centering == 'C'):
            # h+k is even
            seo = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2)
            mask = seo == 0
        elif(centering == 'R'):
            # -h+k+l is divisible by 3
            seo = np.mod(-hkllist[:, 0]+hkllist[:, 1]+hkllist[:, 2]+90, 3)
            mask = seo == 0
        else:
            raise RuntimeError(
                'IsGAllowed: unknown lattice centering encountered.')
        hkls = hkllist[mask, :]

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
                raise RuntimeError(
                    'omitscrewaxisabsences: monoclinic systems\
                     can only have 2_1 screw axis')
            '''
                only unique b-axis will be encoded
                it is the users responsibility to input
                lattice parameters in the standard setting
                with b-axis having the 2-fold symmetry
            '''
            if(iax == 1):
                mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
                mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            else:
                raise RuntimeError(
                    'omitscrewaxisabsences: only b-axis\
                     can have 2_1 screw axis')
        elif(self.latticeType == 'orthorhombic'):
            if(ax != '2_1'):
                raise RuntimeError(
                    'omitscrewaxisabsences: orthorhombic systems\
                     can only have 2_1 screw axis')
            '''
            2_1 screw on primary axis
            h00 ; h = 2n
            '''
            if(iax == 0):
                mask1 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
                mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(iax == 1):
                mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
                mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(iax == 2):
                mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
                mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'tetragonal'):
            if(iax == 0):
                mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
                if(ax == '4_2'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(ax in ['4_1', '4_3']):
                    mask2 = np.mod(hkllist[:, 2]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(iax == 1):
                mask1 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
                mask2 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
                if(ax == '2_1'):
                    mask3 = np.mod(hkllist[:, 0]+100, 2) != 0
                    mask4 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask1 = np.logical_not(np.logical_and(mask1, mask3))
                mask2 = np.logical_not(np.logical_and(mask2, mask4))
                mask = ~np.logical_or(~mask1, ~mask2)
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'trigonal'):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            if(iax == 0):
                if(ax in ['3_1', '3_2']):
                    mask2 = np.mod(hkllist[:, 2]+90, 3) != 0
            else:
                raise RuntimeError(
                    'omitscrewaxisabsences: trigonal \
                    systems can only have screw axis')
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif(self.latticeType == 'hexagonal'):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            if(iax == 0):
                if(ax == '6_3'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(ax in['3_1', '3_2', '6_2', '6_4']):
                    mask2 = np.mod(hkllist[:, 2]+90, 3) != 0
                elif(ax in ['6_1', '6_5']):
                    mask2 = np.mod(hkllist[:, 2]+120, 6) != 0
            else:
                raise RuntimeError(
                    'omitscrewaxisabsences: hexagonal \
                    systems can only have screw axis')
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif(self.latticeType == 'cubic'):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            mask2 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
            mask3 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
            if(ax in ['2_1', '4_2']):
                mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                mask5 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask6 = np.mod(hkllist[:, 0]+100, 2) != 0
            elif(ax in ['4_1', '4_3']):
                mask4 = np.mod(hkllist[:, 2]+100, 4) != 0
                mask5 = np.mod(hkllist[:, 1]+100, 4) != 0
                mask6 = np.mod(hkllist[:, 0]+100, 4) != 0
            mask1 = np.logical_not(np.logical_and(mask1, mask4))
            mask2 = np.logical_not(np.logical_and(mask2, mask5))
            mask3 = np.logical_not(np.logical_and(mask3, mask6))
            mask = ~np.logical_or(~mask1, np.logical_or(~mask2, ~mask3))
            hkllist = hkllist[mask, :]
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
                mask1 = hkllist[:, 1] == 0
                if(plane == 'c'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(plane == 'a'):
                    mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'orthorhombic'):
            if(ip == 0):
                mask1 = hkllist[:, 0] == 0
                if(plane == 'b'):
                    mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                elif(plane == 'c'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(ip == 1):
                mask1 = hkllist[:, 1] == 0
                if(plane == 'c'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(plane == 'a'):
                    mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(ip == 2):
                mask1 = hkllist[:, 2] == 0
                if(plane == 'a'):
                    mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'b'):
                    mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'tetragonal'):
            if(ip == 0):
                mask1 = hkllist[:, 2] == 0
                if(plane == 'a'):
                    mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'b'):
                    mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(ip == 1):
                mask1 = hkllist[:, 0] == 0
                mask2 = hkllist[:, 1] == 0
                if(plane in ['a', 'b']):
                    mask3 = np.mod(hkllist[:, 1]+100, 2) != 0
                    mask4 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'c'):
                    mask3 = np.mod(hkllist[:, 2]+100, 2) != 0
                    mask4 = mask3
                elif(plane == 'n'):
                    mask3 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 2) != 0
                    mask4 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
                elif(plane == 'd'):
                    mask3 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 4) != 0
                    mask4 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                mask1 = np.logical_not(np.logical_and(mask1, mask3))
                mask2 = np.logical_not(np.logical_and(mask2, mask4))
                mask = ~np.logical_or(~mask1, ~mask2)
                hkllist = hkllist[mask, :]
            elif(ip == 2):
                mask1 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 1])
                if(plane in ['c', 'n']):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(2*hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'trigonal'):
            if(plane != 'c'):
                raise RuntimeError(
                    'omitglideplaneabsences: only c-glide \
                    allowed for trigonal systems')

            if(ip == 1):
                mask1 = hkllist[:, 0] == 0
                mask2 = hkllist[:, 1] == 0
                mask3 = hkllist[:, 0] == -hkllist[:, 1]
                if(plane == 'c'):
                    mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                else:
                    raise RuntimeError(
                        'omitglideplaneabsences: only c-glide \
                        allowed for trigonal systems')
            elif(ip == 2):
                mask1 = hkllist[:, 1] == hkllist[:, 0]
                mask2 = hkllist[:, 0] == -2*hkllist[:, 1]
                mask3 = -2*hkllist[:, 0] == hkllist[:, 1]
                if(plane == 'c'):
                    mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                else:
                    raise RuntimeError(
                        'omitglideplaneabsences: only c-glide \
                        allowed for trigonal systems')
            mask1 = np.logical_and(mask1, mask4)
            mask2 = np.logical_and(mask2, mask4)
            mask3 = np.logical_and(mask3, mask4)
            mask = np.logical_not(np.logical_or(
                mask1, np.logical_or(mask2, mask3)))
            hkllist = hkllist[mask, :]
        elif(self.latticeType == 'hexagonal'):
            if(plane != 'c'):
                raise RuntimeError(
                    'omitglideplaneabsences: only c-glide \
                    allowed for hexagonal systems')
            if(ip == 2):
                mask1 = hkllist[:, 0] == hkllist[:, 1]
                mask2 = hkllist[:, 0] == -2*hkllist[:, 1]
                mask3 = -2*hkllist[:, 0] == hkllist[:, 1]
                mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                mask1 = np.logical_and(mask1, mask4)
                mask2 = np.logical_and(mask2, mask4)
                mask3 = np.logical_and(mask3, mask4)
                mask = np.logical_not(np.logical_or(
                    mask1, np.logical_or(mask2, mask3)))
            elif(ip == 1):
                mask1 = hkllist[:, 1] == 0
                mask2 = hkllist[:, 0] == 0
                mask3 = hkllist[:, 1] == -hkllist[:, 0]
                mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
            mask1 = np.logical_and(mask1, mask4)
            mask2 = np.logical_and(mask2, mask4)
            mask3 = np.logical_and(mask3, mask4)
            mask = np.logical_not(np.logical_or(
                mask1, np.logical_or(mask2, mask3)))
            hkllist = hkllist[mask, :]
        elif(self.latticeType == 'cubic'):
            if(ip == 0):
                mask1 = hkllist[:, 0] == 0
                mask2 = hkllist[:, 1] == 0
                mask3 = hkllist[:, 2] == 0
                mask4 = np.mod(hkllist[:, 0]+100, 2) != 0
                mask5 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask6 = np.mod(hkllist[:, 2]+100, 2) != 0
                if(plane == 'a'):
                    mask1 = np.logical_or(np.logical_and(
                        mask1, mask5), np.logical_and(mask1, mask6))
                    mask2 = np.logical_or(np.logical_and(
                        mask2, mask4), np.logical_and(mask2, mask6))
                    mask3 = np.logical_and(mask3, mask4)

                    mask = np.logical_not(np.logical_or(
                        mask1, np.logical_or(mask2, mask3)))
                elif(plane == 'b'):
                    mask1 = np.logical_and(mask1, mask5)
                    mask3 = np.logical_and(mask3, mask5)
                    mask = np.logical_not(np.logical_or(mask1, mask3))
                elif(plane == 'c'):
                    mask1 = np.logical_and(mask1, mask6)
                    mask2 = np.logical_and(mask2, mask6)
                    mask = np.logical_not(np.logical_or(mask1, mask2))
                elif(plane == 'n'):
                    mask4 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 2) != 0
                    mask5 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
                    mask6 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2) != 0
                    mask1 = np.logical_not(np.logical_and(mask1, mask4))
                    mask2 = np.logical_not(np.logical_and(mask2, mask5))
                    mask3 = np.logical_not(np.logical_and(mask3, mask6))
                    mask = ~np.logical_or(
                        ~mask1, np.logical_or(~mask2, ~mask3))
                elif(plane == 'd'):
                    mask4 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 4) != 0
                    mask5 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                    mask6 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
                    mask1 = np.logical_not(np.logical_and(mask1, mask4))
                    mask2 = np.logical_not(np.logical_and(mask2, mask5))
                    mask3 = np.logical_not(np.logical_and(mask3, mask6))
                    mask = ~np.logical_or(
                        ~mask1, np.logical_or(~mask2, ~mask3))
                else:
                    raise RuntimeError(
                        'omitglideplaneabsences: unknown glide \
                        plane encountered.')
                hkllist = hkllist[mask, :]
            if(ip == 2):
                mask1 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 1])
                mask2 = np.abs(hkllist[:, 1]) == np.abs(hkllist[:, 2])
                mask3 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 2])
                if(plane in ['a', 'b', 'c', 'n']):
                    mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                    mask5 = np.mod(hkllist[:, 0]+100, 2) != 0
                    mask6 = np.mod(hkllist[:, 1]+100, 2) != 0
                elif(plane == 'd'):
                    mask4 = np.mod(2*hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                    mask5 = np.mod(hkllist[:, 0]+2*hkllist[:, 1]+100, 4) != 0
                    mask6 = np.mod(2*hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
                else:
                    raise RuntimeError(
                        'omitglideplaneabsences: unknown glide \
                        plane encountered.')
                mask1 = np.logical_not(np.logical_and(mask1, mask4))
                mask2 = np.logical_not(np.logical_and(mask2, mask5))
                mask3 = np.logical_not(np.logical_and(mask3, mask6))
                mask = ~np.logical_or(~mask1, np.logical_or(~mask2, ~mask3))
                hkllist = hkllist[mask, :]
        return hkllist

    def NonSymmorphicAbsences(self, hkllist):
        '''
        this function prunes hkl list for the screw axis and glide
        plane absences
        '''
        planes = constants.SYS_AB[self.sgnum][0]
        for ip, p in enumerate(planes):
            if(p != ''):
                hkllist = self.omitglideplaneabsences(hkllist, p, ip)
        axes = constants.SYS_AB[self.sgnum][1]
        for iax, ax in enumerate(axes):
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
        mask = np.ones(hkllist.shape[0], dtype=np.bool)
        laue = InversionSymmetry
        for i, g in enumerate(hkllist):
            if(mask[i]):
                geqv = self.CalcStar(g, 'r', applyLaue=laue)
                for r in geqv[1:, ]:
                    rid = np.where(np.all(r == hkllist, axis=1))
                    mask[rid] = False
        hkl = hkllist[mask, :].astype(np.int32)
        hkl_max = []
        for g in hkl:
            geqv = self.CalcStar(g, 'r', applyLaue=laue)
            loc = np.argmax(np.sum(geqv, axis=1))
            gmax = geqv[loc, :]
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
            glen.append(np.round(self.CalcLength(g, 'r'), 8))
        # glen = np.atleast_2d(np.array(glen,dtype=np.float)).T
        dtype = [('glen', float), ('max', int), ('sum', int),
                 ('h', int), ('k', int), ('l', int)]
        a = []
        for i, gl in enumerate(glen):
            g = hkllist[i, :]
            a.append((gl, np.max(g), np.sum(g), g[0], g[1], g[2]))
        a = np.array(a, dtype=dtype)
        isort = np.argsort(a, order=['glen', 'max', 'sum', 'l', 'k', 'h'])
        return hkllist[isort, :]

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
        hkllist = np.array([[ih, ik, il] for ih in np.arange(hmax, hmin, -1)
                            for ik in np.arange(kmax, kmin, -1)
                            for il in np.arange(lmax, lmin, -1)])
        hkl_allowed = self.Allowed_HKLs(hkllist)
        hkl = []
        dsp = []
        hkl_dsp = []
        for g in hkl_allowed:
            # ignore [0 0 0] as it is the direct beam
            if(np.sum(np.abs(g)) != 0):
                dspace = 1./self.CalcLength(g, 'r')
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
        self.hkl = self.SortHKL(hkl)
        return self.hkl

    def Required_lp(self, p):
        return _rqpDict[self.latticeType][1](p)


class Phases_LeBail:
    """
    ========================================================================================
    ========================================================================================
    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/20/2020 SS 1.0 original
    >> @DETAILS:    class to handle different phases in the LeBail fit. this is a stripped down
                    version of main Phase class for efficiency. only the 
                    components necessary for calculating peak positions are retained. further 
                    this will have a slight modification to account for different wavelengths 
                    in the same phase name
    =========================================================================================
    =========================================================================================
    """
    def _kev(x):
        return valWUnit('beamenergy', 'energy', x, 'keV')

    def _nm(x):
        return valWUnit('lp', 'length', x, 'nm')

    def __init__(self, material_file=None,
                 material_keys=None,
                 dmin=_nm(0.05),
                 wavelength={'alpha1': [_nm(0.15406), 1.0],
                 'alpha2': [_nm(0.154443), 0.52]}
                 ):

        self.phase_dict = {}
        self.num_phases = 0

        '''
        set wavelength. check if wavelength is supplied in A, if it is
        convert to nm since the rest of the code assumes those units
        '''
        wavelength_nm = {}
        for k, v in wavelength.items():
            wavelength_nm[k] = [valWUnit('lp', 'length', \
                v[0].getVal('nm'), 'nm'), v[1]]

        self.wavelength = wavelength_nm

        self.dmin = dmin

        if(material_file is not None):
            if(material_keys is not None):
                if(type(material_keys) is not list):
                    self.add(material_file, material_keys)
                else:
                    self.add_many(material_file, material_keys)

    def __str__(self):
        resstr = 'Phases in calculation:\n'
        for i, k in enumerate(self.phase_dict.keys()):
            resstr += '\t'+str(i+1)+'. '+k+'\n'
        return resstr

    def __getitem__(self, key):
        if(key in self.phase_dict.keys()):
            return self.phase_dict[key]
        else:
            raise ValueError('phase with name not found')

    def __setitem__(self, key, mat_cls):

        if(key in self.phase_dict.keys()):
            warnings.warn('phase already in parameter \
                list. overwriting ...')
        if(isinstance(mat_cls, Material_LeBail)):
            self.phase_dict[key] = mat_cls
        else:
            raise ValueError('input not a material class')

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if(self.n < len(self.phase_dict.keys())):
            res = list(self.phase_dict.keys())[self.n]
            self.n += 1
            return res
        else:
            raise StopIteration

    def __len__(self):
        return len(self.phase_dict)

    def add(self, material_file, material_key):

        self[material_key] = Material_LeBail(
            fhdf=material_file, xtal=material_key, dmin=self.dmin)

    def add_many(self, material_file, material_keys):

        for k in material_keys:

            self[k] = Material_LeBail(
                fhdf=material_file, xtal=k, dmin=self.dmin)

            self.num_phases += 1

        for k in self:
            self[k].pf = 1.0/len(self)

        self.material_file = material_file
        self.material_keys = material_keys

    def load(self, fname):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       06/08/2020 SS 1.0 original
            >> @DETAILS:    load parameters from yaml file
        '''
        with open(fname) as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        for mfile in dic.keys():
            mat_keys = list(dic[mfile])
            self.add_many(mfile, mat_keys)

    def dump(self, fname):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       06/08/2020 SS 1.0 original
            >> @DETAILS:    dump parameters to yaml file
        '''
        dic = {}
        k = self.material_file
        dic[k] = [m for m in self]

        with open(fname, 'w') as f:
            data = yaml.dump(dic, f, sort_keys=False)

    def dump_hdf5(self, file):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       01/15/2021 SS 1.0 original
        >> @ DETAILS    dumps the information from each material in the phase class
                        to a hdf5 file specified by filename or h5py.File object
        '''
        if(isinstance(file, str)):
            fexist = path.isfile(file)
            if(fexist):
                fid = h5py.File(file, 'r+')
            else:
                fid = h5py.File(file, 'x')

        elif(isinstance(file, h5py.File)):
            fid = file

        else:
            raise RuntimeError(
                'Parameters: dump_hdf5 Pass in a filename \
                string or h5py.File object')

        if("/Phases" in fid):
            del(fid["Phases"])
        gid_top = fid.create_group("Phases")

        for p in self:
            mat = self[p]

            sgnum = mat.sgnum
            sgsetting = mat.sgsetting
            lparms = mat.lparms
            dmin = mat.dmin
            hkls = mat.hkls

            gid = gid_top.create_group(p)

            did = gid.create_dataset("SpaceGroupNumber", (1, ), dtype=np.int32)
            did.write_direct(np.array(sgnum, dtype=np.int32))

            did = gid.create_dataset(
                "SpaceGroupSetting", (1, ), dtype=np.int32)
            did.write_direct(np.array(sgsetting, dtype=np.int32))

            did = gid.create_dataset(
                "LatticeParameters", (6, ), dtype=np.float64)
            did.write_direct(np.array(lparms, dtype=np.float64))

            did = gid.create_dataset("dmin", (1, ), dtype=np.float64)
            did.attrs["units"] = "nm"
            did.write_direct(np.array(dmin, dtype=np.float64))


class LeBail:
    """
    ========================================================================================================
    ========================================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/19/2020 SS 1.0 original
                    09/11/2020 SS 1.1 expt_spectrum, params and phases now have multiple input
                    option for easy integration with hexrdgui
                    09/14/2020 SS 1.2 bkgmethod is now a dictionary. if method is 'chebyshev'
                    the the value specifies the degree of the polynomial to use for background
                    estimation
                    01/22/2021 SS 1.3 added intensity_init option to initialize intensity with 
                    structure factors if the user so chooses
                    01/22/2021 SS 1.4 added option to specify background via a filename or numpy array

    >> @DETAILS:    this is the main LeBail class and contains all the refinable parameters
                    for the analysis. Since the LeBail method has no structural information
                    during refinement, the refinable parameters for this model will be:

                    1. a, b, c, alpha, beta, gamma : unit cell parameters
                    2. U, V, W : cagliotti paramaters
                    3. 2theta_0 : Instrumental zero shift error
                    4. eta1, eta2, eta3 : weight factor for gaussian vs lorentzian

                    @NOTE: All angles are always going to be in degrees

    >> @PARAMETERS  expt_spectrum: name of file or numpy array or Spectrum class of experimental intensity
                    params: yaml file or dictionary or Parameter class
                    phases: yaml file or dictionary or Phases_Lebail class
                    wavelength: dictionary of wavelengths
                    bkgmethod: method to estimate background. either spline or chebyshev fit
                    or filename or numpy array (last two options added 01/22/2021 SS)
                    Intensity_init: if set to none, then some power of 10 is used. User has option
                    to pass in dictionary of structure factors. must ensure that the size of structure
                    factor matches the possible reflections (added 01/22/2021 SS)
    ========================================================================================================
    ========================================================================================================
    """
    def _nm(x):
        return valWUnit('lp', 'length', x, 'nm')

    def __init__(self,
                 expt_spectrum=None,
                 params=None,
                 phases=None,
                 wavelength={'kalpha1': [_nm(0.15406), 1.0],
                 'kalpha2': [_nm(0.154443), 1.0]},
                 bkgmethod={'spline': None},
                 intensity_init=None):

        self.bkgmethod = bkgmethod
        self.intensity_init = intensity_init

        self.initialize_expt_spectrum(expt_spectrum)

        if(wavelength is not None):
            self.wavelength = wavelength

        self._tstart = time.time()

        self.initialize_phases(phases)

        self.initialize_parameters(params)

        self.initialize_Icalc()

        self.computespectrum()

        self._tstop = time.time()
        self.tinit = self._tstop - self._tstart
        self.niter = 0
        self.Rwplist = np.empty([0])
        self.gofFlist = np.empty([0])

    def __str__(self):
        resstr = '<LeBail Fit class>\nParameters of \
        the model are as follows:\n'
        resstr += self.params.__str__()
        return resstr

    def checkangle(ang, name):

        if(np.abs(ang) > 180.):
            warnings.warn(name + " : the absolute value of angles \
                                seems to be large > 180 degrees")

    def initialize_parameters(self, param_info):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
                        09/11/2020 SS 1.1 modified to accept multiple input types
        >> @DETAILS:    initialize parameter list from file. if no file given, then initialize
                        to some default values (lattice constants are for CeO2)

        '''
        if(param_info is not None):
            if(isinstance(param_info, Parameters)):
                '''
                directly passing the parameter class
                '''
                self.params = param_info

            else:
                params = Parameters()

                if(isinstance(param_info, dict)):
                    '''
                    initialize class using dictionary read from the yaml file
                    '''
                    for k, v in param_info.items():
                        params.add(k, value=np.float(v[0]),
                                   lb=np.float(v[1]), ub=np.float(v[2]),
                                   vary=np.bool(v[3]))

                elif(isinstance(param_info, str)):
                    '''
                    load from a yaml file
                    '''
                    if(path.exists(param_info)):
                        params.load(param_info)
                    else:
                        raise FileError('input spectrum file doesn\'t exist.')

                '''
                this part initializes the lattice parameters in the
                '''
                for p in self.phases:

                    mat = self.phases[p]
                    lp = np.array(mat.lparms)
                    rid = list(_rqpDict[mat.latticeType][0])

                    lp = lp[rid]
                    name = _lpname[rid]

                    for n, l in zip(name, lp):
                        nn = p+'_'+n
                        '''
                        is l is small, it is one of the length units
                        else it is an angle
                        '''
                        if(l < 10.):
                            params.add(nn, value=l, lb=l-0.05,
                                       ub=l+0.05, vary=True)
                        else:
                            params.add(nn, value=l, lb=l-1.,
                                       ub=l+1., vary=True)

                self.params = params
        else:
            '''
                first entry is the lattice paramaters
                next three are cagliotti parameters
                next are the three gauss+lorentz mixing paramters
                final is the zero instrumental peak position error
            '''
            params = Parameters()
            names = []
            values = []
            for p in self.phases:
                names.append(p+'_a')
                values.append(self.phases[p].lparms[0])

            valdict = {'U': 1e-2, 'V': 1e-2, 'W': 1e-2,
                       'X': 1e-2, 'Y': 1e-2, 'zero_error': 0.}

            for k, v in valdict.items():
                names.append(k)
                values.append(v)

            names = tuple(names)
            values = tuple(values)
            lbs = (-np.Inf,) * len(names)
            ubs = (np.Inf,) * len(names)
            varies = (True,) * len(names)

            params.add_many(names, values=values,
                            varies=varies, lbs=lbs, ubs=ubs)

            self.params = params

        self._U = self.params['U'].value
        self._V = self.params['V'].value
        self._W = self.params['W'].value
        self._X = self.params['X'].value
        self._Y = self.params['Y'].value
        self._zero_error = self.params['zero_error'].value

    def params_vary_off(self):
        '''
            no params are varied
        '''
        for p in self.params:
            self.params[p].vary = False

    def params_vary_on(self):
        '''
            all params are varied
        '''
        for p in self.params:
            self.params[p].vary = True

    def dump_hdf5(self, file):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       01/19/2021 SS 1.0 original
        >> @DETAILS:    write out the hdf5 file with all the spectrum, parameters
                        and phases pecified by filename or h5py.File object
        '''
        if(isinstance(file, str)):
            fexist = path.isfile(file)
            if(fexist):
                fid = h5py.File(file, 'r+')
            else:
                fid = h5py.File(file, 'x')

        elif(isinstance(file, h5py.File)):
            fid = file

        else:
            raise RuntimeError(
                'Parameters: dump_hdf5 Pass in a filename \
                string or h5py.File object')

        self.phases.dump_hdf5(fid)
        self.params.dump_hdf5(fid)
        self.spectrum_expt.dump_hdf5(fid, 'experimental')
        self.spectrum_sim.dump_hdf5(fid, 'simulated')
        self.background.dump_hdf5(fid, 'background')

    def initialize_expt_spectrum(self, expt_spectrum):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
                        09/11/2020 SS 1.1 multiple data types accepted as input
                        09/14/2020 SS 1.2 background method chebyshev now has user specified
                        polynomial degree
        >> @DETAILS:    load the experimental spectum of 2theta-intensity
        '''
        if(expt_spectrum is not None):
            if(isinstance(expt_spectrum, Spectrum)):
                '''
                directly passing the spectrum class
                '''
                self.spectrum_expt = expt_spectrum
                self.spectrum_expt.nan_to_zero()
            elif(isinstance(expt_spectrum, np.ndarray)):
                '''
                initialize class using a nx2 array
                '''
                max_ang = expt_spectrum[-1, 0]
                if(max_ang < np.pi):
                    warnings.warn('angles are small and appear to \
                        be in radians. please check')

                self.spectrum_expt = Spectrum(x=expt_spectrum[:, 0],
                                              y=expt_spectrum[:, 1],
                                              name='expt_spectrum')
                self.spectrum_expt.nan_to_zero()

            elif(isinstance(expt_spectrum, str)):
                '''
                load from a text file
                '''
                if(path.exists(expt_spectrum)):
                    self.spectrum_expt = Spectrum.from_file(
                        expt_spectrum, skip_rows=0)
                    self.spectrum_expt.nan_to_zero()
                else:
                    raise FileError('input spectrum file doesn\'t exist.')

            self.tth_max = self.spectrum_expt._x[-1]
            self.tth_min = self.spectrum_expt._x[0]

            """
            @date 09/24/2020 SS
            catching the cases when intensity is zero.
            for these points the weights will become
            infinite. therefore, these points will be
            masked out and assigned a weight of zero.
            In addition, if any points have negative
            intensity, they will also be assigned a zero
            weight
            """
            mask = self.spectrum_expt._y <= 0.0

            self.weights = np.zeros(self.spectrum_expt.y.shape)
            """also initialize statistical weights 
            for the error calculation"""
            self.weights[~mask] = 1.0 / \
                np.sqrt(self.spectrum_expt.y[~mask])
            self.initialize_bkg()

    def initialize_bkg(self):
        '''
            the cubic spline seems to be the ideal route in terms
            of determining the background intensity. this involves
            selecting a small (~5) number of points from the spectrum,
            usually called the anchor points. a cubic spline interpolation
            is performed on this subset to estimate the overall background.
            scipy provides some useful routines for this

            the other option implemented is the chebyshev polynomials. this
            basically automates the background determination and removes the
            user from the loop which is required for the spline type background.
        '''
        if(self.bkgmethod is None):
            self.background = Spectrum(
                x=self.tth_list, y=np.zeros(self.tth_list.shape))

        elif('spline' in self.bkgmethod.keys()):
            self.selectpoints()
            x = self.points[:, 0]
            y = self.points[:, 1]
            self.splinefit(x, y)

        elif('chebyshev' in self.bkgmethod.keys()):
            self.chebyshevfit()

        elif('file' in self.bkgmethod.keys()):
            bkg = Spectrum.from_file(self.bkgmethod['file'])
            x = bkg.x
            y = bkg.y
            cs = CubicSpline(x, y)

            yy = cs(self.tth_list)

            self.background = Spectrum(x=self.tth_list, y=yy)

        elif('array' in self.bkgmethod.keys()):
            x = self.bkgmethod['array'][:, 0]
            y = self.bkgmethod['array'][:, 1]
            cs = CubicSpline(x, y)

            yy = cs(self.tth_list)

            self.background = Spectrum(x=self.tth_list, y=yy)

    def chebyshevfit(self):
        degree = self.bkgmethod['chebyshev']
        p = np.polynomial.Chebyshev.fit(
            self.tth_list, self.spectrum_expt._y, degree, w=self.weights**4)
        self.background = Spectrum(x=self.tth_list, y=p(self.tth_list))

    def selectpoints(self):

        title(
            'Select points for background estimation; \
            click middle mouse button when done.')

        plot(self.tth_list, self.spectrum_expt._y, '-k')  # 5 points tolerance

        self.points = np.asarray(ginput(0, timeout=-1))
        close()

    # cubic spline fit of background using custom points chosen from plot
    def splinefit(self, x, y):
        cs = CubicSpline(x, y)
        bkg = cs(self.tth_list)
        self.background = Spectrum(x=self.tth_list, y=bkg)

    def initialize_phases(self, phase_info):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
                        09/11/2020 SS 1.1 multiple different ways to initialize phases
                        09/14/2020 SS 1.2 added phase initialization from material.Material class
        >> @DETAILS:    load the phases for the LeBail fits
        '''

        if(phase_info is not None):
            if(isinstance(phase_info, Phases_LeBail)):
                '''
                directly passing the phase class
                '''
                self.phases = phase_info
            else:

                if(hasattr(self, 'wavelength')):
                    if(self.wavelength is not None):
                        p = Phases_LeBail(wavelength=self.wavelength)
                else:
                    p = Phases_LeBail()

                if(isinstance(phase_info, dict)):
                    '''
                    initialize class using a dictionary with key as
                    material file and values as the name of each phase
                    '''
                    for material_file in phase_info:
                        material_names = phase_info[material_file]
                        if(not isinstance(material_names, list)):
                            material_names = [material_names]
                        p.add_many(material_file, material_names)

                elif(isinstance(phase_info, str)):
                    '''
                    load from a yaml file
                    '''
                    if(path.exists(phase_info)):
                        p.load(phase_info)
                    else:
                        raise FileError('phase file doesn\'t exist.')

                elif(isinstance(phase_info, Material)):
                    p[phase_info.name] = Material_LeBail(
                        fhdf=None,
                        xtal=None,
                        dmin=None,
                        material_obj=phase_info)

                elif(isinstance(phase_info, list)):
                    for mat in phase_info:
                        p[mat.name] = Material_LeBail(
                            fhdf=None,
                            xtal=None,
                            dmin=None,
                            material_obj=mat)

                        p.num_phases += 1

                    for mat in p:
                        p[mat].pf = 1.0/p.num_phases
                self.phases = p

        self.calctth()

    def calctth(self):
        self.tth = {}
        self.hkls = {}
        for p in self.phases:
            self.tth[p] = {}
            self.hkls[p] = {}
            for k, l in self.phases.wavelength.items():
                t = self.phases[p].getTTh(l[0].value)
                allowed = self.phases[p].wavelength_allowed_hkls
                hkl = self.phases[p].hkls[allowed,:]
                limit = np.logical_and(t >= self.tth_min,
                                       t <= self.tth_max)
                self.tth[p][k] = t[limit]
                self.hkls[p][k] = hkl[limit,:]

    def initialize_Icalc(self):
        '''
        @DATE 01/22/2021 SS modified the function so Icalc can be initialized with
        a dictionary of structure factors
        '''

        self.Icalc = {}

        if(self.intensity_init is None):
            n10 = np.floor(np.log10(self.spectrum_expt._y.max())) - 2

            for p in self.phases:
                self.Icalc[p] = {}
                for k, l in self.phases.wavelength.items():

                    self.Icalc[p][k] = (10**n10) * \
                        np.ones(self.tth[p][k].shape)

        elif(isinstance(self.intensity_init, dict)):
            '''
                first check if intensities for all phases are present in the 
                passed dictionary
            '''
            for p in self.phases:
                if p not in self.intensity_init:
                    raise RuntimeError("LeBail: Intensity was initialized\
                     using custom values. However, initial values for one \
                     or more phases seem to be missing from the dictionary.")
                self.Icalc[p] = {}

                '''
                now check that the size of the initial intensities provided is consistent
                with the number of reflections (size of initial intensity > size of hkl is allowed.
                the unused values are ignored.)

                for this we need to step through the different wavelengths in the spectrum and check
                each of them
                '''
                for l in self.phases.wavelength:
                    if l not in self.intensity_init[p]:
                        raise RuntimeError("LeBail: Intensity was initialized\
                         using custom values. However, initial values for one \
                         or more wavelengths in spectrum seem to be missing \
                         from the dictionary.")

                    if(self.tth[p][l].shape[0] <=
                       self.intensity_init[p][l].shape[0]):
                        self.Icalc[p][l] = \
                            self.intensity_init[p][l][0:self.tth[p]
                                                      [l].shape[0]]

        else:
            raise RuntimeError(
                "LeBail: Intensity_init must be either\
                 None or a dictionary")

    def CagliottiH(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/20/2020 SS 1.0 original
        >> @DETAILS:    calculates the cagiotti parameter for the peak width
        '''
        th = np.radians(0.5*tth)
        tanth = np.tan(th)
        Hsq = self.U * tanth**2 + self.V * tanth + self.W
        if(Hsq < 0.):
            Hsq = 1.0e-12
        self.Hcag = np.sqrt(Hsq)

    def LorentzH(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       07/20/2020 SS 1.0 original
        >> @DETAILS:    calculates the size and strain broadening for Lorentzian peak
        '''
        th = np.radians(0.5*tth)
        tanth = np.tan(th)
        cth = np.cos(th)
        self.gamma = self.X/cth + self.Y * tanth

    def MixingFact(self, tth):
        '''
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:    05/20/2020 SS 1.0 original
                     01/29/2021 SS 2.0 updated to depend only on fwhm of profile
                     P. Thompson, D.E. Cox & J.B. Hastings, J. Appl. Cryst.,20,79-83, 
                     1987
        >> @DETAILS: calculates the mixing factor eta
        '''
        fwhm_g = self.Hcag
        fwhm_l = self.gamma

        fwhm = fwhm_g**5 + 2.69269 * fwhm_g**4 * fwhm_l + \
            2.42843 * fwhm_g**3 * fwhm_l**2 + \
            4.47163 * fwhm_g**2 * fwhm_l**3 +\
            0.07842 * fwhm_g * fwhm_l**4 +\
            fwhm_l**5

        fwhm = fwhm**0.20

        self.eta = 1.36603 * (fwhm_l/fwhm) - \
            0.47719 * (fwhm_l/fwhm)**2 + \
            0.11116 * (fwhm_l/fwhm)**3

        if self.eta < 0.:
            self.eta = 0.
        elif self.eta > 1.:
            self.eta = 1.

    def Gaussian(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/20/2020 SS 1.0 original
        >> @DETAILS:    this routine computes the gaussian peak profile
        '''

        H = self.Hcag
        cg = 4.*np.log(2.)
        self.GaussianI = (np.sqrt(cg/np.pi)/H) * \
            np.exp(-cg * ((self.tth_list - tth)/H)**2)

    def Lorentzian(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/20/2020 SS 1.0 original
        >> @DETAILS:    this routine computes the lorentzian peak profile
        '''

        H = self.gamma
        cl = 4.
        self.LorentzI = (2./np.pi/H) / (1. + cl*((self.tth_list - tth)/H)**2)

    def PseudoVoight(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/20/2020 SS 1.0 original
        >> @DETAILS:    this routine computes the pseudo-voight function as weighted
                        average of gaussian and lorentzian
        '''

        self.CagliottiH(tth)
        self.Gaussian(tth)
        self.LorentzH(tth)
        self.Lorentzian(tth)
        self.MixingFact(tth)
        self.PV = (1.0 - self.eta) * self.GaussianI + \
            self.eta * self.LorentzI

    def IntegratedIntensity(self):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
        >> @DETAILS:    Integrated intensity of the pseudo-voight peak
        '''
        return np.trapz(self.PV, self.tth_list)

    def computespectrum(self):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
        >> @DETAILS:    compute the simulated spectrum
        '''
        x = self.tth_list
        y = np.zeros(x.shape)

        for iph, p in enumerate(self.phases):

            for k, l in self.phases.wavelength.items():

                Ic = self.Icalc[p][k]

                tth = self.tth[p][k] + self.zero_error
                n = np.min((tth.shape[0], Ic.shape[0]))

                for i in range(n):

                    t = tth[i]
                    self.PseudoVoight(t)

                    y += Ic[i] * self.PV

        self.spectrum_sim = Spectrum(x=x, y=y)
        self.spectrum_sim = self.spectrum_sim + self.background

    def CalcIobs(self):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
        >> @DETAILS:    this is one of the main functions to partition the expt intensities
                        to overlapping peaks in the calculated pattern
        '''

        self.Iobs = {}
        for iph, p in enumerate(self.phases):

            self.Iobs[p] = {}

            for k, l in self.phases.wavelength.items():
                Ic = self.Icalc[p][k]

                tth = self.tth[p][k] + self.zero_error

                Iobs = []
                n = np.min((tth.shape[0], Ic.shape[0]))

                for i in range(n):
                    t = tth[i]
                    self.PseudoVoight(t)

                    y = self.PV * Ic[i]
                    _, yo = self.spectrum_expt.data
                    _, yc = self.spectrum_sim.data

                    """ 
                    @TODO if yc has zeros in it, then this
                    the next line will not like it. need to 
                    address that 
                    """
                    I = np.trapz(yo * y / yc, self.tth_list)
                    Iobs.append(I)

                self.Iobs[p][k] = np.array(Iobs)

    def calcRwp(self, params):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
        >> @DETAILS:    this routine computes the weighted error between calculated and
                        experimental spectra. goodness of fit is also calculated. the
                        weights are the inverse squareroot of the experimental intensities
        '''

        '''
        the err variable is the difference between simulated and experimental spectra
        '''
        for p in params:
            if(hasattr(self, p)):
                setattr(self, p, params[p].value)

        self.updated_lp = False
        for p in self.phases:

            mat = self.phases[p]

            '''
            PART 1: update the lattice parameters
            '''
            lp = []

            pre = p + '_'

            if(pre+'a' in params):
                if(params[pre+'a'].vary):
                    lp.append(params[pre+'a'].value)
            if(pre+'b' in params):
                if(params[pre+'b'].vary):
                    lp.append(params[pre+'b'].value)
            if(pre+'c' in params):
                if(params[pre+'c'].vary):
                    lp.append(params[pre+'c'].value)
            if(pre+'alpha' in params):
                if(params[pre+'alpha'].vary):
                    lp.append(params[pre+'alpha'].value)
            if(pre+'beta' in params):
                if(params[pre+'beta'].vary):
                    lp.append(params[pre+'beta'].value)
            if(pre+'gamma' in params):
                if(params[pre+'gamma'].vary):
                    lp.append(params[pre+'gamma'].value)

            if(not lp):
                pass
            else:
                lp = self.phases[p].Required_lp(lp)
                mat.lparms = np.array(lp)
                mat._calcrmt()
                self.updated_lp = True

        if(self.updated_lp):
            self.calctth()

        self.computespectrum()

        self.err = (self.spectrum_sim - self.spectrum_expt)

        errvec = np.sqrt(self.weights * self.err._y**2)

        ''' weighted sum of square '''
        wss = np.trapz(self.weights * self.err._y**2, self.err._x)

        den = np.trapz(self.weights * self.spectrum_sim._y **
                       2, self.spectrum_sim._x)

        ''' standard Rwp i.e. weighted residual '''
        Rwp = np.sqrt(wss/den)

        ''' number of observations to fit i.e. number of data points '''
        N = self.spectrum_sim._y.shape[0]

        ''' number of independent parameters in fitting '''
        P = len(params)
        if den > 0.:
            Rexp = np.sqrt((N-P)/den)
        else:
            Rexp = np.inf

        # Rwp and goodness of fit parameters
        self.Rwp = Rwp
        if Rexp > 0.:
            self.gofF = (Rwp / Rexp)**2
        else:
            self.gofF = np.inf

        return errvec

    def initialize_lmfit_parameters(self):

        params = lmfit.Parameters()

        for p in self.params:
            par = self.params[p]
            if(par.vary):
                params.add(p, value=par.value, min=par.lb, max=par.ub)

        return params

    def update_parameters(self):

        for p in self.res.params:
            par = self.res.params[p]
            self.params[p].value = par.value

    def RefineCycle(self, print_to_screen=True):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
                        01/28/2021 SS 1.1 added optional print_to_screen argument
        >> @DETAILS:    this is one refinement cycle for the least squares, typically few
                        10s to 100s of cycles may be required for convergence
        '''
        self.CalcIobs()
        self.Icalc = self.Iobs

        self.res = self.Refine()
        self.update_parameters()
        self.niter += 1
        self.Rwplist = np.append(self.Rwplist, self.Rwp)
        self.gofFlist = np.append(self.gofFlist, self.gofF)

        if print_to_screen:
            print('Finished iteration. Rwp: \
                {:.3f} % goodness of fit: {:.3f}'.format(
                self.Rwp*100., self.gofF))

    def Refine(self):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
        >> @DETAILS:    this routine performs the least squares refinement for all variables
                        which are allowed to be varied.
        '''

        params = self.initialize_lmfit_parameters()

        fdict = {'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4,
                 'verbose': 0, 'max_nfev': 100}

        fitter = lmfit.Minimizer(self.calcRwp, params)

        res = fitter.least_squares(**fdict)
        return res

    def updatespectrum(self):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       11/23/2020 SS 1.0 original
        >> @DETAILS:    this routine computes the spectrum for an updated list of parameters
                        intended to be used for sensitivity and identifiability analysis
        '''

        '''
        the err variable is the difference between simulated and experimental spectra
        '''
        params = self.initialize_lmfit_parameters()

        for p in params:
            if(hasattr(self, p)):
                setattr(self, p, params[p].value)

        self.updated_lp = False
        for p in self.phases:

            mat = self.phases[p]

            '''
            PART 1: update the lattice parameters
            '''
            lp = []

            pre = p + '_'

            if(pre+'a' in params):
                if(params[pre+'a'].vary):
                    lp.append(params[pre+'a'].value)
            if(pre+'b' in params):
                if(params[pre+'b'].vary):
                    lp.append(params[pre+'b'].value)
            if(pre+'c' in params):
                if(params[pre+'c'].vary):
                    lp.append(params[pre+'c'].value)
            if(pre+'alpha' in params):
                if(params[pre+'alpha'].vary):
                    lp.append(params[pre+'alpha'].value)
            if(pre+'beta' in params):
                if(params[pre+'beta'].vary):
                    lp.append(params[pre+'beta'].value)
            if(pre+'gamma' in params):
                if(params[pre+'gamma'].vary):
                    lp.append(params[pre+'gamma'].value)

            if(not lp):
                pass
            else:
                lp = self.phases[p].Required_lp(lp)
                mat.lparms = np.array(lp)
                mat._calcrmt()
                self.updated_lp = True

        if(self.updated_lp):
            self.calctth()

        self.computespectrum()

    @property
    def U(self):
        return self._U

    @U.setter
    def U(self, Uinp):
        self._U = Uinp
        # self.computespectrum()
        return

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, Vinp):
        self._V = Vinp
        # self.computespectrum()
        return

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, Winp):
        self._W = Winp
        # self.computespectrum()
        return

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, Xinp):
        self._X = Xinp
        return

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Yinp):
        self._Y = Yinp
        return

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val

    @property
    def Hcag(self):
        return self._Hcag

    @Hcag.setter
    def Hcag(self, val):
        self._Hcag = val

    @property
    def tth_list(self):
        return self.spectrum_expt._x

    @property
    def zero_error(self):
        return self._zero_error

    @zero_error.setter
    def zero_error(self, value):
        self._zero_error = value
        # self.computespectrum()
        return


def _nm(x):
    return valWUnit('lp', 'length', x, 'nm')


def extract_intensities(polar_view,
                        tth_array,
                        detector_mask,
                        params=None,
                        phases=None,
                        wavelength={'kalpha1': _nm(
                            0.15406), 'kalpha2': _nm(0.154443)},
                        bkgmethod={'chebyshev': 10},
                        intensity_init=None,
                        termination_condition={'rwp_perct_change': 0.05,
                                               'max_iter': 100}):
    ''' 
    ==============================================================================================
    ==============================================================================================

    >> @AUTHOR:     Saransh Singh, Lanwrence Livermore National Lab,
                    saransh1@llnl.gov
    >> @DATE:       01/28/2021 SS 1.0 original
    >> @DETAILS:    this function is used for extracting the experimental pole figure 
                    intensities from the polar 2theta-eta map. The workflow is to simply 
                    run the LeBail class, in parallel, over the different azimuthal profiles
                    and return the Icalc values for the different wavelengths in the 
                    calculation. For now, the multiprocessing is done using the multiprocessing
                    module which comes natively with python. Extension to MPI will be done
                    later if necessary.
    >> @PARAMS      polar_view: mxn array with the polar view. the parallelization is done 
                    over "m" i.e. the eta dimension
                    tth_array: nx1 array with two theta values at each sampling point
                    params: parameter values for the LeBail class. Could be in the form of
                    yaml file, dictionary or Parameter class
                    phases: materials to use in intensity extraction. could be a list of 
                    material objects, or file or dictionary
                    wavelength: dictionary of wavelengths to be used in the computation
                    bkgmethod: "spline" or "chebyshev" or "snip" default is chebyshev
                    intensity_init: initial intensities for each reflection. If none, then 
                    it is specified to some power of 10 depending on maximum intensity in 
                    spectrum (only used for powder simulator)
    ==============================================================================================
    ==============================================================================================
    '''

    # prepare the data file to distribute suing multiprocessing
    data_inp_list = []

    # check if the dimensions all match
    if polar_view.shape[1] != tth_array.shape[0]:
        raise RuntimeError("WPPF : extract_intensities : \
                            inconsistent dimensions \
                            of polar_view and tth_array variables.")

    non_zeros_index = []
    for i in range(polar_view.shape[0]):
        mask = detector_mask[i, :]
        # make sure that there is atleast one nonzero pixel

        if np.sum(mask) > 1:
            data = np.array([tth_array[mask], polar_view[i, :][mask]]).T
            data_inp_list.append(data)
            non_zeros_index.append(i)

    kwargs = {
        'params': params,
        'phases': phases,
        'wavelength': wavelength,
        'bkgmethod': bkgmethod,
        'termination_condition': termination_condition
    }

    P = GenericMultiprocessing()
    results = P.parallelise_function(
        data_inp_list, single_azimuthal_extraction, **kwargs)

    """
    process the outputs from the multiprocessing to make the 
    simulated polar views, tables of hkl <--> intensity etc. at 
    each azimuthal location
    in this section, all the rows which had no pixels 
    falling on detector will be handles
    separately
    """
    pv_simulated = np.zeros(polar_view.shape)
    extracted_intensities = []
    hkls = []
    tths = []
    for i in range(len(non_zeros_index)):
        idx = non_zeros_index[i]
        xp, yp, rwp, \
        Icalc, hkl, tth = results[i]

        intp_int = np.interp(tth_array, xp, yp, left=0., right=0.)

        pv_simulated[idx, :] = intp_int

        extracted_intensities.append(Icalc)
        hkls.append(hkl)
        tths.append(tth)

    return extracted_intensities, \
           hkls, \
           tths, \
           non_zeros_index, \
           pv_simulated


def single_azimuthal_extraction(expt_spectrum,
                                params=None,
                                phases=None,
                                wavelength={'kalpha1': _nm(
                                    0.15406), 'kalpha2': _nm(0.154443)},
                                bkgmethod={'chebyshev': 10},
                                intensity_init=None,
                                termination_condition=None):

    kwargs = {
        'expt_spectrum': expt_spectrum,
        'params': params,
        'phases': phases,
        'wavelength': wavelength,
        'bkgmethod': bkgmethod
    }

    # get termination conditions for the LeBail refinement
    del_rwp = termination_condition['rwp_perct_change']
    max_iter = termination_condition['max_iter']

    L = LeBail(**kwargs)

    rel_error = 1.
    init_error = 1.
    niter = 0

    # when change in Rwp < 0.05% or reached maximum iteration
    while rel_error > del_rwp and niter < max_iter:
        L.RefineCycle(print_to_screen=False)
        rel_error = 100.*np.abs((L.Rwp - init_error))
        init_error = L.Rwp
        niter += 1

    res = (L.spectrum_sim._x, L.spectrum_sim._y, \
           L.Rwp, L.Icalc, L.hkls, L.tth)
    return res


class Material_Rietveld:
    """
    ===========================================================================================
    ===========================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
                    02/01/2021 SS 1.1 class can now be initialized using a 
                    material.Material class instance
    >> @DETAILS:    Material_LeBail class is a stripped down version of the materials.Material
                    class.this is done to keep the class lightweight and make sure only the 
                    information necessary for the Rietveld fit is kept
    ===========================================================================================
     ==========================================================================================
    """

    def __init__(self,
                 fhdf=None,
                 xtal=None,
                 dmin=None,
                 kev=None,
                 material_obj=None):
        if(material_obj is None):
            '''
            dmin in nm
            '''
            self.dmin = dmin.value

            '''
            voltage in ev
            '''
            self.voltage = kev.value * 1000.0

            self._readHDF(fhdf, xtal)
            self._calcrmt()

            if(self.aniU):
                self.calcBetaij()

            self.SYM_SG, self.SYM_PG_d, self.SYM_PG_d_laue, \
                self.centrosymmetric, self.symmorphic = \
                symmetry.GenerateSGSym(self.sgnum, self.sgsetting)
            self.latticeType = symmetry.latticeType(self.sgnum)
            self.sg_hmsymbol = symbols.pstr_spacegroup[self.sgnum-1].strip()
            self.GenerateRecipPGSym()
            self.CalcMaxGIndex()
            self._calchkls()
            self.InitializeInterpTable()
            self.CalcWavelength()
            self.CalcPositions()

        else:
            if(isinstance(material_obj, Material)):
                self._init_from_materials(material_obj)
            else:
                raise ValueError(
                    "Invalid material_obj argument. \
                    only Material class can be passed here.")

    def _init_from_materials(self, material_obj):
        """

        """

        # min d-spacing for sampling hkl
        self.dmin = material_obj.dmin

        # acceleration voltage and wavelength
        self.voltage = material_obj.unitcell.voltage
        self.wavelength = material_obj.unitcell.wavelength

        # space group number
        self.sgnum = material_obj.sgnum

        # space group setting
        self.sgsetting = material_obj.sgsetting

        # lattice type from sgnum
        self.latticeType = material_obj.unitcell.latticeType

        # Herman-Maugauin symbol
        self.sg_hmsymbol = material_obj.unitcell.sg_hmsymbol

        # lattice parameters
        self.lparms = np.array([material_obj.unitcell.a,
                                material_obj.unitcell.b,
                                material_obj.unitcell.c,
                                material_obj.unitcell.alpha,
                                material_obj.unitcell.beta,
                                material_obj.unitcell.gamma])

        # asymmetric atomic positions
        self.atom_pos = material_obj.unitcell.atom_pos

        # Debye-Waller factors including anisotropic ones
        self.U = material_obj.unitcell.U
        self.aniU = False
        if(self.U.ndim > 1):
            self.aniU = True

        # atom types i.e. Z and number of different atom types
        self.atom_type = material_obj.unitcell.atom_type
        self.atom_ntype = material_obj.unitcell.atom_ntype

        self._calcrmt()

        """ get all space and point group symmetry operators
         in direct space, including the laue group. reciprocal
         space point group symmetries also included """
        self.SYM_SG = material_obj.unitcell.SYM_SG
        self.SYM_PG_d = material_obj.unitcell.SYM_PG_d
        self.SYM_PG_d_laue = material_obj.unitcell.SYM_PG_d_laue
        self.centrosymmetric = material_obj.unitcell.centrosymmetric
        self.symmorphic = material_obj.unitcell.symmorphic
        self.SYM_PG_r = material_obj.unitcell.SYM_PG_r
        self.SYM_PG_r_laue = material_obj.unitcell.SYM_PG_r_laue

        # get maximum indices for sampling hkl
        self.ih = material_obj.unitcell.ih
        self.ik = material_obj.unitcell.ik
        self.il = material_obj.unitcell.il

        # copy over the hkl but calculate the multiplicities
        self.hkls = material_obj.planeData.getHKLs()
        multiplicity = []
        for g in self.hkls:
            multiplicity.append(self.CalcStar(g, 'r').shape[0])

        multiplicity = np.array(multiplicity)
        self.multiplicity = multiplicity

        # interpolation tables and anomalous form factors
        self.f1 = material_obj.unitcell.f1
        self.f2 = material_obj.unitcell.f2
        self.f_anam = material_obj.unitcell.f_anam

        # final step is to calculate the asymmetric positions in
        # the unit cell
        self.numat = material_obj.unitcell.numat
        self.asym_pos = material_obj.unitcell.asym_pos

    def _readHDF(self, fhdf, xtal):

        # fexist = path.exists(fhdf)
        # if(fexist):
        fid = h5py.File(fhdf, 'r')
        xtal = "/"+xtal
        if xtal not in fid:
            raise IOError('crystal doesn''t exist in material file.')
        # else:
        #   raise IOError('material file does not exist.')

        gid = fid.get(xtal)

        self.sgnum = np.asscalar(np.array(gid.get('SpaceGroupNumber'),
                                          dtype=np.int32))
        self.sgsetting = np.asscalar(np.array(gid.get('SpaceGroupSetting'),
                                              dtype=np.int32))
        """
            IMPORTANT NOTE:
            note that the latice parameters in EMsoft is nm by default
            hexrd on the other hand uses A as the default units, so we
            need to be careful and convert it right here, so there is no
            confusion later on
        """
        self.lparms = list(gid.get('LatticeParameters'))

        # the last field in this is already
        self.atom_pos = np.transpose(
            np.array(gid.get('AtomData'), dtype=np.float64))

        # the U factors are related to B by the relation B = 8pi^2 U
        self.U = np.transpose(np.array(gid.get('U'), dtype=np.float64))

        self.aniU = False
        if(self.U.ndim > 1):
            self.aniU = True
            self.betaij = material_obj.unitcell.betaij

        # read atom types (by atomic number, Z)
        self.atom_type = np.array(gid.get('Atomtypes'), dtype=np.int32)
        self.atom_ntype = self.atom_type.shape[0]

        fid.close()

    def calcBetaij(self):

        self.betaij = np.zeros([self.atom_ntype, 3, 3])
        for i in range(self.U.shape[0]):
            U = self.U[i, :]
            self.betaij[i, :, :] = np.array([[U[0], U[3], U[4]],
                                             [U[3], U[1], U[5]],
                                             [U[4], U[5], U[2]]])

            self.betaij[i, :, :] *= 2. * np.pi**2 * self.aij

    def CalcWavelength(self):
        # wavelength in nm
        self.wavelength = constants.cPlanck * \
            constants.cLight /  \
            constants.cCharge / \
            self.voltage
        self.wavelength *= 1e9
        self.CalcAnomalous()

    def CalcKeV(self):
        self.kev = constants.cPlanck * \
            constants.cLight /  \
            constants.cCharge / \
            self.wavelength

        self.kev *= 1e-3

    def _calcrmt(self):

        a = self.lparms[0]
        b = self.lparms[1]
        c = self.lparms[2]

        alpha = np.radians(self.lparms[3])
        beta = np.radians(self.lparms[4])
        gamma = np.radians(self.lparms[5])

        ca = np.cos(alpha)
        cb = np.cos(beta)
        cg = np.cos(gamma)
        sa = np.sin(alpha)
        sb = np.sin(beta)
        sg = np.sin(gamma)
        tg = np.tan(gamma)

        '''
            direct metric tensor
        '''
        self.dmt = np.array([[a**2, a*b*cg, a*c*cb],
                             [a*b*cg, b**2, b*c*ca],
                             [a*c*cb, b*c*ca, c**2]])
        self.vol = np.sqrt(np.linalg.det(self.dmt))

        if(self.vol < 1e-5):
            warnings.warn('unitcell volume is suspiciously small')

        '''
            reciprocal metric tensor
        '''
        self.rmt = np.linalg.inv(self.dmt)

        ast = self.CalcLength([1, 0, 0], 'r')
        bst = self.CalcLength([0, 1, 0], 'r')
        cst = self.CalcLength([0, 0, 1], 'r')

        self.aij = np.array([[ast**2, ast*bst, ast*cst],
                             [bst*ast, bst**2, bst*cst],
                             [cst*ast, cst*bst, cst**2]])

    def _calchkls(self):
        self.hkls, self.multiplicity = self.getHKLs(self.dmin)

    ''' calculate dot product of two vectors in any space 'd' 'r' or 'c' '''

    def CalcLength(self, u, space):

        if(space == 'd'):
            vlen = np.sqrt(np.dot(u, np.dot(self.dmt, u)))
        elif(space == 'r'):
            vlen = np.sqrt(np.dot(u, np.dot(self.rmt, u)))
        elif(spec == 'c'):
            vlen = np.linalg.norm(u)
        else:
            raise ValueError('incorrect space argument')

        return vlen

    def getTTh(self, wavelength):

        tth = []
        tth_mask = []
        for g in self.hkls:
            glen = self.CalcLength(g, 'r')
            sth = glen*wavelength/2.
            if(np.abs(sth) <= 1.0):
                t = 2. * np.degrees(np.arcsin(sth))
                tth.append(t)
                tth_mask.append(True)
            else:
                tth_mask.append(False)

        tth = np.array(tth)
        tth_mask = np.array(tth_mask)
        return (tth, tth_mask)

    def GenerateRecipPGSym(self):

        self.SYM_PG_r = self.SYM_PG_d[0, :, :]
        self.SYM_PG_r = np.broadcast_to(self.SYM_PG_r, [1, 3, 3])
        self.SYM_PG_r_laue = self.SYM_PG_d[0, :, :]
        self.SYM_PG_r_laue = np.broadcast_to(self.SYM_PG_r_laue, [1, 3, 3])

        for i in range(1, self.SYM_PG_d.shape[0]):
            g = self.SYM_PG_d[i, :, :]
            g = np.dot(self.dmt, np.dot(g, self.rmt))
            g = np.round(np.broadcast_to(g, [1, 3, 3]))
            self.SYM_PG_r = np.concatenate((self.SYM_PG_r, g))

        for i in range(1, self.SYM_PG_d_laue.shape[0]):
            g = self.SYM_PG_d_laue[i, :, :]
            g = np.dot(self.dmt, np.dot(g, self.rmt))
            g = np.round(np.broadcast_to(g, [1, 3, 3]))
            self.SYM_PG_r_laue = np.concatenate((self.SYM_PG_r_laue, g))

        self.SYM_PG_r = self.SYM_PG_r.astype(np.int32)
        self.SYM_PG_r_laue = self.SYM_PG_r_laue.astype(np.int32)

    def CalcMaxGIndex(self):
        self.ih = 1
        while (1.0 / self.CalcLength(
                np.array([self.ih, 0, 0], dtype=np.float64), 'r')
                > self.dmin):
            self.ih = self.ih + 1
        self.ik = 1
        while (1.0 / self.CalcLength(
                np.array([0, self.ik, 0], dtype=np.float64), 'r') >
                self.dmin):
            self.ik = self.ik + 1
        self.il = 1
        while (1.0 / self.CalcLength(
                np.array([0, 0, self.il], dtype=np.float64), 'r') >
                self.dmin):
            self.il = self.il + 1

    def CalcStar(self, v, space, applyLaue=False):
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
            vp = np.dot(s, v)
            # check if this is new
            isnew = True
            for vec in vsym:
                if(np.sum(np.abs(vp - vec)) < 1E-4):
                    isnew = False
                    break
            if(isnew):
                vsym = np.vstack((vsym, vp))
        return vsym

    def Allowed_HKLs(self, hkllist):
        '''
        this function checks if a particular g vector is allowed
        by lattice centering, screw axis or glide plane
        '''
        hkllist = np.atleast_2d(hkllist)
        centering = self.sg_hmsymbol[0]
        if(centering == 'P'):
            # all reflections are allowed
            mask = np.ones([hkllist.shape[0], ], dtype=np.bool)
        elif(centering == 'F'):
            # same parity
            seo = np.sum(np.mod(hkllist+100, 2), axis=1)
            mask = np.logical_not(np.logical_or(seo == 1, seo == 2))
        elif(centering == 'I'):
            # sum is even
            seo = np.mod(np.sum(hkllist, axis=1)+100, 2)
            mask = (seo == 0)

        elif(centering == 'A'):
            # k+l is even
            seo = np.mod(np.sum(hkllist[:, 1:3], axis=1)+100, 2)
            mask = seo == 0
        elif(centering == 'B'):
            # h+l is even
            seo = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2)
            mask = seo == 0
        elif(centering == 'C'):
            # h+k is even
            seo = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2)
            mask = seo == 0
        elif(centering == 'R'):
            # -h+k+l is divisible by 3
            seo = np.mod(-hkllist[:, 0]+hkllist[:, 1]+hkllist[:, 2]+90, 3)
            mask = seo == 0
        else:
            raise RuntimeError(
                'IsGAllowed: unknown lattice centering encountered.')
        hkls = hkllist[mask, :]

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

                raise RuntimeError(
                    'omitscrewaxisabsences: monoclinic \
                     systems can only have 2_1 screw axis')
            '''
                only unique b-axis will be encoded
                it is the users responsibility to input
                lattice parameters in the standard setting
                with b-axis having the 2-fold symmetry
            '''
            if(iax == 1):
                mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
                mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            else:
                raise RuntimeError(
                    'omitscrewaxisabsences: only b-axis \
                     can have 2_1 screw axis')
        elif(self.latticeType == 'orthorhombic'):
            if(ax != '2_1'):
                raise RuntimeError(
                    'omitscrewaxisabsences: orthorhombic \
                     systems can only have 2_1 screw axis')
            '''
            2_1 screw on primary axis
            h00 ; h = 2n
            '''
            if(iax == 0):
                mask1 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
                mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(iax == 1):
                mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
                mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(iax == 2):
                mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
                mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'tetragonal'):
            if(iax == 0):
                mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
                if(ax == '4_2'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(ax in ['4_1', '4_3']):
                    mask2 = np.mod(hkllist[:, 2]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(iax == 1):
                mask1 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
                mask2 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
                if(ax == '2_1'):
                    mask3 = np.mod(hkllist[:, 0]+100, 2) != 0
                    mask4 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask1 = np.logical_not(np.logical_and(mask1, mask3))
                mask2 = np.logical_not(np.logical_and(mask2, mask4))
                mask = ~np.logical_or(~mask1, ~mask2)
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'trigonal'):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            if(iax == 0):
                if(ax in ['3_1', '3_2']):
                    mask2 = np.mod(hkllist[:, 2]+90, 3) != 0
            else:
                raise RuntimeError(
                    'omitscrewaxisabsences: trigonal \
                     systems can only have screw axis')
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif(self.latticeType == 'hexagonal'):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            if(iax == 0):
                if(ax == '6_3'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(ax in['3_1', '3_2', '6_2', '6_4']):
                    mask2 = np.mod(hkllist[:, 2]+90, 3) != 0
                elif(ax in ['6_1', '6_5']):
                    mask2 = np.mod(hkllist[:, 2]+120, 6) != 0
            else:
                raise RuntimeError(
                    'omitscrewaxisabsences: hexagonal \
                     systems can only have screw axis')
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif(self.latticeType == 'cubic'):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            mask2 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
            mask3 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
            if(ax in ['2_1', '4_2']):
                mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                mask5 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask6 = np.mod(hkllist[:, 0]+100, 2) != 0
            elif(ax in ['4_1', '4_3']):
                mask4 = np.mod(hkllist[:, 2]+100, 4) != 0
                mask5 = np.mod(hkllist[:, 1]+100, 4) != 0
                mask6 = np.mod(hkllist[:, 0]+100, 4) != 0
            mask1 = np.logical_not(np.logical_and(mask1, mask4))
            mask2 = np.logical_not(np.logical_and(mask2, mask5))
            mask3 = np.logical_not(np.logical_and(mask3, mask6))
            mask = ~np.logical_or(~mask1, np.logical_or(~mask2, ~mask3))
            hkllist = hkllist[mask, :]
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
                mask1 = hkllist[:, 1] == 0
                if(plane == 'c'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(plane == 'a'):
                    mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'orthorhombic'):
            if(ip == 0):
                mask1 = hkllist[:, 0] == 0
                if(plane == 'b'):
                    mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                elif(plane == 'c'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(ip == 1):
                mask1 = hkllist[:, 1] == 0
                if(plane == 'c'):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(plane == 'a'):
                    mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(ip == 2):
                mask1 = hkllist[:, 2] == 0
                if(plane == 'a'):
                    mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'b'):
                    mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'tetragonal'):
            if(ip == 0):
                mask1 = hkllist[:, 2] == 0
                if(plane == 'a'):
                    mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'b'):
                    mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
                elif(plane == 'n'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
            elif(ip == 1):
                mask1 = hkllist[:, 0] == 0
                mask2 = hkllist[:, 1] == 0
                if(plane in ['a', 'b']):
                    mask3 = np.mod(hkllist[:, 1]+100, 2) != 0
                    mask4 = np.mod(hkllist[:, 0]+100, 2) != 0
                elif(plane == 'c'):
                    mask3 = np.mod(hkllist[:, 2]+100, 2) != 0
                    mask4 = mask3
                elif(plane == 'n'):
                    mask3 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 2) != 0
                    mask4 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
                elif(plane == 'd'):
                    mask3 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 4) != 0
                    mask4 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                mask1 = np.logical_not(np.logical_and(mask1, mask3))
                mask2 = np.logical_not(np.logical_and(mask2, mask4))
                mask = ~np.logical_or(~mask1, ~mask2)
                hkllist = hkllist[mask, :]
            elif(ip == 2):
                mask1 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 1])
                if(plane in ['c', 'n']):
                    mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
                elif(plane == 'd'):
                    mask2 = np.mod(2*hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                mask = np.logical_not(np.logical_and(mask1, mask2))
                hkllist = hkllist[mask, :]
        elif(self.latticeType == 'trigonal'):
            if(plane != 'c'):
                raise RuntimeError(
                    'omitglideplaneabsences: only c-glide\
                     allowed for trigonal systems')

            if(ip == 1):
                mask1 = hkllist[:, 0] == 0
                mask2 = hkllist[:, 1] == 0
                mask3 = hkllist[:, 0] == -hkllist[:, 1]
                if(plane == 'c'):
                    mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                else:
                    raise RuntimeError(
                        'omitglideplaneabsences: only c-glide \
                         allowed for trigonal systems')
            elif(ip == 2):
                mask1 = hkllist[:, 1] == hkllist[:, 0]
                mask2 = hkllist[:, 0] == -2*hkllist[:, 1]
                mask3 = -2*hkllist[:, 0] == hkllist[:, 1]
                if(plane == 'c'):
                    mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                else:
                    raise RuntimeError(
                        'omitglideplaneabsences: only c-glide \
                         allowed for trigonal systems')
            mask1 = np.logical_and(mask1, mask4)
            mask2 = np.logical_and(mask2, mask4)
            mask3 = np.logical_and(mask3, mask4)
            mask = np.logical_not(np.logical_or(
                mask1, np.logical_or(mask2, mask3)))
            hkllist = hkllist[mask, :]
        elif(self.latticeType == 'hexagonal'):
            if(plane != 'c'):
                raise RuntimeError(
                    'omitglideplaneabsences: only c-glide \
                     allowed for hexagonal systems')
            if(ip == 2):
                mask1 = hkllist[:, 0] == hkllist[:, 1]
                mask2 = hkllist[:, 0] == -2*hkllist[:, 1]
                mask3 = -2*hkllist[:, 0] == hkllist[:, 1]
                mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                mask1 = np.logical_and(mask1, mask4)
                mask2 = np.logical_and(mask2, mask4)
                mask3 = np.logical_and(mask3, mask4)
                mask = np.logical_not(np.logical_or(
                    mask1, np.logical_or(mask2, mask3)))
            elif(ip == 1):
                mask1 = hkllist[:, 1] == 0
                mask2 = hkllist[:, 0] == 0
                mask3 = hkllist[:, 1] == -hkllist[:, 0]
                mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
            mask1 = np.logical_and(mask1, mask4)
            mask2 = np.logical_and(mask2, mask4)
            mask3 = np.logical_and(mask3, mask4)
            mask = np.logical_not(np.logical_or(
                mask1, np.logical_or(mask2, mask3)))
            hkllist = hkllist[mask, :]
        elif(self.latticeType == 'cubic'):
            if(ip == 0):
                mask1 = hkllist[:, 0] == 0
                mask2 = hkllist[:, 1] == 0
                mask3 = hkllist[:, 2] == 0
                mask4 = np.mod(hkllist[:, 0]+100, 2) != 0
                mask5 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask6 = np.mod(hkllist[:, 2]+100, 2) != 0
                if(plane == 'a'):
                    mask1 = np.logical_or(np.logical_and(
                        mask1, mask5), np.logical_and(mask1, mask6))
                    mask2 = np.logical_or(np.logical_and(
                        mask2, mask4), np.logical_and(mask2, mask6))
                    mask3 = np.logical_and(mask3, mask4)

                    mask = np.logical_not(np.logical_or(
                        mask1, np.logical_or(mask2, mask3)))
                elif(plane == 'b'):
                    mask1 = np.logical_and(mask1, mask5)
                    mask3 = np.logical_and(mask3, mask5)
                    mask = np.logical_not(np.logical_or(mask1, mask3))
                elif(plane == 'c'):
                    mask1 = np.logical_and(mask1, mask6)
                    mask2 = np.logical_and(mask2, mask6)
                    mask = np.logical_not(np.logical_or(mask1, mask2))
                elif(plane == 'n'):
                    mask4 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 2) != 0
                    mask5 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
                    mask6 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2) != 0
                    mask1 = np.logical_not(np.logical_and(mask1, mask4))
                    mask2 = np.logical_not(np.logical_and(mask2, mask5))
                    mask3 = np.logical_not(np.logical_and(mask3, mask6))
                    mask = ~np.logical_or(
                        ~mask1, np.logical_or(~mask2, ~mask3))
                elif(plane == 'd'):
                    mask4 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 4) != 0
                    mask5 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                    mask6 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
                    mask1 = np.logical_not(np.logical_and(mask1, mask4))
                    mask2 = np.logical_not(np.logical_and(mask2, mask5))
                    mask3 = np.logical_not(np.logical_and(mask3, mask6))
                    mask = ~np.logical_or(
                        ~mask1, np.logical_or(~mask2, ~mask3))
                else:
                    raise RuntimeError(
                        'omitglideplaneabsences: unknown \
                         glide plane encountered.')
                hkllist = hkllist[mask, :]
            if(ip == 2):
                mask1 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 1])
                mask2 = np.abs(hkllist[:, 1]) == np.abs(hkllist[:, 2])
                mask3 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 2])
                if(plane in ['a', 'b', 'c', 'n']):
                    mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                    mask5 = np.mod(hkllist[:, 0]+100, 2) != 0
                    mask6 = np.mod(hkllist[:, 1]+100, 2) != 0
                elif(plane == 'd'):
                    mask4 = np.mod(2*hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                    mask5 = np.mod(hkllist[:, 0]+2*hkllist[:, 1]+100, 4) != 0
                    mask6 = np.mod(2*hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
                else:
                    raise RuntimeError(
                        'omitglideplaneabsences: unknown glide \
                         plane encountered.')
                mask1 = np.logical_not(np.logical_and(mask1, mask4))
                mask2 = np.logical_not(np.logical_and(mask2, mask5))
                mask3 = np.logical_not(np.logical_and(mask3, mask6))
                mask = ~np.logical_or(~mask1, np.logical_or(~mask2, ~mask3))
                hkllist = hkllist[mask, :]
        return hkllist

    def NonSymmorphicAbsences(self, hkllist):
        '''
        this function prunes hkl list for the screw axis and glide
        plane absences
        '''
        planes = constants.SYS_AB[self.sgnum][0]
        for ip, p in enumerate(planes):

            if(p != ''):
                hkllist = self.omitglideplaneabsences(hkllist, p, ip)
        axes = constants.SYS_AB[self.sgnum][1]
        for iax, ax in enumerate(axes):
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
        mask = np.ones(hkllist.shape[0], dtype=np.bool)
        laue = InversionSymmetry
        for i, g in enumerate(hkllist):
            if(mask[i]):
                geqv = self.CalcStar(g, 'r', applyLaue=laue)
                for r in geqv[1:, ]:
                    rid = np.where(np.all(r == hkllist, axis=1))
                    mask[rid] = False
        hkl = hkllist[mask, :].astype(np.int32)
        hkl_max = []
        for g in hkl:
            geqv = self.CalcStar(g, 'r', applyLaue=laue)
            loc = np.argmax(np.sum(geqv, axis=1))
            gmax = geqv[loc, :]
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
            glen.append(np.round(self.CalcLength(g, 'r'), 8))
        # glen = np.atleast_2d(np.array(glen,dtype=np.float)).T
        dtype = [('glen', float), ('max', int), ('sum', int),
                 ('h', int), ('k', int), ('l', int)]
        a = []
        for i, gl in enumerate(glen):
            g = hkllist[i, :]
            a.append((gl, np.max(g), np.sum(g), g[0], g[1], g[2]))
        a = np.array(a, dtype=dtype)
        isort = np.argsort(a, order=['glen', 'max', 'sum', 'l', 'k', 'h'])
        return hkllist[isort, :]

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
        hkllist = np.array([[ih, ik, il] for ih in np.arange(hmax, hmin, -1)
                            for ik in np.arange(kmax, kmin, -1)
                            for il in np.arange(lmax, lmin, -1)])
        hkl_allowed = self.Allowed_HKLs(hkllist)
        hkl = []
        dsp = []
        hkl_dsp = []
        for g in hkl_allowed:
            # ignore [0 0 0] as it is the direct beam
            if(np.sum(np.abs(g)) != 0):
                dspace = 1./self.CalcLength(g, 'r')
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
        hkls = self.SortHKL(hkl)

        multiplicity = []
        for g in hkls:
            multiplicity.append(self.CalcStar(g, 'r').shape[0])

        multiplicity = np.array(multiplicity)
        return hkls, multiplicity

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
            r = self.atom_pos[i, 0:3]
            r = np.hstack((r, 1.))

            asym_pos.append(np.broadcast_to(r[0:3], [1, 3]))

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
                    if(np.sum(np.abs(rr - asym_pos[i][j, :])) < 1E-4):
                        isnew = False
                        break

                # if its new add this to the list
                if(isnew):
                    asym_pos[i] = np.vstack((asym_pos[i], rr))
                    n += 1

            numat.append(n)

        self.numat = np.array(numat)
        self.asym_pos = asym_pos

    def InitializeInterpTable(self):

        self.f1 = {}
        self.f2 = {}
        self.f_anam = {}

        data = importlib.resources.open_binary(hexrd.resources, 'Anomalous.h5')
        with h5py.File(data, 'r') as fid:
            for i in range(0, self.atom_ntype):

                Z = self.atom_type[i]
                elem = constants.ptableinverse[Z]
                gid = fid.get('/'+elem)
                data = gid.get('data')

                self.f1[elem] = interp1d(data[:, 7], data[:, 1])
                self.f2[elem] = interp1d(data[:, 7], data[:, 2])

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
        s = 0.25 * self.CalcLength(hkl, 'r')**2 * 1E-2
        sf = np.complex(0., 0.)

        for i in range(0, self.atom_ntype):

            Z = self.atom_type[i]
            ff = self.CalcXRFormFactor(Z, s)

            if(self.aniU):
                T = np.exp(-np.dot(hkl, np.dot(self.betaij[i, :, :], hkl)))
            else:
                T = np.exp(-8.0*np.pi**2 * self.U[i]*s)

            ff *= self.atom_pos[i, 3] * T

            for j in range(self.asym_pos[i].shape[0]):
                arg = 2.0 * np.pi * np.sum(hkl * self.asym_pos[i][j, :])
                sf = sf + ff * np.complex(np.cos(arg), -np.sin(arg))

        return np.abs(sf)**2

    def Required_lp(self, p):
        return _rqpDict[self.latticeType][1](p)


class Phases_Rietveld:
    """
    ==============================================================================================
    ==============================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/20/2020 SS 1.0 original
    >> @DETAILS:    class to handle different phases in the LeBail fit. this is a stripped down
                    version of main Phase class for efficiency. only the components necessary for
                    calculating peak positions are retained. further this will have a slight
                    modification to account for different wavelengths in the same phase name
    ==============================================================================================
     =============================================================================================
    """
    def _kev(x):
        return valWUnit('beamenergy', 'energy', x, 'keV')

    def _nm(x):
        return valWUnit('lp', 'length', x, 'nm')

    def __init__(self, material_file=None,
                 material_keys=None,
                 dmin=_nm(0.05),
                 wavelength={'alpha1': [_nm(0.15406), 1.], 'alpha2': [
                     _nm(0.154443), 0.52]}
                 ):

        self.phase_dict = {}
        self.num_phases = 0

        '''
        set wavelength. check if wavelength is supplied in A, if it is
        convert to nm since the rest of the code assumes those units
        '''
        wavelength_nm = {}
        for k, v in wavelength.items():
            if(v[0].unit == 'angstrom'):
                wavelength_nm[k] = [
                    valWUnit('lp', 'length', v[0].value*10., 'nm'), v[1]]
            else:
                wavelength_nm[k] = v
        self.wavelength = wavelength_nm

        self.dmin = dmin

        if(material_file is not None):
            if(material_keys is not None):
                if(type(material_keys) is not list):
                    self.add(material_file, material_keys)
                else:
                    self.add_many(material_file, material_keys)

    def __str__(self):
        resstr = 'Phases in calculation:\n'
        for i, k in enumerate(self.phase_dict.keys()):
            resstr += '\t'+str(i+1)+'. '+k+'\n'
        return resstr

    def __getitem__(self, key):
        if(key in self.phase_dict.keys()):
            return self.phase_dict[key]
        else:
            raise ValueError('phase with name not found')

    def __setitem__(self, key, mat_cls):

        if(key in self.phase_dict.keys()):
            warnings.warn('phase already in parameter list. overwriting ...')
        # if(isinstance(mat_cls, Material_Rietveld)):
        self.phase_dict[key] = mat_cls
        # else:
        # raise ValueError('input not a material class')

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if(self.n < len(self.phase_dict.keys())):
            res = list(self.phase_dict.keys())[self.n]
            self.n += 1
            return res
        else:
            raise StopIteration

    def __len__(self):
        return len(self.phase_dict)

    def add(self, material_file, material_key):
        self[material_key] = {}
        self.num_phases += 1
        for l in self.wavelength:
            lam = self.wavelength[l][0].value * 1e-9
            E = constants.cPlanck * constants.cLight / constants.cCharge / lam
            E *= 1e-3
            kev = valWUnit('beamenergy', 'energy', E, 'keV')
            self[material_key][l] = Material_Rietveld(
                material_file, material_key, dmin=self.dmin, kev=kev)

    def add_many(self, material_file, material_keys):

        for k in material_keys:
            self[k] = {}
            self.num_phases += 1
            for l in self.wavelength:
                lam = self.wavelength[l][0].value * 1e-9
                E = constants.cPlanck * constants.cLight / \
                    constants.cCharge / lam
                E *= 1e-3
                kev = valWUnit('beamenergy', 'energy', E, 'keV')
                self[k][l] = Material_Rietveld(
                    material_file, k, dmin=self.dmin, kev=kev)

        for k in self:
            for l in self.wavelength:
                self[k][l].pf = 1.0/self.num_phases

        self.material_file = material_file
        self.material_keys = material_keys

    def load(self, fname):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       06/08/2020 SS 1.0 original
            >> @DETAILS:    load parameters from yaml file
        '''
        with open(fname) as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        for mfile in dic.keys():
            mat_keys = list(dic[mfile])
            self.add_many(mfile, mat_keys)

    def dump(self, fname):
        '''
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       06/08/2020 SS 1.0 original
            >> @DETAILS:    dump parameters to yaml file
        '''
        dic = {}
        k = self.material_file
        dic[k] = [m for m in self]

        with open(fname, 'w') as f:
            data = yaml.dump(dic, f, sort_keys=False)


class Rietveld:
    """
    ===================================================================================================
    ===================================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       01/08/2020 SS 1.0 original
                    07/13/2020 SS 2.0 complete rewrite to include new parameter/material/pattern class
                    02/01/2021 SS 2.1 peak shapes from Thompson,Cox and Hastings formula using X anf Y
                    parameters for the lorentzian peak widths

    >> @DETAILS:    this is the main rietveld class and contains all the refinable parameters
                    for the analysis. the member classes are as follows (in order of initialization):

                    1. Spectrum         contains the experimental spectrum
                    2. Background       contains the background extracted from spectrum
                    3. Refine           contains all the machinery for refinement
    ===================================================================================================
    ===================================================================================================
    """

    def __init__(self,
                 expt_spectrum=None,
                 params=None,
                 phases=None,
                 wavelength={'kalpha1': [_nm(
                     0.15406), 1.0],
                     'kalpha2': [_nm(0.154443), 0.52]},
                 bkgmethod={'spline': None}):

        self.bkgmethod = bkgmethod

        self.initialize_expt_spectrum(expt_spectrum)

        self._tstart = time.time()

        if(wavelength is not None):
            self.wavelength = wavelength
            for k, v in self.wavelength.items():
                v[0] = valWUnit('lp', 'length',
                                v[0].getVal('nm'), 'nm')

        self.initialize_phases(phases)

        self.initialize_parameters(params)

        self.PolarizationFactor()
        self.computespectrum()

        self._tstop = time.time()
        self.tinit = self._tstop - self._tstart

        self.niter = 0
        self.Rwplist = np.empty([0])
        self.gofFlist = np.empty([0])

    def __str__(self):
        resstr = '<Rietveld Fit class>\nParameters of \
        the model are as follows:\n'
        resstr += self.params.__str__()
        return resstr

    def initialize_parameters(self, param_info):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
        >>              07/15/2020 SS 1.1 modified to add lattice parameters, atom positions
                        and isotropic DW factors
                        02/01/2021 SS 2.0 modified to follow same input style as LeBail class
                        with inputs of Parameter class, dict or filename valid
        >> @DETAILS:    initialize parameter list from file. if no file given, then initialize
                        to some default values (lattice constants are for CeO2)
        '''
        if(param_info is not None):
            if(isinstance(param_info, Parameters)):
                '''
                directly passing the parameter class
                '''
                self.params = param_info

            else:
                params = Parameters()

                if(isinstance(param_info, dict)):
                    '''
                    initialize class using dictionary read from the yaml file
                    '''
                    for k, v in param_info.items():
                        params.add(k, value=np.float(v[0]),
                                   lb=np.float(v[1]), ub=np.float(v[2]),
                                   vary=np.bool(v[3]))

                elif(isinstance(param_info, str)):
                    '''
                    load from a yaml file
                    '''
                    if(path.exists(param_info)):
                        params.load(param_info)
                    else:
                        raise FileError('input spectrum file doesn\'t exist.')

                    '''
                    this part initializes the lattice parameters, atom positions in asymmetric
                    unit, occupation and the isotropic debye waller factor. the anisotropic DW
                    factors will be added in the future
                    '''
                    for p in self.phases:
                        l = list(self.phases[p].keys())[0]

                        mat = self.phases[p][l]
                        lp = np.array(mat.lparms)
                        rid = list(_rqpDict[mat.latticeType][0])

                        lp = lp[rid]
                        name = _lpname[rid]

                        for n, l in zip(name, lp):
                            nn = p+'_'+n
                            '''
                            is l is small, it is one of the length units
                            else it is an angle
                            '''
                            if(l < 10.):
                                params.add(nn, value=l, lb=l-0.05,
                                           ub=l+0.05, vary=False)
                            else:
                                params.add(nn, value=l, lb=l-1.,
                                           ub=l+1., vary=False)

                        atom_pos = mat.atom_pos[:, 0:3]
                        occ = mat.atom_pos[:, 3]
                        atom_type = mat.atom_type

                        atom_label = _getnumber(atom_type)

                        for i in range(atom_type.shape[0]):

                            Z = atom_type[i]
                            elem = constants.ptableinverse[Z]

                            nn = p+'_'+elem+str(atom_label[i])+'_x'
                            params.add(
                                nn, value=atom_pos[i, 0],
                                lb=0.0, ub=1.0,
                                vary=False)

                            nn = p+'_'+elem+str(atom_label[i])+'_y'
                            params.add(
                                nn, value=atom_pos[i, 1],
                                lb=0.0, ub=1.0,
                                vary=False)

                            nn = p+'_'+elem+str(atom_label[i])+'_z'
                            params.add(
                                nn, value=atom_pos[i, 2],
                                lb=0.0, ub=1.0,
                                vary=False)

                            nn = p+'_'+elem+str(atom_label[i])+'_occ'
                            params.add(nn, value=occ[i],
                                       lb=0.0, ub=1.0,
                                       vary=False)

                            if(mat.aniU):
                                U = mat.U
                                for j in range(6):
                                    nn = p+'_'+elem + \
                                        str(atom_label[i])+'_'+_nameU[j]
                                    params.add(
                                        nn, value=U[i, j],
                                        lb=-1e-3,
                                        ub=np.inf,
                                        vary=False)
                            else:

                                nn = p+'_'+elem+str(atom_label[i])+'_dw'
                                params.add(
                                    nn, value=mat.U[i],
                                    lb=0.0, ub=np.inf,
                                    vary=False)

                self.params = params

        else:
            '''
                first 6 are the lattice paramaters
                next three are cagliotti parameters
                next are the three gauss+lorentz mixing paramters
                final is the zero instrumental peak position error
            '''
            raise RuntimeError("Please pass information for parameter \
                class in the form of Parameter object, dictionary or \
                yaml file. Default values are only initialized for the \
                LeBail class, not the Rietveld class.")
            # names = ('a', 'b', 'c', 'alpha', 'beta', 'gamma',
            #          'U', 'V', 'W', 'X', 'Y', 'tth_zero',
            #          'scale')
            # values = (5.415, 5.415, 5.415, 90., 90., 90.,
            #           0.5, 0.5, 0.5, 1e-3, 1e-3, 1e-3, 0.,
            #           1.0)

            # lbs = (-np.Inf,) * len(names)
            # ubs = (np.Inf,) * len(names)
            # varies = (False,) * len(names)

            # params.add_many(names, values=values,
            #                 varies=varies, lbs=lbs, ubs=ubs)

        self._scale = self.params['scale'].value
        self._U = self.params['U'].value
        self._V = self.params['V'].value
        self._W = self.params['W'].value
        self._X = self.params['X'].value
        self._Y = self.params['Y'].value
        self._zero_error = self.params['zero_error'].value

    def initialize_expt_spectrum(self, expt_spectrum):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
                        02/01/2021 SS 2.0 modified to follow same input style as LeBail class
                        with inputs of Spectrum class, array or filename valid
        >> @DETAILS:    load the experimental spectum of 2theta-intensity
        '''
        # self.spectrum_expt = Spectrum.from_file()
        if(expt_spectrum is not None):
            if(isinstance(expt_spectrum, Spectrum)):
                '''
                directly passing the spectrum class
                '''
                self.spectrum_expt = expt_spectrum
                self.spectrum_expt.nan_to_zero()
            elif(isinstance(expt_spectrum, np.ndarray)):
                '''
                initialize class using a nx2 array
                '''
                max_ang = expt_spectrum[-1, 0]
                if(max_ang < np.pi):
                    warnings.warn('angles are small and appear to \
                        be in radians. please check')

                self.spectrum_expt = Spectrum(x=expt_spectrum[:, 0],
                                              y=expt_spectrum[:, 1],
                                              name='expt_spectrum')
                self.spectrum_expt.nan_to_zero()

            elif(isinstance(expt_spectrum, str)):
                '''
                load from a text file
                '''
                if(path.exists(expt_spectrum)):
                    self.spectrum_expt = Spectrum.from_file(
                        expt_spectrum, skip_rows=0)
                    self.spectrum_expt.nan_to_zero()
                else:
                    raise FileError('input spectrum file doesn\'t exist.')

            self.tth_max = self.spectrum_expt._x[-1]
            self.tth_min = self.spectrum_expt._x[0]

            """
            @date 09/24/2020 SS
            catching the cases when intensity is zero.
            for these points the weights will become
            infinite. therefore, these points will be
            masked out and assigned a weight of zero.
            In addition, if any points have negative
            intensity, they will also be assigned a zero
            weight
            """
            mask = self.spectrum_expt._y <= 0.0

            self.weights = np.zeros(self.spectrum_expt.y.shape)
            ''' also initialize statistical weights for
             the error calculation'''
            self.weights[~mask] = 1.0 / np.sqrt(self.spectrum_expt.y[~mask])
            self.initialize_bkg()

    def initialize_bkg(self):
        '''
            the cubic spline seems to be the ideal route in terms
            of determining the background intensity. this involves
            selecting a small (~5) number of points from the spectrum,
            usually called the anchor points. a cubic spline interpolation
            is performed on this subset to estimate the overall background.
            scipy provides some useful routines for this

            the other option implemented is the chebyshev polynomials. this
            basically automates the background determination and removes the
            user from the loop which is required for the spline type background.
        '''
        if(self.bkgmethod is None):
            self.background = Spectrum(
                x=self.tth_list, y=np.zeros(self.tth_list.shape))

        elif('spline' in self.bkgmethod.keys()):
            self.selectpoints()
            x = self.points[:, 0]
            y = self.points[:, 1]
            self.splinefit(x, y)

        elif('chebyshev' in self.bkgmethod.keys()):
            self.chebyshevfit()

        elif('file' in self.bkgmethod.keys()):
            bkg = Spectrum.from_file(self.bkgmethod['file'])
            x = bkg.x
            y = bkg.y
            cs = CubicSpline(x, y)

            yy = cs(self.tth_list)

            self.background = Spectrum(x=self.tth_list, y=yy)

        elif('array' in self.bkgmethod.keys()):
            x = self.bkgmethod['array'][:, 0]
            y = self.bkgmethod['array'][:, 1]
            cs = CubicSpline(x, y)

            yy = cs(self.tth_list)

            self.background = Spectrum(x=self.tth_list, y=yy)

    def chebyshevfit(self):
        degree = self.bkgmethod['chebyshev']
        p = np.polynomial.Chebyshev.fit(
            self.tth_list, self.spectrum_expt._y, degree, w=self.weights**4)
        self.background = Spectrum(x=self.tth_list, y=p(self.tth_list))

    def selectpoints(self):

        title(
            'Select points for background estimation;\
             click middle mouse button when done.')

        plot(self.tth_list, self.spectrum_expt._y, '-k')  # 5 points tolerance

        self.points = np.asarray(ginput(0, timeout=-1))
        close()

    # cubic spline fit of background using custom points chosen from plot
    def splinefit(self, x, y):
        cs = CubicSpline(x, y)
        bkg = cs(self.tth_list)
        self.background = Spectrum(x=self.tth_list, y=bkg)

    def initialize_phases(self, phase_info):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
                        02/01/2021 SS 2.0 modified to follow same input style as LeBail class
                        with inputs of Material class, dict, list or filename valid
        >> @DETAILS:    load the phases for the LeBail fits
        '''

        if(phase_info is not None):
            if(isinstance(phase_info, Phases_Rietveld)):
                '''
                directly passing the phase class
                '''
                self.phases = phase_info
            else:

                if(hasattr(self, 'wavelength')):
                    if(self.wavelength is not None):
                        p = Phases_Rietveld(wavelength=self.wavelength)
                else:
                    p = Phases_Rietveld()

                if(isinstance(phase_info, dict)):
                    '''
                    initialize class using a dictionary with key as
                    material file and values as the name of each phase
                    '''
                    for material_file in phase_info:
                        material_names = phase_info[material_file]
                        if(not isinstance(material_names, list)):
                            material_names = [material_names]
                        p.add_many(material_file, material_names)

                elif(isinstance(phase_info, str)):
                    '''
                    load from a yaml file
                    '''
                    if(path.exists(phase_info)):
                        p.load(phase_info)
                    else:
                        raise FileError('phase file doesn\'t exist.')

                elif(isinstance(phase_info, Material)):
                    if not p.phase_dict:
                        p[phase_info.name] = {}

                    for k, v in self.wavelength.items():
                        E = 1.e6 * constants.cPlanck * \
                            constants.cLight /  \
                            constants.cCharge / \
                            v[0].value
                        phase_info.beamEnergy = valWUnit(
                            "kev", "ENERGY", E, "keV")
                        p[phase_info.name][k] = Material_Rietveld(
                            fhdf=None,
                            xtal=None,
                            dmin=None,
                            material_obj=phase_info)
                        p[phase_info.name][k].pf = 1.0
                    p.num_phases = 1

                elif(isinstance(phase_info, list)):
                    for mat in phase_info:
                        p[mat.name] = {}
                        for k, v in self.wavelength.items():
                            E = 1.e6 * constants.cPlanck * \
                                constants.cLight /  \
                                constants.cCharge / \
                                v[0].value
                            mat.beamEnergy = valWUnit(
                                "kev", "ENERGY", E, "keV")
                            p[mat.name][k] = Material_Rietveld(
                                fhdf=None,
                                xtal=None,
                                dmin=None,
                                material_obj=mat)
                        p.num_phases += 1

                    for mat in p:
                        for k, v in self.wavelength.items():
                            p[mat][k].pf = 1.0/p.num_phases
                self.phases = p

        self.calctth()
        self.calcsf()

    def calctth(self):
        self.tth = {}
        for p in self.phases:
            self.tth[p] = {}
            for k, l in self.phases.wavelength.items():
                t, _ = self.phases[p][k].getTTh(l[0].value)
                limit = np.logical_and(t >= self.tth_min,
                                       t <= self.tth_max)
                self.tth[p][k] = t[limit]

    def calcsf(self):
        self.sf = {}
        for p in self.phases:
            self.sf[p] = {}
            for k, l in self.phases.wavelength.items():
                w_int = l[1]
                t, tmask = self.phases[p][k].getTTh(l[0].value)
                limit = np.logical_and(t >= self.tth_min,
                                       t <= self.tth_max)
                hkl = self.phases[p][k].hkls[tmask][limit]
                multiplicity = self.phases[p][k].multiplicity[tmask][limit]
                sf = []
                for m, g in zip(multiplicity, hkl):
                    sf.append(w_int * m * self.phases[p][k].CalcXRSF(g))
                self.sf[p][k] = np.array(sf)

    def CagliottiH(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/20/2020 SS 1.0 original
        >> @DETAILS:    calculates the cagiotti parameter for the peak width
        '''
        th = np.radians(0.5*tth)
        tanth = np.tan(th)
        Hsq = self.U * tanth**2 + self.V * tanth + self.W
        if(Hsq < 0.):
            Hsq = 1.0e-12
        self.Hcag = np.sqrt(Hsq)

    def LorentzH(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       07/20/2020 SS 1.0 original
        >> @DETAILS:    calculates the size and strain broadening for Lorentzian peak
        '''
        th = np.radians(0.5*tth)
        tanth = np.tan(th)
        cth = np.cos(th)
        self.gamma = self.X/cth + self.Y * tanth

    def MixingFact(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/20/2020 SS 1.0 original
        >> @DETAILS:    calculates the mixing factor eta
        '''
        fwhm_g = self.Hcag
        fwhm_l = self.gamma

        fwhm = fwhm_g**5 + 2.69269 * fwhm_g**4 * fwhm_l + \
            2.42843 * fwhm_g**3 * fwhm_l**2 + \
            4.47163 * fwhm_g**2 * fwhm_l**3 +\
            0.07842 * fwhm_g * fwhm_l**4 +\
            fwhm_l**5

        fwhm = fwhm**0.20

        self.eta = 1.36603 * (fwhm_l/fwhm) - \
            0.47719 * (fwhm_l/fwhm)**2 + \
            0.11116 * (fwhm_l/fwhm)**3

        if self.eta < 0.:
            self.eta = 0.
        elif self.eta > 1.:
            self.eta = 1.

    def Gaussian(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/20/2020 SS 1.0 original
        >> @DETAILS:    this routine computes the gaussian peak profile
        '''

        H = self.Hcag
        cg = 4.*np.log(2.)
        self.GaussianI = (np.sqrt(cg/np.pi)/H) * \
            np.exp(-cg * ((self.tth_list - tth)/H)**2)

    def Lorentzian(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/20/2020 SS 1.0 original
        >> @DETAILS:    this routine computes the lorentzian peak profile
        '''

        H = self.gamma
        cl = 4.
        self.LorentzI = (2./np.pi/H) / (1. + cl*((self.tth_list - tth)/H)**2)

    def PseudoVoight(self, tth):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/20/2020 SS 1.0 original
        >> @DETAILS:    this routine computes the pseudo-voight function as weighted
                        average of gaussian and lorentzian
        '''

        self.CagliottiH(tth)
        self.Gaussian(tth)
        self.LorentzH(tth)
        self.Lorentzian(tth)
        self.MixingFact(tth)

        self.PV = (1. - self.eta) * self.GaussianI + \
            self.eta * self.LorentzI

    def PolarizationFactor(self):

        tth = self.tth
        self.LP = {}
        for p in self.phases:
            self.LP[p] = {}
            for k, l in self.phases.wavelength.items():
                t = np.radians(self.tth[p][k])
                self.LP[p][k] = (1 + np.cos(t)**2) / \
                    np.cos(0.5*t)/np.sin(0.5*t)**2

    def computespectrum(self):

        I = np.zeros(self.tth_list.shape)
        for p in self.tth:
            for l in self.tth[p]:

                tth = self.tth[p][l]
                sf = self.sf[p][l]
                pf = self.phases[p][l].pf / self.phases[p][l].vol**2
                lp = self.LP[p][l]

                for i, (t, fsq) in enumerate(zip(tth, sf)):
                    self.PseudoVoight(t+self.zero_error)
                    I += self.scale * pf * self.PV * fsq * lp[i]

        self.spectrum_sim = Spectrum(self.tth_list, I) + self.background

    def calcRwp(self, params):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
        >> @DETAILS:    this routine computes the weighted error between calculated and
                        experimental spectra. goodness of fit is also calculated. the
                        weights are the inverse squareroot of the experimental intensities
        '''

        '''
        the err variable is the difference between simulated and experimental spectra
        '''
        for p in params:
            if(hasattr(self, p)):
                setattr(self, p, params[p].value)

        self.updated_lp = False
        self.updated_atominfo = False
        for p in self.phases:
            for l in self.phases[p]:

                mat = self.phases[p][l]

                '''
                PART 1: update the lattice parameters
                '''
                lp = []

                pre = p + '_'
                if(pre+'a' in params):
                    if(params[pre+'a'].vary):
                        lp.append(params[pre+'a'].value)
                if(pre+'b' in params):
                    if(params[pre+'b'].vary):
                        lp.append(params[pre+'b'].value)
                if(pre+'c' in params):
                    if(params[pre+'c'].vary):
                        lp.append(params[pre+'c'].value)
                if(pre+'alpha' in params):
                    if(params[pre+'alpha'].vary):
                        lp.append(params[pre+'alpha'].value)
                if(pre+'beta' in params):
                    if(params[pre+'beta'].vary):
                        lp.append(params[pre+'beta'].value)
                if(pre+'gamma' in params):
                    if(params[pre+'gamma'].vary):
                        lp.append(params[pre+'gamma'].value)

                if(not lp):
                    pass
                else:
                    lp = self.phases[p][l].Required_lp(lp)
                    self.phases[p][l].lparms = np.array(lp)
                    self.updated_lp = True
                '''
                PART 2: update the atom info
                '''

                atom_type = mat.atom_type
                atom_label = _getnumber(atom_type)

                for i in range(atom_type.shape[0]):

                    Z = atom_type[i]
                    elem = constants.ptableinverse[Z]
                    nx = p+'_'+elem+str(atom_label[i])+'_x'
                    ny = p+'_'+elem+str(atom_label[i])+'_y'
                    nz = p+'_'+elem+str(atom_label[i])+'_z'
                    oc = p+'_'+elem+str(atom_label[i])+'_occ'

                    if(mat.aniU):
                        Un = []
                        for j in range(6):
                            Un.append(
                                p+'_'+elem +
                                str(atom_label[i]) +
                                '_'+_nameU[j])
                    else:
                        dw = p+'_'+elem+str(atom_label[i])+'_dw'

                    if(nx in params):
                        x = params[nx].value
                        self.updated_atominfo = True
                    else:
                        x = self.params[nx].value

                    if(ny in params):
                        y = params[ny].value
                        self.updated_atominfo = True
                    else:
                        y = self.params[ny].value

                    if(nz in params):
                        z = params[nz].value
                        self.updated_atominfo = True
                    else:
                        z = self.params[nz].value

                    if(oc in params):
                        oc = params[oc].value
                        self.updated_atominfo = True
                    else:
                        oc = self.params[oc].value

                    if(mat.aniU):
                        U = []
                        for j in range(6):
                            if(Un[j] in params):
                                self.updated_atominfo = True
                                U.append(params[Un[j]].value)
                            else:
                                U.append(self.params[Un[j]].value)
                        U = np.array(U)
                        mat.U[i, :] = U
                    else:
                        if(dw in params):
                            dw = params[dw].value
                            self.updated_atominfo = True
                        else:
                            dw = self.params[dw].value
                        mat.U[i] = dw

                    mat.atom_pos[i, :] = np.array([x, y, z, oc])

                if(mat.aniU):
                    mat.calcBetaij()
                if(self.updated_lp):
                    mat._calcrmt()

        if(self.updated_lp):
            self.calctth()
        if(self.updated_lp or self.updated_atominfo):
            self.calcsf()

        self.computespectrum()

        self.err = (self.spectrum_sim - self.spectrum_expt)

        errvec = np.sqrt(self.weights * self.err._y**2)

        ''' weighted sum of square '''
        wss = np.trapz(self.weights * self.err._y**2, self.err._x)

        den = np.trapz(self.weights * self.spectrum_sim._y **
                       2, self.spectrum_sim._x)

        ''' standard Rwp i.e. weighted residual '''
        Rwp = np.sqrt(wss/den)

        ''' number of observations to fit i.e. number of data points '''
        N = self.spectrum_sim._y.shape[0]

        ''' number of independent parameters in fitting '''
        P = len(params)
        if den > 0.:
            Rexp = np.sqrt((N-P)/den)
        else:
            Rexp = np.inf

        # Rwp and goodness of fit parameters
        self.Rwp = Rwp
        if Rexp > 0.:
            self.gofF = (Rwp / Rexp)**2
        else:
            self.gofF = np.inf
        Rexp = np.sqrt((N-P)/den)

        # Rwp and goodness of fit parameters
        self.Rwp = Rwp
        self.gofF = (Rwp / Rexp)**2

        return errvec

    def initialize_lmfit_parameters(self):

        params = lmfit.Parameters()

        for p in self.params:
            par = self.params[p]
            if(par.vary):
                params.add(p, value=par.value, min=par.lb, max=par.ub)

        return params

    def update_parameters(self):

        for p in self.res.params:
            par = self.res.params[p]
            self.params[p].value = par.value

    def Refine(self):
        '''
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
        >> @DETAILS:    this routine performs the least squares refinement for all variables
                        which are allowed to be varied.
        '''

        params = self.initialize_lmfit_parameters()

        fdict = {'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4,
                 'verbose': 0, 'max_nfev': 8}

        fitter = lmfit.Minimizer(self.calcRwp, params)

        self.res = fitter.least_squares(**fdict)

        self.update_parameters()

        self.niter += 1
        self.Rwplist = np.append(self.Rwplist, self.Rwp)
        self.gofFlist = np.append(self.gofFlist, self.gofF)

        print('Finished iteration. Rwp: {:.3f} % goodness of \
              fit: {:.3f}'.format(self.Rwp*100., self.gofF))

    @property
    def U(self):
        return self._U

    @U.setter
    def U(self, Uinp):
        self._U = Uinp
        return

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, Vinp):
        self._V = Vinp
        return

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, Winp):
        self._W = Winp
        return

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, Xinp):
        self._X = Xinp
        return

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Yinp):
        self._Y = Yinp
        return

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val

    @property
    def Hcag(self):
        return self._Hcag

    @Hcag.setter
    def Hcag(self, val):
        self._Hcag = val

    @property
    def tth_list(self):
        return self.spectrum_expt._x

    @property
    def zero_error(self):
        return self._zero_error

    @zero_error.setter
    def zero_error(self, value):
        self._zero_error = value
        return

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        return


_rqpDict = {
    'triclinic': (tuple(range(6)), lambda p: p),  # all 6
    # note beta
    'monoclinic': ((0, 1, 2, 4), lambda p: (p[0], p[1], p[2], 90, p[3], 90)),
    'orthorhombic': ((0, 1, 2), lambda p: (p[0], p[1], p[2], 90, 90,   90)),
    'tetragonal': ((0, 2), lambda p: (p[0], p[0], p[1], 90, 90,   90)),
    'trigonal': ((0, 2), lambda p: (p[0], p[0], p[1], 90, 90,  120)),
    'hexagonal': ((0, 2), lambda p: (p[0], p[0], p[1], 90, 90,  120)),
    'cubic': ((0,), lambda p: (p[0], p[0], p[0], 90, 90,   90)),
}

_lpname = np.array(['a', 'b', 'c', 'alpha', 'beta', 'gamma'])
_nameU = np.array(['U11', 'U22', 'U33', 'U12', 'U13', 'U23'])


def _getnumber(arr):

    res = np.ones(arr.shape)
    for i in range(arr.shape[0]):
        res[i] = np.sum(arr[0:i+1] == arr[i])
    res = res.astype(np.int32)

    return res


def generate_pole_figures(hkls, tth, Icalc):
    """
    >> @AUTHOR  Saransh Singh Lawrence Livermore national Lab
    >> @DATE    02/05/2021 SS 1.0 original
    >> @DETAILS this is the section where we'll do the texture 
    computations for now. the idea is to get the A(h,y) function 
    for the determination of the ODF. Using spherical harmonics 
    for now nut will switch to discrete harmonics in the future
    """

