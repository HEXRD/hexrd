import numpy as np
import h5py
import matplotlib.pyplot as plt
from os import path

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
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       01/15/2021 SS 1.0 original
            >> @DETAILS:    dump the class to a hdf5 file. the file argument could either be a 
                            string or a h5.File instance. If it is a filename, then HDF5 file
                            is created, a Spectrum group is created and data is written out.
                            Else data written to Spectrum group in existing file object
            >> @PARAMS file file name string or h5py.File object
                       name name ID of the spectrum e.g. experimental or simulated or background
        """

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