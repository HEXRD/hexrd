import numpy as np
import warnings
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from hexrd.wppf.peakfunctions import (
    _split_unit_gaussian as sp_gauss)
    #_split_unit_pv as sp_pv)

class Amorphous:
    '''
    >> @AUTHOR:     Saransh Singh,
                    Lawrence Livermore National Lab,
                    saransh1@llnl.gov
    >> @DATE:       06/09/2025 SS 1.0 original
    >> @DETAILS:    amorphous class can be used to include a
                    broad, diffuse signal in the Rietveld 
                    refinement. The primary purpose is to
                    extract an approximate solid/liquid phase
                    fraction. This is best used for solid and 
                    liquid coming from materials with the same
                    chemistry. The technique is most similar to
                    degree of crystallinity (DOC) as documented
                    in 
                    Dinnebier and Kern,
                    "Quantification of amorphous phases - theory",
                    PPXRD-13 workshop, Quantitative phase analysis 
                    by XRPD (2015)

    Attributes
    ----------
    model_type : str
        allowed model types are "split_pv" for split pseudo-voight
        "split_gaussian" for split gaussian and "experimental" for
        an experimentally measured lineoiut of the amorphous phase

    model_data: numpy.ndarray
        if the "experimental model type is used, then model_data 
        is a numpy array containing the 2theta-intensity of the 
        measured amorphous signal. the signal in model_data will
        be shifted and scaled to get the best fit with the observed
        data."

    scale: if model is "experimental", then this quantifies the
        scale factor. otherwise, not present

    shift: if model is "experimental", then this quantifies the
        shift in 2theta. otherwise, not present

    smoothing: if model is "experimental", then this specifies how
        much (if any) gaussian smoothing to apply to the lineout
    '''
    def __init__(self,
                 tth_list,
                 model_type='split_gaussian',
                 model_data=None,
                 scale={'c1':1.},
                 shift={'c1':0.},
                 smoothing=0,
                 center={'c1': 30.},
                 fwhm={'c1': np.array([5, 5])}):
        '''
        Parameters
        ----------
        tth_list: numpy.ndarray
            list of two-theta values for which amorphous
            intensity is computed

        model_type: str
            type of model to use for amorphous peak.
            this could be a predefined model such as
            "split_pv", or an experimentally measured 
            pattern "experimental"

        model_data: numpy.ndarray, optional
            if the model is "experimental", then this
            optional array input is used as the model
            for amporphous peak. this model will be 
            shifted and scaled to minimize the difference
            between observed and calculated intensities

        scale: dict
            scaling factor for the experimentally measured
            signal

        shift: float
            shift in two-theta for the experimental signal
            to match the observations

        smoothing: int
            width of gaussian kernel smoothing function

        center: dict
            center of split gaussian or pseudo-voight function.
            should have same keys as scale

        fwhm: dict
            dictionary of arrays of shape [2,] with [
            fwhm_l, fwhm_r] of the two halves. should have
            same keys as scale
        '''
        self.tth_list = tth_list

        self.model_type = model_type
        self.model_data = model_data

        self.scale = scale
        
        self._shift = shift
        self._smoothing = smoothing

        self._center = center
        self._fwhm = fwhm

    @property
    def model_type(self):
        return self._model_type
    
    @model_type.setter
    def model_type(self, mtype):
        if mtype.lower() in ["split_pv",
                             "split_gaussian",
                             "experimental"]:
            self._model_type = mtype
        else:
            msg = (f'{mtype} is an unknown model type')
            raise ValueError(msg)

    @property
    def tth_list(self):
        return self._tth_list

    @tth_list.setter
    def tth_list(self, val):
        if isinstance(val, np.ndarray):
            self._tth_list = val
        elif isinstance(val, (list, tuple)):
            self._tth_list = np.array(val)
        else:
            msg = f'{type(val)} not supported for tth_list'
            raise ValueError(msg)

    @property
    def model_data(self):
        return self._model_data

    @model_data.setter
    def model_data(self, data):
        if self.model_type.lower() == "experimental":
            if data is not None:
                if isinstance(data, dict):
                    self._model_data = data
                else:
                    msg = f'data should be passed as a dictionary'
                    raise ValueError(msg)
            else:
                msg = (f'experimental model is being used. '
                       f'please supply the data array')
                raise ValueError(msg)
        else:
            if data is not None:
                msg = (f'model data supplied will be ignored'
                       f'for model type {self.model_type}')
                warnings.warn(msg)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        if isinstance(val, dict):
            self._scale = val
        else:
            msg = f'scale should be passed as a dictionary'
            raise ValueError(msg)

    @property
    def shift(self):
        if self.model_type == "experimental":
            return self._shift
        return None

    @shift.setter
    def shift(self, val):
        if self.model_type == "experimental":
            if isinstance(val, dict):
                self._shift = val
            else:
                msg = f'shift should be passed as a dictionary'
                raise ValueError(msg)
        else:
            msg = (f'can not set shift for '
                   f'model_type {self.model_type}')
            warnings.warn(msg)

    @property
    def smoothing(self):
        if self.model_type == "experimental":
            return self._smoothing
        return None

    @smoothing.setter
    def smoothing(self, val):
        if self.model_type == "experimental":
            self._smoothing = val
        else:
            msg = (f'can not set smoothing for '
                   f'model_type {self.model_type}')
            warnings.warn(msg)

    @property
    def center(self):
        if self.model_type in ["split_gaussian",
                               "split_pv"]:
            return self._center

        return None

    @center.setter
    def center(self, val):
        if self.model_type in ["split_gaussian",
                               "split_pv"]:
            if isinstance(val, dict):
                self._center = val
            else:
                msg = f'center should be passed as a dictionary'
                raise ValueError(msg)
        else:
            msg = (f'can not set center for '
                   f'model_type {self.model_type}')
            warnings.warn(msg)

    @property
    def fwhm(self):
        if self.model_type in ["split_gaussian",
                               "split_pv"]:
            return self._fwhm

        return None

    @fwhm.setter
    def fwhm(self, val):
        if self.model_type in ["split_gaussian",
                               "split_pv"]:
            if isinstance(fwhm, dict):
                self._fwhm = val
            else:
                msg = f'fwhm should be passed as a dictionary'
                raise ValueError(msg)
        else:
            msg = (f'can not set fwhm for '
                   f'model_type {self.model_type}')
            warnings.warn(msg)

    @property
    def amorphous_lineout(self):
        if self.model_type == "experimental":

            for key in self.shift:
                lo = np.zeros_like(self.tth_list)
                smooth_model_data = gaussian_filter(
                                   self.model_data[key],
                                   self.smoothing
                                   )
                
                lo  += self.scale[key]*np.interp(
                                    self.tth_list,
                                    self.tth_list+self.shift[key],
                                    smooth_model_data,
                                    left=0.,
                                    right=0.)
            return lo

        elif self.model_type == "split_gaussian":
            lo = np.zeros_like(self.tth_list)
            for key in self.center:
                p = np.hstack((self.center[key],
                           self.fwhm[key]))
                lo += self.scale[key]*sp_gauss(p, self.tth_list)
            return lo

    @property
    def integrated_area(self):
        x = self.tth_list
        y = self.amorphous_lineout
        return np.trapz(y, x)

