import numpy as np
import warnings
from scipy.interpolate import CubicSpline

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
                 model_type='split_pv',
                 model_data=None):
        '''
        Parameters
        ----------
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
        '''

        self.model_type = model_type
        self.model_data = model_data

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

        if self.model_type.lower() == "experimental":
            if model_data is not None:
                self.model_data = model_data
            else:
                msg = (f'experimental model is being used. '
                       f'Please supply the data array')
                raise ValueError(msg)
        else:
            if model_data is not None:
                msg = (f'model data supplied will be ignored'
                       f'for model type {mtype}.')
                warnings.warn(msg)

