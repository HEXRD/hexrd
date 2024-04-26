import numpy as np
from hexrd.material import Material
from hexrd.instrument import HEDMInstrument
from hexrd.transforms.xfcapi import anglesToDVec

class filterpack(object):
    """this class deals with intensity corrections
    related to absorption by the filter pack.

    Attributes
    ----------
    instr : HEDMInstrument
        instrument for absorption correction.
    rmat_s : numpy.ndarray
        3x3 array which specifies sample orientation
    sample : dict
        dictionary containing information about the sample
    window : dict
        dictionary containing information about window
    transmission : dict
        dictionary over detectors with each key,value 
        pair containing fraction of transmitted xrays

    Methods
    -------
    calc_transmission_filter
        compute transmission in filter
    calc_transmission_coating
        compute transmission in coating
    calc_transmission
        compute overall transmission over
        all detectors
    """
    def __init__(
                self,
                instr=None,
                filterdict=None,
                coatingdict=None
                ):
        self.instr = instr
        self.filter = filterdict
        self.coating = coatingdict
        self.transmission = dict.fromkeys(self.instr.detectors.keys())

    def calc_transmission(self):
        filter_thickness = self.filter_thickness
        al_f = self.absorption_length
        detector_normal = self.detector_normal

        for det_name, det in self.instr.detectors.items():

            tth, eta = det.pixel_angles()
            angs = np.vstack((tth.flatten(), eta.flatten(),
                              np.zeros(tth.flatten().shape))).T

            dvecs = anglesToDVec(angs, bHat_l=bvec)

            det_norm = detector_normal[det_name]
            t_f = filter_thickness[det_name]
            secb = 1./np.dot(dvecs, det_norm).reshape(det.shape)

            self.transmission[det_name] = self.calc_transmission_filter(secb, t_f, al_f)


    def calc_transmission_filter(self,
                            secb,
                            thickness_f,
                            absorption_length_filter):
        mu_f = 1./absorption_length # in microns^-1
        return np.exp(-thickness_f*mu_f*secb)

    def calc_transmission_coating(self,
                            secb,
                            thickness_c,
                            absorption_length_coating):
        mu_c = 1./absorption_length_coating # in microns^-1
        return np.exp(-thickness_c*mu_c*secb)

    @property
    def instr(self):
        return self._instr
    
    @instr.setter
    def instr(self, ins):
        if not isinstance(ins, HEDMInstrument):
            raise ValueError(f'instr must be of type HEDMInstrument')
        self._instr = ins

    @property
    def detector_normal(self):
        detector_normal = dict.fromkeys(self.instr.detectors.keys())
        for det_name, det in self.instr.detectors.items():
            rmat = self.instr.detectors[det_name].rmat
            detector_normal[det_name] = np.dot(rmat, [0., 0., -1.])
        return detector_normal

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, filterdict):
        if not isinstance(filterdict, dict):
            raise ValueError(f'filterdict needs to be a dict')
        self._filter = filterdict

    @property
    def absorption_length_filter(self):
        if self.filter is None:
            return np.inf
        elif isinstance(self.filter['material'], Material):
            return self.filter['material'].absorption_length
        elif isinstance(self.filter['material'], float):
            return self.filter['material'] # in micron

    @property
    def absorption_length_coating(self):
        if self.coating is None:
            return np.inf
        elif isinstance(self.coating['material'], Material):
            return self.coating['material'].absorption_length
        elif isinstance(self.coating['material'], float):
            return self.coating['material'] # in micron

    @property
    def filter_thickness(self):
        return dict((k, self.filter[k]) 
                    for k in self.instr.detectors.keys())
    