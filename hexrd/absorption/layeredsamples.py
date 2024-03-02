import numpy as np
from hexrd.material import Material
from hexrd.rotations import make_rmat_euler
from hexrd.instrument import HEDMInstrument

class layeredsamples(object):
    """this class deals with intensity corrections
    related to self-absorption by a layered physics
    package. the class uses information from a material
    and instrument class to determine these corrections.

    we will only consider absorption after the
    diffraction has occured i.e. absorption by the 
    sample and the window only. absorption by the 
    filterpacks etc. will be in a separate class

    The most important equation to use will be eqn. 42
    from Rygg et al., X-ray diffraction at the National 
    Ignition Facility, Rev. Sci. Instrum. 91, 043902 (2020)

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
    calc_transmission_sample
        compute transmission in sample
    calc_transmission_window
        compute transmission in window
    calc_transmission
        compute overall transmission over
        all detectors
    """
    def __init__(
                self,
                instr=None,
                rmat_s=None,
                sample=None,
                window=None):

        self.instr = instr
        self.sample = sample
        self.window = window
        self.rmat_s = rmat_s
        self.transmission = dict.fromkeys(self.instr.detectors.keys())

    def calc_transmission(self):
        bvec = self.instr.beam_vector
        seca = 1./np.dot(bvec, self.sample_normal)
        for det_name, det in self.instr.detectors.items():
            x,y = det.pixel_coords
            xy_data = np.vstack((x.flatten(), y.flatten())).T

            dvecs = det.cart_to_dvecs(xy_data)
            dvecs = dvecs/np.tile(np.linalg.norm(dvecs, axis=1), [3,1]).T

            secb = np.abs(1./np.dot(dvecs, self.sample_normal).reshape(det.shape))

            T_sample = self.calc_transmission_sample(seca, secb)
            T_window = self.calc_transmission_window(secb)

            self.transmission[det_name] = T_sample*T_window

    def calc_transmission_sample(self, seca, secb):
        thickness_s = self.sample_thickness # in microns
        mu_s = 1./self.sample_absorption_length # in microns^-1
        x = (mu_s*thickness_s)
        pre = 1./x/(secb - seca)
        num = np.exp(-x*seca) - np.exp(-x*secb)
        return pre * num

    def calc_transmission_window(self, secb):
        thickness_w = self.window_thickness # in microns
        mu_w = 1./self.window_absorption_length # in microns^-1
        return np.exp(-thickness_w*mu_w*secb)

    @property
    def sample_rmat(self):
        return self._rmat_s

    @sample_rmat.setter
    def sample_rmat(self, rmat_s):
        if rmat_s is None:
            self._rmat_s = np.eye(3)
        else:
            self._rmat_s = rmat_s

    @property
    def sample_normal(self):
        return np.dot(self.rmat_s, [0., 0., -1.])

    @property
    def sample_thickness(self):
        return self._sample_thickness

    @property
    def transmission(self):
        return self._transmission
    
    @transmission.setter
    def transmission(self, trans):
        if not isinstance(trans, dict):
            raise ValueError(f'transmission should be a dict')
        self._transmission = trans

    @property
    def instr(self):
        return self._instr
    
    @instr.setter
    def instr(self, ins):
        if not isinstance(ins, HEDMInstrument):
            raise ValueError(f'instr must be of type HEDMInstrument')
        self._instr = ins

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, sam):
        if not isinstance(sam, dict):
            raise ValueError(f'sample must be a dict')
        self._sample = sam

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, win):
        if not isinstance(win, dict):
            raise ValueError(f'window must be a dict')
        self._window = win

    @property
    def sample_thickness(self):
        if self.sample is None:
            return 0.0
        return self.sample['thickness']

    @property
    def sample_absorption_length(self):
        if self.sample is None:
            return np.inf
        return self.sample['material'].absorption_length

    @property
    def window_thickness(self):
        if self.window is None:
            return 0.0
        return self.window['thickness']

    @property
    def window_absorption_length(self):
        if self.window is None:
            return np.inf
        return self.window['material'].absorption_length
    
    