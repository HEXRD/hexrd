import numpy as np
from hexrd.material import Material
from hexrd.rotations import make_rmat_euler
from hexrd.instrument import HEDMInstrument

class layeredsamples():
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
    transmission_sample
        compute transmission in sample
    transmission_window
        compute transmission in window
    transmission
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

    def transmission(
                    self,
                    detector):

        seca = 1./np.dot(det.bvec, self.sample_normal)

        for det_name, det in self.instr.detectors.items():
            x,y = det.pixel_coords
            xy_data = np.vstack((x.flatten(), y.flatten())).T

            dvecs = det.cart_to_dvecs(xy_data)
            dvecs = dvecs/np.tile(np.linalg.norm(dvecs, axis=1), [3,1]).T

            secb = 1./np.dot(dvecs, self.sample_normal).reshape(det.shape)

            T_sample = self.transmission_sample(seca, secb)
            T_window = self.transmission_window(secb)

    def transmission_sample(self, seca, secb):
        thickness_s = self.sample_thickness # in microns
        mu_s = 1./self.sample_absorption_length
        x = (mu_s*thickness_s)
        pre = 1./x/(secb - seca)

        num = np.exp(-x*seca) - np.exp(-x*secb)

        return pre * (num/den)

    def transmission_window(self, secb):
        thickness_w = self.window_thickness # in microns
        mu_w = 1./self.window_absorption_length # in microns
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
        if not instance(ins, HEDMInstrument):
            raise ValueError(f'instr must be of type HEDMInstrument')

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
        return self.sample['thickness']

    @property
    def sample_absorption_length(self):
        return self.sample['material'].absorption_length

    @property
    def window_thickness(self):
        return self.window['thickness']

    @property
    def window_absorption_length(self):
        return self.window['material'].absorption_length
    
    