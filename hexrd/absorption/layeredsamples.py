import numpy as np
from hexrd.material import Material

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
    """
    def __init__(
                self,
                instr=None,
                sample_tilt=None,
                sample=None,
                window=None):

    def transmission(
                    self,
                    detector):

        sample_normal = np.array([0.,0.,-1.])
        seca = 1./np.dot(det.bvec, sample_normal)

        for det_name, det in self.instr.detectors.items():
            x,y = det.pixel_coords
            xy_data = np.vstack((x.flatten(), y.flatten())).T

            dvecs = det.cart_to_dvecs(xy_data)
            dvecs = dvecs/np.tile(np.linalg.norm(dvecs, axis=1), [3,1]).T

            secb = 1./np.dot(dvecs, sample_normal).reshape(det.shape)

            T_sample = self.transmission_sample(seca, secb)
            T_window = self.transmission_window(secb)

    def transmission_sample(self, seca, secb):
        thickness_s = self.sample.thickness # in microns
        mu_s = 1./self.sample.absorption_length
        x = (mu_s*thickness_s)
        pre = 1./x/(secb - seca)

        num = np.exp(-x*seca) - np.exp(-x*secb)

        return pre * (num/den)

    def transmission_window(self, secb):
        thickness_w = self.window.thickness # in microns
        mu_w = 1./self.window.absorption_length # in microns
        return np.exp(-thickness_w*mu_w*secb)