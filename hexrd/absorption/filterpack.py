import numpy as np
from hexrd.material import Material
from hexrd.instrument import HEDMInstrument


class filterpack(object):
    """this class deals with intensity corrections
    related to absorption by the filter pack.
    """
    def __init__(
                self,
                instr=None,
                filterdict=None
                ):
        self.instr = instr
        self.filter = filterdict
        self.transmission = dict.fromkeys(self.instr.detectors.keys())

    def calc_transmission(self):
        filter_thickness = self.filter_thickness
        al_f = self.absorption_length
        detector_normal = self.detector_normal

        for det_name, det in self.instr.detectors.items():
            x,y = det.pixel_coords
            xy_data = np.vstack((x.flatten(), y.flatten())).T

            dvecs = det.cart_to_dvecs(xy_data)
            dvecs = dvecs/np.tile(np.linalg.norm(dvecs, axis=1), [3,1]).T

            det_norm = detector_normal[det_name]
            t_f = filter_thickness[det_name]
            secb = 1./np.dot(dvecs, det_norm).reshape(det.shape)

            self.transmission[det_name] = self.calc_transmission_filter(secb, t_f, al_f)


    def calc_transmission_filter(self,
                            secb,
                            thickness_f,
                            absorption_length):
        mu_f = 1./absorption_length # in microns^-1
        return np.exp(-thickness_f*mu_f*secb)

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
    def absorption_length(self):
        if self.filter is None:
            return np.inf
        return self.filter['material'].absorption_length

    @property
    def filter_thickness(self):
        return dict((k, self.filter[k]) 
                    for k in self.instr.detectors.keys())
    