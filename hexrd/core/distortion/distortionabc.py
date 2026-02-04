import abc


class DistortionABC(metaclass=abc.ABCMeta):

    maptype: str | None = None

    @abc.abstractmethod
    def apply(self, xy_in):
        """Apply distortion mapping"""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_inverse(self, xy_in):
        """Apply inverse distortion mapping"""
        raise NotImplementedError
