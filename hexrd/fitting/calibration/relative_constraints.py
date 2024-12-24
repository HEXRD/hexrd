from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from hexrd.instrument import HEDMInstrument


class RelativeConstraintsType(Enum):
    """These are relative constraints between the detectors"""
    # 'none' means no relative constraints
    none = 'None'
    # 'group' means constrain tilts/translations within a group
    group = 'Group'
    # 'system' means constrain tilts/translations within the whole system
    system = 'System'


class RotationCenter(Enum):
    """These are different centers for relative constraint rotations"""
    # Rotate about the mean center of all the detectors
    instrument_mean_center = 'InstrumentMeanCenter'

    # Rotate about lab origin, which is (0, 0, 0)
    lab_origin = 'Origin'


class RelativeConstraints(ABC):
    @property
    @abstractmethod
    def type(self) -> RelativeConstraintsType:
        pass

    @property
    @abstractmethod
    def params(self) -> dict:
        pass

    @property
    @abstractmethod
    def rotation_center(self) -> RotationCenter:
        pass

    @abstractmethod
    def reset(self):
        # Reset everything
        pass

    @property
    @abstractmethod
    def reset_params(self):
        # Reset the parameters
        pass


class RelativeConstraintsNone(RelativeConstraints):
    type = RelativeConstraintsType.none

    @property
    def params(self) -> dict:
        return {}

    @property
    def rotation_center(self) -> RotationCenter:
        return RotationCenter.instrument_mean_center

    def reset(self):
        pass

    def reset_params(self):
        pass


class RelativeConstraintsGroup(RelativeConstraints):
    type = RelativeConstraintsType.group

    def __init__(self, instr: HEDMInstrument):
        self._groups = []
        for panel in instr.detectors.values():
            if panel.group is not None and panel.group not in self._groups:
                self._groups.append(panel.group)

        self.reset()

    def reset(self):
        self.reset_params()
        self.reset_rotation_center()

    def reset_params(self):
        self.group_params = {}

        for group in self._groups:
            self.group_params[group] = {
                'tilt': np.array([0, 0, 0], dtype=float),
                'translation': np.array([0, 0, 0], dtype=float),
            }

    def reset_rotation_center(self):
        self._rotation_center = RotationCenter.instrument_mean_center

    @property
    def params(self) -> dict:
        return self.group_params

    @property
    def rotation_center(self):
        return self._rotation_center

    @rotation_center.setter
    def rotation_center(self, v: RotationCenter):
        self._rotation_center = v


class RelativeConstraintsSystem(RelativeConstraints):
    type = RelativeConstraintsType.system

    def __init__(self):
        self.reset()

    @property
    def params(self) -> dict:
        return self._params

    @property
    def rotation_center(self):
        return self._rotation_center

    @rotation_center.setter
    def rotation_center(self, v: RotationCenter):
        self._rotation_center = v

    def reset(self):
        self.reset_params()
        self.reset_rotation_center()

    def reset_params(self):
        self._params = {
            'tilt': np.array([0, 0, 0], dtype=float),
            'translation': np.array([0, 0, 0], dtype=float),
        }

    def reset_rotation_center(self):
        self._rotation_center = RotationCenter.instrument_mean_center

    def center_of_rotation(self, instr: HEDMInstrument) -> np.ndarray:
        if self.rotation_center == RotationCenter.instrument_mean_center:
            return instr.mean_detector_center
        elif self.rotation_center == RotationCenter.lab_origin:
            return np.array([0.0, 0.0, 0.0])

        raise NotImplementedError(self.rotation_center)


def create_relative_constraints(type: RelativeConstraintsType,
                                instr: HEDMInstrument):
    types = {
        'None': RelativeConstraintsNone,
        'Group': RelativeConstraintsGroup,
        'System': RelativeConstraintsSystem,
    }

    kwargs = {}
    if type == 'System':
        kwargs['instr'] = instr

    return types[type.value](**kwargs)
