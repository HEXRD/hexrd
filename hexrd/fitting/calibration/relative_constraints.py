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


class RelativeConstraints(ABC):
    @property
    @abstractmethod
    def type(self) -> RelativeConstraintsType:
        pass

    @property
    @abstractmethod
    def params(self) -> dict:
        pass

    @abstractmethod
    def reset(self):
        # Reset the parameters
        pass


class RelativeConstraintsNone(RelativeConstraints):
    type = RelativeConstraintsType.none

    @property
    def params(self) -> dict:
        return {}

    def reset(self):
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
        self.group_params = {}

        for group in self._groups:
            self.group_params[group] = {
                'tilt': np.array([0, 0, 0], dtype=float),
                'translation': np.array([0, 0, 0], dtype=float),
            }

    @property
    def params(self) -> dict:
        return self.group_params


class RelativeConstraintsSystem(RelativeConstraints):
    type = RelativeConstraintsType.system

    def __init__(self):
        self.reset()

    @property
    def params(self) -> dict:
        return self._params

    def reset(self):
        self._params = {
            'tilt': np.array([0, 0, 0], dtype=float),
            'translation': np.array([0, 0, 0], dtype=float),
        }


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
