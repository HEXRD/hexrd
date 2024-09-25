import importlib.resources
import hexrd.resources
from hexrd.constants import cClassicalelectronRad as re,\
cAvogadro, ATOM_WEIGHTS_DICT
import chemparse
import numpy as np
import h5py

"""
calculate the molecular weight given the formula unit
@author Saransh Singh, LLNL
@date   1.0 original 02/16/2022
"""
def interpret_formula(formula):
    """
    first interpret if the formula is a dictionary
    or just a string and get number of different
    elements in the formula
    """
    if isinstance(formula, dict):
        return formula

    elif isinstance(formula, str):
        # interpret string to a dictionary

        return chemparse.parse_formula(formula)

def calculate_molecular_mass(formula):
    """
    interpret the formula as either a dictionary 
    or a chemical formula
    """
    formula_dict = interpret_formula(formula)
    M = 0.
    for k,v in formula_dict.items():
        M += v * ATOM_WEIGHTS_DICT[k]

    return M

"""
calculate the number density of element or compound
number density is the number of atoms per unit volume
@author Saransh Singh, LLNL
@date   1.0 original 02/16/2022
"""
def calculate_number_density(density, 
                             formula):

    molecular_mass = calculate_molecular_mass(formula)
    return 1e-21*density*cAvogadro/molecular_mass

def calculate_linear_absorption_length(density, 
                                       formula,
                                       energy_vector):
    """
    this function calculates the absorption length (in mm)
    based on both coherent and incoherent scattering cross
    sections. This gives the total absorption length, instead
    of the one due to only photoeffect cross-section

    linear attenuation coefficient  = 1/linear absorption length

    @author Saransh Singh, LLNL
    @date   04/19/2023 1.0 original

    Parameters
    ----------
    density : float
        density of material in g/cm^3.
    formula : str/dict
        chemical formula of compound either as a string
        or a dict. eg. "H2O" and {"H":2, "O":1} are both
        acceptable
    energy_vector: list/numpy.ndarray
        energy (units keV) list or array of 1D vector 
        for which beta values are calculated for.

    Returns
    -------
    numpy.ndarray
        the attenuation length in microns

    """
    data = importlib.resources.open_binary(hexrd.resources, 'mu_en.h5')
    fid = h5py.File(data, 'r')

    formula_dict =  interpret_formula(formula)
    molecular_mass = calculate_molecular_mass(formula)

    density_conv = density

    mu_rho = 0.0
    for k, v in formula_dict.items():
        wi = v*ATOM_WEIGHTS_DICT[k]/molecular_mass

        d = np.array(fid[f"/{k}/data"])

        E   = d[:,0]
        mu_rho_tab = d[:,1]

        val = np.interp(np.log(energy_vector),
                        np.log(E),
                        np.log(mu_rho_tab),
                        left=0.0,
                        right=0.0)

        val = np.exp(val)
        mu_rho += wi * val

    mu = mu_rho * density_conv # this is in cm^-1
    mu = mu * 1E-4 # this is in mm^-1
    absorption_length = 1./mu

    return absorption_length

def calculate_energy_absorption_length(density, 
                                       formula,
                                       energy_vector):
    """
    this function calculates the absorption length (in mm)
    based on the total energy absorbed by the medium. this
    function is used in calculating the scintillator response
    to x-rays

    @author Saransh Singh, LLNL
    @date   04/25/2023 1.0 original

    Parameters
    ----------
    density : float
        density of material in g/cm^3.
    formula : str/dict
        chemical formula of compound either as a string
        or a dict. eg. "H2O" and {"H":2, "O":1} are both
        acceptable
    energy_vector: list/numpy.ndarray
        energy (units keV) list or array of 1D vector 
        for which beta values are calculated for.

    Returns
    -------
    numpy.ndarray
        the attenuation length in microns

    """
    data = importlib.resources.open_binary(hexrd.resources, 'mu_en.h5')
    fid = h5py.File(data, 'r')

    formula_dict =  interpret_formula(formula)
    molecular_mass = calculate_molecular_mass(formula)

    density_conv = density

    mu_rho = 0.0
    for k, v in formula_dict.items():
        wi = v*ATOM_WEIGHTS_DICT[k]/molecular_mass

        d = np.array(fid[f"/{k}/data"])

        E   = d[:,0]
        mu_rho_tab = d[:,2]

        val = np.interp(np.log(energy_vector),
                        np.log(E),
                        np.log(mu_rho_tab),
                        left=0.0,
                        right=0.0)
        val = np.exp(val)

        mu_rho += wi * val

    mu = mu_rho * density_conv # this is in cm^-1
    mu = mu * 1E-4 # this is in microns^-1
    absorption_length = 1./mu

    return absorption_length
