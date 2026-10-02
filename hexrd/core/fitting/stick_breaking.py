"""Fit lmfit parameters that are fractions of a whole.

Use this for groups of parameters that must each stay non-negative and sum
to one, usually with one of them an expression of the others, like
"C = 1 - A - B". lmfit cannot enforce this: bounds do not limit a sum, and
lmfit clamps an expression to its bounds, which lets a fit run over budget
without penalty. Instead, fit the params from `add_params`, which keeps
every group physical, and map the result back with `strip_params`.

For example, Rietveld refinement fits the fraction of each phase. For CeO2,
Si and Ni, hexrd's parameters are CeO2_phase_fraction, Si_phase_fraction,
and Ni_phase_fraction = 1 - CeO2_phase_fraction - Si_phase_fraction, and
`Rietveld.Refine` fits them with:

    groups = wppfsupport.fraction_groups(self.params)
    fit_params = stick_breaking.add_params(self.params, groups)
    fitter = lmfit.Minimizer(self.calcRwp, fit_params)
    self.res = fitter.least_squares(**fdict)
    self.res.params = stick_breaking.strip_params(self.res.params, self.params)
"""

import logging

import lmfit
import numpy as np

logger = logging.getLogger(__name__)


def add_params(params: lmfit.Parameters, groups: list[list[str]]) -> lmfit.Parameters:
    """Return a copy of `params` to fit, in which each group of fractions
    stays non-negative and sums to one.

    Fixed fractions stay fixed, and the free ones share the rest through
    "stick-breaking" parameters t, each in [0, 1]:

        f_1 = R t_1,  f_2 = R (1 - t_1) t_2,  ...,  f_m = R (1 - t_1)...

    where R = 1 - sum(fixed). Narrower bounds on free fractions would clamp
    the fitted values, so they are reset to [0, 1] in `params` (fix a
    fraction to constrain it instead), and an expression in a group is
    rewritten as one minus the others.
    """
    validate(params, groups)
    for group in groups:
        _, free = _split_fixed_free(params, group)
        if narrowed := [k for k in free if params[k].min > 0 or params[k].max < 1]:
            logger.warning(f'Bounds on varying fractions reset to [0, 1]: {narrowed}')
            for k in narrowed:
                params[k].set(min=0, max=1)

        for k in group:
            if params[k].expr is not None:
                others = [x for x in group if x != k]
                params[k].expr = f'1 - {" - ".join(others)}' if others else '1'

    fit_params = params.copy()
    for group in groups:
        fixed, free = _split_fixed_free(params, group)
        remaining = 1 - sum(params[k].value for k in fixed)
        factors = [f'(1 - {" - ".join(fixed)})'] if fixed else ['1']
        for k in free[:-1]:
            # Start from the current fraction, projected if the fractions do
            # not add up. The name must not end like the group's members.
            t = np.clip(params[k].value / remaining, 0, 1) if remaining > 0 else 0.5
            t_name = f'_stick_{k}_t'
            fit_params.add(t_name, value=float(t), min=0, max=1)
            fit_params[k].set(expr=' * '.join(factors + [t_name]))
            factors.append(f'(1 - {t_name})')
            remaining *= 1 - t

        if free:
            fit_params[free[-1]].set(expr=' * '.join(factors))

    return fit_params


def strip_params(
    fit_params: lmfit.Parameters, params: lmfit.Parameters
) -> lmfit.Parameters:
    """Map fitted params from `add_params` back onto a copy of `params`."""
    result = params.copy()
    for k, par in result.items():
        fit_par = fit_params[k]
        par.value = fit_par.value  # A no-op for an expression
        par.stderr = fit_par.stderr
        # Drop correlations with the t parameters
        par.correl = fit_par.correl and {
            n: c for n, c in fit_par.correl.items() if n in result
        }

    return result


def validate(params: lmfit.Parameters, groups: list[list[str]]) -> None:
    """Raise a ValueError if a group has more than one expression, or if its
    fixed fractions are outside [0, 1] or sum to more than one."""
    for group in groups:
        if sum(params[k].expr is not None for k in group) > 1:
            raise ValueError(f'Only one of {group} can be an expression')

        fixed, _ = _split_fixed_free(params, group)
        values = {k: params[k].value for k in fixed}
        if any(not -1e-8 <= v <= 1 + 1e-8 for v in values.values()):
            raise ValueError(f'Fixed fractions must be within [0, 1]: {values}')

        if sum(values.values()) > 1 + 1e-8:
            raise ValueError(f'Fixed fractions sum to more than 1: {values}')


def _split_fixed_free(
    params: lmfit.Parameters, group: list[str]
) -> tuple[list[str], list[str]]:
    # Fixed fractions are neither varied nor expressions
    fixed = [k for k in group if not params[k].vary and params[k].expr is None]
    return fixed, [k for k in group if k not in fixed]
