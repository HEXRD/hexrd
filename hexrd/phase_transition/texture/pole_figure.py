"""
Pole figures from orientation distribution functions.

A pole figure is the projection of an ODF onto the sphere: for a crystal
direction h, the pole density P_h(r) is the total density of orientations
that put h along the specimen direction r,

    P_h(r) = (1 / |G_h|) * integral over {g : g h = r} of f(g).

That fibre integral is the Radon transform of the ODF. For an ODF built
from a radially symmetric kernel it has a closed form: the Radon transform
of the kernel, evaluated at the cosine between each rotated crystal
direction and each specimen direction. Concretely, for modes g_j with
weights w_j,

    P_h(r) = 1/(|h_sym| |S|) * sum_j w_j sum_i sum_s RK(<s g_j h_i, r>)

where h_sym is the set of distinct symmetry equivalents of h, S is the
sample symmetry group, and RK is the kernel's Radon transform.

Because the transform is linear, pole figures of sums of ODFs are sums of
pole figures, so this works for CompositeODF too.

Why the closed form
-------------------
The same quantity can be obtained by expanding the pole density in
spherical harmonics and summing the series. Any such expansion must be
truncated at a finite degree, and truncating a series whose terms have not
yet decayed produces Gibbs ringing: oscillations around sharp peaks that
drive the density slightly negative, which a probability density cannot be.
Evaluating the Radon transform in closed form avoids the expansion
entirely, so the result is exact to floating point and non-negative by
construction, the kernel's Radon transform being non-negative itself.
"""

from typing import Optional

import numpy as np

from hexrd.phase_transition.texture import plotting as pf_plotting


def regular_s2_grid(
    n_polar: int = 30, n_azimuth: int = 72
) -> tuple[np.ndarray, np.ndarray]:
    """
    Regular grid of specimen directions over the upper hemisphere.

    Both endpoints are included in each direction: the pole is therefore
    repeated `n_azimuth` times and the 0 / 2*pi meridian appears twice.
    That redundancy is harmless for evaluation and is what makes the
    contour plots close cleanly at the seam.

    Parameters
    ----------
    n_polar : int, optional
        Number of polar angles from 0 to pi/2 inclusive, default 30.
    n_azimuth : int, optional
        Number of azimuths from 0 to 2*pi inclusive, default 72.

    Returns
    -------
    tuple of numpy.ndarray
        Polar and azimuthal angles in radians, each of length
        `n_polar * n_azimuth`, with the polar angle varying fastest.
    """
    azimuth = np.linspace(0.0, 2.0 * np.pi, n_azimuth)
    polar = np.linspace(0.0, 0.5 * np.pi, n_polar)
    grid_azimuth, grid_polar = np.meshgrid(azimuth, polar)
    return grid_polar.ravel(order='F'), grid_azimuth.ravel(order='F')


def directions_from_angles(
    polar: np.ndarray, azimuth: np.ndarray
) -> np.ndarray:
    """
    Unit vectors from polar and azimuthal angles.

    Parameters
    ----------
    polar, azimuth : array_like
        Angles in radians, broadcastable to a common shape.

    Returns
    -------
    numpy.ndarray
        Unit vectors, shape (..., 3).
    """
    polar = np.asarray(polar, dtype=float)
    azimuth = np.asarray(azimuth, dtype=float)
    sin_polar = np.sin(polar)
    return np.stack(
        [sin_polar * np.cos(azimuth), sin_polar * np.sin(azimuth),
         np.cos(polar)],
        axis=-1,
    )


def angles_from_directions(
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Polar and azimuthal angles of unit vectors.

    Parameters
    ----------
    directions : array_like
        Vectors of shape (..., 3); need not be normalized.

    Returns
    -------
    tuple of numpy.ndarray
        Polar angle in [0, pi] and azimuth in (-pi, pi], in radians.
    """
    directions = _unit_vectors(directions)
    polar = np.arccos(np.clip(directions[..., 2], -1.0, 1.0))
    azimuth = np.arctan2(directions[..., 1], directions[..., 0])
    return polar, azimuth


def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize an array of vectors of shape (..., 3)."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    if vectors.shape[-1] != 3:
        raise ValueError(
            f"Directions must have shape (..., 3), got {vectors.shape}"
        )
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Directions must be non-zero vectors")
    return vectors / norms


def equivalent_directions(
    direction: np.ndarray,
    operations: np.ndarray,
    antipodal: bool = True,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Distinct symmetry equivalents of a crystal direction.

    Parameters
    ----------
    direction : array_like
        A single unit vector of shape (3,).
    operations : array_like
        Symmetry rotations, shape (n, 3, 3).
    antipodal : bool, optional
        Treat h and -h as equivalent, default True. This composes the
        symmetry group with inversion; for a centrosymmetric (Laue) group
        the antipode is already present and the flag changes nothing.
        Friedel's law makes it the right default for diffraction, which
        cannot distinguish the two.
    tol : float, optional
        Distance below which two directions count as the same.

    Returns
    -------
    numpy.ndarray
        Distinct equivalent directions, shape (m, 3).
    """
    direction = np.asarray(direction, dtype=float).reshape(3)
    candidates = np.asarray(operations, dtype=float) @ direction
    if antipodal:
        candidates = np.concatenate([candidates, -candidates])

    distinct: list[np.ndarray] = []
    for candidate in candidates:
        if not any(
            np.linalg.norm(candidate - kept) < tol for kept in distinct
        ):
            distinct.append(candidate)
    return np.array(distinct)


def pole_density(
    odf,
    crystal_direction: np.ndarray,
    specimen_directions: np.ndarray,
    antipodal: bool = True,
) -> np.ndarray:
    """
    Pole density of an ODF for one crystal direction.

    Evaluates the Radon transform of the ODF for a single h: the numerical
    core that `calc_pole_figure` is built on.

    Parameters
    ----------
    odf : ODF object
        A UniformODF, UnimodalODF or CompositeODF, or anything else
        providing a `pole_density` method.
    crystal_direction : array_like
        Crystal direction as a cartesian vector of shape (3,), in the same
        crystal frame the ODF's orientations are expressed in. Need not be
        normalized. Convert from Miller indices with the material's
        `TransSpace(hkl, 'r', 'c')`.
    specimen_directions : array_like
        Specimen directions of shape (..., 3), likewise cartesian.
    antipodal : bool, optional
        Treat h and -h as equivalent, default True.

    Returns
    -------
    numpy.ndarray
        Pole density in MRD, shape matching the leading dimensions of
        `specimen_directions`.

    Raises
    ------
    TypeError
        If the ODF does not support pole figure calculation.
    """
    method = getattr(odf, 'pole_density', None)
    if not callable(method):
        raise TypeError(
            f"{type(odf).__name__} does not support pole figure "
            f"calculation; it provides no pole_density() method"
        )
    return method(
        crystal_direction, specimen_directions, antipodal=antipodal
    )


def unimodal_pole_density(
    odf,
    crystal_direction: np.ndarray,
    specimen_directions: np.ndarray,
    antipodal: bool = True,
) -> np.ndarray:
    """
    Pole density of a kernel-based ODF, via the kernel's Radon transform.

    Implements the closed form described in the module docstring. Used by
    `UnimodalODF.pole_density`; kept here so the pole figure mathematics
    stays in one module.

    Parameters
    ----------
    odf : UnimodalODF
        ODF with `modal_orientations`, `weights` and `kernel`.
    crystal_direction : array_like
        Cartesian crystal direction, shape (3,).
    specimen_directions : array_like
        Cartesian specimen directions, shape (..., 3).
    antipodal : bool, optional
        Treat h and -h as equivalent, default True.

    Returns
    -------
    numpy.ndarray
        Pole density in MRD.
    """
    specimen = np.asarray(specimen_directions, dtype=float)
    output_shape = specimen.shape[:-1]
    specimen_flat = _unit_vectors(specimen.reshape(-1, 3))

    kernel = odf.kernel
    crystal_ops = kernel.crystal_symmetry_operations
    sample_ops = kernel.sample_symmetry_operations

    equivalents = equivalent_directions(
        _unit_vectors(crystal_direction)[0], crystal_ops, antipodal=antipodal
    )

    total = np.zeros(specimen_flat.shape[0])
    for orientation, weight in zip(odf.modal_orientations, odf.weights):
        # Specimen directions that this mode sends each equivalent crystal
        # direction to, then every sample symmetry image of those.
        poles = equivalents @ orientation.T
        for sample_op in sample_ops:
            cosines = (poles @ sample_op.T) @ specimen_flat.T
            total += weight * kernel.radon(cosines).sum(axis=0)

    # Average over the equivalent directions and the sample symmetry group,
    # which keeps the pole figure in MRD (mean 1 over the sphere).
    total /= len(equivalents) * len(sample_ops)

    if output_shape == ():
        return float(total[0])
    return total.reshape(output_shape)


class PoleFigures:
    """
    Pole figures computed from an ODF.

    Mirrors the accessors of the WPPF harmonic model's pole figure support
    (`pfdata`, `intensities`, `angs`, `stereo_radius`, `num_pfs`,
    `plot_pf`) so both can be consumed the same way, and draws them the
    same way so the plots match.

    Normally built by :func:`calc_pole_figure` rather than directly.

    Parameters
    ----------
    crystal_directions : array_like
        Cartesian crystal directions, shape (n_figures, 3).
    specimen_directions : array_like
        Cartesian specimen directions, shape (n_points, 3).
    intensities : array_like
        Pole density in MRD, shape (n_figures, n_points).
    hkls : array_like, optional
        Miller indices labelling each pole figure, shape
        (n_figures, 3) or (n_figures, 4). Used for dict keys and plot
        titles; when absent, figures are keyed by integer index.

    Attributes
    ----------
    num_pfs : int
        Number of pole figures
    keys : tuple
        Dict key of each pole figure: an hkl tuple, or an index
    intensities : dict
        Pole density per figure, in MRD
    angs : dict
        (polar, azimuth) angles per figure, in radians, shape (n_points, 2)
    stereo_radius : dict
        Stereographic projection radius per figure
    pfdata : dict
        (x, y, z, intensity) per figure, matching the WPPF layout
    """

    def __init__(
        self,
        crystal_directions: np.ndarray,
        specimen_directions: np.ndarray,
        intensities: np.ndarray,
        hkls: Optional[np.ndarray] = None,
    ) -> None:
        self._crystal_directions = _unit_vectors(crystal_directions)
        self._specimen_directions = _unit_vectors(specimen_directions)
        self._values = np.atleast_2d(np.asarray(intensities, dtype=float))

        n_figures = len(self._crystal_directions)
        if self._values.shape != (
            n_figures, len(self._specimen_directions)
        ):
            raise ValueError(
                f"intensities must have shape ({n_figures}, "
                f"{len(self._specimen_directions)}), got {self._values.shape}"
            )

        if hkls is None:
            self._hkls = None
            self._keys = tuple(range(n_figures))
        else:
            self._hkls = np.atleast_2d(np.asarray(hkls))
            if len(self._hkls) != n_figures:
                raise ValueError(
                    f"hkls must have one row per pole figure: got "
                    f"{len(self._hkls)} for {n_figures} figures"
                )
            self._keys = tuple(tuple(int(i) for i in h) for h in self._hkls)

        polar, azimuth = angles_from_directions(self._specimen_directions)
        self._polar = polar
        self._azimuth = azimuth
        self._radius = pf_plotting.stereographic_radius(polar)

    @property
    def crystal_directions(self) -> np.ndarray:
        """numpy.ndarray: Cartesian crystal directions, shape (n, 3)."""
        return self._crystal_directions.copy()

    @property
    def specimen_directions(self) -> np.ndarray:
        """numpy.ndarray: Cartesian specimen directions, shape (m, 3)."""
        return self._specimen_directions.copy()

    @property
    def hkls(self) -> Optional[np.ndarray]:
        """numpy.ndarray or None: Miller indices, if labels were given."""
        return None if self._hkls is None else self._hkls.copy()

    @property
    def keys(self) -> tuple:
        """tuple: Dict key of each pole figure."""
        return self._keys

    @property
    def num_pfs(self) -> int:
        """int: Number of pole figures."""
        return len(self._crystal_directions)

    @property
    def values(self) -> np.ndarray:
        """numpy.ndarray: Pole density, shape (n_figures, n_points), MRD."""
        return self._values.copy()

    @property
    def intensities(self) -> dict:
        """dict: Pole density per figure, in MRD."""
        return {k: self._values[i] for i, k in enumerate(self._keys)}

    @property
    def angs(self) -> dict:
        """dict: (polar, azimuth) per figure, radians, shape (m, 2)."""
        angles = np.column_stack((self._polar, self._azimuth))
        return {k: angles for k in self._keys}

    @property
    def stereo_radius(self) -> dict:
        """dict: Stereographic projection radius per figure."""
        return {k: self._radius for k in self._keys}

    @property
    def pfdata(self) -> dict:
        """dict: (x, y, z, intensity) per figure, as WPPF lays it out."""
        return {
            k: np.column_stack(
                (self._specimen_directions, self._values[i])
            )
            for i, k in enumerate(self._keys)
        }

    def label(self, index: int) -> str:
        """Plot title for one pole figure."""
        if self._hkls is None:
            return f'Pole figure {index + 1}'
        return pf_plotting.hkl_label(self._hkls[index])

    def plot_pf(
        self,
        filled: bool = False,
        grid: bool = False,
        cmap: str = pf_plotting.DEFAULT_CMAP,
        colorbar: bool = True,
        colorbar_label: str = pf_plotting.DEFAULT_COLORBAR_LABEL,
        show: bool = True,
        window_title: Optional[str] = None,
    ):
        """
        Plot the pole figures, one panel each.

        Renders through :mod:`hexrd.phase_transition.texture.plotting`,
        which draws them the way the WPPF harmonic model does, so the two
        look the same.

        Parameters
        ----------
        filled : bool, optional
            Filled contours rather than lines, default False.
        grid : bool, optional
            Show the polar grid, default False.
        cmap : str, optional
            Colormap name, default 'jet'.
        colorbar : bool, optional
            Attach a colorbar, default True.
        colorbar_label : str, optional
            Colorbar label, default 'm.r.d.'.
        show : bool, optional
            Call `show()` on the figure, default True.
        window_title : str, optional
            Title for the figure window.

        Returns
        -------
        matplotlib.figure.Figure
            The figure, also stored as `self.fig`.
        """
        self.fig, self.ax = pf_plotting.plot_pole_figures(
            [self.label(i) for i in range(self.num_pfs)],
            [self._azimuth] * self.num_pfs,
            [self._radius] * self.num_pfs,
            list(self._values),
            filled=filled,
            grid=grid,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            tight_layout=True,
            window_title=window_title,
        )

        if show:
            self.fig.show()

        return self.fig

    def __repr__(self) -> str:
        """String representation of PoleFigures."""
        return (
            f"PoleFigures(num_pfs={self.num_pfs}, "
            f"n_directions={len(self._specimen_directions)}, "
            f"keys={self._keys!r})"
        )

    def __str__(self) -> str:
        """Human-readable description."""
        lines = [
            f"{self.num_pfs} pole figure(s) on "
            f"{len(self._specimen_directions)} specimen directions"
        ]
        for i, key in enumerate(self._keys):
            values = self._values[i]
            lines.append(
                f"  {self.label(i)}: "
                f"{values.min():.3f} - {values.max():.3f} MRD"
            )
        return '\n'.join(lines)


def calc_pole_figure(
    odf,
    crystal_directions: Optional[np.ndarray] = None,
    specimen_directions: Optional[np.ndarray] = None,
    hkls: Optional[np.ndarray] = None,
    material=None,
    antipodal: bool = True,
) -> PoleFigures:
    """
    Compute pole figures from an ODF.

    Parameters
    ----------
    odf : ODF object
        A UniformODF, UnimodalODF or CompositeODF.
    crystal_directions : array_like, optional
        Cartesian crystal directions, shape (3,) or (n, 3), in the same
        crystal frame as the ODF's orientations. Either this or `hkls`
        must be given.
    specimen_directions : array_like, optional
        Cartesian specimen directions, shape (m, 3). Defaults to a regular
        30 x 72 grid over the upper hemisphere.
    hkls : array_like, optional
        Miller indices, shape (3,), (4,), (n, 3) or (n, 4). Used to label
        the pole figures, and - when `material` is given - to derive
        `crystal_directions`.
    material : object, optional
        Anything providing `TransSpace(hkl, 'r', 'c')`, e.g. a hexrd
        `Material`. Required to convert `hkls` into cartesian crystal
        directions. Note this uses hexrd's crystal frame (a || x1,
        c* || x3), so the ODF's orientations must use it too.
    antipodal : bool, optional
        Treat h and -h as equivalent, default True. Diffraction cannot
        distinguish them.

    Returns
    -------
    PoleFigures
        One pole figure per crystal direction.

    Raises
    ------
    ValueError
        If neither `crystal_directions` nor a convertible `hkls` is
        given, or if 4-index Miller-Bravais indices are passed without a
        material to convert them.

    Examples
    --------
    >>> from hexrd.phase_transition.texture import (
    ...     DeLaValleePoussinKernel, UnimodalODF, calc_pole_figure
    ... )
    >>> kernel = DeLaValleePoussinKernel(
    ...     halfwidth=np.radians(10), crystal_symmetry='oh'
    ... )
    >>> odf = UnimodalODF(np.eye(3), kernel)
    >>> pfs = calc_pole_figure(odf, [[1, 1, 1], [2, 0, 0]])
    >>> pfs.num_pfs
    2
    """
    if hkls is not None:
        hkls = np.atleast_2d(np.asarray(hkls))

    if crystal_directions is None:
        if hkls is None:
            raise ValueError(
                "Provide either crystal_directions or hkls"
            )
        if material is not None:
            crystal_directions = np.array(
                [material.TransSpace(h, 'r', 'c') for h in hkls]
            )
        elif hkls.shape[-1] == 3:
            # Only a cubic-like frame makes (h, k, l) directly cartesian,
            # so require a material for anything that needs the lattice.
            raise ValueError(
                "Converting hkls to cartesian crystal directions requires a "
                "material; pass material=..., or pass crystal_directions "
                "directly"
            )
        else:
            raise ValueError(
                "4-index Miller-Bravais hkls require a material to convert "
                "them to cartesian crystal directions"
            )

    crystal_directions = _unit_vectors(crystal_directions)

    if specimen_directions is None:
        polar, azimuth = regular_s2_grid()
        specimen_directions = directions_from_angles(polar, azimuth)
    specimen_directions = _unit_vectors(specimen_directions)

    values = np.array([
        pole_density(odf, h, specimen_directions, antipodal=antipodal)
        for h in crystal_directions
    ])

    return PoleFigures(
        crystal_directions, specimen_directions, values, hkls=hkls
    )
