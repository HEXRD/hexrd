"""
Rendering for ODF pole figures.

Draws a pole figure the way the WPPF harmonic model does (see
`HarmonicModel.plot_pf` in :mod:`hexrd.powder.wppf.texture`): each
figure is a polar axes carrying a 20-level contour plot of intensity
against azimuth and projected radius, laid out three to a row.
"""

from typing import Optional, Sequence, Union

from matplotlib import pyplot as plt
import numpy as np

DEFAULT_CMAP = 'jet'
DEFAULT_COLORBAR_LABEL = 'm.r.d.'
DEFAULT_N_COLUMNS = 3
DEFAULT_N_LEVELS = 20


def stereographic_radius(polar_angle: np.ndarray) -> np.ndarray:
    """
    Stereographic projection radius for a polar angle.

    Projects as `sin(t) / (1 + |cos(t)|)`, mapping the pole to 0 and the
    equator to 1. The projection is conformal: it preserves angles, but
    not areas.

    Parameters
    ----------
    polar_angle : array_like
        Polar angle(s) from the sample normal, in radians.

    Returns
    -------
    numpy.ndarray
        Projected radius, in [0, 1] over the upper hemisphere.
    """
    polar_angle = np.asarray(polar_angle, dtype=float)
    return np.sin(polar_angle) / (1.0 + np.abs(np.cos(polar_angle)))


def hkl_label(hkl: Sequence[float]) -> str:
    """Format a reflection as `(1, 0, -1, 0)`, for use as a plot title."""
    return '(' + ', '.join(str(int(x)) for x in hkl) + ')'


def subplot_shape(
    n_figures: int, n_columns: int = DEFAULT_N_COLUMNS
) -> tuple[int, int]:
    """
    Rows and columns needed to lay out `n_figures` pole figures.

    Parameters
    ----------
    n_figures : int
        Number of pole figures to show.
    n_columns : int, optional
        Maximum number of columns, default 3.

    Returns
    -------
    tuple of (int, int)
        Number of rows and columns. A single row is narrowed to the number
        of figures so a lone pole figure is not stretched across the width.
    """
    n_rows = int(n_figures / n_columns) + 1
    if n_rows == 1:
        return 1, min(n_columns, max(n_figures, 1))
    return n_rows, n_columns


def create_axes(
    n_figures: int,
    n_columns: int = DEFAULT_N_COLUMNS,
    tight_layout: bool = False,
    window_title: Optional[str] = None,
):
    """
    Create a blank grid of polar axes to draw pole figures on.

    Every axes is stripped of its frame, ticks and grid; :func:`draw`
    restores the grid per axes if asked for. Axes beyond `n_figures` are
    left blank.

    Parameters
    ----------
    n_figures : int
        Number of pole figures the grid must hold.
    n_columns : int, optional
        Maximum number of columns, default 3.
    tight_layout : bool, optional
        Whether to use matplotlib's tight layout, default False.
    window_title : str, optional
        Title for the figure window, if the backend supports one.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, numpy.ndarray)
        The figure and a 2-D object array of axes.
    """
    n_rows, n_columns = subplot_shape(n_figures, n_columns)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        subplot_kw={'projection': 'polar'},
        figsize=(12, 4 * n_rows),
        tight_layout=tight_layout,
    )
    axes = np.atleast_2d(axes)

    if window_title is not None:
        manager = fig.canvas.manager
        if manager is not None:
            manager.set_window_title(window_title)

    for ax in axes.flatten():
        ax.set_axis_off()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(False)

    return fig, axes


def axes_for(axes: np.ndarray, index: int, n_columns: int = DEFAULT_N_COLUMNS):
    """Return the axes that the `index`-th pole figure belongs on."""
    return axes[index // n_columns][index % n_columns]


def close_azimuth_seam(
    azimuth: np.ndarray,
    radius: np.ndarray,
    intensities: np.ndarray,
):
    """
    Duplicate the points sitting exactly on the +/-pi seam.

    Triangulating on a polar axes treats azimuth as a linear coordinate,
    so no triangle spans the branch cut at +/-pi and a wedge of the disk
    can be left blank. Copying the points on one edge across to the other
    gives the triangulation something to span.

    .. note::

       FIXME: there is a bug during plots when the 0/2pi or -pi/pi points
       are not filled in for some set of inputs. This carries the same
       limitation as `HarmonicModel.plot_pf`, which it mirrors: only
       points lying exactly on +/-pi are duplicated, and a grid need not
       contain any.

    Parameters
    ----------
    azimuth, radius, intensities : array_like
        Point coordinates and values. Azimuth is in radians and assumed to
        lie in (-pi, pi].

    Returns
    -------
    tuple of numpy.ndarray
        The inputs with the seam points appended.
    """
    azimuth = np.asarray(azimuth, dtype=float)
    radius = np.asarray(radius, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    mask = np.isclose(np.abs(azimuth), np.pi)
    if not np.any(mask):
        return azimuth, radius, intensities

    return (
        np.concatenate((azimuth, -azimuth[mask])),
        np.concatenate((radius, radius[mask])),
        np.concatenate((intensities, intensities[mask])),
    )


def draw(
    ax,
    azimuth: np.ndarray,
    radius: np.ndarray,
    intensities: np.ndarray,
    label: Optional[str] = None,
    filled: bool = False,
    grid: bool = False,
    cmap: str = DEFAULT_CMAP,
    colorbar: bool = True,
    colorbar_label: str = DEFAULT_COLORBAR_LABEL,
    levels: Union[int, Sequence[float]] = DEFAULT_N_LEVELS,
):
    """
    Draw one pole figure onto a polar axes.

    Any contour set and colorbar left from a previous call is removed
    first, so this doubles as the redraw path for interactive updates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A polar axes, e.g. from :func:`create_axes`.
    azimuth, radius, intensities : array_like
        Point coordinates and values, of equal length.
    label : str, optional
        Title for the axes.
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
    levels : int or sequence of float, optional
        Number of contour levels, or the level values themselves, as
        matplotlib's contour functions take them. Default 20.

    Returns
    -------
    matplotlib.contour.ContourSet
        The contour set that was drawn.
    """
    azimuth, radius, intensities = close_azimuth_seam(
        azimuth, radius, intensities
    )

    # Remove any artists left by a previous draw on this axes.
    for attribute in ('_hexrd_pf_colorbar', '_hexrd_pf_contours'):
        artist = getattr(ax, attribute, None)
        if artist is not None:
            artist.remove()
            setattr(ax, attribute, None)

    contour = ax.tricontourf if filled else ax.tricontour
    contours = contour(azimuth, radius, intensities, levels=levels,
                       cmap=cmap)
    ax._hexrd_pf_contours = contours

    ax.set_yticklabels([])
    ax.grid(grid)
    if label is not None:
        ax.set_title(label)

    if colorbar:
        ax._hexrd_pf_colorbar = ax.figure.colorbar(
            contours, ax=ax, label=colorbar_label
        )

    return contours


def plot_pole_figures(
    labels: Sequence[Optional[str]],
    azimuths: Sequence[np.ndarray],
    radii: Sequence[np.ndarray],
    intensities: Sequence[np.ndarray],
    filled: bool = False,
    grid: bool = False,
    cmap: str = DEFAULT_CMAP,
    colorbar: bool = True,
    colorbar_label: str = DEFAULT_COLORBAR_LABEL,
    n_columns: int = DEFAULT_N_COLUMNS,
    tight_layout: bool = False,
    window_title: Optional[str] = None,
):
    """
    Lay out and draw a set of pole figures, one panel each.

    Parameters
    ----------
    labels : sequence of str or None
        Axes title for each pole figure.
    azimuths, radii, intensities : sequence of array_like
        Point coordinates and values for each pole figure.
    filled : bool, optional
        Filled contours rather than lines, default False.
    grid : bool, optional
        Show the polar grid, default False.
    cmap : str, optional
        Colormap name, default 'jet'.
    colorbar : bool, optional
        Attach a colorbar to each panel, default True.
    colorbar_label : str, optional
        Colorbar label, default 'm.r.d.'.
    n_columns : int, optional
        Maximum number of columns, default 3.
    tight_layout : bool, optional
        Whether to use matplotlib's tight layout, default False.
    window_title : str, optional
        Title for the figure window.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, numpy.ndarray)
        The figure and its 2-D array of axes.

    Raises
    ------
    ValueError
        If the sequences do not all have the same length.
    """
    lengths = {
        len(labels), len(azimuths), len(radii), len(intensities)
    }
    if len(lengths) != 1:
        raise ValueError(
            f"labels, azimuths, radii and intensities must have the same "
            f"length; got {len(labels)}, {len(azimuths)}, {len(radii)} "
            f"and {len(intensities)}"
        )

    fig, axes = create_axes(
        len(labels),
        n_columns=n_columns,
        tight_layout=tight_layout,
        window_title=window_title,
    )

    for index, label in enumerate(labels):
        draw(
            axes_for(axes, index, n_columns),
            azimuths[index],
            radii[index],
            intensities[index],
            label=label,
            filled=filled,
            grid=grid,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
        )

    return fig, axes
