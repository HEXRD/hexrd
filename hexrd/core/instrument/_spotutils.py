"""Numerical helpers for reflection extraction."""

from functools import lru_cache
from weakref import WeakKeyDictionary

import numpy as np
from scipy import ndimage

_frame_views = WeakKeyDictionary()


def clear_sparse_frame_views():
    """Start a fresh immutable image snapshot for a fit or public spot pull."""
    _frame_views.clear()


def _comparison_threshold(dtype, threshold):
    # NumPy's weak Python scalars may compare a float32 image in float32,
    # whereas a NumPy float64 scalar promotes that image to float64.
    if np.dtype(dtype).kind == 'f':
        resolved = np.result_type(np.empty(0, dtype=dtype), threshold)
        return np.asarray(threshold, dtype=resolved).item()
    return threshold


def native_threshold_supported(dtype, threshold):
    try:
        comparison = np.result_type(np.empty(0, dtype=dtype), threshold)
        return (
            np.ndim(threshold) == 0
            and comparison.kind in 'biuf'
            and comparison.itemsize <= 8
        )
    except (TypeError, ValueError, OverflowError):
        return False


class _SpotData:
    """Columnar spot records used internally during grain fitting.

    Keep arrays compact while culling, optimizing, and crossing worker IPC.
    Convert to the historical list of nine-field records only at the API edge.
    """

    fields = (
        'peak_id',
        'hkl_id',
        'hkls',
        'sum_int',
        'max_int',
        'pred_angs',
        'meas_angs',
        'meas_xy',
        'pred_xy',
    )

    def __init__(self, *columns):
        for name, column in zip(self.fields, columns):
            setattr(self, name, column)

    def __len__(self):
        return len(self.peak_id)

    def __getitem__(self, i):
        valid = self.peak_id[i] >= 0
        return [
            int(self.peak_id[i]),
            self.hkl_id[i],
            self.hkls[i],
            self.sum_int[i] if valid else np.nan,
            self.max_int[i] if valid else np.nan,
            self.pred_angs[i],
            self.meas_angs[i],
            self.meas_xy[i],
            self.pred_xy[i],
        ]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def take(self, indices):
        return type(self)(*(getattr(self, name)[indices] for name in self.fields))

    @classmethod
    def concatenate(cls, chunks):
        return cls(
            *(np.concatenate([getattr(c, name) for c in chunks]) for name in cls.fields)
        )

    def as_fit_data(self):
        return {
            'hkls': self.hkls.T,
            'meas_xyo': np.column_stack((self.meas_xy, self.meas_angs[:, 2])),
        }


def pull_spots_batched(
    instrument,
    panel,
    series,
    view,
    hkl_ids,
    hkls,
    predicted,
    predicted_xy,
    angular_pixels,
    tth_tol,
    eta_tol,
    npdiv,
    ome_grid,
    delta_omega,
    ome_period,
    threshold,
    rmat_c,
    tvec_c,
    invalid_id,
):
    """Bounded extraction batches, preserving the historical spot record order."""
    from hexrd.core.extensions import transforms as native
    from hexrd.core.rotations import mapAngle
    import logging

    from hexrd.hedm._fit_progress import current_reporter
    reporter = current_reporter()
    threshold = _comparison_threshold(view.dtype, threshold)
    omega_eval = np.degrees(predicted[:, 2, None]) + ome_grid
    frame_ids = omega_to_frames(series, omega_eval)
    valid_windows = np.all(frame_ids >= 0, axis=1)
    for hkl in hkls[~valid_windows]:
        logging.getLogger().warning(f'window for {hkl} falls outside omega range')
    valid_ids = np.flatnonzero(valid_windows)
    if reporter is not None:
        reporter.panel(len(valid_ids))
    patches = instrument._get_panel_patches(
        panel,
        predicted[valid_ids],
        angular_pixels[valid_ids],
        tth_tol,
        eta_tol,
        rmat_c,
        tvec_c,
        npdiv,
        packed=True,
        max_points=max(1, 524288 // len(ome_grid)),
    )
    chunks = []

    for start, edges, patch_shapes, offsets, rows, cols in patches:
        ids = valid_ids[start : start + len(patch_shapes)]
        shapes = np.column_stack((np.full(len(ids), len(ome_grid)), patch_shapes))
        centers, sums, maxima, valid = view.extract(
            frame_ids[ids], offsets, rows, cols, shapes, threshold
        )
        if reporter is not None:
            reporter.reflections(len(ids), np.count_nonzero(valid))
        angles = np.full((len(ids), 3), np.nan)
        angles[valid, :2] = (
            edges[valid, :2] + (0.5 + centers[valid, :0:-1]) * edges[valid, 2:]
        )
        omega_start = series.omega[frame_ids[ids[valid], 0], 0]
        angles[valid, 2] = np.radians(
            omega_start + (0.5 + centers[valid, 0]) * delta_omega
        )
        xy = np.full((len(ids), 2), np.nan)
        peak_ids = ids.copy()
        bad = ~valid
        peak_ids[bad] = invalid_id - np.arange(np.count_nonzero(bad))
        invalid_id -= np.count_nonzero(bad)
        chunks.append(
            _SpotData(
                peak_ids,
                hkl_ids[ids],
                hkls[ids],
                sums,
                maxima,
                predicted[ids],
                angles,
                xy,
                predicted_xy[ids],
            )
        )
    result = _SpotData.concatenate(chunks) if chunks else None
    if result is not None:
        valid = result.peak_id >= 0
        result.meas_angs[valid, 2] = mapAngle(result.meas_angs[valid, 2], ome_period)
        measured = native._spotMeasuredXY(
            result.meas_angs[valid],
            panel.rmat,
            rmat_c,
            instrument.chi,
            panel.tvec,
            instrument.tvec,
            tvec_c,
            instrument.beam_vector,
        )
        if panel.distortion is not None and len(measured):
            # Keep per-spot distortion calls: user-provided mappings may have
            # convergence criteria depending on the supplied point batch.
            measured = np.array(
                [panel.distortion.apply_inverse(p[None, :]).ravel() for p in measured]
            )
        result.meas_xy[valid] = measured
    return result, invalid_id


def sparse_frame_view(series):
    """Borrow CSR buffers, including zero-copy ROI views, without densifying.

    Native views live outside the series' pickle state. Fork workers inherit
    the parent's immutable views; other workers construct their own on demand.
    """
    from hexrd.core.experiment import ImageSeries, RoiImageSeries
    from hexrd.core.imageseries.baseclass import ImageSeries as AdapterSeries
    from hexrd.core.imageseries.omega import OmegaImageSeries
    from hexrd.core.imageseries.process import ProcessedImageSeries
    from hexrd.core.imageseries.load.framecache import FrameCacheImageSeriesAdapter
    from hexrd.hedm.experiment import _AdapterImageSeries
    from hexrd.core.extensions import transforms as native

    if series in _frame_views:
        return _frame_views[series]
    from hexrd.hedm._image_cache import BoundedSeries

    if isinstance(series, BoundedSeries):
        result = series.native_view()
    elif isinstance(series, RoiImageSeries):
        if series._images is not None:
            result = _native_frames(series._images, native)
        else:
            parent = sparse_frame_view(series._parent)
            result = (
                None if parent is None else parent.view(*series._roi, *series._shape)
            )
    elif type(series) in (AdapterSeries, OmegaImageSeries):
        result = sparse_frame_view(series._adapter)
    elif type(series) is ProcessedImageSeries:
        result = None
        if (
            not series._hasframelist
            and len(series.oplist) == 1
            and series.oplist[0][0] == series.RECT
        ):
            parent = sparse_frame_view(series._imser)
            (r0, r1), (c0, c1) = series.oplist[0][1]
            if parent is not None:
                try:
                    result = parent.view(r0, c0, r1 - r0, c1 - c0)
                except ValueError:
                    pass
    elif type(series) is FrameCacheImageSeriesAdapter:
        series._load_framelist_if_needed()
        frames = series._framelist
        result = _native_frames(frames, native)
    elif type(series) is _AdapterImageSeries:
        # Registered frame-cache formats exposed by HedmExperiment.
        result = (
            _native_frames(series._images, native)
            if series._images is not None
            else sparse_frame_view(series._series)
        )
    elif type(series) is ImageSeries:
        result = _native_frames(series.images, native)
    else:
        result = None
    _frame_views[series] = result
    return result


def _native_frames(frames, native):
    if not frames or frames[0].dtype not in (
        np.dtype(x) for x in ('u1', 'i2', 'u2', 'i4', 'u4', 'f4', 'f8')
    ):
        return None
    if any(not f.has_sorted_indices for f in frames):
        return None
    return native._SparseFrames(frames)


def omega_to_frames(series, angles):
    """Vector form of OmegaImageSeries.omega_to_frame, preserving wedge order."""
    frames = np.full(angles.shape, -1, dtype=np.int64)
    for wedge, (start, stop, step) in enumerate(series._wedge_om):
        checked = start + np.mod(angles - start, series.TAU)
        use = (frames < 0) & (checked < stop)
        frames[use] = series._wedge_f[wedge, 0] + np.floor(
            (checked[use] - start) / step
        ).astype(np.int64)
    return frames


@lru_cache(maxsize=256)
def _moment_coordinates(length):
    # Zeroth and first powers for raw moments. The higher powers computed by
    # regionprops are unnecessary for weighted centroids.
    coordinates = np.column_stack((np.ones(length), np.arange(length)))
    coordinates.flags.writeable = False
    return coordinates


def _weighted_centroids(labels, intensity, num_labels):
    """Compute regionprops' weighted centroids using only first moments.

    Retain its bounding boxes, float64 intensities, axis contraction order,
    and local-coordinate arithmetic so the spot positions stay identical.
    """
    centers = np.empty((num_labels, labels.ndim))
    for i, region in enumerate(ndimage.find_objects(labels, num_labels)):
        image = intensity[region] * (labels[region] == i + 1)
        moments = image.astype(np.float64, copy=False)
        for axis, length in enumerate(image.shape):
            moments = np.moveaxis(
                np.tensordot(_moment_coordinates(length), moments, axes=(0, axis)),
                0,
                axis,
            )
        mass = moments[(0,) * labels.ndim]
        for axis, bounds in enumerate(region):
            index = [0] * labels.ndim
            index[axis] = 1
            centers[i, axis] = moments[tuple(index)] / mass + bounds.start
    return centers
