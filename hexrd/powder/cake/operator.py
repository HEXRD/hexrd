"""Persistent bilinear caking with compact detector inputs.

The file contract remains provisional pending workflow validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import h5py
import numba
import numpy as np

from hexrd.core.instrument import HEDMInstrument
from hexrd.core.utils.hdf5 import unwrap_h5_to_dict


from ._builder import (
    FORMAT_NAME,
    FORMAT_VERSION_MAJOR,
    FORMAT_VERSION_MINOR,
    REPRESENTATION,
    SCHEMA_STATUS,
    COMPRESSION,
    COMPRESSION_LEVEL,
    STRING_DTYPE as STRING,
    _write_array as _compressed,
    _write_mapping,
    _mapping_digest,
    _attribute_value,
    _package_version,
)

HEXRD_VERSION = _package_version("hexrd")


def _image_array(value):
    if np.ma.isMaskedArray(value):
        raise ValueError(
            "masked inputs must be explicitly filled (for example with NaN)"
        )
    array = np.asarray(value)
    if array.dtype.kind not in "buif" or array.dtype.itemsize > 8:
        raise ValueError("pixels must have a real numeric dtype of at most 64 bits")
    if array.dtype == np.float16:
        array = array.astype(np.float32)
    return array


@numba.njit(nogil=True, parallel=True, cache=True)
def _apply_output_blocks(output_indptr, sources, weights, active_pixels):
    output = np.empty(output_indptr.size - 1, dtype=np.float64)
    for row in numba.prange(output.size):
        value = 0.0
        for group in range(output_indptr[row], output_indptr[row + 1]):
            source = sources[group]
            weight = weights[group]
            value += (
                weight[0] * active_pixels[source[0]]
                + weight[1] * active_pixels[source[1]]
                + weight[2] * active_pixels[source[2]]
                + weight[3] * active_pixels[source[3]]
            )
        output[row] = value
    return output


@numba.njit(nogil=True, parallel=True, cache=True)
def _apply_output_blocks_batch(output_indptr, sources, weights, active_pixels):
    output = np.empty(
        (output_indptr.size - 1, active_pixels.shape[1]), dtype=np.float64
    )
    for row in numba.prange(output.shape[0]):
        for frame in range(output.shape[1]):
            value = 0.0
            for group in range(output_indptr[row], output_indptr[row + 1]):
                source = sources[group]
                weight = weights[group]
                value += (
                    weight[0] * active_pixels[source[0], frame]
                    + weight[1] * active_pixels[source[1], frame]
                    + weight[2] * active_pixels[source[2], frame]
                    + weight[3] * active_pixels[source[3], frame]
                )
            output[row, frame] = value
    return output


def _payload_digest(operator):
    digest = hashlib.sha256()

    def update(name, array):
        value = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype="<u8").tobytes())
        digest.update(value.tobytes())

    digest.update("\0".join(operator.panel_names).encode())
    update("tth_edges", operator.grid.tth_edges)
    update("eta_edges", operator.grid.eta_edges)
    update("panel_shapes", operator.panel_shapes)
    update("active_pixel_indptr", operator.active_pixel_indptr)
    update("active_local_indices", operator.active_local_indices)
    update("output_indptr", operator.output_indptr)
    update("source_indices", operator.source_indices)
    update("weights", operator.weights)
    if operator.active_frame_indices is not None:
        update("active_frame_indices", operator.active_frame_indices)
        update("composite_frame_shape", operator.composite_frame_shape)
    return digest.hexdigest()


@dataclass(frozen=True)
class CakeGrid:
    """The logical polar output grid, independent of scan orchestration."""

    tth_edges: np.ndarray
    eta_edges: np.ndarray
    output_mask: np.ndarray
    angle_units: str = "degrees"
    array_order: str = "eta,tth (C order)"

    def __post_init__(self):
        tth = np.asarray(self.tth_edges, dtype=np.float64)
        eta = np.asarray(self.eta_edges, dtype=np.float64)
        mask = np.asarray(self.output_mask)
        if mask.dtype != np.dtype(bool):
            raise ValueError("output_mask must be boolean")
        if tth.ndim != 1 or eta.ndim != 1:
            raise ValueError("tth_edges and eta_edges must be one-dimensional")
        if tth.size < 2 or eta.size < 2:
            raise ValueError("angular edge arrays must contain at least two values")
        if not np.isfinite(tth).all() or not np.isfinite(eta).all():
            raise ValueError("angular edges must be finite")
        if self.angle_units != "degrees" or self.array_order != "eta,tth (C order)":
            raise ValueError("unsupported angular units or array order")
        if np.any(np.diff(tth) <= 0) or np.any(np.diff(eta) <= 0):
            raise ValueError("angular edges must be strictly increasing")
        expected = (eta.size - 1, tth.size - 1)
        if mask.shape != expected:
            raise ValueError(f"output_mask has shape {mask.shape}, expected {expected}")
        object.__setattr__(self, "tth_edges", tth)
        object.__setattr__(self, "eta_edges", eta)
        object.__setattr__(self, "output_mask", mask)

    @property
    def tth_centers(self):
        return 0.5 * (self.tth_edges[:-1] + self.tth_edges[1:])

    @property
    def eta_centers(self):
        return 0.5 * (self.eta_edges[:-1] + self.eta_edges[1:])

    @property
    def shape(self):
        return self.output_mask.shape

    @property
    def size(self):
        return self.output_mask.size


@dataclass
class CakeOperator:
    """Reusable geometry transform from compact detector pixels to polar bins.

    Scan parsing, dark correction, omega integration, process orchestration,
    and result-file writing deliberately remain outside this class.
    """

    grid: CakeGrid
    panel_names: tuple[str, ...]
    panel_shapes: np.ndarray
    active_pixel_indptr: np.ndarray
    active_local_indices: np.ndarray
    output_indptr: np.ndarray
    source_indices: np.ndarray
    weights: np.ndarray
    instrument_config: dict[str, Any] | None = None
    active_frame_indices: np.ndarray | None = None
    composite_frame_shape: tuple[int, ...] | None = None
    eta_wedge_bounds: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    reduction_modes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    _source_path: Path | None = field(default=None, init=False, repr=False)
    _instrument_cache: HEDMInstrument | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        self.panel_names = tuple(self.panel_names)
        for name in (
            "panel_shapes",
            "active_pixel_indptr",
            "active_local_indices",
            "output_indptr",
            "source_indices",
            "active_frame_indices",
        ):
            value = getattr(self, name)
            if value is None and name == "active_frame_indices":
                continue
            array = np.asarray(value)
            if array.dtype.kind not in "iu" or np.any(array < 0):
                raise ValueError(f"{name} must contain nonnegative integers")
            setattr(self, name, np.ascontiguousarray(array))
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float64)
        if self.composite_frame_shape is not None:
            shape = np.asarray(self.composite_frame_shape)
            if (
                shape.shape != (2,)
                or shape.dtype.kind not in "iu"
                or np.any(shape <= 0)
            ):
                raise ValueError(
                    "composite_frame_shape must contain two positive integers"
                )
            self.composite_frame_shape = tuple(int(x) for x in shape)
        self.eta_wedge_bounds = np.asarray(self.eta_wedge_bounds, dtype=np.float64)
        self.reduction_modes = tuple(self.reduction_modes)
        self.validate()

    @classmethod
    def build(cls, path, instrument, spec, **kwargs):
        """Stream geometry to a file, then load the execution arrays.

        Use ``export_cake_operator`` directly to build a large operator without
        loading all of its execution arrays into memory.
        """
        from ._builder import export_cake_operator

        export_cake_operator(path, instrument, spec, **kwargs)
        return cls.load(path)

    @property
    def instrument(self):
        """Lazily reconstruct the embedded instrument only when requested."""
        if self._instrument_cache is None:
            if self.instrument_config is None:
                if self._source_path is None:
                    raise RuntimeError("no embedded instrument source is available")
                config = {}
                with h5py.File(self._source_path, "r") as stream:
                    unwrap_h5_to_dict(stream["cake_operator/instrument"], config)
                self.instrument_config = config
            self._instrument_cache = HEDMInstrument(deepcopy(self.instrument_config))
        return self._instrument_cache

    @property
    def active_pixel_count(self):
        return self.active_local_indices.size

    @property
    def group_count(self):
        return self.source_indices.shape[0]

    @property
    def term_count(self):
        return self.weights.size

    def validate(self):
        """Check every index range before entering the compiled executor."""
        panel_count = len(self.panel_names)
        if not panel_count or any(
            not isinstance(x, str) or not x for x in self.panel_names
        ):
            raise ValueError("panel_names must contain nonempty strings")
        if len(set(self.panel_names)) != panel_count:
            raise ValueError("panel_names must be unique")
        if self.panel_shapes.shape != (panel_count, 2) or np.any(
            self.panel_shapes == 0
        ):
            raise ValueError("panel_shapes must have positive shape (panel_count, 2)")
        if self.active_local_indices.ndim != 1:
            raise ValueError("active_local_indices must be one-dimensional")
        if self.source_indices.ndim != 2 or self.source_indices.shape[1] != 4:
            raise ValueError("source_indices must have shape (group_count, 4)")
        for name, length, end in (
            ("active_pixel_indptr", panel_count + 1, self.active_pixel_count),
            ("output_indptr", self.grid.size + 1, self.group_count),
        ):
            pointer = getattr(self, name)
            if pointer.shape != (length,):
                raise ValueError(f"{name} has the wrong length")
            # Subtraction wraps for unsigned pointers: compare adjacent values.
            if (
                pointer[0] != 0
                or np.any(pointer[1:] < pointer[:-1])
                or pointer[-1] != end
            ):
                raise ValueError(
                    f"{name} must start at zero, be monotonic, and span its records"
                )
        for i, shape in enumerate(self.panel_shapes):
            start, stop = self.active_pixel_indptr[i : i + 2]
            local = self.active_local_indices[start:stop]
            if np.any(local >= int(shape[0]) * int(shape[1])):
                raise ValueError("active_local_indices exceed panel shape")
            if np.any(local[1:] <= local[:-1]):
                raise ValueError(
                    "active_local_indices must be sorted and unique per panel"
                )
        if (
            self.weights.shape != self.source_indices.shape
            or not np.isfinite(self.weights).all()
        ):
            raise ValueError("weights must be finite with shape (group_count, 4)")
        if self.group_count and self.source_indices.max() >= self.active_pixel_count:
            raise ValueError("source_indices exceed compact active pixels")
        empty = self.output_indptr[1:] == self.output_indptr[:-1]
        if np.any(empty.reshape(self.grid.shape) & ~self.grid.output_mask):
            raise ValueError("output_mask must mask every empty output row")
        if self.active_frame_indices is not None:
            if self.composite_frame_shape is None:
                raise ValueError("composite_frame_shape is required with an adapter")
            if self.active_frame_indices.shape != (self.active_pixel_count,):
                raise ValueError("active_frame_indices has the wrong length")
            if self.active_frame_indices.size and self.active_frame_indices.max() >= (
                self.composite_frame_shape[0] * self.composite_frame_shape[1]
            ):
                raise ValueError("active_frame_indices exceed composite frame")
        elif self.composite_frame_shape is not None:
            raise ValueError("composite_frame_shape requires active_frame_indices")
        if self.eta_wedge_bounds.ndim != 2 or self.eta_wedge_bounds.shape[1] != 2:
            raise ValueError("eta_wedge_bounds must have shape (wedges, 2)")
        if len(self.reduction_modes) != len(self.eta_wedge_bounds):
            raise ValueError("one reduction mode is required per eta wedge")
        if any(x not in {"sum", "mean", "masked_mean"} for x in self.reduction_modes):
            raise ValueError("unsupported reduction mode")

    def compact_input(self):
        """Drop active inputs not referenced by this loaded output region."""
        used, inverse = np.unique(self.source_indices, return_inverse=True)
        if used.size == self.active_pixel_count:
            return self

        old_indptr = self.active_pixel_indptr
        new_indptr = [0]
        for panel_index in range(len(self.panel_names)):
            start = np.searchsorted(used, old_indptr[panel_index], side="left")
            stop = np.searchsorted(used, old_indptr[panel_index + 1], side="left")
            new_indptr.append(new_indptr[-1] + stop - start)

        self.active_local_indices = self.active_local_indices[used]
        self.active_pixel_indptr = np.asarray(new_indptr, dtype=np.uint64)
        self.source_indices = inverse.reshape(self.source_indices.shape)
        if used.size < 2**32:
            self.source_indices = self.source_indices.astype(np.uint32)
        else:
            self.source_indices = self.source_indices.astype(np.uint64)
        if self.active_frame_indices is not None:
            self.active_frame_indices = self.active_frame_indices[used]
        self.metadata["input_compacted_after_load"] = True
        self.validate()
        return self

    def gather_active_pixels(self, images: Mapping[str, np.ndarray]):
        """Gather only the panel pixels referenced by the operator."""
        if set(images) != set(self.panel_names):
            missing = set(self.panel_names) - set(images)
            extra = set(images) - set(self.panel_names)
            raise ValueError(f"panel mismatch; missing={missing}, extra={extra}")
        dtype = np.result_type(*(np.asarray(images[x]).dtype for x in images))
        active = np.empty(self.active_pixel_count, dtype=dtype)
        for panel_index, (name, shape) in enumerate(
            zip(self.panel_names, self.panel_shapes)
        ):
            image = _image_array(images[name])
            if image.shape != tuple(shape):
                raise ValueError(
                    f"{name} has shape {image.shape}, expected {tuple(shape)}"
                )
            start = self.active_pixel_indptr[panel_index]
            stop = self.active_pixel_indptr[panel_index + 1]
            local = self.active_local_indices[start:stop]
            if image.flags.c_contiguous:
                active[start:stop] = image.ravel()[local]
            else:
                rows, cols = np.divmod(local, int(shape[1]))
                active[start:stop] = image[rows, cols]
        return active

    def gather_active_composite_pixels(self, frame: np.ndarray):
        """Gather active values with a precompiled flat-index adapter."""
        frame = _image_array(frame)
        if frame.ndim != 2:
            raise ValueError("composite frame must be two-dimensional")
        if self.active_frame_indices is not None:
            if frame.shape != self.composite_frame_shape:
                raise ValueError(
                    f"frame has shape {frame.shape}, expected "
                    f"{self.composite_frame_shape}"
                )
            if frame.flags.c_contiguous:
                return frame.ravel()[self.active_frame_indices]
            rows, cols = np.divmod(self.active_frame_indices, frame.shape[1])
            return frame[rows, cols]

        active = np.empty(self.active_pixel_count, dtype=frame.dtype)
        for panel_index, (name, shape) in enumerate(
            zip(self.panel_names, self.panel_shapes)
        ):
            panel = self.instrument.detectors[name]
            if panel.roi is None:
                raise ValueError(f"panel {name} has no ROI for composite input")
            start = self.active_pixel_indptr[panel_index]
            stop = self.active_pixel_indptr[panel_index + 1]
            local = self.active_local_indices[start:stop]
            local_rows, local_columns = np.divmod(local, int(shape[1]))
            (row_start, row_stop), (column_start, column_stop) = panel.roi
            if row_stop > frame.shape[0] or column_stop > frame.shape[1]:
                raise ValueError(f"composite frame is too small for panel {name}")
            active[start:stop] = frame[
                row_start + local_rows, column_start + local_columns
            ]
        return active

    def apply_active_pixels(self, active_pixels, mask_output=True):
        """Execute directly from the output-major four-corner layout."""
        active = _image_array(active_pixels)
        if active.ndim == 1:
            if active.shape != (self.active_pixel_count,):
                raise ValueError("active pixel vector has the wrong shape")
            output = _apply_output_blocks(
                self.output_indptr,
                self.source_indices,
                self.weights,
                active,
            ).reshape(self.grid.shape)
            if mask_output:
                return np.ma.array(output, mask=self.grid.output_mask)
            return output

        if active.ndim == 2 and active.shape[0] == self.active_pixel_count:
            output = _apply_output_blocks_batch(
                self.output_indptr,
                self.source_indices,
                self.weights,
                active,
            ).reshape((*self.grid.shape, active.shape[1]))
            if mask_output:
                mask = np.broadcast_to(self.grid.output_mask[..., None], output.shape)
                return np.ma.array(output, mask=mask)
            return output
        raise ValueError("active pixels must have shape (pixels,) or (pixels, frames)")

    def apply(self, images: Mapping[str, np.ndarray], mask_output=True):
        return self.apply_active_pixels(
            self.gather_active_pixels(images), mask_output=mask_output
        )

    def apply_composite(self, frame: np.ndarray, mask_output=True):
        return self.apply_active_pixels(
            self.gather_active_composite_pixels(frame),
            mask_output=mask_output,
        )

    def save(self, path, *, overwrite=False):
        """Atomically save a self-contained operator."""
        self.validate()
        path = Path(path).expanduser().resolve()
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        try:
            writer = self
            counts = np.diff(self.output_indptr)
            if np.any(self.grid.output_mask.ravel() & (counts != 0)):
                # Older files can mask rows that still have contributions.
                # Remove those records when migrating; preserve the caller.
                writer = copy(self)
                writer.metadata = dict(self.metadata)
                keep = np.repeat(~self.grid.output_mask.ravel(), counts.astype(np.intp))
                counts = np.where(self.grid.output_mask.ravel(), 0, counts)
                writer.output_indptr = np.concatenate(
                    (np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64))
                )
                writer.source_indices = self.source_indices[keep]
                writer.weights = self.weights[keep]
                writer.compact_input()
                writer.validate()
            writer._save_file(temporary, overwrite=True)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _save_file(self, path, *, overwrite=False):
        """Write the proposed self-contained, compressed HDF5 contract."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        if (
            self.instrument_config is None
            and self._source_path is not None
            and path.resolve() == self._source_path.resolve()
        ):
            _ = self.instrument

        with h5py.File(path, "w", libver="latest") as stream:
            root = stream.create_group("cake_operator", track_order=True)
            root.attrs["format"] = FORMAT_NAME
            root.attrs["format_version_major"] = FORMAT_VERSION_MAJOR
            root.attrs["format_version_minor"] = FORMAT_VERSION_MINOR
            root.attrs["schema_status"] = SCHEMA_STATUS
            root.attrs["hexrd_version"] = HEXRD_VERSION
            root.attrs["representation"] = REPRESENTATION
            root.attrs["array_order"] = self.grid.array_order
            root.attrs["compression"] = f"{COMPRESSION}-{COMPRESSION_LEVEL}"
            root.attrs["payload_sha256"] = _payload_digest(self)
            if self.instrument_config is not None:
                instrument_sha256 = _mapping_digest(self.instrument_config)
            elif self._source_path is not None:
                with h5py.File(self._source_path, "r") as source:
                    instrument_sha256 = source["cake_operator"].attrs[
                        "instrument_sha256"
                    ]
            else:
                raise RuntimeError("an embedded instrument is required")
            root.attrs["instrument_sha256"] = instrument_sha256
            root.attrs["required_features_json"] = json.dumps(
                ["core-v2", REPRESENTATION]
            )
            optional = []
            if self.active_frame_indices is not None:
                optional.append("composite-flat-index-v1")
            if len(self.eta_wedge_bounds):
                optional.append("named-reduction-recipe-v1")
            root.attrs["optional_features_json"] = json.dumps(optional)
            for key, value in self.metadata.items():
                if key in root.attrs:
                    raise ValueError(f"reserved metadata key: {key}")
                root.attrs[key] = _attribute_value(value)

            root.create_dataset(
                "README",
                data=(
                    "Draft self-contained persistent HEXRD cake operator. "
                    "Unknown optional groups may be ignored. Unknown required "
                    "features must be rejected."
                ),
                dtype=STRING,
            )

            if self.instrument_config is not None:
                instrument_group = root.create_group("instrument")
                _write_mapping(instrument_group, self.instrument_config)
            elif self._source_path is not None:
                with h5py.File(self._source_path, "r") as source:
                    source.copy(
                        source["cake_operator/instrument"],
                        root,
                        name="instrument",
                    )
            else:
                raise RuntimeError("an embedded instrument is required")

            grid = root.create_group("grid", track_order=True)
            grid.attrs["angle_units"] = self.grid.angle_units
            grid.attrs["array_order"] = self.grid.array_order
            _compressed(grid, "tth_edges", self.grid.tth_edges)
            _compressed(grid, "eta_edges", self.grid.eta_edges)
            grid.create_dataset("shape", data=np.asarray(self.grid.shape))

            inputs = root.create_group("input", track_order=True)
            inputs.attrs["indexing"] = "zero-based, panel-local flattened C order"
            inputs.attrs["active_pixel_count"] = self.active_pixel_count
            inputs.attrs["full_panel_pixel_count"] = int(
                np.prod(self.panel_shapes, axis=1).sum()
            )
            inputs.create_dataset(
                "panel_names",
                data=np.asarray(self.panel_names, dtype=object),
                dtype=STRING,
            )
            _compressed(inputs, "panel_shapes", self.panel_shapes)
            _compressed(inputs, "active_pixel_indptr", self.active_pixel_indptr)
            _compressed(inputs, "active_local_indices", self.active_local_indices)

            if self.active_frame_indices is not None:
                adapters = root.create_group("input_adapters")
                composite = adapters.create_group("composite")
                composite.attrs["kind"] = "flat-index-v1"
                composite.attrs["frame_order"] = "C"
                composite.create_dataset("frame_shape", data=self.composite_frame_shape)
                _compressed(
                    composite,
                    "active_frame_indices",
                    self.active_frame_indices,
                )

            transform = root.create_group("operator", track_order=True)
            transform.attrs["semantics"] = (
                "For output row r and groups g in "
                "[output_indptr[r], output_indptr[r+1]), accumulate "
                "sum(weights[g, :] * active_pixels[source_indices[g, :]])."
            )
            transform.attrs["output_size"] = self.grid.size
            transform.attrs["group_count"] = self.group_count
            transform.attrs["term_count"] = self.term_count
            transform.attrs["weight_dtype"] = str(self.weights.dtype)
            _compressed(transform, "output_indptr", self.output_indptr)
            _compressed(transform, "source_indices", self.source_indices)
            _compressed(transform, "weights", self.weights)

            if len(self.eta_wedge_bounds):
                recipes = root.create_group("recipes")
                default = recipes.create_group("default")
                reductions = default.create_group("reductions")
                reductions.attrs["full_eta_mode"] = "sum"
                reductions.attrs["status"] = (
                    "Optional workflow recipe, not part of core geometry."
                )
                bounds = _compressed(
                    reductions,
                    "eta_wedge_bounds",
                    self.eta_wedge_bounds,
                )
                bounds.attrs["units"] = self.grid.angle_units
                reductions.create_dataset(
                    "modes",
                    data=np.asarray(self.reduction_modes, dtype=object),
                    dtype=STRING,
                )

            provenance = root.create_group("provenance", track_order=True)
            for key, value in self.provenance.items():
                provenance.attrs[key] = _attribute_value(value)

    @classmethod
    def load(
        cls,
        path,
        *,
        eta_slice=None,
        compact_input=False,
        verify_digest=False,
    ):
        """Load hot arrays eagerly and leave the instrument snapshot lazy.

        eta_slice may select contiguous eta rows. The output-major layout lets
        HDF5 read only the corresponding transform records.
        """
        path = Path(path).expanduser().resolve()
        if verify_digest:
            from ._builder import verify_cake_operator

            verify_cake_operator(path)
        with h5py.File(path, "r") as stream:
            root = stream["cake_operator"]
            if root.attrs["format"] != FORMAT_NAME:
                raise ValueError(f"{path} is not a HEXRD cake operator")
            if int(root.attrs["format_version_major"]) not in {1, FORMAT_VERSION_MAJOR}:
                raise ValueError(
                    "unsupported major format version "
                    f"{root.attrs['format_version_major']}"
                )
            required = set(json.loads(root.attrs["required_features_json"]))
            understood = {"core-v1", "core-v2", REPRESENTATION}
            unknown = required - understood
            if unknown:
                raise ValueError(f"unknown required features: {sorted(unknown)}")
            if root.attrs["representation"] != REPRESENTATION:
                raise ValueError(
                    f"unsupported representation {root.attrs['representation']}"
                )

            grid_group = root["grid"]
            tth_edges = grid_group["tth_edges"][:]
            full_eta_edges = grid_group["eta_edges"][:]
            full_eta_count = len(full_eta_edges) - 1
            tth_count = len(tth_edges) - 1

            if eta_slice is None:
                eta_start, eta_stop = 0, full_eta_count
            else:
                eta_start, eta_stop, step = eta_slice.indices(full_eta_count)
                if step != 1:
                    raise ValueError("eta_slice must be contiguous")
            if eta_stop <= eta_start:
                raise ValueError("eta_slice must select at least one row")
            row_start = eta_start * tth_count
            row_stop = eta_stop * tth_count

            stored_indptr = root["operator/output_indptr"]
            output_indptr = stored_indptr[row_start : row_stop + 1]
            if output_indptr.dtype.kind not in "iu" or np.any(output_indptr < 0):
                raise ValueError("output_indptr must contain nonnegative integers")
            group_start = int(output_indptr[0])
            group_stop = int(output_indptr[-1])
            if np.any(output_indptr[1:] < output_indptr[:-1]):
                raise ValueError("output_indptr must be monotonic")
            if group_stop > root["operator/source_indices"].shape[0]:
                raise ValueError("output_indptr exceeds stored records")
            if eta_slice is None and group_start != 0:
                raise ValueError("output_indptr must start at zero")
            if (
                eta_slice is None
                and group_stop != root["operator/source_indices"].shape[0]
            ):
                raise ValueError("output_indptr does not span stored records")
            if root["operator/weights"].shape != root["operator/source_indices"].shape:
                raise ValueError("stored source and weight shapes differ")
            output_indptr = output_indptr - group_start

            if int(root.attrs["format_version_major"]) == 1:
                output_mask = grid_group["output_mask"][eta_start:eta_stop]
            else:
                if "output_mask" in grid_group:
                    raise ValueError("version 2 derives output_mask from contributions")
                output_mask = (output_indptr[1:] == output_indptr[:-1]).reshape(
                    eta_stop - eta_start, tth_count
                )

            grid = CakeGrid(
                tth_edges=tth_edges,
                eta_edges=full_eta_edges[eta_start : eta_stop + 1],
                output_mask=output_mask,
                angle_units=grid_group.attrs["angle_units"],
                array_order=grid_group.attrs["array_order"],
            )

            if tuple(grid_group["shape"][:]) != (full_eta_count, tth_count):
                raise ValueError("stored grid shape does not match angular edges")

            recipe_path = "recipes/default/reductions"
            if recipe_path in root:
                reductions = root[recipe_path]
                eta_wedge_bounds = reductions["eta_wedge_bounds"][:]
                reduction_modes = tuple(reductions["modes"].asstr()[:])
            else:
                eta_wedge_bounds = np.empty((0, 2))
                reduction_modes = ()

            adapter_path = "input_adapters/composite"
            if adapter_path in root:
                adapter = root[adapter_path]
                if (
                    adapter.attrs["kind"] != "flat-index-v1"
                    or adapter.attrs["frame_order"] != "C"
                ):
                    raise ValueError("unsupported composite adapter")
                active_frame_indices = adapter["active_frame_indices"][:]
                composite_frame_shape = tuple(adapter["frame_shape"][:])
            else:
                active_frame_indices = None
                composite_frame_shape = None

            metadata = {
                str(key): value
                for key, value in root.attrs.items()
                if key
                not in {
                    "format",
                    "format_version_major",
                    "format_version_minor",
                    "schema_status",
                    "hexrd_version",
                    "representation",
                    "array_order",
                    "compression",
                    "payload_sha256",
                    "instrument_sha256",
                    "required_features_json",
                    "optional_features_json",
                }
            }
            if eta_slice is not None:
                metadata["loaded_eta_slice"] = [eta_start, eta_stop]

            result = cls(
                grid=grid,
                panel_names=tuple(root["input/panel_names"].asstr()[:]),
                panel_shapes=root["input/panel_shapes"][:],
                active_pixel_indptr=root["input/active_pixel_indptr"][:],
                active_local_indices=root["input/active_local_indices"][:],
                output_indptr=output_indptr,
                source_indices=root["operator/source_indices"][group_start:group_stop],
                weights=root["operator/weights"][group_start:group_stop],
                active_frame_indices=active_frame_indices,
                composite_frame_shape=composite_frame_shape,
                eta_wedge_bounds=eta_wedge_bounds,
                reduction_modes=reduction_modes,
                metadata=metadata,
                provenance={
                    str(key): value for key, value in root["provenance"].attrs.items()
                },
            )
            result._source_path = path

            if compact_input:
                result.compact_input()
            return result

    def summary(self):
        full_pixels = int(np.prod(self.panel_shapes, axis=1).sum())
        return {
            "panels": len(self.panel_names),
            "grid_shape": self.grid.shape,
            "active_pixels": self.active_pixel_count,
            "full_panel_pixels": full_pixels,
            "active_fraction": self.active_pixel_count / full_pixels,
            "operator_groups": self.group_count,
            "operator_terms": self.term_count,
            "eta_wedges": len(self.eta_wedge_bounds),
            "representation": REPRESENTATION,
            "format_version": (
                FORMAT_VERSION_MAJOR,
                FORMAT_VERSION_MINOR,
            ),
            "instrument_loaded": self._instrument_cache is not None,
            "composite_adapter": self.active_frame_indices is not None,
        }
