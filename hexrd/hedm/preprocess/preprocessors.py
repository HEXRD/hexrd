import logging
import os
import time
from typing import Any, Optional, Union, Sequence, cast

import h5py
from numpy.typing import NDArray
from numpy import float32

from hexrd.core import imageseries
from hexrd.core.imageseries.baseclass import ImageSeries
from hexrd.core.imageseries.omega import OmegaWedges
from hexrd.core.imageseries.process import ProcessedImageSeries
from hexrd.hedm.preprocess.profiles import (
    Eiger_Arguments,
    Dexelas_Arguments,
    HexrdPPScript_Arguments,
)

logger = logging.getLogger(__name__)


class PP_Base(object):
    PROCFMT: Optional[str] = None

    def __init__(
        self,
        fname: str,
        omw: OmegaWedges,
        panel_opts: list,
        frame_start: int = 0,
        style: str = "npz",
        **kwargs: Any,
    ):
        self.fname = fname
        self.omwedges = omw
        self.panel_opts = panel_opts
        self.frame_start = frame_start
        self.style = style
        self.use_frame_list = False
        self.raw = ImageSeries(adapter=None)

    @property
    def oplist(self) -> list:
        return self.panel_opts

    @property
    def framelist(self) -> Sequence[int]:
        return range(self.frame_start, self.nframes + self.frame_start)

    @property
    def nframes(self) -> int:
        return self.omwedges.nframes

    @property
    def omegas(self) -> NDArray:
        return self.omwedges.omegas

    def processed(self) -> ProcessedImageSeries:
        kw = {}
        if self.use_frame_list:
            kw = dict(frame_list=self.framelist)
        return ProcessedImageSeries(self.raw, self.oplist, **kw)

    def _attach_metadata(self, metadata: dict) -> None:
        metadata["omega"] = self.omegas

    def save_processed(
        self, name: str, threshold: int, output_dir: Optional[str] = None
    ) -> None:
        if output_dir is None:
            output_dir = os.getcwd()
        else:
            os.mkdir(output_dir)

        # add omegas
        pims = self.processed()
        self._attach_metadata(pims.metadata)
        cache = f"{name}-cachefile." + f"{self.style}"
        imageseries.write(
            pims,
            "dummy",
            self.PROCFMT,
            style=self.style,
            threshold=threshold,
            cache_file=cache,
        )


class PP_Eiger(PP_Base):
    """PP_Eiger"""

    PROCFMT = "frame-cache"

    def __init__(
        self,
        fname: str,
        omw: OmegaWedges,
        panel_opts: list = [],
        frame_start: int = 0,
        style: str = "npz",
        eiger_stream_v2_threshold: str = "threshold_1",
        eiger_stream_v2_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            fname=fname,
            omw=omw,
            panel_opts=panel_opts,
            frame_start=frame_start,
            style=style,
        )
        self.raw = imageseries.open(self.fname, format=self.rawfmt)
        self.use_frame_list = self.nframes != len(self.raw)
        self.eiger_stream_v2_threshold = eiger_stream_v2_threshold
        self.eiger_stream_v2_multiplier = eiger_stream_v2_multiplier
        logger.info(
            f"On Init:\n\t{self.fname}, {self.nframes} frames, "
            f"{self.omwedges.nframes} omw, {len(self.raw)} total"
        )

        if self.is_eiger_stream_v2:
            logger.info(
                "Eiger stream v2 file format detected. Threshold setting is: "
                + self.eiger_stream_v2_threshold
            )
            if self.eiger_stream_v2_threshold == "man_diff":
                logger.info(
                    f"Eiger stream v2 multiplier is: {self.eiger_stream_v2_multiplier}"
                )

            self.raw.set_option('threshold_setting', self.eiger_stream_v2_threshold)
            self.raw.set_option('multiplier', self.eiger_stream_v2_multiplier)

    @property
    def rawfmt(self) -> str:
        # Open the file and check if it is eiger-stream-v1 or eiger-stream-v2
        # Only check the format once. Cache it.
        if not hasattr(self, '_rawfmt'):
            v2_str = 'CHESS_EIGER_STREAM_V2'
            fmt = 'eiger-stream-v1'
            with h5py.File(self.fname, 'r') as f:
                if f.attrs.get('version', None) == v2_str:
                    fmt = 'eiger-stream-v2'

            self._raw_fmt = fmt

        return self._raw_fmt

    @property
    def is_eiger_stream_v2(self) -> bool:
        return self.rawfmt == 'eiger-stream-v2'


class PP_Dexela(PP_Base):
    """PP_Dexela"""

    PROCFMT = "frame-cache"

    RAWPATH = "/imageseries"
    DARKPCTILE = 50

    def __init__(
        self,
        fname: str,
        omw: OmegaWedges,
        panel_opts: list,
        panel_id: str,
        frame_start: int = 0,
        style: str = "npz",
        raw_format: str = "hdf5",
        dark: Union[NDArray, float32] = None,
    ) -> None:
        super().__init__(
            fname=fname,
            omw=omw,
            panel_opts=panel_opts,
            frame_start=frame_start,
            style=style,
        )

        self.rawfmt = "hdf5"

        self._panel_id = panel_id
        # TODO is this logic applicable also for Eiger ?
        if raw_format.lower() == "hdf5":
            self.raw = imageseries.open(self.fname, self.rawfmt, path=self.RAWPATH)
        else:
            self.raw = imageseries.open(self.fname, raw_format.lower())
        self._dark = dark

        self.use_frame_list = self.nframes != len(
            self.raw
        )  # Framelist fix, DCP 6/18/18
        logger.info(
            f"On Init:\n\t{self.fname}, {self.nframes} frames, "
            f"{self.omwedges.nframes} omw, {len(self.raw)} total"
        )

    def _attach_metadata(self, metadata: dict) -> None:
        super()._attach_metadata(metadata)
        metadata["panel_id"] = self.panel_id

    @property
    def panel_id(self) -> str:
        return self._panel_id

    @property
    def oplist(self) -> list:
        return [('dark', self.dark)] + self.panel_opts

    @property
    def dark(self, nframes: int = 100) -> Union[NDArray, float32]:
        """Build and return dark image"""
        if self._dark is None:
            usenframes = min(nframes, self.nframes)
            logger.info(f"Building dark images using {usenframes} frames...")

            start = time.time()
            self._dark = imageseries.stats.median(self.raw, nframes=usenframes)
            elapsed = time.time() - start
            logger.info(f"Done building background (dark) image. Took {elapsed} sec.")

        return self._dark


def preprocess(args: HexrdPPScript_Arguments) -> None:
    if type(args) == Eiger_Arguments:
        omw = imageseries.omega.OmegaWedges(args.num_frames)
        omw.addwedge(args.ome_start, args.ome_end, args.num_frames)
        ppe = PP_Eiger(
            fname=args.file_name,
            omw=omw,
            frame_start=args.start_frame,
            style=args.style,
            eiger_stream_v2_threshold=args.eiger_stream_v2_threshold,
            eiger_stream_v2_multiplier=args.eiger_stream_v2_multiplier,
        )
        ppe.save_processed(args.output, args.threshold)
    elif type(args) == Dexelas_Arguments:
        omw = imageseries.omega.OmegaWedges(args.num_frames)
        omw.addwedge(args.ome_start, args.ome_end, args.num_frames)
        for file_name in args.file_names:
            for key in args.panel_opts.keys():
                if key.lower() in file_name:
                    ppd = PP_Dexela(
                        fname=file_name,
                        omw=omw,
                        panel_opts=args.panel_opts[key],
                        panel_id=key,
                        frame_start=args.start_frame,
                        style=args.style,
                    )

                    samp_name = cast(str, args.samp_name)
                    scan_number = cast(int, args.scan_number)
                    output_name = (
                        samp_name
                        + "_"
                        + str(scan_number)
                        + "_"
                        + file_name.split("/")[-1].split(".")[0]
                    )
                    ppd.save_processed(output_name, args.threshold)
    else:
        raise AttributeError(f"Unknown argument type: {type(args)}")
