from hexrd.core.imageseries.baseclass import ImageSeries
from hexrd.core.imageseries.omega import OmegaWedges
from hexrd.hedm.preprocess.profiles import Eiger_Arguments, Dexelas_Arguments, HexrdPPScript_Arguments
from hexrd.core import imageseries
from hexrd.core.imageseries.process import ProcessedImageSeries
import os
import time
from typing import Any, Optional, Union, Sequence, cast
from numpy.typing import NDArray
from numpy import float32


class PP_Base(object):
    PROCFMT: Optional[str] = None
    RAWFMT: Optional[str] = None

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
    RAWFMT = "eiger-stream-v1"

    def __init__(
        self,
        fname: str,
        omw: OmegaWedges,
        panel_opts: list = [],
        frame_start: int = 0,
        style: str = "npz",
    ) -> None:
        super().__init__(
            fname=fname,
            omw=omw,
            panel_opts=panel_opts,
            frame_start=frame_start,
            style=style,
        )
        self.raw = imageseries.open(self.fname, format=self.RAWFMT)
        self.use_frame_list = self.nframes != len(self.raw)
        print(
            f"On Init:\n\t{self.fname}, {self.nframes} frames,"
            f"{self.omwedges.nframes} omw, {len(self.raw)} total"
        )


class PP_Dexela(PP_Base):
    """PP_Dexela"""

    PROCFMT = "frame-cache"
    RAWFMT = "hdf5"

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

        self._panel_id = panel_id
        # TODO is this logic applicable also for Eiger ?
        if raw_format.lower() == "hdf5":
            self.raw = imageseries.open(
                self.fname, self.RAWFMT, path=self.RAWPATH
            )
        else:
            self.raw = imageseries.open(self.fname, raw_format.lower())
        self._dark = dark

        self.use_frame_list = self.nframes != len(
            self.raw
        )  # Framelist fix, DCP 6/18/18
        print(
            f"On Init:\n\t{self.fname}, {self.nframes} frames,"
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
        """build and return dark image"""
        if self._dark is None:
            usenframes = min(nframes, self.nframes)
            print(
                "building dark images using %s frames (may take a while)..."
                % usenframes
            )
            start = time.time()
            #            self._dark = imageseries.stats.percentile(
            #                    self.raw, self.DARKPCTILE, nframes=usenframes
            #            )
            self._dark = imageseries.stats.median(
                self.raw, nframes=usenframes
            )  # changed to median by DCP 11/18/17
            elapsed = time.time() - start
            print(
                "done building background (dark) image: "
                + "elapsed time is %f seconds" % elapsed
            )

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
