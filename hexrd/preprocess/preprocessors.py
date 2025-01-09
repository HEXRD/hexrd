from hexrd.preprocess.profiles import Eiger_Arguments, Dexelas_Arguments
from hexrd import imageseries
from hexrd.imageseries.process import ProcessedImageSeries
import os
import time


class PP_Base(object):
    PROCFMT = None
    RAWFMT = None

    def __init__(
        self, fname, omw, panel_opts, frame_start=0, style="npz", **kwargs
    ):
        self.fname = fname
        self.omwedges = omw
        self.panel_opts = panel_opts
        self.frame_start = frame_start
        self.style = style

    @property
    def oplist(self):
        return self.panel_opts

    @property
    def framelist(self):
        return range(self.frame_start, self.nframes + self.frame_start)

    @property
    def nframes(self):
        return self.omwedges.nframes

    @property
    def omegas(self):
        return self.omwedges.omegas

    def processed(self):
        kw = {}
        if self.use_frame_list:
            kw = dict(frame_list=self.framelist)
        return ProcessedImageSeries(self.raw, self.oplist, **kw)

    def _attach_metadata(self, metadata):
        metadata["omega"] = self.omegas

    def save_processed(self, name, threshold, output_dir=None):
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

    def __init__(self, fname, omw, panel_opts, frame_start=0, style="npz"):
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
        fname,
        omw,
        panel_opts,
        panel_id,
        frame_start=0,
        style="npz",
        raw_format="hdf5",
        dark=None,
    ):
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

    def _attach_metadata(self, metadata):
        super()._attach_metadata(metadata)
        metadata["panel_id"] = self.panel_id

    @property
    def panel_id(self):
        return self._panel_id

    @property
    def dark(self, nframes=100):
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


def preprocess(args):
    omw = imageseries.omega.OmegaWedges(args.num_frames)
    omw.addwedge(args.ome_start, args.ome_end, args.num_frames)

    if type(args) == Eiger_Arguments:
        pp = PP_Eiger(
            fname=args.file_name,
            omw=omw,
            panel_opts=args.panel_opts,
            frame_start=args.start_frame,
            style=args.style,
        )
        pp.save_processed(args.output, args.threshold)
    elif type(args) == Dexelas_Arguments:
        for file_name in args.file_names:
            for key in args.panel_keys:
                if key.lower() in file_name:
                    pp = PP_Dexela(
                        fname=args.file_name,
                        omw=omw,
                        panel_opts=args.panel_opts[key],
                        panel_id=key,
                        frame_start=args.start_frame,
                        style=args.style,
                    )

                    output_name = (
                        args.samp_name
                        + "_"
                        + str(args.scan_number)
                        + "_"
                        + file_name.split("/")[-1].split(".")[0]
                    )
                    pp.save_processed(output_name, args.threshold)
    else:
        raise AttributeError(f"Unknown argument type: {type(args)}")
