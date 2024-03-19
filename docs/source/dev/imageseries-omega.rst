.. _omega:

Omega Module
^^^^^^^^^^^^^^^^^^^^
This module has two classes. The ``OmegaImageSeries`` is used for the analysis.
It is basically am imageseries with omega metadata (and n x 2 numpy array of
rotation angle ranges for each frame) and methods for associating the frames
with the omega angles. The ``OmegaWedges`` is used for setting up the omega
metadata. During a scan, the specimen is rotated through an angular range
while frames are being written. We call a continous angular range a wedge.
We commonly use a single wedge of 360 degrees or 180 degrees, but sometimes
there are multiple wedges, e.g. if there is some fixture in the way.

Examples
+++++++++++++++
Start with a couple examples. In the first example, we have 3 files, each
with 240 frames, going through 180 degrees in quarter degree increments. The
omega array is saved into a numpy file.

.. code-block:: python

    nf = 3*240
    omw = OmegaWedges(nf)
    omw.addwedge(0, nf*0.25, nf)
    omw.save_omegas('ti7-145-147-omegas')

In the second example, there are four wedges, each with 240 frames. The wedges
go through angular ranges of 0 to 60, 120 to 180, 180 to 240, and 300 to 360.
The omega array is then added to the imageseries metadata.

.. code-block:: python

    nsteps = 240
    totalframes = 4*nsteps
    omwedges = omega.OmegaWedges(totalframes)
    omwedges.addwedge(0, 60, nsteps)
    omwedges.addwedge(120, 180, nsteps)
    omwedges.addwedge(180, 240, nsteps)
    omwedges.addwedge(300, 360, nsteps)

    ims.metadata['omega'] = omwedges.omegas


OmegaWedges Class
++++++++++++++++++++++++++++++

``__init__(self, nframes)``
  instatiate with the number of frames.

``omegas``
  (property) n x 2 array of omega values

``nwedges``
  (property) number of wedges

``addwedge_(self, ostart, ostop, nsteps, loc=None)``:
  add a new wedge to wedge list; take starting omega, end omega, and number of
  steps; the keyword argument is where to insert the wedge (at the end by
  default)

``delwedge_(self, i)``:
  delete wedge i

``wframes``
  (property) number of steps in each wedge

``save_omegas(self, fname)``
  save the omega array to a numpy file (for use with YAML-based formats)
