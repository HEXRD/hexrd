.. _examples:

Example of Imageseries Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is an example of how the imageseries is used. One thing we commonly do is
to process the raw image files by adding flips and subtract off the background.
Then we save it as a new imageseries. This example saves it into a HDF5 file,
but it is more common to use the frame-cache (sparse matrices), which is way
smaller.

.. code-block:: python

    import numpy as np

    from hexrd import imageseries
    from hexrd.imageseries.process import ProcessedImageSeries as ProcessedIS
    from hexrd.imageseries.omega import OmegaWedges

    # Inputs
    darkfile = 'dark-50pct-100f'
    h5file = 'example.h5'
    fname = 'example-images.yml'
    mypath = '/example'

    # Raw image series: directly from imagefiles
    imgs = imageseries.open(fname, 'image-files')
    print(
       "number of frames: ", len(imgs),
       "\ndtype: ", imgs.dtype,
       "\nshape: ", imgs.shape
    )

    # Make dark image from first 100 frames
    pct = 50
    nf_to_use = 100
    dark = imageseries.stats.percentile(imgs, pct, nf_to_use)
    np.save(darkfile, dark)


    # Now, apply the processing options
    ops = [('dark', dark), ('flip', 'h')]
    pimgs = ProcessedIS(imgs, ops)


    # Save the processed imageseries in HDF5 format
    print(f"writing HDF5 file (may take a while): {h5file}")
    imageseries.write(pimgs, h5file, 'hdf5', path=mypath)

Here is the YAML file for the raw image-series.

.. code-block:: yaml

    image-files:
      directory: GE
      files: "ti7_*.ge2"
    options:
      empty-frames: 0
      max-frames: 2
    meta:
      omega: "! load-numpy-array example-omegas.npy"
