ImageSeries
------------
Originally, hexrd could only process GE images. We developed the imageseries
package to allow for other image formats. The imageseries package provides a
standard interface for images coming from different sources. The idea is that
we could be working with a large number of images that we don't want to keep
in memory. Instead we load the images from a file or generate them dynamically,
but the interface is independent of the source.

See :ref:`examples` for an example of usage.

**Open and Write.**
The imageseries package has two main functions: open and save.

.. code-block:: python

    ims = imageseries.open(file, format, **kwargs)
    imageseries.write(ims, file, format, **kwargs):

The format refers to the source of the images; file and kwargs depend on the
format. Possible formats currently are:

* ``hdf5`` - the images are stored in an HDF5 file and loaded on demand.
  ``file`` is the name of the HDF5 file.
* ``frame-cache`` - the images are stored sparse matrices in a numpy .npz
  file; all of the sparse arrays are loaded on open, and a full (not sparse)
  array is delivered on request for a frame. There are two ways this can be
  done. In one, ``file`` is the name of the npz, and metadata is stored in the
  npz file. In the other, ``file`` is a YAML file that includes the name of
  the npz file as well as the metadata.
* ``image-files`` - the images are stored as one or more regular image files on
  the file system. ``file`` is a YAML file describing listing a sequence of
  image files and metadata.
* ``array`` - images are stored as a 3D numpy array; used for testing.

See also [load options](https://github.com/donald-e-boyce/hexrd/wiki/ims-load-options).

**Processed Imageseries.**
This is a subclass of imageseries. It has a number of built-in operations, such as flipping, dark subtraction, restriction to a sub-rectangle. It was intended to be further subclassed by adding more operations. It is instantiated with an existing imageseries and a list of operations. When a frame is requested, the processed imageseries gets the frame from the original image series and applies the operations in order. It can then be saved as a regular imageseries and loaded as usual.

For more detail, see [processed imageseries](https://github.com/donald-e-boyce/hexrd/wiki/processed-ims).

**Interface.**
The imageseries provides a standard interface, somewhat like a 3D array.

If `ims` is an imageseries instance:
* `len(ims)` is the number of frames
* `ims[j]` returns the j'th frame
* `ims.shape` is the shape of each frame
* `ims.dtype` is the numpy.dtype of each frame
* `ims.metadata` is a dictionary of metadata

**Stats module.**
This module delivers pixel by pixel stats on the imageseries. Functions are:

* `max(ims, nframes=0)` gives a single image that is the max over all frames
* `average(ims, nframes=0)` gives the mean pixel value over all the frames
* `median(ims, nframes=0)` gives median
* `percentile(ims, pct, nframes=0)` gives the percentile over all frames

The median is typically used to generate background images, but percentile could also be used too.

**Omega module.**
For the hexrd work, we usually have a sequence of rotations about the vertical axis. Omega refers to the angle of rotation. The OmegaImageSeries is a subclass that has metadata for the rotation angles.

See [omega](https://github.com/donald-e-boyce/hexrd/wiki/imageseries-omega).

.. include:: imageseries-usage.rst
