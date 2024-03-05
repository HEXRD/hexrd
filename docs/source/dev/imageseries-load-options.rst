.. _keyword-options:

Keyword Options for imageseries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each type of imageseries has its own keyword options for loading and saving.

Image Files
+++++++++++++

The format name is ``image-files``.

This is usually written by hand. It is a YAML-based format, so the options are
in the file, not passed as keyword arguments. The file defines a
list of image files. It could be a list of single images or a list of
multi-imagefiles.

YAML keywords are:

``image-files``
    dictionary defining the image files

    - ``directory``: the directory containing the images
    - ``files``:  the list of images; it is a space separated list of file
      names or glob patterns

``empty-frames``
    (optional) number of frames to skip at the beginning of
    each multiframe file; this is a commonly used option

``max-total-frames``
    (optional) the maximum number of frames in the imageseries; this option
     might be used for testing the data on a small number of frames

``max-file-frames``
    (optional) the maximum number of frames to read per file; this would
    be unusual

``metadata``
    (required) it usually contains array data or string, but it can be empty

There is actually no write function for this type of imageseries. It is
usually used to load image data to be sparsed and saved in another (usually
frame-cache) format.



HDF5
++++++++++

The format name is ``hdf5``.

This is used at CHESS (Cornell High Energy Synchrotron Source). Raw data from
the Dexela detectors comes out in HDF5 format. We still will do the dark
subtraction and flipping.

**On Write.**

``path``
    (required) path to directory containing data group (data set is named
    `images`

``shuffle``
    (default=True) HDF5 write option

``gzip``
    (default=1) compression level

``chunk_rows``
    (default=all) sets HDF5 chunk size in terms of number of rows in image

**On Open.**

``path``
    (required) path to directory containing data group (data set is named
    `images`)

Frame Cache
++++++++++++++++++++
The format name is ``frame-cache``.

A better name might be sparse matrix format because the images are stored as
sparse matrices in numpy npz file. There are actually two forms of the
frame-cache. The original is a YAML-based format, which is now deprecated.
The other format is a single .npz file that includes array data and metadata.

**On Write.**

``threshold``
    (required) this is the main option; all data below the threshold is ignored;
    be careful because a too small threshold creates huge files; normally,
    however, we get a massive savings of file size since the images are
    usually over 99% sparse.

``output_yaml``
    (default=False) This is deprecated.

**On Open.**
No options are available on open.
