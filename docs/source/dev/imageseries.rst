imageseries package
===============
The *imageseries* package provides a standard API for accessing image-based data sets.  The primary tool in the package is the ImageSeries class. It's interface is analagous to a list of images with  associated image metadata. The number of images is given by the len() function. Properties are defined for image shape (shape), data type (dtype) and metadata (metadata). Individual images are accessed by standard subscripting (e.g. image[i]).

The package contains interfaces for loading (load) and saving (save) imageseries. Images can be loaded in three formats: 'array', 'hdf5' and 'frame-cache'. The 'array' format takes the images from a 3D numpy array. With 'hdf5', images are stored in hdf5 file and accessed on demand. The 'frame-cache' is a list of sparse matrices, useful for thresholded images. An imageseries can be saved in 'hdf5' or 'frame-cache' format.

The imageseries package also contains a  module for modifying the images (process). The process module provides the ProcessedImageSeries class, which takes a given imageseries and produces a new one by modifying the images. It has certain built-in image operations including transposition, flipping, dark subtraction and restriction to a subset.


Metadata
----------------

The metadata property is generically a dictionary. The actual contents depends on the application. For common hexrd applications in which the specimen is rotated while being exposed to x-rays, the metadata has an 'omega' key with an associated value being an nx2 numpy array where n is the number of frames and the two associated values give the omega (rotation) range for that frame.

Reader Refactor
-------------
While the imageseries package is in itself indpendent of hexrd, it was used as the basis of a refactoring of the reader classes originally found in the detector module. The main reader class was ReadGE. In the refactored code, the reader classes are now in their own module, image_io, but imported into detector to preserve the interface. The image_io module contains a generic OmegaImageSeries class for working with imageseries having omega metadata. The refactored ReadGE class simply uses the OmegaImageSeries class to provide the same methods as the old class. New code should use the OmegaImageSeries (or the standard ImageSeries) class directly.
