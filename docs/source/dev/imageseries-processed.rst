.. _processed-ims:

Processed Image Series
^^^^^^^^^^^^^^^^^^^^^^^
This class is intended for image manipulations applied to all images of the
imageseries. It is instantiated with an existing imageseries and a sequence of
operations to be performed on each image. The class has built-in operations for
common transformations and a mechanism for adding new operations. This class is
typically used for preparing raw detector images for analysis, but other uses
are possible. The `rectangle` operation is used in ``stats.percentile`` to
compute percentiles one image section at a time to avoid loading all images at
once.

Instantiation
++++++++++++++++++++
Here is an example:

.. code-block:: python

    oplist = [('dark', darkimg), ('flip', 'v')]
    frames = range(2, len(ims))
    pims = ProcessedImageSeries(ims, oplist, frame_list=frames)

Here, `ims` is an existing imageseries with two empty frames. The operation
list has two operations. First, the a dark (background) image is subtracted.
Then it is *flipped* about a vertical axis. Order is important here; operations
do not always commute. Note that the dark image is usually constructed from
the raw images, so if you flipped first, the dark subtraction would be wrong.
Finally, the only keyword argument available is `frame_list`; it takes a
sequence of frames. In the example, the first two frames are skipped.


Built-In Operations
++++++++++++++++++++
The operation list is a sequence of (key, data) pairs. The key specifies the
operation, and the data is passed with the image to the requested function.
Here are the built-in functions by key.

``dark``
    dark subtraction; it's data is an image

``flip``
    These are simple image reorientations; the data is a short string;
    possible values are:

    - ``y`` or ``v``: flip about y-axis (vertical)
    - ``x`` or ``h``: flip about x-axis (horizontal)
    - ``vh``, ``hv`` or ``r180``: 180 degree rotation
    - ``t`` or ``T``: transpose
    - ``ccw90`` or ``r90``: rotate 90 degrees
    - ``cw90`` or ``r270``: rotate 270

    Note there are possible image shape changes in the last three.

``rectangle``
    restriction to a sub-rectangle of the image; data is a 2x2 array with each
    row giving the range of rows and columns forming the rectangle

Methods
++++++++++
In addition to the usual imageseries methods, there are:

.. code-block:: python

    @classmethod
    def addop(cls, key, func):
        """Add operation to processing options

        *key* - string to use to specify this op
        *func* - function to call for this op: f(img, data)
        """
    @property
    def oplist(self):
        """list of operations to apply"""
