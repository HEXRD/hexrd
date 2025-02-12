GrainData Class
================

The `GrainData` class loads the grain data output files, either
`grains.out` or `grains.npz`. The intent of the class is to simplify
post-processing by making it easy to load and save the grain data files
and providing commonly used data attributes.


Loading and Saving
-----------------------
You can load the newer `grains.npz` file using the `load` method, or you
can load the `grains.out` file directly using the `from_grains_out`
method. For example:

.. code-block::

   from hexrd.fitgrains import GrainData

   # Load an npz file.
   gd = GrainData.load("graindata.npz")

   # Or load the standard `grains.out` file.
   gd = GrainData.from_grains_out("grains.out")

You can also write a new `.npz` file or a `.out` file.

.. code-block::

   gd.save("new-grains.npz")
   gd.write_grains_out("new-grains.out")



Working With Grain Data
------------------------

After loading data you have access to the following attributes. Notice that
the shape of orientation arrays (`expmap`, `rotation_matrices`, `quaterions`)
is different from core hexrd and more pythonic in that each orientation
occupies a contiguous section of memory.

num_grains
    the number of grains

id
    *array(num_grains)*, grain ID in original output file

completeness
    *array(num_grains)*, completeness value for each grain

chisq
    *array(num_grains)*, goodness of fit values

expmap
    *array(num_grains, 3)*, exponential map parameters for orientations

centroid
    *array(num_grains), 3*, grain centroid values

inv_Vs
    *array(num_grains, 6)*, inverse of symmetric left stretch tensor

ln_Vs
    *array(num_grains, 6)*, matrix logarithm of symmetric left stretch tensor

rotation_matrices
    *array(num_grains, 3)*, rotation matrices for orientations

quaternions
    *array(num_grains, 4)*, unit quaternions for orientations

strain
    *array(num_grains, 6)*, convenience function for `ln_Vs`

You are also able to select a subset of grains based on completeness or
goodness of fit using the `select()` method. It returns a new `GrainData`
instance with all arrays filtered by the selected IDs. Note that the
new `id` attribute shows the original IDs and will no longer be contiguous.
Here is an example of usage:

.. code-block::

   # Select grains at 80% completeness or better.
   gd_new = gd.select(min_completeness=0.8)

   # Select grains with chi-squared at most 0.5.
   gd_new = gd.select(max_chisq=0.5)

   # Or both.
   gd_new = gd.select(min_completeness=0.8, max_chisq=0.5)
