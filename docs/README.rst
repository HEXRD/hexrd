HEXRD Documentation
-------------------

This directory contains the majority of the documentation for HEXRD.

Requirements
------------
The following tools are needed to build the documentation:

sphinx

With `Anaconda <https://store.continuum.io/cshop/anaconda/>`_-based Python
environments, you should be able to run::

    conda install sphinx

The documentation is built using ``make``:

``make html`` - build the API and narrative documentation web pages. This
is the the default ``make`` target; running ``make`` is equivalent to
``make html``. 

``make html_noapi`` - same as ``make html``, but without auto-generating API
docs, the most time-consuming portion  of the build process.

``make pdf`` compiles the documentation into a redistributable pdf file.

``make help`` provides information on all supported make targets.
