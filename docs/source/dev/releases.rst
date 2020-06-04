Releases
========

Steps for creating an official release, for example version 1.0.0.


Preliminary Checks
------------------

First, check that your working directory and environment is clean::

  git checkout head
  python setup.py develop --uninstall
  rm -rf build
  git clean -n

Next, test the  conda package (see below), which will only succeed if the
project can be built and installed. Building the conda package also runs
the test suite.


Update Version Information
--------------------------

First, tag the commit that corresponds to version 1.0.0::

  git tag v1.0.0 [head]
  git push --tags

The `head`, or `commit hash` is not necessary if you are tagging the most
recent commit in your active branch as v1.0.0. The hexrd library will report
its version based on the git tag.


Create Conda Packages
---------------------

In the `hexrd/conda.recipe` project directory, temporarily modify `git_tag`
in `meta.yaml` to point to the version you want to build (don't commit this
change to git)::

  #git_tag: master
  git_tag: v1.0.0

and then run::

  conda build conda.recipe

The resulting conda package can be uploaded to binstar::

  binstar upload -u praxes /path/to/hexrd-1.0.0-np19py27_0.tar.bz2

Finally, change `git_tag` back to the default `master`.


Update the Documentation
------------------------

Pushing the branch and tags to github will trigger the documentation to build
automatically. Visit the `hexrd documentation dashboard
<https://readthedocs.org/dashboard/hexrd/>`_, the new version may need to be
activated.
