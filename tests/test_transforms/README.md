# Tests for the transforms module #

This directory contains the tests for the transforms module. You can
run them by running `pytest <this directory>` in the command line.


## Physical structure of the transform package unit tests ##

The tests are organized in a test file for each function supported in
the API. The test file is named `test_<function>.py`.

The tests are parametrized so that the same tests are run in all
available implementations. This is achieved by using
`pytest.mark.parametrize`.  Typically, a decorator `all_impls` (short
for "all implementations") is defined as a `pytest.mark.parametrize`
that define two arguments, the function implementation and a submodule
name (for reporting mostly). A list of parametrizations defines the
implementations to test. The decorator will be added to each test
function so that the test is run on all the different implementations.

An example, for an API function named `foobar`, having a numpy and a numba
implementation (but not capi) would be:

```python
from .. import foobar as default_foobar
from ..xf_numpy import foobar as numpy_foobar
from ..xf_numba import foobar as numba_foobar

all_impls = pytest.mark.parametrize('foobar_impl', 'module_name',
	[(numpy_foobar, 'numpy'),
	 (numba_foobar, 'numba'),
	 (default_foobar, 'default')]
)

@allimpls
def test_sample_test(foobar_impl, module_name):
	<Your test goes here>
```

In the example, the test `test_sample_test` will be run thrice. First
using the implementation of `foobar` in `xf_numpy`. Then using the
implementation of `foobar` in `xf_numba`. Finally using the
implementation exported at the module level.

Typically the implementation exported at the module level will be one
of the implementations explicitly tested from one of the
implementation packages. In that case the tests on that implementation
will be run twice, once as the module level export and again for the
submodule implementation.


## Recommented tests ##

When writing the unit tests for transforms functions, testing the
following features should be considered:

* Test basic functionality, preferably using trivial/known setups as
  arguments.
  
* Test default arguments against their equivalent explicit versions.
  In many cases the default argument for array arguments is `None`,
  but semantically a `None` value is logically equivalent to some
  array constant. The explicit check should use that constant, not
  `None`.
  
* Test error cases that should raise exceptions, if they exist. Check
  that the exception happens. If the numerical code has some kind of
  singularity, even if it does not raise exceptions, write a test that
  checks the singularity is handled in a consistent (and preferably,
  documented) way.

* If the function broadcasts, test that it broadcasts properly. But
  check also the arguments that do not broadcast. Some implementations
  may use different code paths for broadcasting and scalar versions.

* If a function broadcasts and it has some kind of singularity, make
  sure that tests exist checking arguments where a singularity happens
  while broadcasting, mixed with cases where the singularity doesn't
  happen.
  
* Functions taking numpy arrays as arguments should be tested with
  arguments that are strided in a non natural way. Implementations
  based on C code are prone to not handling those properly if not
  carefully implemented/wrapped.
  
Testing behavior on handling special values on input, like `NaN`, may
be important in some functions.

Finally, when a bug is found, it is desirable to add a test that
reproduces that bug and leave it so that any regression does not go
unnoticed.
