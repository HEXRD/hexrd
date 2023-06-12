# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup, find_packages, Extension
from pathlib import Path
import numpy
np_include_dir = numpy.get_include()

install_reqs = [
    'fabio>=0.11',
    'fast-histogram',
    'h5py',
    'lmfit',
    'numba',
    'numpy<1.25',  # NOTE: bump this to support the latest version numba supports
    'psutil',
    'pycifrw',
    'pyyaml',
    'scikit-learn',
    'scipy',
    'tqdm',
    'xxhash',
]

# This a hack to get around the fact that scikit-image on conda-forge doesn't install
# dist info so setuptools can't find it, even though its there, which results in
# pkg_resources.DistributionNotFound, even though the package is available. So we
# only added it if we aren't building with conda.
# appdirs has the same issue.
if os.environ.get('CONDA_BUILD') != '1':
    install_reqs.append('scikit-image')
    install_reqs.append('appdirs')


# extension for convolution from astropy
def get_convolution_extensions():
    c_convolve_pkgdir = Path('hexrd') / 'convolution'

    src_files = [str(c_convolve_pkgdir / 'src/convolve.c')]

    extra_compile_args=['-UNDEBUG']
    if not sys.platform.startswith('win'):
        extra_compile_args.append('-fPIC')
    # Add '-Rpass-missed=.*' to ``extra_compile_args`` when compiling with clang
    # to report missed optimizations
    _convolve_ext = Extension(name='hexrd.convolution._convolve', sources=src_files,
                              extra_compile_args=extra_compile_args,
                              include_dirs=[numpy.get_include()],
                              language='c')

    return [_convolve_ext]


def get_old_xfcapi_extension_modules():
    # for transforms
    srclist = ['transforms_CAPI.c', 'transforms_CFUNC.c']
    srclist = [os.path.join('hexrd/transforms', f) for f in srclist]
    transforms_mod = Extension(
        'hexrd.extensions._transforms_CAPI',
        sources=srclist,
        include_dirs=[np_include_dir],
    )

    return [transforms_mod]


def get_new_xfcapi_extension_modules():
    transforms_mod = Extension(
        'hexrd.extensions._new_transforms_capi',
        sources=['hexrd/transforms/new_capi/module.c'],
        include_dirs=[np_include_dir],
    )

    return [transforms_mod]


def get_extension_modules():
    # Flatten the lists
    return [item for sublist in (
        get_old_xfcapi_extension_modules(),
        get_new_xfcapi_extension_modules(),
        get_convolution_extensions(),
    ) for item in sublist]


ext_modules = get_extension_modules()

# use entry_points, not scripts:
entry_points = {
    'console_scripts': ["hexrd = hexrd.cli.main:main"]
    }

setup(
    name='hexrd',
    use_scm_version=True,
    setup_requires=['setuptools-scm'],
    description = 'hexrd X-ray diffraction data analysis tool',
    long_description = open('README.md').read(),
    author='The hexrd Development Team',
    author_email='joelvbernier@me.com',
    url='https://github.com/HEXRD/hexrd',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Web Environment',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'

    ],
    entry_points = entry_points,
    ext_modules=ext_modules,
    packages=find_packages(),
    include_package_data=True,
    package_data={'':['Anomalous.h5']},
    python_requires='>=3.8',
    install_requires=install_reqs
)

# ext_modules = get_extension_modulesf()
# setupF(
#     name='hexrd',
#     url='https://github.com/cryos/hexrd',
#     license='BSD',
#     ext_modules=ext_modules,
#     packages=find_packages(),
# )
