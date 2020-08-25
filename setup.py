# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup, find_packages, Extension
from pathlib import Path
import numpy
np_include_dir = os.path.join(numpy.get_include(), 'numpy')

install_reqs = [
    'fabio@git+https://github.com/joelvbernier/fabio.git@master',  # until patch is pushed to PyPI
    'fast-histogram',
    'h5py',
<<<<<<< HEAD
    'psutil',
    'scipy',
    'pycifrw',
=======
>>>>>>> 4d488e4a4d9e73e20570913f26780a926835c0a4
    'numba',
    'psutil',
    'pyyaml',
    'scipy',
    'scikit-learn'
]

# This a hack to get around the fact that scikit-image on conda-forge doesn't install
# dist info so setuptools can't find it, even though its there, which results in
# pkg_resources.DistributionNotFound, even though the package is available. So we
# only added it if we aren't building with conda.
if os.environ.get('CONDA_BUILD') != '1':
    install_reqs.append('scikit-image')


# extension for convolution from astropy
def get_convolution_extensions():
    c_convolve_pkgdir = Path(__file__).parent / 'hexrd' / 'convolution'

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

def get_extension_modules():
    # for SgLite
    srclist = [
        'sgglobal.c', 'sgcb.c', 'sgcharmx.c', 'sgfile.c', 'sggen.c', 'sghall.c',
        'sghkl.c', 'sgltr.c', 'sgmath.c', 'sgmetric.c', 'sgnorm.c', 'sgprop.c',
        'sgss.c', 'sgstr.c', 'sgsymbols.c', 'sgtidy.c', 'sgtype.c', 'sgutil.c',
        'runtests.c', 'sglitemodule.c'
        ]
    srclist = [os.path.join('hexrd/sglite', f) for f in srclist]
    sglite_mod = Extension(
        'hexrd.extensions.sglite',
        sources=srclist,
        define_macros=[('PythonTypes', 1)]
        )

    # for transforms
    srclist = ['transforms_CAPI.c', 'transforms_CFUNC.c']
    srclist = [os.path.join('hexrd/transforms', f) for f in srclist]
    transforms_mod = Extension(
        'hexrd.extensions._transforms_CAPI',
        sources=srclist,
        include_dirs=[np_include_dir]
        )

    return [sglite_mod, transforms_mod] + get_convolution_extensions()

ext_modules = get_extension_modules()

# use entry_points, not scripts:
entry_points = {
    'console_scripts': ["hexrd = hexrd.cli.main:main"]
    }

setup(
    name='hexrd',
    setup_requires=['setuptools-scm'],
    use_scm_version=True,
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
        'Programming Language :: Python :: 3.6'
    ],
    entry_points = entry_points,
    ext_modules=ext_modules,
    packages=find_packages(),
    package_data={'':['Anomalous.h5']},
    include_package_data=True,
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