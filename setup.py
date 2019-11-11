# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages, Extension

import numpy
np_include_dir = os.path.join(numpy.get_include(), 'numpy')

install_reqs = [
    'h5py',
    'scipy',
    'numba'
]

# This a hack to get around the fact that scikit-image on conda-forge doesn't
# install dist info so setuptools can't find it, even though its there, which
# results in pkg_resources.DistributionNotFound, even though the package is
# available. So we only added it if we aren't buildding with conda.
if os.environ.get('CONDA_BUILD') != '1':
    install_reqs.append('scikit-image')

##############################################################################
# Extensions
EXTENSIONS_BASE_PATH = 'hexrd'
NEW_EXTENSIONS_BASE_PATH = 'extensions'
def make_hexrd_extension_name(name):
    return '.'.join(('hexrd.extensions', name))

## sglite
sglite_location = os.path.join(EXTENSIONS_BASE_PATH, 'sglite')

sglite_srcs = [
    'sgglobal.c', 'sgcb.c', 'sgcharmx.c', 'sgfile.c', 'sggen.c', 'sghall.c',
    'sghkl.c', 'sgltr.c', 'sgmath.c', 'sgmetric.c', 'sgnorm.c', 'sgprop.c',
    'sgss.c', 'sgstr.c', 'sgsymbols.c', 'sgtidy.c', 'sgtype.c', 'sgutil.c',
    'runtests.c', 'sglitemodule.c'
]

sglite_extension = Extension(
    make_hexrd_extension_name('sglite'),
    sources=[os.path.join(sglite_location, f) for f in sglite_srcs],
    define_macros=[('PythonTypes', 1)]
)

## legacy transforms module
legacy_transforms_location = os.path.join(EXTENSIONS_BASE_PATH,
                                          'legacy_transforms')
legacy_transforms_srcs = ['transforms_CAPI.c', 'transforms_CFUNC.c']
legacy_transforms_extension = Extension(
    make_hexrd_extension_name('_transforms_CAPI'),
    sources=[os.path.join(legacy_transforms_location, f)
               for f in legacy_transforms_srcs],
    include_dirs=[np_include_dir]
)


## transforms module
transforms_module_location = os.path.join(NEW_EXTENSIONS_BASE_PATH,
                                          '_transforms_CAPI')
transforms_module_srcs = ['transforms_CAPI.c', 'transforms_CFUNC.c']
transforms_module_extension = Extension(
    make_hexrd_extension_name('transforms_CAPI'),
    sources=[os.path.join(transforms_module_location, f)
             for f in transforms_module_srcs],
    include_dirs=[numpy.get_include()]
)

## transforms module
new_transforms_module_location = os.path.join(NEW_EXTENSIONS_BASE_PATH,
                                          '_transforms_CAPI_new')
new_transforms_module_srcs = ['transforms_CAPI.c', 'transforms_CFUNC.c']
new_transforms_module_extension = Extension(
    make_hexrd_extension_name('transforms_CAPI_new'),
    sources=[os.path.join(new_transforms_module_location, 'module.c')],
    include_dirs=[numpy.get_include()]
)

## list of modules to include
ext_modules = [
    sglite_extension,
    legacy_transforms_extension,
    transforms_module_extension,
    new_transforms_module_extension
]

##############################################################################
# Module configuration
setup(
    name='hexrd',
    use_scm_version=True,
    setup_requires=['setuptools-scm'],
    description='',
    long_description='',
    author='Kitware, Inc.',
    author_email='kitware@kitware.com',
    url='https://github.com/cryos/hexrd',
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
    ext_modules=ext_modules,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=install_reqs
)
