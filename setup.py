# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages, Extension

import numpy
np_include_dir = os.path.join(numpy.get_include(), 'numpy')

install_reqs = [
    'h5py',
    'scipy',
    'pycifrw',
    'f90wrap',
    'numba'
]

# This a hack to get around the fact that scikit-image on conda-forge doesn't install
# dist info so setuptools can't find it, even though its there, which results in
# pkg_resources.DistributionNotFound, even though the package is available. So we
# only added it if we aren't buildding with conda.
if os.environ.get('CONDA_BUILD') != '1':
    install_reqs.append('scikit-image')

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

    '''
        this is the fortran compilation part. pretty janky for now 
    '''
    srclist = ['local.f90', 'io.f90', 'error.f90', 'typedefs.f90',
                'constants.f90', 'math.f90', 'quaternions.f90', 
                'crystal.f90','symmetry.f90','Lambert.f90',
                'rotations.f90', 'others.f90', 'diffraction.f90'  
              ]
    srclist = [os.path.join('hexrd/EMsoft', f) for f in srclist]
    srclist_str = ''
    for f in srclist:
        srclist_str = srclist_str+f+' '


    # this is the kind map for f90wrap
    kmap = 'kind_map'
    kmap = os.path.join('hexrd/EMsoft',kmap)


    cmd = 'f90wrap -k '+ kmap + ' -m EMsoft --package ' + srclist_str
    os.system(cmd)

    cmd = 'gfortran -c '+ srclist_str + ' -fno-unsafe-math-optimizations -frounding-math -fsignaling-nans'
    os.system(cmd)

    cmd = 'ar cr libEMsoft.a *.o'
    os.system(cmd)

    cmd = 'f2py-f90wrap -c -m _EMsoft --build-dir build/ \
            f90wrap_*.f90 --link-lapack_opt -L/Users/saransh1/hexrd3 -lEMsoft'
    os.system(cmd)

    return [sglite_mod, transforms_mod]

ext_modules = get_extension_modules()

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
