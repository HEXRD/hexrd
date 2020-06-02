# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages, Extension
import versioneer

import numpy
np_include_dir = os.path.join(numpy.get_include(), 'numpy')

install_reqs = [
    'fabio@git+https://github.com/joelvbernier/fabio.git@master',
    'h5py',
    'psutil',
    'scipy',
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

    return [sglite_mod, transforms_mod]

ext_modules = get_extension_modules()

# use entry_points, not scripts:
entry_points = {
    'console_scripts': ["hexrd = hexrd.cli.main:main"]
    }

setup(
    name='hexrd',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
    python_requires='>=3.8',
    install_requires=install_reqs
)
