# -*- coding: utf-8 -*-
import distutils.ccompiler
import os
from pathlib import Path
import platform
from setuptools import setup, find_packages, Extension
import subprocess
import sys

import numpy
np_include_dir = numpy.get_include()

install_reqs = [
    'appdirs',
    'chemparse',
    'fabio>=0.11',
    'fast-histogram',
    'h5py<3.12',  # Currently, h5py 3.12 on Windows fails to import.
                  # We can remove this version pin when that is fixed.
    'lmfit',
    'matplotlib',
    'numba',
    'numpy<1.27',  # noqa NOTE: bump this to support the latest version numba supports
    'psutil',
    'pycifrw',
    'pyyaml',
    'scikit-image',
    'scikit-learn',
    'scipy',
    'tqdm',
    'xxhash',
]

if platform.machine() == 'x86_64':
    # tbb is an optional dependency, but it is nice to have
    # Add it to the requirements automatically for intel computers.
    install_reqs.append('tbb')

# Determine which compiler is being used to build the C/C++ modules
compiler_type = distutils.ccompiler.get_default_compiler()
if compiler_type in ("unix", "mingw32"):
    compiler_flags = ['-O3', '-ftree-vectorize', '-Wall', '-funroll-loops']
    if not sys.platform.startswith('win'):
        compiler_flags.append('-fPIC')
elif compiler_type == "msvc":
    compiler_flags = ['/Ox', '/GL', '/std:c++14']
else:
    compiler_flags = []

# Extension for convolution from astropy
def get_convolution_extensions():
    c_convolve_pkgdir = Path('hexrd') / 'convolution'

    src_files = [str(c_convolve_pkgdir / 'src/convolve.c')]

    extra_compile_args = ['-UNDEBUG'] + compiler_flags
    # Add '-Rpass-missed=.*' to ``extra_compile_args`` when compiling with
    # clang to report missed optimizations
    _convolve_ext = Extension(
        name='hexrd.convolution._convolve',
        sources=src_files,
        extra_compile_args=extra_compile_args,
        include_dirs=[numpy.get_include()],
        language='c'
    )

    return [_convolve_ext]

def get_include_path(library_name):
    env_var_hint = os.getenv(f"{library_name.upper()}_INCLUDE_DIR")
    if env_var_hint is not None and os.path.exists(env_var_hint):
        return env_var_hint

    conda_include_dir = os.getenv('CONDA_PREFIX')
    if conda_include_dir:
        full_path = Path(conda_include_dir) / 'include' / library_name
        if full_path.exists():
            return full_path

    build_include_dir = Path(__file__).parent / 'build/include'
    full_path = build_include_dir / library_name
    if full_path.exists():
        return full_path

    # If the path doesn't exist, then install it there
    scripts_path = Path(__file__).parent / 'scripts'
    install_script = scripts_path / 'install/install_build_dependencies.py'

    args = [
        sys.executable,
        install_script,
        library_name,
        build_include_dir,
    ]

    subprocess.run(args, check=True)

    # It should exist now
    return full_path

def get_pybind11_include_path():
    # If we can import pybind11, use that include path
    try:
        import pybind11
    except ImportError:
        pass
    else:
        return pybind11.get_include()

    # Otherwise, we will download the source and include that
    return get_include_path('pybind11')

def get_cpp_extensions():
    cpp_transform_pkgdir = Path('hexrd') / 'transforms/cpp_sublibrary'
    src_files = [str(cpp_transform_pkgdir / 'src/inverse_distortion.cpp')]

    # Define include directories
    include_dirs = [
        get_include_path('xsimd'),
        get_include_path('eigen3'),
        get_pybind11_include_path(),
        numpy.get_include(),
    ]

    inverse_distortion_ext = Extension(
        name='hexrd.extensions.inverse_distortion',
        sources=src_files,
        extra_compile_args=compiler_flags+['-std=c++14'],
        include_dirs=include_dirs,
        language='c++',
    )

    return [inverse_distortion_ext]

def get_old_xfcapi_extension_modules():
    # for transforms
    srclist = ['transforms_CAPI.c', 'transforms_CFUNC.c']
    srclist = [os.path.join('hexrd/transforms', f) for f in srclist]
    transforms_mod = Extension(
        'hexrd.extensions._transforms_CAPI',
        sources=srclist,
        include_dirs=[np_include_dir],
        extra_compile_args=compiler_flags,
    )

    return [transforms_mod]

def get_new_xfcapi_extension_modules():
    transforms_mod = Extension(
        'hexrd.extensions._new_transforms_capi',
        sources=['hexrd/transforms/new_capi/module.c'],
        include_dirs=[np_include_dir],
        extra_compile_args=compiler_flags,
    )

    return [transforms_mod]

def get_extension_modules():
    # Flatten the lists
    return [item for sublist in (
        get_old_xfcapi_extension_modules(),
        get_new_xfcapi_extension_modules(),
        get_convolution_extensions(),
        get_cpp_extensions(),
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
    description='hexrd X-ray diffraction data analysis tool',
    long_description=open('README.md').read(),
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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points=entry_points,
    ext_modules=ext_modules,
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['Anomalous.h5']},
    python_requires='>=3.9',
    install_requires=install_reqs
)
