#!/usr/bin/env python3

import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request


def get_file_md5(filepath):
    with open(filepath, 'rb') as rf:
        return hashlib.md5(rf.read()).hexdigest()


def download_and_extract_tgz(url, md5sum, path):
    temp_dir = tempfile.gettempdir()
    temp_file = Path(temp_dir) / '_hexrd_temp_file'
    if temp_file.exists():
        temp_file.unlink()

    urllib.request.urlretrieve(url, temp_file)

    file_md5sum = get_file_md5(temp_file)
    if file_md5sum != md5sum:
        raise Exception(
            f'md5sum "{file_md5sum}" of file from "{url}" '
            f'does not match expected md5sum "{md5sum}"'
        )

    os.makedirs(path, exist_ok=True)
    with tarfile.open(temp_file, 'r:gz') as tarball:
        tarball.extractall(path)

    temp_file.unlink()


def download_xtensor(path):
    url = 'https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.27.1.tar.gz'  # noqa
    md5sum = 'ef143422b31b94dd0f5b95b69388cd48'
    out_dir_name = 'xtensor-0.27.1'

    with tempfile.TemporaryDirectory() as temp_dir:
        download_and_extract_tgz(url, md5sum, temp_dir)

        target_path = Path(path) / 'xtensor'
        if target_path.exists():
            shutil.rmtree(target_path)

        os.makedirs(path, exist_ok=True)
        shutil.move(
            str(Path(temp_dir) / out_dir_name / 'include/xtensor'),
            str(Path(path) / 'xtensor/xtensor'),
        )

    return str(target_path)


def download_xtensor_python(path):
    url = 'https://github.com/xtensor-stack/xtensor-python/archive/refs/tags/0.29.0.tar.gz'  # noqa
    md5sum = 'd943e73a742241931579444f19846872'
    out_dir_name = 'xtensor-python-0.29.0'

    with tempfile.TemporaryDirectory() as temp_dir:
        download_and_extract_tgz(url, md5sum, temp_dir)

        target_path = Path(path) / 'xtensor-python'
        if target_path.exists():
            shutil.rmtree(target_path)

        os.makedirs(path, exist_ok=True)
        shutil.move(
            str(Path(temp_dir) / out_dir_name / 'include/xtensor-python'),
            str(Path(path) / 'xtensor-python/xtensor-python'),
        )

    return str(target_path)


def download_xtl(path):
    url = 'https://github.com/xtensor-stack/xtl/archive/refs/tags/0.8.2.tar.gz'  # noqa
    md5sum = 'a5a5b86d5695baff39fc9943481b2b5b'
    out_dir_name = 'xtl-0.8.2'

    with tempfile.TemporaryDirectory() as temp_dir:
        download_and_extract_tgz(url, md5sum, temp_dir)

        target_path = Path(path) / 'xtl'
        if target_path.exists():
            shutil.rmtree(target_path)

        os.makedirs(path, exist_ok=True)
        shutil.move(
            str(Path(temp_dir) / out_dir_name / 'include/xtl'),
            str(Path(path) / 'xtl/xtl'),
        )

    return str(target_path)


def download_xsimd(path):
    url = 'https://github.com/xtensor-stack/xsimd/archive/refs/tags/13.2.0.tar.gz'  # noqa
    md5sum = 'f451a1c57d2a4fdc0ba663be438dced4'
    out_dir_name = 'xsimd-13.2.0'

    with tempfile.TemporaryDirectory() as temp_dir:
        download_and_extract_tgz(url, md5sum, temp_dir)

        target_path = Path(path) / 'xsimd'
        if target_path.exists():
            shutil.rmtree(target_path)

        os.makedirs(path, exist_ok=True)
        shutil.move(
            str(Path(temp_dir) / out_dir_name / 'include/xsimd'),
            str(Path(path) / 'xsimd/xsimd'),
        )

    return str(target_path)


def download_eigen3(path):
    url = 'https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz'  # noqa
    md5sum = '77f2c00ce620767e4df74958c6d5c822'
    out_dir_name = 'eigen-3.4.1'

    download_and_extract_tgz(url, md5sum, path)

    target_path = Path(path) / 'eigen3'
    if target_path.exists():
        shutil.rmtree(target_path)

    shutil.move(str(Path(path) / out_dir_name), str(Path(path) / 'eigen3'))

    return str(target_path)


def download_pybind11(path):
    url = 'https://github.com/pybind/pybind11/archive/refs/tags/v3.0.4.tar.gz'
    md5sum = '933fa1b6b1fe34c9945ecb3fe67f5c4b'
    out_dir_name = 'pybind11-3.0.4'

    with tempfile.TemporaryDirectory() as temp_dir:
        download_and_extract_tgz(url, md5sum, temp_dir)

        target_path = Path(path) / 'pybind11'
        if target_path.exists():
            shutil.rmtree(target_path)

        os.makedirs(path, exist_ok=True)
        shutil.move(
            str(Path(temp_dir) / out_dir_name / 'include/pybind11'),
            str(Path(path) / 'pybind11/pybind11'),
        )

    return str(target_path)


INSTALL_FUNCTIONS = {
    'eigen3': download_eigen3,
    'pybind11': download_pybind11,
    'xsimd': download_xsimd,
    'xtensor': download_xtensor,
    'xtensor-python': download_xtensor_python,
    'xtl': download_xtl,
}


def install(library, destination):
    if library not in INSTALL_FUNCTIONS:
        raise NotImplementedError(library)

    return INSTALL_FUNCTIONS[library](destination)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        sys.exit('<script> <library> <destination>')

    library = sys.argv[1]
    destination = sys.argv[2]

    install(library, destination)
