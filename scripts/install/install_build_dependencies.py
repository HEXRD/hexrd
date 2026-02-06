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
    url = 'https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.24.7.tar.gz'  # noqa
    md5sum = 'e21a14d679db71e92a703bccd3c5866a'
    out_dir_name = 'xtensor-0.24.7'

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
    url = 'https://github.com/xtensor-stack/xtensor-python/archive/refs/tags/0.26.1.tar.gz'  # noqa
    md5sum = '5d05edf71ac948dc620968229320c791'
    out_dir_name = 'xtensor-python-0.26.1'

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
    url = 'https://github.com/xtensor-stack/xtl/archive/refs/tags/0.7.7.tar.gz'  # noqa
    md5sum = '6df56ae8bc30471f6773b3f18642c8ab'
    out_dir_name = 'xtl-0.7.7'

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
    url = 'https://github.com/xtensor-stack/xsimd/archive/refs/tags/12.1.1.tar.gz'  # noqa
    md5sum = 'e8887de343bd6036bdfa1f4a4752dc64'
    out_dir_name = 'xsimd-12.1.1'

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
    url = 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz'  # noqa
    md5sum = '4c527a9171d71a72a9d4186e65bea559'
    out_dir_name = 'eigen-3.4.0'

    download_and_extract_tgz(url, md5sum, path)

    target_path = Path(path) / 'eigen3'
    if target_path.exists():
        shutil.rmtree(target_path)

    shutil.move(str(Path(path) / out_dir_name), str(Path(path) / 'eigen3'))

    return str(target_path)


def download_pybind11(path):
    url = 'https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz'
    md5sum = '81399a5277559163b3ee912b41de1b76'
    out_dir_name = 'pybind11-3.0.1'

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

    logger.debug(f'Installing "{library}" to "{destination}"')
    install(library, destination)
