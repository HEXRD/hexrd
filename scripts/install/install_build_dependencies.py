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


def download_xsimd(path):
    url = 'https://github.com/xtensor-stack/xsimd/archive/refs/tags/7.6.0.tar.gz'  # noqa
    md5sum = '6e52c2af8b3cb4688993a0e70825f4e8'
    out_dir_name = 'xsimd-7.6.0'

    with tempfile.TemporaryDirectory() as temp_dir:
        download_and_extract_tgz(url, md5sum, temp_dir)

        target_path = Path(path) / 'xsimd'
        if target_path.exists():
            shutil.rmtree(target_path)

        os.makedirs(path, exist_ok=True)
        shutil.move(str(Path(temp_dir) / out_dir_name / 'include/xsimd'),
                    str(Path(path) / 'xsimd/xsimd'))

    return str(target_path)


def download_eigen(path):
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
    url = 'https://github.com/pybind/pybind11/archive/refs/tags/v2.11.0.tar.gz'
    md5sum = '90c4946e87c64d8d8fc0ae4edf35d780'
    out_dir_name = 'pybind11-2.11.0'

    download_and_extract_tgz(url, md5sum, path)

    target_path = Path(path) / 'pybind11'
    if target_path.exists():
        shutil.rmtree(target_path)

    shutil.move(str(Path(path) / out_dir_name), str(Path(path) / 'pybind11'))

    return str(target_path)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        sys.exit('<script> <destination')

    destination = sys.argv[1]

    # Call the functions to download and place the libraries
    download_eigen(destination)
    download_xsimd(destination)
