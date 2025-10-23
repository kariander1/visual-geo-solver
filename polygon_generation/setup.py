#!/usr/bin/env python3

from setuptools import setup, Extension
import pybind11

# Define the extension module
ext_modules = [
    Extension(
        "max_area_polygon_cpp",
        [
            "max_area_polygon_cpp.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=["-std=c++14", "-O3", "-march=native", "-DPYTHON_BINDING"],
    ),
]

setup(
    name="max_area_polygon_cpp",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.6",
)