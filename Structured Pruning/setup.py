from setuptools import setup, Extension
import pybind11
import sys

# MSVC (Windows) flags
if sys.platform == 'win32':
    extra_compile_args = ['/O2', '/EHsc']
else:
    extra_compile_args = ['-O3', '-std=c++11']

ext_modules = [
    Extension(
        'assign_PE_max_output_filter_cpp', # This is the name you will import in Python
        ['assign_PE_cpp.cpp'],             # This MUST match your filename
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name='assign_PE_max_output_filter_cpp',
    ext_modules=ext_modules,
)