from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'assign_PE_max_output_filter_cpp',
        ['assign_PE_cpp.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        # MSVC uses /O2 for optimization, not -O3
        extra_compile_args=['/O2', '/EHsc'], 
    ),
]

setup(
    name='assign_PE_max_output_filter_cpp',
    ext_modules=ext_modules,
)