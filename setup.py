from setuptools import setup, Extension
from torch.utils import cpp_extension
import pybind11

setup(
    name='gpt2_cpp',
    ext_modules=[
        cpp_extension.CppExtension(
            'gpt2_cpp',
            ['python_cpp/gpt2_cpp.cpp'],
            include_dirs=['python_cpp', pybind11.get_include()],
            extra_compile_args=['-g', '-std=c++17']
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=['torch', 'pybind11']
)