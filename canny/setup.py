from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("canny", ["filter.pyx", "conv2d.pyx"]),
    Extension("canny_tst", ["main.py"]),
]
setup(
    name="cython canny",
    ext_modules=cythonize(extensions),
)
