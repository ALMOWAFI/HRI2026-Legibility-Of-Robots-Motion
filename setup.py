from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup(**generate_distutils_setup(
    packages=["hri_legible_motion"],
    package_dir={"": "src"},
))
