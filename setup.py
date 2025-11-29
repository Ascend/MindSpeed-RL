# coding=utf-8

import os
import sys
from pathlib import Path

import setuptools

__description = 'MindSpeed-RL for LLMs of Ascend'
__version = '2.1.0'
__author = 'Ascend'
__long_description = 'MindSpeed-RL for LLMs of Ascend'
__keywords = 'Ascend, langauge, deep learning, NLP'
__package_name = 'mindspeed_rl'
__contact_names = 'Ascend'

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by mindspeed_rl.")

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ''

###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

cmd_class = {}
exts = []


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mindspeed_rl')

setuptools.setup(
    name=__package_name,
    version=__version,
    description=__description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    author=__contact_names,
    maintainer=__contact_names,
    # The licence under which the project is released
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        # Indicate what your project relates to
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Supported python versions
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=Path("requirements.txt").read_text().splitlines(),
    packages=setuptools.find_packages(),
    # Add in any packaged data.
    include_package_data=True,
    install_package_data=True,
    exclude_package_data={'': ['**/*.md']},
    package_data={'': package_files(src_path)},
    bug_data={'mindspeed_rl': ['**/*.h', '**/*.cpp', '*/*.sh', '**/*.patch']},
    zip_safe=False,
    # PyPI package information.
    keywords=__keywords,
    cmdclass={},
    entry_points={
        "console_scripts": [
            "mindspeed_rl = mindspeed_rl.run.run:main",
        ]
    },
    ext_modules=exts
)
