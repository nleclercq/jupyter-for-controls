#!/usr/bin/env python

# ===========================================================================
#  This file is part of the Flyscan Ecosystem
#
#  Copyright 2014-EOT Synchrotron SOLEIL, St.Aubin, France
#
#  This is free software: you can redistribute it and/or modify it under the
#  terms of the GNU Lesser General Public License as published by the Free
#  Software Foundation, either version 3 of the License, or (at your option)
#  any later version.
#
#  This is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
#  more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with This.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================

import ast
from distutils.core import setup

setup(
    name="JupyTango",
    packages=[
        'jupytango',
        'jupytango.jupyter',
        'jupytango.tools',
        'jupytango.tools.tango',
	'jupytango.tools.task'
    ],
    version='0.0.1',
    author='Nicolas Leclercq',
    author_email='nicolas.leclercq@synchrotron-soleil.fr',
    maintainer='Nicolas Leclercq',
    maintainer_email='nicolas.leclercq@synchrotron-soleil.fr',
    description='Jupyter Tools for Tango',
    long_description='Jupyter Tools for Tango',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License (LGPL)',
        'Environment :: Console',
        'Operating System :: POSIX',
        'Framework :: IPython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
    ],
    platforms=[
        'Linux',
        'Windows',
    ],
    license="LGPL",
    requires=[
	'jupyter',
	'bokeh',
        'PyTango',
        'IPython',
        'prettytable',
        'h5py',
        'numpy',
        'six'
    ],
)
