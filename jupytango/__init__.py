# ===========================================================================
#  This file is part of the Flyscan Ecosystem
#
#  Copyright 2017-EOT Synchrotron SOLEIL, St.Aubin, France
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

"""jupytango"""

import itango
from jupytango import magics as jptm


def load_ipython_extension(app):
    try:
        itango.load_ipython_extension(app)
    except Exception as e:
        print(e)
    try:
        jptm.load_ipython_extension(app)
    except Exception as e:
        print(e)


def unload_ipython_extension(app):
  try:
      itango.unload_ipython_extension(app)
  except Exception as e:
      print(e)
  try:
      jptm.unload_ipython_extension(app)
  except Exception as e:
      print(e)


