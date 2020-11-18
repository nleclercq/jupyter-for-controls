# ===========================================================================
#  This file is part of the Tango Ecosystem
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

from __future__ import print_function
import numpy as np

from jupytango.tools import *  

# ------------------------------------------------------------------------------
module_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
class ChannelData(object):
    """channel data"""

    Format = enum(
        'SCALAR',
        'SPECTRUM',
        'IMAGE',
        'UNKNOWN'
    )

    def __init__(self, name='anonymous'):
        # name
        self._name = name
        # format
        self._format = ChannelData.Format.UNKNOWN
        # data buffer (numpy ndarray)
        self._buffer = np.zeros((1, 1))
        # time buffer (numpy ndarray)
        self._time_buffer = None
        # error flag (error could have occured but data can be valid)
        self._has_error = False
        # data flag 
        self._has_valid_data = False
        # error txt
        self._error = "no error"
        # exception caught
        self._exception = None

    @property
    def name(self):
        return self._name

    @property
    def format(self):
        return self._format

    @property
    def has_error(self):
        return self._has_error

    @property
    def error(self):
        return self._error

    @property
    def exception(self):
        return self._exception

    @property
    def has_valid_data(self):
        return self._has_valid_data

    @property
    def dim_x(self):
        num_dims = len(self._buffer.shape)
        if num_dims >= 1:
            return self._buffer.shape[num_dims - 1]
        else:
            return 0

    @property
    def dim_y(self):
        num_dims = len(self._buffer.shape)
        if num_dims >= 2:
            return self._buffer.shape[num_dims - 2]
        else:
            return 0

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, b):
        self._buffer = b
        self._has_valid_data = True

    @property
    def time_buffer(self):
        return self._time_buffer

    def set_data(self, data_buffer, time_buffer=None, format=None):
        assert (isinstance(data_buffer, np.ndarray))
        #TODO: check data & time buffers have same size 
        self._buffer = data_buffer
        self._time_buffer = time_buffer
        self._format = format
        self._has_valid_data = True

    def clear_data(self):
        self._buffer = None
        self._time_buffer = None
        self._has_valid_data = False

    def set_error(self, err=None, exc=None, rst_data=False):
        if not self._has_error:
            self._has_error = True
            self._error = "unknown error" if err is None else err
            self._exception = Exception("unknown error") if exc is None else exc

    def clear_error(self):
        self._has_error = False
        self._error = "no error"
        self._exception = None
        

# ------------------------------------------------------------------------------
class DataSource(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def pull_data(self):
        return ChannelData()

    def cleanup(self):
        pass
