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

plots_module_logger_name = "fs.client.jupyter.datasource"

# ------------------------------------------------------------------------------
def enum(*sequential):
    enums = dict(zip(sequential, range(len(sequential))))
    enums['len'] = len(sequential)
    return type('Enum', (), enums)

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
        # update failed - data is invalid
        self._has_failed = False
        # has new data (updated since last read)
        self.has_been_updated = False
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
    def has_failed(self):
        return self._has_failed

    @property
    def error(self):
        return self._error

    @property
    def exception(self):
        return self._exception

    @property
    def is_valid(self):
        return not self._has_failed and self._buffer is not None

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

    @property
    def time_buffer(self):
        return self._time_buffer

    def set_data(self, data_buffer, time_buffer=None, format=None):
        assert (isinstance(data_buffer, np.ndarray))
        self._buffer = data_buffer
        self._time_buffer = time_buffer
        self._format = format
        self.has_been_updated = True
        self.reset_error()

    def reset_error(self):
        self._has_failed = False
        self._error = "no error"
        self._exception = None

    def set_error(self, err, exc):
        if not self._has_failed:
            self._has_failed = True
            self._error = "unknown error" if not err else err
            self._exception = Exception("unknown error") if not exc else exc
            self.__reset_data()

    def __reset_data(self):
        self._buffer = None
        self._time_buffer = None
        self._has_been_updated = False
        

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
