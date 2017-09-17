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

from six import iteritems, string_types
import time

import numpy
import PyTango


# def convert_dtype(dtype):
#     """Returns the corresponding Tango type from a numpy dtype"""
#     if dtype.str.startswith('|S'):
#         return PyTango.DevString
#     return {
#         '|b1': PyTango.DevBoolean,
#         '|i1': PyTango.DevUChar,
#         '|u1': PyTango.DevUChar,
#         '<i2': PyTango.DevShort,
#         '<u2': PyTango.DevUShort,
#         '<i4': PyTango.DevLong,
#         '<u4': PyTango.DevULong,
#         '<i8': PyTango.DevLong64,
#         '<u8': PyTango.DevULong64,
#         '<f4': PyTango.DevFloat,
#         '<f8': PyTango.DevDouble,
#     }[dtype.str]


TANGO_TYPES = {
    numpy.bool_: PyTango.DevBoolean,
    numpy.int8: PyTango.DevUChar,
    numpy.uint8: PyTango.DevUChar,
    numpy.int16: PyTango.DevShort,
    numpy.uint16: PyTango.DevUShort,
    numpy.int32: PyTango.DevLong,
    numpy.uint32: PyTango.DevULong,
    numpy.int64: PyTango.DevLong64,
    numpy.uint64: PyTango.DevULong64,
    numpy.float32: PyTango.DevFloat,
    numpy.float64: PyTango.DevDouble,
    numpy.string_: PyTango.DevString,
}


NUMPY_TYPES = {v: k for k, v in iteritems(TANGO_TYPES)}


class AttributeInfo(object):

    def __init__(self):
        self.name = None
        self.numpy_data_type = None
        self.tango_data_type = None
        self.data_format = None
        self.x_dim = 0
        self.y_dim = 0

    def extract_from_device_attribute(self, da):
        assert isinstance(da, PyTango.DeviceAttribute)
        self.name = da.name
        self.data_format = da.data_format
        self.x_dim = da.r_dimension.dim_x
        self.y_dim = da.r_dimension.dim_y
        self.tango_data_type = da.type
        try:
            self.numpy_data_type = numpy.dtype(NUMPY_TYPES[self.tango_data_type])
        except KeyError:
            self.numpy_data_type = None


class TangoHelper(object):
    """A Tango helper class"""

    @classmethod
    def wait_device_state(cls, device, expected_state, timeout_in_sec):
        """
        Waits up to 'timeout_in_sec' seconds for the specified device to switch to the specified PyTango.DevState
        :param device: a Tango device name or proxy
        :type device: str, unicode or PyTango.DeviceProxy
        :param expected_state: the expected Tango state
        :type expected_state: PyTango.DevState.
        :param timeout_in_sec: the wait timeout in second
        :type timeout_in_sec: float
        """
        cls.wait_device_state(device, [expected_state], timeout_in_sec)

    @classmethod
    def wait_device_states(cls, device, expected_states, timeout_in_sec):
        """
        Waits up to 'timeout_in_sec' seconds for the specified device to switch to one of the specified PyTango.DevState
        :param device: a Tango device name or proxy
        :type device: str, unicode or PyTango.DeviceProxy
        :param expected_states: the expected Tango states
        :type expected_states: an iterable object containing instances of PyTango.DevState.
        :param timeout_in_sec: the wait timeout in second
        :type timeout_in_sec: float
        """
        if isinstance(device, string_types):
            dev_proxy = PyTango.DeviceProxy(str(device))
        elif isinstance(device, PyTango.DeviceProxy):
            dev_proxy = device
        else:
            raise TypeError("invalid argument 'device' argument specified to TangoHelper.wait_device_states")
        t0 = time.time()
        while (time.time() - t0) < timeout_in_sec:
            if dev_proxy.State() in expected_states:
                return
        raise RuntimeError("timeout expired while waiting for {} to switch to {}".format(dev_proxy.dev_name(), expected_states))

    @classmethod
    def get_attribute_info(cls, fully_specified_attribute_name):
        """
        Returns the AttributeInfo of the specified attribute
        :param fully_specified_attribute_name: the fully specified Tango attribute name
        :type fully_specified_attribute_name: str or unicode
        :returns the requested attribute info as an instance of AttributeInfo
        :rtype: AttributeInfo
        """
        assert isinstance(fully_specified_attribute_name, string_types)
        attr = PyTango.AttributeProxy(fully_specified_attribute_name)
        attr_info = AttributeInfo()
        attr_info.extract_from_device_attribute(attr.read())
        return attr_info
