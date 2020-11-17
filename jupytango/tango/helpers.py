from six import iteritems, string_types
import time
import numpy
import PyTango as tango

# def convert_dtype(dtype):
#     """Returns the corresponding Tango type from a numpy dtype"""
#     if dtype.str.startswith('|S'):
#         return tango.DevString
#     return {
#         '|b1': tango.DevBoolean,
#         '|i1': tango.DevUChar,
#         '|u1': tango.DevUChar,
#         '<i2': tango.DevShort,
#         '<u2': tango.DevUShort,
#         '<i4': tango.DevLong,
#         '<u4': tango.DevULong,
#         '<i8': tango.DevLong64,
#         '<u8': tango.DevULong64,
#         '<f4': tango.DevFloat,
#         '<f8': tango.DevDouble,
#     }[dtype.str]

# ------------------------------------------------------------------------------
TANGO_TYPES = {
    numpy.bool_: tango.DevBoolean,
    numpy.int8: tango.DevUChar,
    numpy.uint8: tango.DevUChar,
    numpy.int16: tango.DevShort,
    numpy.uint16: tango.DevUShort,
    numpy.int32: tango.DevLong,
    numpy.uint32: tango.DevULong,
    numpy.int64: tango.DevLong64,
    numpy.uint64: tango.DevULong64,
    numpy.float32: tango.DevFloat,
    numpy.float64: tango.DevDouble,
    numpy.string_: tango.DevString,
}

# ------------------------------------------------------------------------------
NUMPY_TYPES = {v: k for k, v in iteritems(TANGO_TYPES)}

# ------------------------------------------------------------------------------
class AttributeInfo(object):

    def __init__(self):
        self.name = None
        self.numpy_data_type = None
        self.tango_data_type = None
        self.data_format = None
        self.x_dim = 0
        self.y_dim = 0

    def extract_from_device_attribute(self, da):
        assert isinstance(da, tango.DeviceAttribute)
        self.name = da.name
        self.data_format = da.data_format
        self.x_dim = da.r_dimension.dim_x
        self.y_dim = da.r_dimension.dim_y
        self.tango_data_type = da.type
        try:
            self.numpy_data_type = numpy.dtype(NUMPY_TYPES[self.tango_data_type])
        except KeyError:
            self.numpy_data_type = None

# ------------------------------------------------------------------------------
class TangoHelper(object):
    """A Tango helper class"""

    @classmethod
    def wait_device_state(cls, device, expected_state, timeout_in_sec):
        """
        Waits up to 'timeout_in_sec' seconds for the specified device to switch to the specified tango.DevState
        :param device: a Tango device name or proxy
        :type device: str, unicode or tango.DeviceProxy
        :param expected_state: the expected Tango state
        :type expected_state: tango.DevState.
        :param timeout_in_sec: the wait timeout in second
        :type timeout_in_sec: float
        """
        cls.wait_device_state(device, [expected_state], timeout_in_sec)

    @classmethod
    def wait_device_states(cls, device, expected_states, timeout_in_sec):
        """
        Waits up to 'timeout_in_sec' seconds for the specified device to switch to one of the specified tango.DevState
        :param device: a Tango device name or proxy
        :type device: str, unicode or tango.DeviceProxy
        :param expected_states: the expected Tango states
        :type expected_states: an iterable object containing instances of tango.DevState.
        :param timeout_in_sec: the wait timeout in second
        :type timeout_in_sec: float
        """
        if isinstance(device, string_types):
            dev_proxy = tango.DeviceProxy(str(device))
        elif isinstance(device, tango.DeviceProxy):
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
        attr = tango.AttributeProxy(fully_specified_attribute_name)
        attr_info = AttributeInfo()
        attr_info.extract_from_device_attribute(attr.read())
        return attr_info
