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

from __future__ import print_function
from six import string_types
import inspect
from contextlib import contextmanager

import numpy
import PyTango

# PyTango versus backward compatibilities -_-
# pylint: disable=import-error,no-name-in-module
try:
    from PyTango.server import run, Device, DeviceMeta, attribute, command, device_property
except ImportError:
    from PyTango.api2 import run, Device, DeviceMeta, attribute, command, device_property

from fs.utils.tango.TangoHelpers import *
from fs.utils.tango.TangoEventsConsumer import *

from fs.utils.display import TreeNode, Ansi
from fs.utils.errors import silent_catch

# =============
# miscellaneous
# =============

tango_state_colors = {
    PyTango.DevState.ON: {'fg': 'green'},
    PyTango.DevState.OFF: {'fg': 'white'},
    PyTango.DevState.CLOSE: {'fg': 'white'},
    PyTango.DevState.OPEN: {'fg': 'green'},
    PyTango.DevState.INSERT: {'fg': 'white'},
    PyTango.DevState.EXTRACT: {'fg': 'green'},
    PyTango.DevState.MOVING: {'fg': 'blue', 'bold': True},
    PyTango.DevState.STANDBY: {'fg': 'yellow', 'bold': True},
    PyTango.DevState.FAULT: {'fg': 'red', 'bold': True},
    PyTango.DevState.INIT: {'fg': 'yellow'},
    PyTango.DevState.RUNNING: {'fg': 'green', 'bold': True},
    PyTango.DevState.ALARM: {'fg': 'yellow'},
    PyTango.DevState.DISABLE: {'fg': 'magenta'},
    PyTango.DevState.UNKNOWN: {'fg': 'white'},
}


def colorized_state(devstate):
    """Returns a colorful representation of a device state"""
    scheme = tango_state_colors.get(devstate, {})
    return Ansi.graphics(devstate.name, **scheme)


def subdevices_tree(dev, seen=None):
    """
    Compute a tree containing recursive device dependencies (name plus state)
    dev may be either a DeviceProxy or a device name
    """
    prop_name = '__SubDevices'
    seen = seen or set()
    # get entry name and put it in the list of nodes seen
    name = dev.dev_name() if isinstance(dev, DeviceProxy) else dev
    if name in seen:
        return TreeNode('{} ...'.format(name))
    seen.add(name)
    # get state
    try:
        if not isinstance(dev, DeviceProxy):
            dev = DeviceProxy(dev)
        state = dev.state()
    except PyTango.DevFailed:
        state = PyTango.DevState.UNKNOWN
    # get direct children
    children = []
    with silent_catch():
        children += dev.get_property(prop_name)[prop_name]
    # get dserver children
    with silent_catch():
        dserver = DeviceProxy("dserver/{}".format(dev.info().server_id))
        children += dserver.get_property(prop_name)[prop_name]
    # make node
    node = TreeNode("{} {}".format(name, colorized_state(state)))
    for child in sorted({v for v in children if 'dserver' not in v}):
        node.add_child(subdevices_tree(child, seen))
    return node


class DeviceLogger(logging.Handler):
    """Logging handler routing records to tango streams"""

    def __init__(self, device, **kwargs):
        logging.Handler.__init__(self, **kwargs)
        self.device = device

    def emit(self, record):
        stream_attr = {
            logging.DEBUG: '__debug_stream',
            logging.INFO: '__info_stream',
            logging.WARNING: '__warn_stream',
            logging.ERROR: '__error_stream',
            logging.FATAL: '__fatal_stream',
        }.get(record.levelno, '__info_stream')
        stream = getattr(self.device, stream_attr)
        stream(str(self.format(record)))


@contextmanager
def try_tango(desc, reason='PYTHON_ERROR', severity=PyTango.ErrSeverity.ERR):
    """
    context manager that adds an additional DevError to DevFailed thrown
    The origin will be set inspecting the caller frame
    """
    caller_stack = inspect.stack()[2]
    try:
        yield
    except PyTango.DevFailed as df:
        filename, line, name = caller_stack[1:4]
        origin = 'In {} [line {}] function {}'.format(filename, line, name)
        PyTango.Except.re_throw_exception(df, reason, desc, origin, severity)


def get_attribute_shape(attr):
    """return the max_shape supported by the attribute as a tuple"""
    try:
        ndim = {PyTango.SCALAR: 0, PyTango.SPECTRUM: 1, PyTango.IMAGE: 2}[attr.get_data_format()]
        max_shape = (attr.get_max_dim_x(), attr.get_max_dim_y())[:ndim]
        return max_shape[::-1] if ndim == 2 else max_shape  # swap rows & cols
    except KeyError:
        raise Exception('unknown attribute data format')


def set_attribute_value(attr, value):
    """set an attribute value from a numpy ndarray checking coherence"""
    attr_shape = get_attribute_shape(attr)
    attr_ndim = len(attr_shape)
    if not (attr_ndim == 0 and value.size == 1):  # do not check single value for scalar type
        if attr_ndim != value.ndim:
            raise Exception('dimension mismatch: attr. shape is {}, got {}'.format(attr_shape, value.shape))
        if any(d > d_max for d_max, d in zip(attr_shape, value.shape)):  # check overflow
            raise Exception('data overflow: attr. shape is {}, got {}'.format(attr_shape, value.shape))
    # extract scalar from array
    v = value.item() if attr_ndim == 0 else value
    # actually set the attribute
    attr.set_value(v)


def attribute_info_to_config(info):
    """
    convert an instance of AttributeInfoEx or AttributeConfig to an AttributeConfig3
    suitable for being used with set_attribute_config in a device implementation
    """
    result = PyTango.AttributeConfig_3()
    result.name = info.name
    result.writable = info.writable
    result.data_format = info.data_format
    result.data_type = info.data_type
    result.max_dim_x = info.max_dim_x
    result.max_dim_y = info.max_dim_y
    result.description = info.description
    result.label = info.label
    result.unit = info.unit
    result.standard_unit = info.standard_unit
    result.display_unit = info.display_unit
    result.format = info.format
    result.min_value = info.min_value
    result.max_value = info.max_value
    result.writable_attr_name = info.writable_attr_name
    if hasattr(info, 'level'):
        result.level = info.level
    elif hasattr(info, 'disp_level'):
        result.level = info.disp_level
    else:
        result.level = PyTango.DispLevel.OPERATOR
    return result


# =============
# DeviceWrapper
# =============

class AttributeWrapper(object):
    """
    An attribute wrapper is a descriptor that can be used with any class that adds a 'proxy' attribute
    binding an instance of DeviceProxy (ie a DeviceWrapper).
    It is a view over an attribute
    """

    def __init__(self, attribute_name):
        self.attribute_name = attribute_name

    def __get__(self, instance, owner):
        del owner  # unused
        return instance.proxy.read_attribute(self.attribute_name).value

    def __set__(self, instance, value):
        return instance.proxy.write_attribute(self.attribute_name, value)

    def __delete__(self, instance):
        raise AttributeError("you can't delete an attribute from a device")


class DeviceWrapper(object):
    """
    A device wrapper is an object that binds a Tango DeviceProxy and
    adds some features linked to the device class
    """

    def __init__(self, dev_class):
        assert isinstance(dev_class, string_types)
        self._dev_class = str(dev_class)
        self.db = PyTango.Database()
        self._proxy = None

    # -----------------------
    # Device Class Management
    # -----------------------

    @property
    def dev_class(self):
        return self._dev_class

    def query_devices(self, pattern='*'):
        """Returns the whole list of devices available existing in the default Tango database"""
        return list(self.db.get_device_name(pattern, self._dev_class))

    # ------------
    # Proxy Access
    # ------------

    def has_proxy(self):
        """
        Returns True if a proxy is bound to this wrapper.
        NOTE that property proxy will raise if False
        """
        return self._proxy is not None

    @property
    def proxy(self):
        """
        Accessor to the bound proxy that checks its existence. It aims
        to raise an explicit error rather than 'NoneType' related exceptions
        that can be hard to diagnose
        """
        if self._proxy is None:
            raise AttributeError('%r is not bound to a device' % self)
        return self._proxy

    @proxy.setter
    def proxy(self, new_proxy):
        """
        Change the Tango device bound to this wrapper.
        A warning is generated if the classes mismatch
        None is accepted and will clear the current proxy
        """
        # pylint: disable=W0212
        if new_proxy is not None:
            # check device's class
            dev_class = self.db.get_class_for_device(new_proxy.dev_name())
            if dev_class.lower() != self._dev_class.lower():
                raise ValueError('wrong class %r, expected %r' % (dev_class, self._dev_class))
            # check that device is accessible
            new_proxy.ping()
        self._proxy = new_proxy

    def relocate(self, dev_name):
        """Change the Tango device bound to this wrapper by its name"""
        assert isinstance(dev_name, string_types)
        self.proxy = PyTango.DeviceProxy(str(dev_name))

    # --------------
    # State / Status
    # --------------

    @property
    def state(self):
        try:
            return self.proxy.state()
        except Exception:
            return PyTango.DevState.UNKNOWN

    @property
    def status(self):
        try:
            return self.proxy.status()
        except Exception:
            return "unknown status"

    # ------
    # String
    # ------

    def __repr__(self):
        return '<DeviceWrapper(%s) at %#x>' % (self._dev_class, id(self))
