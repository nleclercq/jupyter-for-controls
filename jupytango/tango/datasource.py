from __future__ import print_function
import time
import logging
from collections import deque
import threading
from six import iteritems, string_types

import PyTango as tango

from jupytango.datasource import *
from jupytango.tango.eventsconsumer import TangoEventSubscriptionForm, TangoEventsConsumer


# ------------------------------------------------------------------------------
module_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
class RingBuffer(np.ndarray):
    """
    a multidimensional ring buffer
    https://gist.github.com/mtambos/aa435461084b5c0025d1
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def append(self, x):
        """adds element x to the ring buffer"""
        self[:-1] = self[1:]
        self[-1] = x


# ------------------------------------------------------------------------------
class MonitoredAttribute(TangoEventsConsumer):
    """ manages a monitored Tango attribute """

    def __init__(self, fqan, bfdp=0):
        try:
            TangoEventsConsumer.__init__(self)
            # data source url
            self._fqan = fqan
            # data source proxy (connection to Tango device server)
            self._proxy = None
            # fetch data source info (data type, data format, ...)
            self._config = None
            # scalar attributes need a particular treatment
            self._is_scalar = False
            self._buffer_depth = int(bfdp)
            self._data_buffer = None
            self._time_buffer = None
            # data (i.e. Tango attribute value as an instance of ChannelData)
            self._value = None
            # data ready event stuffs
            self._lock = threading.Lock()
            self._value_lock = threading.Lock()
            self._has_tango_evt = False
            self._previous_evt_counter = 0
            self._current_evt_counter = 0
            # initialization
            self.__initialize()
        except Exception:
            pass

    def __reset_connection(self):
        self._proxy = None

    def __connect(self, val=None):
        try:
            # data source proxy (connection to Tango device server)
            self._proxy = tango.AttributeProxy(self._fqan)
        except Exception as e:
            self.__reset_connection()
            err = "failed to initialize attribute monitor for '{}'\n".format(self._fqan)
            err += self.__error_tips(e)
            # print("{}".format(err))
            if val:
                val.set_error(err, e)
            raise

    def __check_device_is_alive(self, val=None):
        if self._proxy is None:
            self.__connect(val)
        try:
            # be sure the device is running (test connection to Tango device server)
            self._proxy.ping()
        except Exception as e:
            err = "failed to read Tango attribute '{}'\n".format(self._fqan)
            err += "please make sure its Tango device is up and running"
            self.__append_tango_exception(err, e)
            #print("{}".format(err))
            if val:
                val.set_error(err, e)
            raise
     
    def __get_attribute_config(self, val=None):
        self.__check_device_is_alive(val)
        try:
            # fetch data source info (data type, data format, ...)
            self._config = self._proxy.get_config()
        except Exception as e:
            err = "failed to obtain Tango attribute configuration for '{}'".format(self._fqan)
            err += "please check state and status of '{}'".format(self._proxy.get_device_proxy().name())
            self.__append_tango_exception(err, e)
            #print("{}".format(err))
            if val:
                val.set_error(err, e)
            raise

    def __error_tips(self, dev_failed=None):
        err = "1. it might be a dynamic attribute which is not yet ready\n"
        err += "\t`-> wait till the data is available...'\n"
        err += "2. the associated Tango device might be down\n"
        err += "\t`-> check the device state and status'\n"
        err += "3. the device name, attribute name or attribute alias might be misspelled\n"
        err += "\t`-> check the specified name then retry"
        self.__append_tango_exception(dev_failed)
        return err

    @staticmethod
    def __append_tango_exception(error_text, dev_failed=None):
        if dev_failed and isinstance(dev_failed, tango.DevFailed):
            error_text += "\nTango exception caught:\n"
            for de in dev_failed.args:
                error_text += "{}\n".format(de.desc)
        return error_text

    def __initialize(self, reset_existing_data=True):
        # reset both proxy and config
        self._proxy = None
        self._config = None
        val = ChannelData()
        try:
            # fetch data source info (data type, data format, ...)
            self.__get_attribute_config(val)
            try:
                # try subscribe to data ready event
                f = TangoEventSubscriptionForm()
                f.dev = self._proxy.get_device_proxy().name()
                f.attr = self._proxy.name()
                f.evt_type = tango.EventType.DATA_READY_EVENT
                self.subscribe_event(f)
                self._has_tango_evt = True
                self._tango_evt_counter = 0
                #print("{}: successfully subscribed to events".format(self.name))
            except Exception as e:
                #print("{}: failed to subscribe to events".format(self.name))
                self._has_tango_evt = False
                pass
            # scalar attributes need a circular buffer and a dummy attribute value
            self._is_scalar = self._config.data_format == tango.AttrDataFormat.SCALAR
            if self._is_scalar and (reset_existing_data or not self._data_buffer):
                array = np.empty((self._buffer_depth,))
                array.fill(np.nan)
                self._data_buffer = RingBuffer(array)
                array = np.empty((self._buffer_depth,), dtype=float)
                array.fill(np.nan)
                self._time_buffer = RingBuffer(array)
            # reset any error
            val.reset_error()
            # print("initialization successful for {}".format(self.name))
        except Exception as e:
            #print("initialization failed for {}".format(self.name))
            #print(e)
            raise
        finally:
            self.__push_current_value(val)

    @property
    def name(self):
        return self._fqan

    @property
    def __initialized(self):
        return (self._proxy is not None) and (self._config is not None)

    @property
    def is_valid(self):
        return self.__initialized

    def _handle_event(self, event):
        with self._lock:
            #print("received data-ready from {} for {}".format(event.data.attr_name, self.name))
            self.__increment_evt_counter()
            self.__update_value()

    def __increment_evt_counter(self):
        self._previous_evt_counter = self._current_evt_counter
        self._current_evt_counter += 1

    def __received_evt(self):
        return self._current_evt_counter != 0

    def __has_new_data_available(self):
        ans = self._current_evt_counter > self._previous_evt_counter
        self._previous_evt_counter = self._current_evt_counter
        return ans

    @staticmethod
    def __tango_to_data_stream_format(format):
        if format == tango.AttrDataFormat.SCALAR:
            return ChannelData.Format.SCALAR
        elif format == tango.AttrDataFormat.SPECTRUM:
            return ChannelData.Format.SPECTRUM
        elif format == tango.AttrDataFormat.IMAGE:
            return ChannelData.Format.IMAGE
        else:
            return ChannelData.Format.UNKNOWN

    def __update_value(self):
        try:
            if not self.__initialized:
                self.__initialize(False)
        except:
            return
        val = ChannelData()
        try:
            try:
                # read data source (i.e. read Tango attribute value)
                t0 = time.time()
                av = self._proxy.read()
                val.read_time = time.time() - t0
                val.counter = self._current_evt_counter
            except Exception as e:
                self.__reset_connection()
                if self._data_buffer:
                    self._data_buffer.append(float('nan'))
                if self._time_buffer:
                    self._time_buffer.append(float('nan'))
                err = "failed to read Tango attribute '{}'\n".format(self._fqan)
                #print("{}".format(err))
                err += self.__error_tips()
                val.set_error(err, e)
                raise
            # be sure av contains valid data
            if av.has_failed or av.is_empty:
                if self._data_buffer:
                    self._data_buffer.append(float('nan'))
                if self._time_buffer:
                    self._time_buffer.append(time.time() * 1000.)
                err = "oops, got invalid data from Tango attribute '{}'".format(self._fqan)
                #print("{}".format(err))
                val.set_error(err, None)
                raise Exception("invalid data")
            # convert 'av' to our own attribute value representation
            dfmt = self.__tango_to_data_stream_format(av.data_format)
            if self._is_scalar:
                self._data_buffer.append(float(av.value))
                self._time_buffer.append(av.time.totime() * 1000.)
                val.set_data(data_buffer=self._data_buffer, time_buffer=self._time_buffer, format=dfmt)
            else:
                val.set_data(data_buffer=av.value, format=dfmt)
            #print("successfully updated value for {}".format(self.name))
        except:
            #print("failed to update value for {}".format(self.name))
            #print(e)
            pass
        finally:
            self.__push_current_value(val)

    def update(self, force=False):
        if not force and self._has_tango_evt:
            with self._lock:
                if self.__received_evt():
                    return self.__update_with_events_enabled()
                else:
                    return self.__update_with_events_disabled()
        else:
            return self.__update_with_events_disabled()

    def __update_with_events_disabled(self):
        #print("update_with_events_disabled for {}".format(self.name))
        try:
            self.__update_value()
        finally:
            return self.__pop_current_value()

    def __update_with_events_enabled(self):
        #print("update_with_events_enabled for {}".format(self.name))
        value = None
        try:
            if self.__has_new_data_available():
                #print("\t `-> new data available - returning last value for {}".format(self.name))
                value = self.__pop_current_value()
            else:
                #print("\t `-> no new data available - returning empty value for {}".format(self.name))
                value = ChannelData()
        finally:
            return value

    def __push_current_value(self, value):
        with self._value_lock:
            self._value = value

    def __pop_current_value(self):
        with self._value_lock:
            current_value = self._value
            self._value = None
            return current_value

    def get_value(self):
        return self.__pop_current_value()

    def is_scalar(self):
        return self.__is_a(tango.AttrDataFormat.SCALAR)

    def is_spectrum(self):
        return self.__is_a(tango.AttrDataFormat.SPECTRUM)

    def is_image(self):
        return self.__is_a(tango.AttrDataFormat.IMAGE)

    def __is_a(self, fmt):
        if not self.has_valid_configuration():
            self.__get_attribute_config()
        return self._config.data_format == fmt

    def has_valid_configuration(self, try_reconnect=True):
        if not self._config and try_reconnect:
            try:
                self.__initialize()
            except:
                return False
        return False if not self._config else True

    def cleanup(self):
        try:
            self.unsubscribe_events()
        except:
            pass
        try:
            self.__reset_connection()
        except:
            pass


# ------------------------------------------------------------------------------
class TangoDataSource(DataSource):
    def __init__(self, dtsn, fqan, bfdp=0):
        DataSource.__init__(self, dtsn)
        self._mta = MonitoredAttribute(fqan, bfdp)

    @property
    def monitored_attribute(self):
        return self._mta

    def pull_data(self):
        return self._mta.update()

    def is_scalar(self):
        return self._mta.is_scalar()

    def is_spectrum(self):
        return self._mta.is_spectrum()

    def is_image(self):
        return self._mta.is_image()

    def cleanup(self):
        self._mta.cleanup()



