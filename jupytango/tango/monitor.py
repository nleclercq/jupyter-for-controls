from __future__ import print_function
import time
import datetime
import logging
from collections import deque
import threading
from six import iteritems, string_types

try:
    # just in case tango stuffs are not installed
    import PyTango as tango
except:
    pass

from bokeh.models import ColumnDataSource
from bokeh.models import CustomJS
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.tools import HoverTool
from bokeh.palettes import Plasma256, Viridis256, Inferno256, Greys256
from bokeh.io import show, output_notebook, push_notebook, reset_output
from bokeh.plotting import figure, Figure
from bokeh.resources import INLINE

from jupytango.tools import silent_catch
from jupytango.plots import GenericChannel, DataStream, DataStreamer, DataStreamerController
from jupytango.tango.datasource import *


# ------------------------------------------------------------------------------
module_logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
bokeh_redirected = False

# ------------------------------------------------------------------------------
tango_monitors = dict()
tango_attribute_plots = dict()

# ------------------------------------------------------------------------------
def redirect_bokeh_output(force=False):
    global bokeh_redirected
    if force or not bokeh_redirected:
        try:
            output_notebook(resources=INLINE, hide_banner=True)
            bokeh_redirected = True
        except:
            pass
        
# ------------------------------------------------------------------------------
def open_tango_monitor(**kwargs):
    GenericMonitor.instanciate(**kwargs)
        
# ------------------------------------------------------------------------------
class Monitor(NotebookCellContent):
    """ base class for Tango attribute monitors """

    def __init__(self, **kwargs):
        try:
            # generate uid
            self._uid = uuid4().int
            # init super class
            NotebookCellContent.__init__(self, str(self._uid))
            # running in standalone mode
            self._standalone_mode = kwargs.get('standalone_mode', True)
            # refresh period in seconds
            rp = kwargs.get('refresh_period', 1.)
            self._refresh_period = self.__compute_refresh_period(rp)
        except:
            self._remove_self_reference()
            raise

    def _remove_self_reference(self):
        try:
            del monitors[self._uid]
        except Exception as e:
            print(e)

    @staticmethod
    def __compute_refresh_period(refresh_period):
        if not refresh_period:
            refresh_period = 1.
        computed_refresh_period = max(0.1, float(refresh_period))
        computed_refresh_period = min(10., computed_refresh_period)
        return computed_refresh_period

    @property
    def standalone_mode(self):
        return self._standalone_mode

    @property
    def refresh_period(self):
        return self._refresh_period

    def close(self):
        self.cleanup()

    def cleanup(self):
        # remove self ref. from monitors repository
        self._remove_self_reference()


# ------------------------------------------------------------------------------
class GenericMonitor(Monitor):
    """ Generic monitor for single Tango attribute """

    @staticmethod
    def instanciate(**kwargs):
        redirect_bokeh_output()
        # fully qualified attribute name
        fqan = kwargs.get('attribute', '')
        if not len(fqan):
            return
        # close  existing monitor (if any)
        for mn, mi in iteritems(tango_monitors):
            if mi.attribute_name == fqan:
                module_logger.debug("Monitor[{}] exist".format(fqan))
                try:
                    module_logger.debug("closing Monitor[{}]...".format(fqan))
                    mi.close()
                    module_logger.debug("Monitor[{}] successfully closed".format(fqan))
                except:
                    module_logger.debug("failed to close Monitor[{}]".format(fqan))
                    pass
                break
        # instanciate the Monitor
        try:
            module_logger.debug("instanciating Monitor for {}".format(fqan))
            m = GenericMonitor(**kwargs)
            tango_monitors[m._uid] = m
            module_logger.debug("Monitor successfully instanciated")
        except Exception as e:
            module_logger.error(e)
            raise

    def __init__(self, **kwargs):
        try:
            # init super class
            Monitor.__init__(self, **kwargs)
            # fully qualified attribute name or alias must be specified
            self._fqan = kwargs.get('attribute', None)
            if self._fqan is None:
                raise Exception("can't open Tango monitor - no Tango attribute name specified")
            # ring buffer depth (for scalar)
            hbd = kwargs.get('history_buffer_depth', None)
            self._buffer_depth = self.__compute_buffer_depth(self._refresh_period, hbd)
            # setup data stream
            self._dsr = None
            self._dsr = self._setup_data_stream(**kwargs)
            if self._standalone_mode:
                self._dsc = DataStreamerController('ctr', self._dsr)
                self._dsc.register_close_callback(self.cleanup)
            else:
                self._dsc = None
        except Exception as e:
            print(e)
            self._dsr = None
            self._dsc = None
            self.cleanup()
            raise

    def _setup_data_stream(self, **kwargs):
        tds = TangoDataSource(self._fqan, self._fqan, self._buffer_depth)
        tch = GenericChannel(self._fqan, model_properties=kwargs)
        tch.set_data_source(tds)
        dsm = DataStream('dsm', channels=[tch])
        dsr = DataStreamer('dsr', data_streams=[dsm], update_period=self._refresh_period)
        return dsr

    def _get_data_streamer(self):
        return self._dsr

    @staticmethod
    def __compute_buffer_depth(refresh_period, buffer_depth):
        if not buffer_depth:
            buffer_depth = 900.
        history_depth = max(1., float(buffer_depth))
        history_depth = min(3600., history_depth)
        computed_buffer_depth = history_depth / refresh_period
        return computed_buffer_depth

    @property
    def attribute_name(self):
        return self._fqan

    @property
    def buffer_depth(self):
        return self._buffer_depth

    def close(self):
        try:
            if self._dsc:
                with silent_catch:
                    self._dsc.close()
            else:
                self.cleanup()
        finally:
            del monitors[self._uid]

    def cleanup(self):
        # cleanup the data streamer
        if self._dsr:
            self._dsr.cleanup()
        # call mother-class' cleanup
        super(GenericMonitor, self).cleanup()


# ------------------------------------------------------------------------------
def plot_tango_attribute(ns):
    redirect_bokeh_output()
    n = ns.attr.count("/")
    if not n:
        ap = tango.AttributeProxy(ns.attr)
        av = ap.read()
        fqan = ap.get_device_proxy().name() + "/" + ap.name()
        title = ns.attr + " [" + fqan + "]"
    elif n == 3:
        dn, _, an = ns.attr.rpartition("/")
        dp = tango.DeviceProxy(dn)
        av = dp.read_attribute(an)
        fqan = ns.attr
    else:
        raise Exception("invalid attribute name specified - expected an alias or something like 'fully/qualified/attribute/name'")
    kwargs = dict()
    kwargs['tools'] = 'pan,wheel_zoom,box_select,reset,hover'
    kwargs['title'] = fqan + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if ns.width is not None:
        kwargs['plot_width'] = ns.width
    if ns.height is not None:
        kwargs['plot_height'] = ns.height
    upsidedown = ns.upsidedown if ns.upsidedown is not None else False
    plot = None
    if av.data_format == tango.AttrDataFormat.SCALAR:
        print(fqan + " = " + str(av.value))
    elif av.data_format == tango.AttrDataFormat.SPECTRUM:
        kwargs['toolbar_location'] = 'above'
        plot = figure(**kwargs)
        plot.line(range(av.value.shape[0]), av.value, line_width=1)
        plot.circle(range(av.value.shape[0]), av.value, size=3)
    elif av.data_format == tango.AttrDataFormat.IMAGE:
        kwargs['toolbar_location'] = 'right'
        lcm = LinearColorMapper(palette=Plasma256)
        ymin = 0 if not upsidedown else av.dim_y
        ymax = av.dim_y if not upsidedown else 0
        plot = figure(x_range=(0, av.dim_x), y_range=(ymin, ymax), **kwargs)
        image = av.value if not upsidedown else av.value[::-1]
        plot.image(image=[image], x=0, y=ymin, dw=av.dim_x, dh=av.dim_y, color_mapper=lcm)
    else:
        print(fqan + " has an unknown/unsupported attribute data format [{}]".format(str(av.data_format)))
    if plot:
        ht = plot.select(HoverTool)[0]
        ht.tooltips = [("index", "$index"), ("(x,y)", "(@x, @y)")]
        plot.toolbar.active_drag = None
        plot.toolbar.active_scroll = None
        plot.toolbar.logo = None
        tango_attribute_plots[fqan] = show(plot, notebook_handle=True)