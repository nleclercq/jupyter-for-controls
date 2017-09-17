# python built-ins
from __future__ import print_function
import time
import datetime
import logging
from math import ceil
from uuid import uuid4
from operator import itemgetter
from collections import deque, OrderedDict
from contextlib import contextmanager
import threading
# python2/ python3 compatibility
from six import iteritems, string_types
# tornado web server
from tornado.ioloop import IOLoop
# numpy
import numpy as np
# ipython
try:
    from IPython import get_ipython
    from IPython.display import display, clear_output
    from IPython.core.interactiveshell import InteractiveShell
except:
    pass
try:
    # IPython 4.x
    from traitlets.config.application import Application
except:
    # IPython < 4.x
    from IPython.config.application import Application
# jupyter ipywidgets module
try:
    import ipywidgets
except:
    pass
# bokeh modules
from bokeh.models import ColumnDataSource
from bokeh.models import CustomJS
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.tools import HoverTool
from bokeh.palettes import Plasma256, Viridis256, Inferno256, Greys256
from bokeh.io import show, output_notebook, push_notebook, reset_output
from bokeh.plotting import figure, Figure
from bokeh.resources import INLINE
bokeh_redirected = False
# tango module
try:
    import tango
except:
    import PyTango as tango
# flyscan modules
from fs.core import configuration as fs_configuration
from fs.core.parameters import Type as fs_param_type
from fs.utils.task import Task
from fs.utils.errors import silent_catch
from fs.utils.logging.LogAdapter import NotebookCellLogger
from fs.client.utils import spawn_atk, spawn_fs_gui_rw
from fs.client.jupyter.plots import ScaleType, Scale, BoxSelectionManager
from fs.client.jupyter.monitors import TangoDataSource, SpectrumChannel, ImageChannel, LayoutChannel
from fs.client.jupyter.monitors import DataStream, DataStreamer, DataStreamerController


# ------------------------------------------------------------------------------
fs_logger = logging.getLogger('fs.client.jupyter.notebook')


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
def reset_bokeh_output():
    global bokeh_redirected
    bokeh_redirected = False
    reset_output()


# ------------------------------------------------------------------------------
def load_ipython_extension(shell):
    pass


# ------------------------------------------------------------------------------
def unload_ipython_extension(shell):
    global bokeh_redirected
    bokeh_redirected = False


# ------------------------------------------------------------------------------
def enum(*sequential):
    enums = dict(zip(sequential, range(len(sequential))))
    enums['len'] = len(sequential)
    return type('Enum', (), enums)


# ------------------------------------------------------------------------------
class NotebookCellContent(object):
    def __init__(self):
        pass

    def clear_output(self):
        clear_output()

    def print(self, some_txt):
        print(some_txt)


# ------------------------------------------------------------------------------
class JnbWidgetsHelper(object):
    """Jupyter notebook widgets helper functions"""

    @classmethod
    def l01a(cls, width='auto', *args, **kwargs):
        return ipywidgets.Layout(flex='0 1 auto', width=width, *args, **kwargs)

    @classmethod
    def l11a(cls, width='auto', *args, **kwargs):
        return ipywidgets.Layout(flex='1 1 auto', width=width, *args, **kwargs)

    @classmethod
    def l21a(cls, width='auto', *args, **kwargs):
        return ipywidgets.Layout(flex='2 1 auto', width=width, *args, **kwargs)

    @classmethod
    def optimize_num_columns(cls, num_items, max_column):
        if num_items <= 1:
            num_columns = 1
        elif num_items <= 6 and not num_items % 2:
            num_columns = 2
        elif num_items <= 12 and not num_items % 3:
            num_columns = 3
        else:
            num_columns = 4
        return int(num_columns) if num_columns <= max_column else int(max_column)

    @classmethod
    def linear_partition(cls, seq, k):
        if k <= 0:
            return []
        n = len(seq) - 1
        if k > n:
            return map(lambda x: [x], seq)
        table, solution = JnbWidgetsHelper.__linear_partition_table(seq, k)
        k, ans = k - 2, []
        while k >= 0:
            ans = [[seq[i] for i in range(solution[n - 1][k] + 1, n + 1)]] + ans
            n, k = solution[n - 1][k], k - 1
        return [[seq[i] for i in range(0, n + 1)]] + ans

    @classmethod
    def __linear_partition_table(cls, seq, k):
        n = len(seq)
        table = [[0] * k for _ in range(n)]
        solution = [[0] * (k - 1) for _ in range(n - 1)]
        for i in range(n):
            table[i][0] = seq[i] + (table[i - 1][0] if i else 0)
        for j in range(k):
            table[0][j] = seq[0]
        for i in range(1, n):
            for j in range(1, k):
                table[i][j], solution[i - 1][j - 1] = min(
                    ((max(table[x][j - 1], table[i][0] - table[x][0]), x) for x in range(i)),
                    key=itemgetter(0))
        return table, solution

    @classmethod
    def device_state_to_widget_style(cls, s):
        if s in [tango.DevState.STANDBY, tango.DevState.ON]:
            return 'warning'
        if s in [tango.DevState.RUNNING]:
            return 'success'
        if s in [tango.DevState.MOVING]:
            return 'primary'
        if s in [tango.DevState.FAULT, tango.DevState.UNKNOWN]:
            return 'danger'
        if s in [tango.DevState.ALARM]:
            return 'danger'
        return ''

    @classmethod
    def device_state_to_widget_color(cls, s):
        if s in [tango.DevState.INIT]:
            return '#CCCC7A'
        if s in [tango.DevState.STANDBY]:
            return '#FFFF00'
        if s in [tango.DevState.ON, tango.DevState.OPEN, tango.DevState.EXTRACT]:
            return '#00FF00'
        if s in [tango.DevState.OFF, tango.DevState.CLOSE, tango.DevState.INSERT]:
            return '#FFFFFF'
        if s in [tango.DevState.MOVING]:
            return '#80A0FF'
        if s in [tango.DevState.RUNNING]:
            return '#228B22' #008000
        if s in [tango.DevState.FAULT]:
            return '#FF0000'
        if s in [tango.DevState.UNKNOWN]:
            return '#808080'
        if s in [tango.DevState.ALARM]:
            return '#FF8C00'
        if s in [tango.DevState.DISABLE]:
            return '#FF00FF'
        return '#808080'

    @classmethod
    def device_state_to_button_color(cls, b, s):
        b.style.button_color = JnbWidgetsHelper.device_state_to_widget_color(s)

    line_colors = {
        0: 'darkblue',
        1: 'crimson',
        2: 'darkgreen',
        3: 'black',
        4: 'darkorchid',
        5: 'darkorange',
        6: 'deepskyblue',
        7: 'slategrey',
        8: 'gold',
        9: 'magenta'
    }

    @classmethod
    def line_color(cls, index):
        i = index % 10
        return JnbWidgetsHelper.line_colors[i]

    @classmethod
    def plot_style(cls, instance, index):
        assert (isinstance(instance, Figure))
        i = index % 3
        if i == 0:
            return instance.circle
        if i == 1:
            return instance.square
        if i == 2:
            return instance.diamond
        return instance.square

    @classmethod
    def return_true(cls, *args):
        return True

    @classmethod
    def return_false(cls, *args):
        return False

    @classmethod
    def return_none(cls, *args):
        return None

    @classmethod
    def do_nothing(cls, *args):
        pass


# ------------------------------------------------------------------------------
jwh = JnbWidgetsHelper


# ------------------------------------------------------------------------------
class AsyncCallerDecorator(object):
    def __init__(self, instance, interval=None):
        self._instance = instance
        self._interval = interval
        self._hdl = None

    def __call__(self, func):
        self.wrapped = func
        self._schedule_next_call(first_call=True)
        return None

    def _schedule_next_call(self, first_call=False, dt=0.):
        loop = IOLoop.current()
        if not self._interval or first_call:
            delay = 0.01
        else:
            if dt > self._interval:
                dt *= 1.25
            delay = max(0.1, self._interval)
            delay = max(dt, self._interval)
            delay = min(3600, delay)
        self._hdl = loop.call_later(delay, self.iterate)

    def interval_changed(self, interval):
        self._interval = interval
        if self._hdl:
            IOLoop.current().remove_timeout(self._hdl)
        self._schedule_next_call()

    def iterate(self):
        if self._instance.stopped:
            return
        dt = 0.
        try:
            t0 = time.time()
            self.wrapped()
            dt = time.time() - t0
        except:
            pass
        if not self._instance.stopped:
            self._schedule_next_call(dt=dt)


# ------------------------------------------------------------------------------
class PeriodicAsyncCaller(object):
    def __init__(self, refresh_period=None, owner_callback=None):
        self._stopped = False
        self._frozen = False
        self._caller = None
        self._owner_callback = owner_callback
        self._refresh_period = refresh_period

    def freeze(self):
        # print("PeriodicAsyncCaller: freeze called")
        self._frozen = True

    def unfreeze(self):
        # print("PeriodicAsyncCaller: unfreeze called")
        self._frozen = False

    @property
    def refresh_period(self):
        return self._refresh_period

    @refresh_period.setter
    def refresh_period(self, refresh_period):
        self._refresh_period = refresh_period
        if self._caller:
            self._caller.interval_changed(self._refresh_period)

    @property
    def frozen(self):
        return self._frozen

    @property
    def stopped(self):
        return self._stopped

    def stop(self, *args, **kwargs):
        # print("PeriodicAsyncCaller: stop called")
        self._stopped = True
        if self._owner_callback:
            self._owner_callback()

    def spawn_async_call(self, callback):
        @self.__loop()
        def iterate():
            try:
                if not self._frozen:
                    callback()
            except Exception as e:
                print(e)

    def __loop(self):
        self._caller = AsyncCallerDecorator(self, interval=self._refresh_period)
        return self._caller


# ------------------------------------------------------------------------------
class ScanDataMonitor(NotebookCellContent):
    """ Base class of the scan data monitors """

    def __init__(self):
        # init super class
        NotebookCellContent.__init__(self)
        # generate uid
        self._uid = uuid4().int
        # belongs to common (i.e. shared data stream)?
        self._ucds = False
        # data stream & data streamer & controller
        self._dts = None
        self._dsr = None
        self._dsc = None
        # associated channel
        self._channels = None

    @property
    def data_stream(self):
        return self._dts

    @property
    def data_streamer(self):
        return self._dsr

    @property
    def data_streamer_ctrl(self):
        return self._dsc

    @property
    def channels(self):
        return self._channels

    def belongs_to_common_data_stream(self):
        return self._ucds

    def _clear(self):
        self._dts = None
        self._dsr = None
        self._dsc = None
        self._channels = None

    def setup_data_stream(self, ds_channels, **kwargs):
        self._ucds = kwargs.get('use_common_datastream', False)
        up = kwargs.get('refresh_period', 1.0)
        if self._ucds:
            self._channels = ds_channels
        else:
            try:
                # DataStream
                self._dts = DataStream(str(self._uid), channels=ds_channels)
                # DataStreamer
                self._dsr = DataStreamer(str(self._uid),
                                         data_streams=[self._dts],
                                         update_period=up,
                                         auto_start=True,
                                         start_delay=0.0)
                # DataStreamerControllers
                # self._dsc = DataStreamerController(str(self._uid), self._dsr)
            except Exception as e:
                print(e)
                raise

    @property
    def __data_streamer(self):
        dsr = None
        if self._dsc:
            dsr = self._dsc.data_streamer
        elif self._dsr:
            dsr = self._dsr
        return dsr

    def delayed_start(self, tmo=10.):
        self.start(tmo)

    def start(self, tmo=0.):
        self.__data_streamer.start(tmo)

    def stop(self):
        self.__data_streamer.stop()

    def close(self):
        self.__data_streamer.close()
        self._clear()


# ------------------------------------------------------------------------------
class SpectrumsMonitor(ScanDataMonitor):
    """ Tango spectrum attributes monitor (multiple-attribute live plot) """

    def __init__(self, **kwargs):
        # init super class
        ScanDataMonitor.__init__(self)
        # be sure we have something to display
        if 'channels' not in kwargs:
            raise Exception("failed to create SpectrumAttributesMonitor: 'channels' argument is missing!")
        if not isinstance(kwargs['channels'], dict):
            raise Exception(
                "failed to create SpectrumAttributesMonitor: invalid 'channels' argument specified [dict expected]")
        if not len(kwargs['channels']):
            raise Exception("failed to create SpectrumAttributesMonitor: unexpected empty dict")
        # data sources
        data_sources = list()
        for dsn, fqan in iteritems(kwargs['channels']):
            #print('SpectrumsMonitor: adding TangoDataSource {}:{}'.format(dsn, fqan))
            data_source = TangoDataSource(dsn, fqan)
            data_sources.append(data_source)
        # model properties
        mp = dict()
        mp['channel_title'] = kwargs.get('channel_title', None)
        mp['show_channel_title'] = kwargs.get('show_channel_title', False)
        mp['merge_tools'] = True
        mp['show_legend'] = True
        mp['show_title'] = False
        mp['x_scale'] = kwargs.get('x_scale', Scale())
        mp['y_scale'] = kwargs.get('y_scale', Scale())
        # setup data stream
        try:
            sc = SpectrumChannel(str(self._uid), data_sources, mp)
            self.setup_data_stream(sc, **kwargs)
        except Exception as e:
            print(e)


# ------------------------------------------------------------------------------
class ImagesMonitor(ScanDataMonitor):
    """ Tango image attributes monitor (multiple-attribute live plot) """

    def __init__(self, **kwargs):
        # init super class
        ScanDataMonitor.__init__(self)
        # be sure we have something to display
        if 'channels' not in kwargs:
            raise Exception("display: no channels specified!")
        if not isinstance(kwargs['channels'], dict):
            raise Exception("display: invalid argument specified [dict expected]")
        if not len(kwargs['channels']):
            raise Exception("display: unexpected empty dict")
        # models properties
        size = kwargs.get('size', None)
        if size is None or not isinstance(size, (list, tuple)) or len(size) != 2:
            size = [300, 300]
        scb = kwargs.get('selection_changed_callback', None)
        rcb = kwargs.get('reset_selection_callback', None)
        full_frame_shape = None
        sd = kwargs.get('scan_dimensions', None)
        if sd is not None and isinstance(sd, (tuple, list)) and len(sd) >= 2 and all(sd):
            full_frame_shape = (sd[1], sd[0])
        mp = dict()
        mp['images_size_threshold'] = kwargs['images_size_threshold']
        mp['full_frame_shape'] = full_frame_shape
        mp['channel_title'] = kwargs.get('channel_title', None)
        mp['show_channel_title'] = kwargs.get('show_channel_title', False)
        mp['layout'] = kwargs.get('layout', 'grid')
        mp['width'] = size[0]
        mp['height'] = size[1]
        mp['merge_tools'] = True
        mp['show_legend'] = False
        mp['palette'] = kwargs.get('palette', Plasma256)
        mp['x_scale'] = kwargs.get('x_scale', Scale())
        mp['y_scale'] = kwargs.get('y_scale', Scale())
        if scb or rcb:
            mp['selection_manager'] = BoxSelectionManager(selection_callback=scb, reset_callback=rcb)
        # data sources & image channels (one by image to monitor)
        image_channels = list()
        for dsn, fqan in iteritems(kwargs['channels']):
            #print('ImagesMonitor: adding TangoDataSource {}:{}'.format(dsn, fqan))
            data_source = TangoDataSource(dsn, fqan)
            image_channel = ImageChannel(dsn, data_source=data_source)
            image_channels.append(image_channel)
        # setup data stream
        try:
            lc = LayoutChannel(str(self._uid), channels=image_channels, model_properties=mp)
            self.setup_data_stream(lc, **kwargs)
        except Exception as e:
            print(e)
            raise

# ------------------------------------------------------------------------------
tango_attribute_plots = dict()


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
        raise Exception(
            "invalid attribute name specified - expected an alias or something like 'fully/qualified/attribute/name'")
    kwargs = dict()
    kwargs['webgl'] = True
    kwargs['tools'] = 'pan,wheel_zoom,box_select,resize,reset,hover'
    kwargs['title'] = fqan + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if ns.width is not None:
        kwargs['plot_width'] = ns.width
    if ns.height is not None:
        kwargs['plot_height'] = ns.height
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
        plot = figure(x_range=(0, av.dim_x), y_range=(0, av.dim_y), **kwargs)
        plot.image(image=[av.value], x=0, y=0, dw=av.dim_x, dh=av.dim_y, color_mapper=lcm)
    else:
        print(fqan + " has an unknown/unsupported attribute data format [{}]".format(str(av.data_format)))
    if plot:
        ht = plot.select(HoverTool)[0]
        ht.tooltips = [("index", "$index"), ("(x,y)", "(@x, @y)")]
        plot.toolbar.active_drag = None
        plot.toolbar.active_scroll = None
        plot.toolbar.logo = None
        tango_attribute_plots[fqan] = show(plot, notebook_handle=True)


# ------------------------------------------------------------------------------
# repository for scans launched from the notebook
#       - key: flyscan configuration name
#       - value: an instance of JnbScanInterface
jnfsi = dict()


# ------------------------------------------------------------------------------
def open_notebook_scan_interface(fss, cfg_name):
    try:
        cfg = fss.get_valid_configuration(cfg_name)
    except Exception as e:
        fs_logger.error(e)
        return
    if not cfg.has_easy_config():
        fs_logger.error("sorry, but the 'notebook flyscan interface' requires the configuration have an easy-config section")
        return
    # instanciate the JnbScanInterface
    try:
        nbs = jnfsi[cfg.name]
    except KeyError:
        nbs = None
    if nbs:
        # fs_logger.debug("JnbScanInterface[{}] exist".format(cfg.name))
        try:
            # fs_logger.debug("asking JnbScanInterface[{}] to exit...".format(cfg.name))
            nbs.close()
            # fs_logger.debug("JnbScanInterface[{}] exited successfully".format(cfg.name))
        except:
            # fs_logger.debug("failed to kill existing JnbScanInterface[{}]".format(cfg.name))
            pass
            # instanciate the JnbScanInterface
            # fs_logger.debug("instanciating JnbScanInterface[{}]".format(cfg_name))
    jnfsi[cfg_name] = JnbScanInterface(fss, cfg)
    # start the notebook scan interface
    # fs_logger.debug("displaying JnbScanInterface[{}]".format(cfg_name))
    jnfsi[cfg_name].show()
    # fs_logger.debug("JnbScanInterface[{}] successfully instanciated".format(cfg_name))


# ------------------------------------------------------------------------------
def open_notebook_fss_monitor(fss):
    # instanciate the JnbScanMonitor
    # fs_logger.debug("instanciating JnbScanMonitor")
    nbsm = JnbScanMonitor(fss)
    # fs_logger.debug("displaying JnbScanMonitor")
    nbsm.show()
    # fs_logger.debug("JnbScanMonitor successfully instanciated")


# ------------------------------------------------------------------------------
class JnbScanMonitor(Task, NotebookCellContent):
    """  a flyScan monitor for the Jupyter notebook"""

    # -----------------------------------
    def __init__(self, fss):
        # init mother classes
        NotebookCellContent.__init__(self)
        Task.__init__(self, "JnbScanMonitor")
        # the attached flyscan configuration
        self._fss = fss
        # logger
        self._logger = NotebookCellLogger(logging.getLogger('fs.client.jupyter'))
        # fss monitor widgets
        self._fss_monitor = JnbFssMonitorWidgets(self._fss, True, self._logger)
        self._fss_monitor.set_stop_handler(self.on_stop_clicked)
        self._fss_monitor.set_close_interface_handler(self.on_close_interface_clicked)
        # mains widgets
        self._main_layout = None
        self.__setup_main_layout()

    # -----------------------------------
    def on_stop_clicked(self):
        try:
            self._logger.print_in_cell("> aborting scan...")
            self._fss.abort(sync=True)
        except Exception as e:
            self._logger.error("oops, an error occurred while trying to stop the scan")
            self._logger.error(e)
    # -----------------------------------
    def on_close_interface_clicked(self):
        self.close()

    # -----------------------------------
    def __setup_main_layout(self):
        self._main_layout = self._fss_monitor.main_widgets_layout

    # -----------------------------------
    def show(self):
        # display the widgets
        if self._main_layout:
            display(self._main_layout)
            # start the underlying task (async. activity)
            if not self.is_alive():
                self.start()
            self.enable_periodic_message(period_in_sec=1.0)

    # -----------------------------------
    def close(self):
        try:
            # disable periodic activity
            self.disable_periodic_message()
            # ask the task to exit BEFORE closing the widgets
            self.exit()
            # close then exit
            if self._main_layout:
                self._main_layout.close()
            # clear cell output
            self.clear_output()
        except:
            pass

    # -----------------------------------
    def on_init(self):
        """
        This is called at Task startup in the context of the underlying thread.
        """
        # self._logger.debug("JnbScanMonitor: on_init called!")
        pass

    # -----------------------------------
    def on_exit(self):
        """
        This is called at Task exit in the context of the underlying thread.
        """
        # self._logger.debug("JnbScanMonitor: on_exit called!")
        pass

    # -----------------------------------
    def handle_message(self, msg):
        """
        This is called upon receipt of a new message in the context of the underlying thread.
        """
        # self._logger.debug("JnbScanMonitor: handle_message called!".format(self.task_name))
        pass

    # -----------------------------------
    def handle_periodic_message(self):
        """
        This is called upon receipt of a periodic message in the context of the underlying thread.
        """
        # self._logger.debug("JnbScanMonitor: received a periodic msg".format(self.task_name))
        self._fss_monitor.update()


# ------------------------------------------------------------------------------
class TextAreaLogger(object):
    """TextArea widget logger"""

    def __init__(self, text_area):
        self._text_area = text_area

    def debug(self, msg, *args, **kwargs):
        self.__log(msg)

    def info(self, msg, *args, **kwargs):
        self.__log(msg)

    def warning(self, msg, *args, **kwargs):
        self.__log(msg)

    def error(self, msg, *args, **kwargs):
        self.__log(msg)

    def critical(self, msg, *args, **kwargs):
        self.__log(msg)

    def exception(self, msg, *args, **kwargs):
        self.__log(msg)

    def print_in_cell(self, msg):
        self.__log(msg, True)

    def __log(self, msg, clear=False):
        if not clear:
            self._text_area.value += "\n{}".format(msg)
        else:
            self._text_area.value = msg
        self._text_area.rows = min(10, 1 + self._text_area.value.count('\n'))


# ------------------------------------------------------------------------------
class JnbScanInterface(NotebookCellContent):
    """ data structure representing a scan interface in a Jupyter notebook"""

    # -----------------------------------
    def __init__(self, fss, cfg):
        # init mother classes
        NotebookCellContent.__init__(self)
        # async. caller
        self._async_caller = None
        # the attached flyscan server proxy
        self._fss = fss
        # the attached flyscan configuration
        self._cfg = cfg
        # running in "fss monitor" mode?
        self._fss_monitor_mode = False if cfg else True
        # logger
        self._log_widget = None
        self.__setup_log_widget()
        self._logger = TextAreaLogger(self._log_widget)
        # setup widgets for easy-config parameters
        self._params_widgets = dict()
        self._params_layout = None
        self.__setup_parameters_widgets()
        # setup widgets for sensors selection
        self._num_sensors_selected = 0
        self._sensors_layout = None
        self.__setup_sensors_widgets()
        # options widgets
        self._options = dict()
        self._option_layout = None
        self.__setup_options_widgets()
        # fss monitor widgets (rework needed on handlers)
        self._fss_monitor = JnbFssMonitorWidgets(self._fss, False, self._logger)
        self._fss_monitor.set_start_handler(self.on_start_clicked)
        self._fss_monitor.set_stop_handler(self.on_stop_clicked)
        self._fss_monitor.set_save_handler(self.on_save_clicked)
        self._fss_monitor.set_close_interface_handler(self.on_close_interface_clicked)
        self._fss_monitor.set_open_close_plots_handler(self.on_open_close_plots_clicked)
        self._fss_monitor.set_end_of_scan_handler(self.on_end_of_scan)
        # mains widgets
        self._main_layout = None
        self._tab_widget = None
        self.__setup_main_layout()
        # spectrums and images monitors
        sccb = self.on_selection_change
        srcb = self.on_selection_reset
        self._monitors = JnbScanDataMonitors(fss,  cfg, sccb, srcb, self._logger)

    # -----------------------------------
    def __setup_log_widget(self):
        v = "flyScan: {} interface".format(self._cfg.name)
        self._log_widget = ipywidgets.Textarea(value=v, layout=jwh.l11a())
        self._log_widget.rows = 1

    # -----------------------------------
    def __filter_parameters(self):
        #  we want the widgets to appear in the easy-config declaration order
        specs = OrderedDict()
        if self._cfg.easy_config.plugin:
            specs.update(self._cfg.easy_config.plugin._declarations)
        specs.update(self._cfg.easy_config._specifications)
        #  here the idea is to separate json parameters from the others
        json_params = OrderedDict()
        non_json_params = OrderedDict()
        for spec_name, spec in iteritems(specs):
            dec = None
            try:
                dec = self._cfg.easy_config_prop._declarations[spec_name]
            except:
                pass
            is_json = dec and dec.type == fs_param_type.JSON
            d = json_params if is_json else non_json_params
            d[spec_name] = spec
        return json_params, non_json_params

    # -----------------------------------
    def __setup_parameters_widgets(self):
        # separate json parameters from the others
        jp, njp = self.__filter_parameters()
        layouts = list()
        njp_widgets = self.__setup_non_json_parameters_widgets(njp)
        if njp_widgets:
            layouts.append(njp_widgets)
        jp_widgets = self.__setup_json_parameters_widgets(jp)
        if jp_widgets:
            layouts.append(jp_widgets)
        self._params_layout = ipywidgets.VBox(layouts, layout=jwh.l11a())

    # -----------------------------------
    def __setup_non_json_parameters_widgets(self, params):
        # any non-json param?
        if not len(params):
            return None
        # num of columns in the widgets layout
        num_columns = jwh.optimize_num_columns(len(params.keys()), 4)
        # create each widget and add it to the right column
        ci = 0
        cl = [list() for _ in range(num_columns)]
        for spec_name, spec in iteritems(params):
            # instanciate the widget
            desc = str(spec_name)
            with silent_catch():
                desc += ' in {}'.format(spec.unit)
            phd = str()
            with silent_catch():
                phd = spec.description
            val = str()
            with silent_catch():
                val = str(spec.value)
            # label + valid in a hbox
            lw = ipywidgets.Label(desc + ":", layout=jwh.l01a())
            vl = ipywidgets.Valid(value=True, layout=jwh.l01a())
            hb = ipywidgets.HBox([lw, vl])
            cl[ci].append(hb)
            # instanciate widget
            tw = ipywidgets.Text(value=val, placeholder=phd, layout=jwh.l11a())
            cl[ci].append(tw)
            # attach the parameter name to its text widget  (see on_text_submitted)
            setattr(tw, "param", spec_name)
            # attach the valid widget to its text widget  (see on_text_submitted)
            setattr(tw, "valid", vl)
            # attach callback
            tw.on_submit(self.on_text_submitted)
            # add text widget to global list
            self._params_widgets[spec_name] = tw
            # next column
            ci = (ci + 1) % num_columns
        # trick to obtain the expected alignment adding some hidden widgets
        lc0 = len(cl[0])
        for c in cl[1:num_columns]:
            d = lc0 - len(c)
            wl = [ipywidgets.Label("", visible=False, disabled=True) for _ in range(d)]
            c.extend(wl)
        # build widgets layout
        w = "{:.2f}%".format(100. / len(cl))
        vbl = [ipywidgets.VBox(c, layout=jwh.l11a(width=w)) for c in cl]
        hb = ipywidgets.HBox(vbl, layout=jwh.l11a())
        return hb

    # -----------------------------------
    def __setup_json_parameters_widgets(self, params):
        # any json param?
        if not len(params):
            return None
        # num of columns in the widgets layout
        num_columns = jwh.optimize_num_columns(len(params.keys()), 2)
        # create each widget and add it to the right column
        ci = 0
        cl = [list() for _ in range(num_columns)]
        for spec_name, spec in iteritems(params):
            # instanciate the widget
            desc = str(spec_name)
            with silent_catch():
                desc += ' in {}'.format(spec.unit)
            phd = str()
            with silent_catch():
                phd = spec.description
            val = str()
            with silent_catch():
                val = str(spec.value)
            # label + valid in a hbox
            lw = ipywidgets.Label(desc + ":", layout=jwh.l01a())
            vl = ipywidgets.Valid(value=True, layout=jwh.l01a())
            cl[ci].append(ipywidgets.HBox([lw, vl]))
            # instanciate widget
            i = len(spec.value)
            s = "{ "
            for k, v in iteritems(spec.value):
                s += "'" + str(k) + "':" + str(v)
                i -= 1
                if i:
                    s += " , "
            s += " }"
            tw = ipywidgets.Textarea(value=s, placeholder=phd, layout=jwh.l11a())
            tw.rows = 1
            cl[ci].append(tw)
            # attach the parameter name to its text widget  (see on_text_submitted)
            setattr(tw, "param", spec_name)
            # attach the valid widget to its text widget  (see on_text_submitted)
            setattr(tw, "valid", vl)
            # add text widget to global list
            self._params_widgets[spec_name] = tw
            # next column
            ci = (ci + 1) % num_columns
        # build widgets layout
        vbl = [ipywidgets.VBox(c, layout=jwh.l11a()) for c in cl]
        hb = ipywidgets.HBox(vbl, layout=jwh.l11a())
        return hb

    # -----------------------------------
    def __enable_parameters_widgets(self):
        for pw in self._params_widgets.values():
            pw.disabled = False

    # -----------------------------------
    def __disable_parameters_widgets(self):
        for pw in self._params_widgets.values():
            pw.disabled = True

    # -----------------------------------
    def __validate_parameter_values(self):
        self._logger.print_in_cell("> validating scan parameters...")
        params_ok = True
        for pw in self._params_widgets.values():
            try:
                # print("validating {} with pv={}".format(pw.param, pw.value))
                ok = self.__validate_parameter_value(pw.param, pw.value, pw.valid)
                params_ok = ok if params_ok else params_ok
            except Exception:
                params_ok = False
        if params_ok:
            self._logger.info("scan parameters validated")
        return params_ok

    # -----------------------------------
    def __validate_parameter_value(self, pn, pv, vw):
        try:
            if not pv or not len(pv):
                why = ["no value specified!"]
                raise ValueError("oops")
            try:
                known_param = pn in self._cfg.easy_config_prop._declarations.keys()
                is_str = known_param and self._cfg.easy_config_prop._declarations[pn].type == fs_param_type.STR
                f = eval if pn in self._cfg.easy_config_prop._declarations.keys() and not is_str else str
                evaluated_pv = f(pv)
            except Exception:
                why = ["invalid syntax detected in parameter value '{}'".format(pv)]
                why.append("this might be due to a missing '[' or ']' bracket in a range specification? "
                           + "an invalid character? "
                           + "a copy/paste error?")
                raise ValueError("oops")
            param_ok, why = self._cfg.easy_config._check_specification(pn, evaluated_pv)
            if not param_ok:
                raise ValueError("oops")
            try:
                self._cfg.easy_config._set_specification(pn, f(pv))
            except Exception:
                why = ["parameter value '{}' has been unexpectedly rejected! [might be a bug in the fs API]".format(pv)]
                raise ValueError("oops")
        except ValueError:
            self._logger.error("- oops, the specified '{}' value is invalid:".format(pn))
            self._logger.error("\n".join(why))
            param_ok = False
        except Exception as e:
            self._logger.error("- oops, the specified '{}' value is invalid:".format(pn))
            self._logger.error(e)
            param_ok = False
        vw.value = param_ok
        return param_ok

    # -----------------------------------
    def on_text_submitted(self, tw):
        try:
            #print("on_text_submitted called for '{}' = {} - disabled:{}".format(tw.param, tw.value, tw.disabled))
            self.__validate_parameter_values()
        except Exception as e:
            self._logger.error(e)

    # -----------------------------------
    def __setup_sensors_widgets(self):
        # checkbox widgets
        sd = self._cfg.sensors.__dict__
        sensors = OrderedDict(sorted(sd.items(), key=lambda x: x[0], reverse=False))
        # create each widget and add it to the right column
        num_columns = jwh.optimize_num_columns(len(sensors.keys()), 4)
        ci = 0
        cl = [list() for _ in range(num_columns)]
        for sn, si in iteritems(sensors):
            self._num_sensors_selected += 1 if si.enable else 0
            label = sn + " [" + str(si.plugin_name) + "]"
            lb = ipywidgets.Label(label, layout=jwh.l11a())
            cb = ipywidgets.Checkbox(value=si.enable, layout=jwh.l01a())
            cb.observe(self.on_sensor_checkbox_clicked, names='value')
            setattr(cb, 'sensor', si)
            hb = ipywidgets.HBox([cb, lb], layout=jwh.l11a())
            cl[ci].append(hb)
            ci = (ci + 1) % num_columns
        lc0 = len(cl[0])
        for c in cl[1:num_columns]:
            d = lc0 - len(c)
            wl = [ipywidgets.Label("", visible=False, disabled=True) for _ in range(d)]
            c.extend(wl)
        # layout
        vbl = [ipywidgets.VBox(c, layout=jwh.l11a()) for c in cl]
        vh1 = ipywidgets.HBox(vbl, layout=jwh.l11a())
        self._sensors_layout = vh1

    # -----------------------------------
    def on_sensor_checkbox_clicked(self, change):
        # print("on_sensor_checkbox_clicked: {} {}".format(change['old'], change['new']))
        nv = change['new']
        self._num_sensors_selected += 1 if nv else -1
        change['owner'].sensor.enable = nv

    # -----------------------------------
    def __setup_options_widgets(self):
        # ----------
        self._options['live_visualization'] = True
        lb = ipywidgets.Label("live data visualization", layout=jwh.l11a())
        lv = ipywidgets.Checkbox(value=True, layout=jwh.l01a())
        lv.observe(self.on_option_checkbox_clicked, names='value')
        setattr(lv, "option", 'live_visualization')
        hb1 = ipywidgets.HBox([lv, lb], layout=jwh.l11a())
        # ----------
        self._options['auto_apply_selection'] = True
        lb = ipywidgets.Label("auto apply 'box selection'", layout=jwh.l11a())
        lv = ipywidgets.Checkbox(value=True, layout=jwh.l01a())
        lv.observe(self.on_option_checkbox_clicked, names='value')
        setattr(lv, "option", 'auto_apply_selection')
        hb2 = ipywidgets.HBox([lv, lb], layout=jwh.l11a())
        # ----------
        self._option_layout = ipywidgets.HBox([hb1, hb2], layout=jwh.l11a())

    # -----------------------------------
    def on_option_checkbox_clicked(self, change):
        # print("on_option_checkbox_clicked: {} {}".format(change['old'], change['new']))
        self._options[change['owner'].option] = change['new']

    # -----------------------------------
    def on_start_clicked(self):
        i = 0
        if self._fss.is_scanning:
            self._logger.print_in_cell("> checking flyScan server state...")
            self._logger.warning("oops, can't launch scan, an acquisition is in progress on server side")
            raise Exception("action failed [start]")
        # close monitors
        self._monitors.close()
        # clear cell output
        self.clear_output()
        # switch to monitor tab
        self.__select_scan_monitor_tab()
        # validate scan parameters
        if not self.__validate_parameter_values():
            self.__select_scan_parameters_tab()
            raise Exception("action failed [start]")
        if not self._num_sensors_selected:
            self.__select_scan_sensors_tab()
            self._logger.error("oops, no sensor selected! please select (at least) one sensor then retry")
            raise Exception("action failed [start]")
        # load scan config on server side
        try:
            self._logger.print_in_cell("> uploading {} configuration on flyScan server...".format(self._cfg.name))
            self._fss.set_cfg(self._cfg)
            self._logger.info("scan configuration set")
        except Exception as e:
            self._logger.error("oops, failed to upload scan configuration on server side")
            self._logger.error(str(e))
            raise Exception("action failed [start]")
        # start the scan
        try:
            self._logger.print_in_cell("> starting scan...")
            self._fss.start(sync=False)
            scanning_states = (tango.DevState.RUNNING, tango.DevState.STANDBY)
            self._fss.wait_states(scanning_states, 15)
            self._logger.info("scan successfully started")
            self.__disable_parameters_widgets()
        except Exception as e:
            self._logger.error("oops, scan failed to start")
            self._logger.error(str(e))
            self.__enable_parameters_widgets()
            raise Exception("action failed [start]")
        # open scan data monitors
        try:
            if self._options['live_visualization']:
                self._monitors.open()
        except Exception as e:
            self._logger.error("oops, failed to display scan plots")
            self._logger.error(str(e))
            raise Exception("action failed [start]")
        finally:
            self._async_caller.refresh_period = 1.0

    # -----------------------------------
    def on_stop_clicked(self):
        self._logger.print_in_cell("> aborting scan...")
        try:
            if not self._fss.is_scanning:
                self._logger.warning("no scan running on server side [abort request ignored]")
                return
        except Exception as e:
            self._logger.error("oops, an error occurred while trying to stop the scan")
            self._logger.error(e)
            raise Exception("action failed [stop]")
        try:
            self._fss.sync_abort(print_func=self._logger.print_in_cell)
            self._logger.info("scan successfully aborted")
        except Exception as e:
            self._logger.error("oops, an error occurred while trying to stop the scan")
            self._logger.error(e)
            raise Exception("action failed [stop]")
        finally:
            self.on_end_of_scan()

    # -----------------------------------
    def on_save_clicked(self):
        self._logger.print_in_cell("> saving scan configuration '{}'...".format(self._cfg.name))
        try:
            pass
        except Exception as e:
            self._logger.error("oops, an error occurred while trying to save scan configuration")
            self._logger.error(e)
            raise Exception("action failed [stop]")
        try:
            self._fss.save_config(self._cfg, confirm=False)
            self._logger.info("scan configuration successfully saved")
        except Exception as e:
            self._logger.error("oops, an error occurred while trying to save scan configuration")
            self._logger.error(e)
            raise Exception("action failed [save scan configuration]")

    # -----------------------------------
    def on_end_of_scan(self):
        self._async_caller.refresh_period = 2.0
        self.__enable_parameters_widgets()
        self._monitors.stop()

    # -----------------------------------
    def on_close_interface_clicked(self):
        self.close()

    # -----------------------------------
    def on_open_close_plots_clicked(self):
        if self._monitors.has_open_plots():
            self._monitors.close()
            self.clear_output()
        else:
            self._monitors.open()
            if self._fss.is_scanning:
                self._monitors.start()

    # -----------------------------------
    def __select_scan_parameters_tab(self):
        self.__select_tab(0)

    # -----------------------------------
    def __select_scan_sensors_tab(self):
        self.__select_tab(1)

    # -----------------------------------
    def __select_scan_monitor_tab(self):
        self.__select_tab(2)

    # -----------------------------------
    def __select_tab(self, t):
        if self._tab_widget:
            self._tab_widget.selected_index = t

    # -----------------------------------
    def on_selection_change(self, rect):
        # rect is {'x0':x0, 'x1':x1, 'y0':y0, 'y1':y1, 'width':w, 'height':h}
        x0, x1, w = rect['x0'], rect['x1'], rect['width']
        y0, y1, h = rect['y0'], rect['y1'], rect['height']
        self._logger.print_in_cell("> selection changed:")
        self._logger.info("x: from {:.2f} to {:.2f} [width: {:.2f}]".format(x0, x1, w))
        self._logger.info("y: from {:.2f} to {:.2f} [height: {:.2f}]".format(y0, y1, h))
        if self._options['auto_apply_selection']:
            try:
                kwargs = dict()
                kwargs['rect'] = rect
                self.__call_easy_config_callback("on_scan_domain_change", **kwargs)
            except:
                pass

    # -----------------------------------
    def on_selection_reset(self):
        pass

    # -------------------- ---------------
    def __call_easy_config_callback(self, cbn, **kwargs):
        try:
            easy_cfg_func = self._cfg.get_easy_config_callback(cbn)
        except:
            self._logger.error("failed to import easy-config callback '{}'".format(cbn))
            return
        if not easy_cfg_func:
            return
        try:
            self._logger.info("> calling easy-config callback {}.{}:".format(self._cfg.name, cbn))
            changes = easy_cfg_func(self._cfg, **kwargs)
        except Exception as e:
            self._logger.error("{}.{} failed".format(self._cfg.name, cbn))
            self._logger.error(e)
            return
        if not changes:
            self._logger.warning("{}.{} returned None [might be a prog. error]".format(self._cfg.name, cbn))
        else:
            if not isinstance(changes, dict):
                self._logger.error("got invalid reply from easy-config callback {}.{}".format(self._cfg.name, cbn))
                self._logger.error("expected a 'dict' of easy-cfg-param:new-value got a '{}'".format(changes.__class__))
                return
            for pn, pv in iteritems(changes):
                try:
                    self._logger.error("{}.{} changed {} to {}".format(self._cfg.name, cbn, pn, pv))
                    self._params_widgets[pn].value = str(pv)
                except KeyError:
                    self._logger.error("oops, got invalid reply from {}.{}".format(self._cfg.name, cbn))
                    self._logger.error("'{}' is not a  valid easy-config parameter name".format(pn))
                except Exception as e:
                    self._logger.error("oops,failed to extract new value for parameter {}".format(pn))
                    self._logger.error(e)

    # -----------------------------------
    def __setup_main_layout(self):
        mdl = self._fss_monitor.main_widgets_layout
        sdfl = self._fss_monitor.scan_data_files_layout
        self._tab_widget = ipywidgets.Tab([self._params_layout, self._sensors_layout, mdl, sdfl, self._option_layout])
        self._tab_widget.set_title(0, "Scan Parameters")
        self._tab_widget.set_title(1, "Sensors Selection")
        self._tab_widget.set_title(2, "Scan Monitor")
        self._tab_widget.set_title(3, "Data Files")
        self._tab_widget.set_title(4, "Options")
        vb = ipywidgets.VBox([self._tab_widget, self._fss_monitor.scan_controls_layout, self._log_widget], layout=jwh.l11a())
        self._main_layout = ipywidgets.VBox([vb], layout=jwh.l11a())

    # -----------------------------------
    def show(self):
        # display the widgets
        if self._main_layout:
            self.clear_output()
            scanning = self._fss.is_scanning
            if scanning:
                self.__select_scan_monitor_tab()
            display(self._main_layout)
            if scanning and self._options['live_visualization']:
                self._monitors.open()
                self._monitors.start()
            self.__spawn_periodic_task()

    # -----------------------------------
    def __spawn_periodic_task(self):
        if not self._async_caller:
            self._async_caller = PeriodicAsyncCaller(refresh_period=1.0)
            self._async_caller.spawn_async_call(self.__periodic_task)

    # -----------------------------------
    def __periodic_task(self):
        try:
            self._fss_monitor.update()
        except:
            pass

    # -----------------------------------
    def clear_output(self):
        try:
            # close monitors
            self._monitors.close()
            # clear cell output
            super(JnbScanInterface, self).clear_output()
        except:
            pass

    # -----------------------------------
    def close(self):
        try:
            # stop periodic activity
            self._async_caller.stop()
            # close monitors
            self._monitors.close()
            # close main widgets layout
            if self._main_layout:
                self._main_layout.close()
            # clear cell output
            self.clear_output()
        except:
            pass
        finally:
            # remove instance from repository
            with silent_catch():
                del jnfsi[self._cfg.name]


# ------------------------------------------------------------------------------
class JnbScanDataMonitors(object):
    """ scan data monitors """

    # -----------------------------------
    def __init__(self, fss, cfg, scc=None, src=None, lgg=None):
        # flyscan server
        self._fss = fss
        # flyscan configuration
        self._cfg = cfg
        # logger
        self._logger = NotebookCellLogger(fs_logger) if not lgg else lgg
        # spectrums monitors
        self._spectrums_monitors = dict()
        # images monitors
        self._images_monitors = dict()
        # selection change callback
        self._selection_change_callback = scc
        # selection reset callback
        self._selection_reset_callback = src
        # common (i.e. shared) data stream (see ScanDataMonitor)
        self._common_data_stream = None

    # -----------------------------------
    def open(self):
        for a in self._cfg.actors:
            c1 = a.enable
            c2 = isinstance(a, fs_configuration.Hook)
            c3 = a.plugin_name == 'DisplayManager'
            if c1 and c2 and c3:
                self.__open_scan_data_monitor(a)
        self.__setup_common_data_stream()

    # -----------------------------------
    def __setup_common_data_stream(self):
        channels = list()
        for sm in self._spectrums_monitors.values():
            if sm.belongs_to_common_data_stream():
                channels.append(sm.channels)
        for im in self._images_monitors.values():
            if im.belongs_to_common_data_stream():
                channels.append(im.channels)
        if len(channels):
            try:
                kwargs = dict()
                kwargs['use_common_datastream'] = False
                kwargs['refresh_period'] = 1.0
                self._common_data_stream = ScanDataMonitor()
                self._common_data_stream.setup_data_stream(channels, **kwargs)
            except Exception as e:
                print(e)
                raise

    # -----------------------------------
    def __open_scan_data_monitor(self, display_manager):
        assert (isinstance(display_manager, fs_configuration.Hook))
        kwargs = dict()
        kwargs['actor'] = display_manager.name
        # title
        try:
            title = display_manager.parameters.title.value
        except AttributeError:
            title = "DisplayManager: {}".format(display_manager.name)
        kwargs['channel_title'] = title
        kwargs['show_channel_title'] = True
        # refresh period specification (in sec)
        try:
            refresh_period = float(display_manager.parameters.refresh_period.value)
        except AttributeError:
            refresh_period = 1.
        kwargs['refresh_period'] = refresh_period
        # refresh mode: all or one-by-one
        try:
            refresh_mode = display_manager.parameters.refresh_mode.value
        except AttributeError:
            refresh_mode = 'all'
        if refresh_mode not in ['all', 'one-by-one']:
            refresh_mode = 'all'
        kwargs['refresh_mode'] = refresh_mode
        # images organization on screen: tabs or grid
        try:
            images_layout = display_manager.parameters.images_layout.value
        except AttributeError:
            images_layout = dict()
        if 'layout' not in images_layout or images_layout['layout'] not in ['grid', 'tabs']:
            images_layout['layout'] = 'grid'
        else:
            images_layout['layout'] = images_layout['layout'].encode('UTF-8')
        if 'size' not in images_layout:
            images_layout['size'] = None
        kwargs['layout'] = images_layout
        # x scale specification
        try:
            x_scale = display_manager.parameters.x_scale.value
        except AttributeError:
            x_scale = None
        kwargs['x_scale'] = x_scale
        # y scale specification
        try:
            y_scale = display_manager.parameters.y_scale.value
        except AttributeError:
            y_scale = None
        kwargs['y_scale'] = y_scale
        # images size threshold
        try:
            kwargs['images_size_threshold'] = int(display_manager.parameters.images_size_threshold.value)
        except AttributeError:
            kwargs['images_size_threshold'] = int(100000) #TODO: don't hardcode default value
        # actual scan dimensions (i.e. post easy-config values)
        try:
            kwargs['scan_dimensions'] = self._fss.actual_scan_dimensions
        except:
            pass
        # spectrums to monitor
        try:
            spectrums = display_manager.parameters.spectrums.value
        except AttributeError:
            spectrums = None
        kwargs['spectrums'] = spectrums
        # images to monitor
        try:
            images = display_manager.parameters.images.value
        except AttributeError:
            images = None
        kwargs['images'] = images
        # something to display?
        if not spectrums and not images:
            return
        # open spectrums and images monitors
        if spectrums:
            self.__open_spectrums_monitor(**kwargs)
        if images:
            self.__open_images_monitor(**kwargs)

    # -----------------------------------
    def __open_spectrums_monitor(self, **kwargs):
        try:
            #print("open_spectrums_monitor: {}".format(kwargs))
            # --------------------------------------------------------------------------------------------------------
            # x_scale = {'label': 'x' and 'actuator': 'rz' or 'attribute': 'tango/device/xpos/data' or 'range':(0,100,1)}
            # spectrums = {'rz': 'flyscan/viewer/1/lsl_rz', 't1': 'flyscan/viewer/1/lsl_tt1dsca_sampling_time'}
            # --------------------------------------------------------------------------------------------------------
            # x_scale specification
            x_scale = Scale()
            x_scale_spec = kwargs.get('x_scale', None)
            if x_scale_spec:
                x_scale = self.__extract_scale_from_specification('x', x_scale_spec)
                # print(x_scale)
            y_scale = Scale()
            y_scale_spec = kwargs.get('y_scale', None)
            if y_scale_spec:
                y_scale = self.__extract_scale_from_specification('y', y_scale_spec)
                # print(y_scale)
            channels = OrderedDict()
            spectrums = kwargs['spectrums']
            if x_scale.type == ScaleType.CHANNEL:
                in_keys = x_scale.label in spectrums.keys()
                in_values = x_scale.channel in spectrums.values()
                is_key_of_value = False
                if in_values:
                    for k, v in iteritems(spectrums):
                        if v == x_scale.range and k == x_scale.label:
                            is_key_of_value = True
                            break
                is_value_of_key = in_keys and spectrums[x_scale.label] == x_scale.channel
                if (in_keys and not is_value_of_key) or (in_values and not is_key_of_value):
                    self._logger.warning(
                        "display: invalid display configuration detected in DisplayManager {}.{}".format(self._cfg.name,
                                                                                                         kwargs['actor']))
                    self._logger.warning(
                        "display: there's a conflict related to '{}' and '{}'".format(x_scale.label, x_scale.channel))
                    self._logger.warning("display: please, check both 'x_scale' and 'spectrums' parameters value")
                    return
                if not in_keys:
                    channels[x_scale.channel] = x_scale.channel
            for sname, sattr in iteritems(spectrums):
                channels[sname] = sattr
            # print(channels)
            spm_kwargs = dict()
            spm_kwargs['scan_dimensions'] = kwargs.get('scan_dimensions', None)
            spm_kwargs['channel_title'] = kwargs['channel_title']
            spm_kwargs['show_channel_title'] = kwargs['show_channel_title']
            spm_kwargs['channels'] = channels
            spm_kwargs['x_scale'] = x_scale
            spm_kwargs['y_scale'] = y_scale
            spm_kwargs['refresh_mode'] = kwargs['refresh_mode']
            spm_kwargs['refresh_period'] = kwargs['refresh_period']
            spm_kwargs['standalone_mode'] = False
            sm = SpectrumsMonitor(**spm_kwargs)
            self._spectrums_monitors[kwargs['actor']] = sm
        except Exception as e:
            # print(e)
            raise

    # -----------------------------------
    def __open_images_monitor(self, **kwargs):
        try:
            #print("open_images_monitor: {}".format(kwargs))
            # --------------------------------------------------------------------------------------------------------
            # x_scale = {'label': 'x' and 'actuator': 'rz' or 'attribute': 'tango/device/xpos/data' or 'range':(0,100,1)}
            # y_scale = {'label': 'y' and 'actuator': 'tx' or 'attribute': 'tango/device/ypos/data' or 'range':(0,100,1)}
            # images = {'img1-name': 'img-attr1', img2-name': 'img-attr2', 'img3-name': 'img-attr3'}
            # --------------------------------------------------------------------------------------------------------
            # x_scale specification
            x_scale = Scale()
            scale_spec = kwargs['x_scale']
            if scale_spec:
                x_scale = self.__extract_scale_from_specification('x', scale_spec)
                #print(x_scale)
                if x_scale.type == ScaleType.CHANNEL:
                    self._logger.warning("display: 'attribute' scales are not yet supported for images ['{}' ignored]".format(x_scale.channel))
            # y_scale specification
            y_scale = Scale()
            scale_spec = kwargs['y_scale']
            if scale_spec:
                y_scale = self.__extract_scale_from_specification('y', scale_spec)
                #print(y_scale)
                if y_scale.type == ScaleType.CHANNEL:
                    self._logger.warning("display: 'attribute' scales are not yet supported for images ['{}' ignored]".format(y_scale.channel))
            # build ImagesMonitor ctor parameter
            channels = OrderedDict()
            for img_name, img_attr in iteritems(kwargs['images']):
                channels[img_name] = img_attr
            im_kwargs = dict()
            im_kwargs['scan_dimensions'] = kwargs.get('scan_dimensions', None)
            im_kwargs['images_size_threshold'] = kwargs['images_size_threshold']
            im_kwargs['channel_title'] = kwargs['channel_title']
            im_kwargs['show_channel_title'] = kwargs['show_channel_title']
            im_kwargs['channels'] = channels
            im_kwargs['refresh_period'] = kwargs['refresh_period']
            im_kwargs['refresh_mode'] = kwargs['refresh_mode']
            im_kwargs['x_scale'] = x_scale
            im_kwargs['y_scale'] = y_scale
            im_kwargs['layout'] = kwargs['layout']['layout']
            im_kwargs['size'] = kwargs['layout']['size']
            im_kwargs['standalone_mode'] = False
            im_kwargs['selection_changed_callback'] = self._selection_change_callback
            im_kwargs['reset_selection_callback'] = self._selection_reset_callback
            im = ImagesMonitor(**im_kwargs)
            self._images_monitors[kwargs['actor']] = im
        except Exception as e:
            print(e)
            raise

    # -----------------------------------
    def __extract_scale_from_specification(self, label, scale_spec):
        scale_cfg = dict()
        scale_cfg['type'] = ScaleType.INDEXES
        scale_cfg['label'] = scale_spec.get('label', label)
        scale_cfg['unit'] = scale_spec.get('unit', '')
        scale_range = None
        if 'actuator' in scale_spec:
            # the scale is the trajectory of the specified actuator
            scale_cfg['type'] = ScaleType.RANGE
            scale_range = self.__extract_scale_from_actuator_trajectory(scale_spec['actuator'])
        elif 'range' in scale_spec:
            # the scale is a specified range
            scale_cfg['type'] = ScaleType.RANGE
            scale_range = self.__extract_scale_from_specified_range(scale_spec['range'])
        elif 'attribute' in scale_spec:
            # the scale is a spectrum attribute
            scale_cfg['type'] = ScaleType.CHANNEL
            scale_cfg['channel'] = self.__extract_scale_from_spectrum_attribute(scale_spec['attribute'])
        if scale_range:
            scale_cfg['start'] = float(scale_range[0])
            scale_cfg['end'] = float(scale_range[1])
            scale_cfg['num_points'] = float(scale_range[2])
        return Scale(**scale_cfg)


    # -----------------------------------
    def __extract_scale_from_actuator_trajectory(self, actuator):
        # TODO: so far we support only single range trajectories: we can build a global scale from ranges
        try:
            fs_range = eval("self._fss.current_cfg.actuators." + actuator + ".trajectory[0]")
            scale_range = (fs_range.start, fs_range.stop, fs_range.num)
        except:
            self._logger.warning("display: failed to obtain data scale from the specified actuator '{}'".format(actuator))
            self._logger.warning("display: check display configuration [is '{}' a valid actuator name?]".format(actuator))
            scale_range = None
        return scale_range

    # -----------------------------------
    def __extract_scale_from_specified_range(self, range_spec):
        try:
            scale_range = tuple(range_spec)
            if len(scale_range) != 3:
                raise Exception("oops")
        except:
            self._logger.warning("display: failed to extract data scale from the specified range '{}'".format(range_spec))
            self._logger.warning("display: expected something like 'range':(from, to, num_points)")
            scale_range = None
        return scale_range

    # -----------------------------------
    def __extract_scale_from_spectrum_attribute(self, range_spec):
        try:
            scale_range = str(range_spec)
        except:
            self._logger.warning("display: failed to extract data scale from the specified attribute '{}'".format(range_spec))
            self._logger.warning("display: expected an attribute name or alias")
            scale_range = None
        return scale_range

    # -----------------------------------
    def has_open_plots(self):
        return len(self._spectrums_monitors) or len(self._images_monitors)

    # -----------------------------------
    def start(self, tmo=0.):
        if self._common_data_stream:
            try:
                self._common_data_stream.start()
            except Exception as e:
                self._logger.exception(e)
        for sm in self._spectrums_monitors.values():
            try:
                sm.start(tmo)
            except Exception as e:
                self._logger.exception(e)
        for im in self._images_monitors.values():
            try:
                im.start(tmo)
            except Exception as e:
                self._logger.exception(e)

    # -----------------------------------
    def stop(self):
        if self._common_data_stream:
            try:
                self._common_data_stream.stop()
            except Exception as e:
                self._logger.exception(e)
        for sm in self._spectrums_monitors.values():
            try:
                sm.stop()
            except Exception as e:
                self._logger.exception(e)
        for im in self._images_monitors.values():
            try:
                im.stop()
            except Exception as e:
                self._logger.exception(e)

    # -----------------------------------
    def close(self):
        if self._common_data_stream:
            try:
                self._common_data_stream.close()
            except Exception as e:
                self._logger.exception(e)
        for sm in self._spectrums_monitors.values():
            try:
                sm.close()
            except Exception as e:
                self._logger.exception(e)
        self._spectrums_monitors.clear()
        for im in self._images_monitors.values():
            try:
                im.close()
            except Exception as e:
                self._logger.exception(e)
        self._images_monitors.clear()

# ------------------------------------------------------------------------------
FssAttributes = enum(
    'STATE',
    'STATUS',
    'CURRENT_CFG',
    'START_DATE',
    'END_DATE',
    'SCAN_POS',
    'SCAN_DIMS',
    'ELAPSED_TIME',
    'REMAINING_TIME',
    'SCAN_PROGRESS',
    'MON_DEVS_NAME',
    'MON_DEVS_STATE',
    'INTERNAL_STATE'
)


# ------------------------------------------------------------------------------
class FssMonitorData(object):
    """flyScan server monitor data"""

    # the fss attributes to monitor
    __fss_attributes = [
        "State",
        "Status",
        "currentConfig",
        "scanStartDate",
        "scanEndDate",
        "scanPosition",
        "scanDimensions",
        "scanElapsedTime",
        "scanRemainingTime",
        "scanProgress",
        "monitoredDeviceNames",
        "monitoredDeviceStates",
        "internalState"
    ]

    # -----------------------------------
    def __init__(self, fss=None):
        self._fss = fss
        self._data = [None for i in range(FssAttributes.len)]

    # -----------------------------------
    def refresh_all(self):
        self._data = self.update_all(self._fss)

    # -----------------------------------
    def refresh(self, fss, *arg):
        an = [FssMonitorData.__fss_attributes[i] for i in arg]
        try:
            al = fss.proxy.read_attributes(an)
            for i, j in zip(arg, range(len(arg))):
                self._data[i] = al[j]
        except:
            self._data = None

    # -----------------------------------
    def value(self, data_id):
        return self._data[data_id].value

    # -----------------------------------
    @classmethod
    def update_all(cls, fss):
        try:
            al = fss.proxy.read_attributes(FssMonitorData.__fss_attributes)
        except:
            al = None
        return al


# ------------------------------------------------------------------------------
class JnbFssMonitorWidgets(object):
    """ a flyScan server monitor for the Jupyter notebook"""

    # -----------------------------------
    def __init__(self, fss, sam=True, lgi=None):
        # running in standalone mode?
        self._standalone_mode = sam
        # logger instance
        if not lgi:
            self._logger = NotebookCellLogger(lgi if lgi else logging.getLogger('fs.client'))
        else:
            self._logger = lgi
        # link to the specified flyScan server or the current one
        self._fss = fss
        # flyscan server data
        self._fss_data = FssMonitorData(self._fss)
        # action callbacks
        self._action_handler = {
            'start': None,
            'stop': None,
            'save': None,
            'close_fsi': None,
            'open_close_plt': None,
            'eos': None
        }
        # fss status widgets
        self._fss_status_layout = None
        # setup monitoring widgets: misc scan progress widgets
        self._scan_progress_widgets = dict()
        # setup data files widgets: list of data files generated by teh DataMerger
        self._scan_data_files_widgets = dict()
        self._scan_data_files_layout = None
        # setup monitoring widgets: monitored devices button
        self._monitored_devices_widgets = dict()
        self._monitored_devices_layout = None
        # setup widgets for scan controls
        self._scan_controls_layout = None
        # main widget (global layout for standalone mode)
        self._start_button = None
        self._stop_button = None
        self._main_layout = None
        # misc helper variables & flags
        self._fss_state = tango.DevState.UNKNOWN
        try:
            self._scan_started_from_this_interface = not sam and self._fss.is_scanning
        except:
            self._scan_started_from_this_interface = not sam
        self._scan_aborted_from_this_interface = False
        # setup the widgets layouts
        self.__setup_scan_progress_widgets()
        self.__setup_monitored_devices_widgets()
        self.__setup_data_files_widgets()
        self.__setup_scan_controls_widgets()
        self.__setup_main_layout()
        # confirm abort pending
        self._confirm_abort_pending = False

    # -----------------------------------
    @property
    def main_widgets_layout(self):
        return self._main_layout

    # -----------------------------------
    @property
    def scan_controls_layout(self):
        return self._scan_controls_layout

    # -----------------------------------
    @property
    def monitored_devices_layout(self):
        return self._monitored_devices_layout

    # -----------------------------------
    @property
    def fss_status_layout(self):
        return self._fss_status_layout

    # -----------------------------------
    @property
    def scan_data_files_layout(self):
        return self._scan_data_files_layout

    # -----------------------------------
    def set_start_handler(self, f):
        self.__set_handler('start', f)

    # -----------------------------------
    def set_stop_handler(self, f):
        self.__set_handler('stop', f)

    # -----------------------------------
    def set_save_handler(self, f):
        self.__set_handler('save', f)

    # -----------------------------------
    def set_close_interface_handler(self, f):
        self.__set_handler('close_fsi', f)

    # -----------------------------------
    def set_open_close_plots_handler(self, f):
        self.__set_handler('open_close_plt', f)

    # -----------------------------------
    def set_end_of_scan_handler(self, f):
        self.__set_handler('eos', f)

    # -----------------------------------
    def __set_handler(self, k, f):
        assert (callable(f))
        self._action_handler[k] = f

    # -----------------------------------
    def __setup_monitored_devices_widgets(self):
        # update the required data from the fss
        try:
            data = [FssAttributes.STATE, FssAttributes.MON_DEVS_NAME, FssAttributes.MON_DEVS_STATE]
            self._fss_data.refresh(self._fss, *data)
        except Exception as e:
            self._logger.error(e)
            return
        # the fss button
        bt1 = ipywidgets.Button(
            description="[##]/[##] - ET: ##s - RT: ##s",
            tooltip="click to open atkpanel",
            layout=jwh.l11a(width="75%"),
            icon='fa-spinner',
            disabled=False
        )
        jwh.device_state_to_button_color(bt1, self._fss_data.value(FssAttributes.STATE))
        bt1.style.font_weight = 'normal'
        bt1.on_click(self.on_fss_button_clicked)
        self._scan_progress_widgets["fss_but"] = bt1
        # the fss internal state button
        bt2 = ipywidgets.Button(
            description="INTERNAL STATE",
            tooltip="click to open atkpanel",
            layout=jwh.l11a(width="25%"),
            disabled=False
        )
        jwh.device_state_to_button_color(bt2, self._fss_data.value(FssAttributes.STATE))
        bt2.on_click(self.on_fss_button_clicked)
        bt2.style.font_weight = 'normal'
        self._scan_progress_widgets["fss_istate_but"] = bt2
        # fss buttons layout
        fss_hb = ipywidgets.HBox([bt1, bt2], layout=jwh.l11a())
        # get list of monitored devices and their current state
        mon_devs_name = self._fss_data.value(FssAttributes.MON_DEVS_NAME)
        mon_devs_state = self._fss_data.value(FssAttributes.MON_DEVS_STATE)
        # num of columns in the widgets layout
        num_columns = jwh.optimize_num_columns(len(mon_devs_name), 3)
        # reset monitored devices dict
        self._monitored_devices_widgets = dict()
        # create each widget and add it to the right column
        ci = 0
        cl = [list() for _ in range(num_columns)]
        for n, s in zip(mon_devs_name, mon_devs_state):
            if n.find("flyscan/core/data-merger") != -1:
                icon = 'fa-files-o'
            elif n.find("flyscan/core/") != -1:
                icon = 'fa-gears'
            elif n.find("flyscan/sensor/") != -1:
                icon = 'fa-signal'
            elif n.find("flyscan/actuator/") != -1:
                icon = 'fa-arrows'
            elif n.find("flyscan/viewer/") != -1:
                icon = 'fa-area-chart'
            elif n.find("flyscan/generator/") != -1:
                icon = 'fa-recycle'
            elif n.find("flyscan/clock/") != -1:
                icon = 'fa-clock-o'
            else:
                icon = ''
            b = ipywidgets.Button(description=n, icon=icon, tooltip="click to open atkpanel", layout=jwh.l11a())
            b.style.font_weight = 'normal'
            self._monitored_devices_widgets[n] = b
            jwh.device_state_to_button_color(b, s)
            b.on_click(self.on_monitored_device_button_clicked)
            cl[ci].append(b)
            ci = (ci + 1) % num_columns
        # trick to obtain the expected alignment adding some hidden widgets
        lc0 = len(cl[0])
        for c in cl[1:num_columns]:
            d = lc0 - len(c)
            wl = list()
            for _ in range(d):
                b = ipywidgets.Button(description="",
                                      visible=False,
                                      disabled=True,
                                      layout=jwh.l11a())
                jwh.device_state_to_button_color(b, tango.DevState.STANDBY)
                wl.append(b)
            c.extend(wl)
        # build widgets layout
        mon_devices_layout = ipywidgets.HBox([ipywidgets.VBox(c, layout=jwh.l11a()) for c in cl], layout=jwh.l11a())
        new_children_layout = [ipywidgets.VBox([fss_hb, mon_devices_layout], layout=jwh.l11a())]
        if not self._monitored_devices_layout:
            self._monitored_devices_layout = ipywidgets.HBox(new_children_layout, layout=jwh.l11a())
        else:
            old_children = self._monitored_devices_layout.children
            self._monitored_devices_layout.children = new_children_layout
            del old_children

    # -----------------------------------
    def _exec_shell_cmd_callback(self):
        cmd = ''
        return CustomJS(args=dict(shell_cmd=cmd), code="""
            /// execute the specified shell command
            var exec = require('child_process').exec, child;
            child = exec(shell_cmd,
                function (error, stdout, stderr) {
                    console.log('stdout: ' + stdout);
                    console.log('stderr: ' + stderr);
                    if (error !== null) {
                         console.log('exec error: ' + error);
                    }
                });
        """)

    # -----------------------------------
    def on_monitored_device_button_clicked(self, b):
        if b.description.count("/") == 2:
            spawn_atk(b.description)

    # -----------------------------------
    def on_fss_button_clicked(self, b):
        spawn_atk(self._fss.name)

    # -----------------------------------
    def __setup_scan_progress_widgets(self):
        # fss button
        try:
            # scan progress bar and the associated label
            pb = ipywidgets.IntProgress(
                value=0.0,
                min=0,
                max=100,
                step=1,
                description='Scan Progress: ',
                orientation='horizontal',
                layout=jwh.l11a()
            )
            pb.style.bar_color = jwh.device_state_to_widget_color(tango.DevState.STANDBY)
            self._scan_progress_widgets["scan_pbar"] = pb
            pl = ipywidgets.Label(value=" 0%", disabled=True, layout=jwh.l01a(width='60px'))
            self._scan_progress_widgets["scan_pval"] = pl
            # merging  progress bar and the associated label
            pb = ipywidgets.IntProgress(
                value=0,
                min=0,
                max=100,
                step=1,
                description='Data Merging: ',
                orientation='horizontal',
                layout=jwh.l11a()
            )
            pb.style.bar_color = jwh.device_state_to_widget_color(tango.DevState.STANDBY)
            self._scan_progress_widgets["merging_pbar"] = pb
            pl = ipywidgets.Label(value=" 0%", disabled=True, layout=jwh.l01a(width='60px'))
            self._scan_progress_widgets["merging_pval"] = pl
            # a fss log area
            ta = ipywidgets.Textarea(value="this is the flyScan server status area...", disabled=True, layout=jwh.l11a())
            ta.rows = 1
            ta.observe(self.__on_fss_status_change, names='value')
            self._scan_progress_widgets["status_txt"] = ta
            self._fss_status_layout = ta
        except Exception as e:
            self._logger.error(e)

    # -----------------------------------
    def __on_fss_status_change(self, change):
        self._scan_progress_widgets["status_txt"].rows = min(3, 1 + change["new"].count('\n'))

    # -----------------------------------
    def __setup_data_files_widgets(self):
        try:
            ta = ipywidgets.Textarea(value="this is the data-merger files area...", layout=jwh.l11a())
            ta.rows = 1
            self._scan_data_files_widgets["files_list"] = ta
            self._scan_data_files_layout = ta
        except Exception as e:
            self._logger.error(e)

    # -----------------------------------
    def __setup_scan_controls_widgets(self):
        icn = "play" if not self._standalone_mode else ""
        desc = "" if not self._standalone_mode else "-"
        ttip = "Start scan" if not self._standalone_mode else ""
        self._start_button = ipywidgets.Button(icon=icn,
                                               description=desc,
                                               tooltip=ttip,
                                               layout=jwh.l01a(width="100px"))
        self._start_button.on_click(self.on_start_clicked)
        self._stop_button = ipywidgets.Button(icon="stop",
                                              description="",
                                              tooltip="Abort scan",
                                              layout=jwh.l01a(width="100px"))
        self._stop_button.on_click(self.on_stop_clicked)
        save_button = ipywidgets.Button(description="Save Config.",
                                        tooltip="Save scan configuration",
                                        icon="fa-sliders",
                                        layout=jwh.l01a(width="120px"))
        save_button.on_click(self.on_save_clicked)
        open_close_plt_button = ipywidgets.Button(icon='fa-area-chart',
                                                  description="",
                                                  tooltip="Open/Close the scan plots",
                                                  layout=jwh.l01a(width="50px"))
        open_close_plt_button.on_click(self.on_open_close_plt_clicked)
        close_fsi_button = ipywidgets.Button(icon='fa-remove',
                                             description="",
                                             tooltip="Close this scan interface",
                                             layout=jwh.l01a(width="50px"))
        close_fsi_button.on_click(self.on_close_interface_clicked)
        spb = self._scan_progress_widgets["scan_pbar"]
        spv = self._scan_progress_widgets["scan_pval"]
        mpb = self._scan_progress_widgets["merging_pbar"]
        mpv = self._scan_progress_widgets["merging_pval"]
        children = list()
        children.extend([self._start_button, self._stop_button])
        children.extend([spb, spv, mpb, mpv])
        children.extend([save_button, open_close_plt_button, close_fsi_button])
        hb2 = ipywidgets.HBox(children, layout=jwh.l11a())
        self._scan_controls_layout = hb2

    # -----------------------------------
    def update(self):
        try:
            self._fss_data.refresh_all()
            self.__update_scan_progress_widgets()
            self.__update_monitored_devices_widgets()
            self.__update_merger_progress_widgets()
        except Exception as e:
            raise

    # -----------------------------------
    def __update_merger_progress_widgets(self):
        try:
            md = self._fss.get_merger_data(['State', 'outputNexusFiles', 'progress'])
            mgs = md[0].value
            dfl = md[1].value
            prg = float(md[2].value)
            dfl_str = "" if not dfl or not len(dfl) else "\n".join(dfl)
        except Exception as e:
            mgs = tango.DevState.UNKNOWN
            dfl_str = "failed to obtain data files list from DataMerger"
            prg = 0.
        if mgs == tango.DevState.STANDBY:
            prg = 0.
        self._scan_data_files_widgets["files_list"].value = dfl_str
        self._scan_progress_widgets["merging_pbar"].value = int(prg)
        self._scan_progress_widgets["merging_pbar"].style.bar_color = jwh.device_state_to_widget_color(mgs)
        self._scan_progress_widgets["merging_pval"].value = "{:.2f}%".format(prg)

    # -----------------------------------
    def __reset_merger_progress_widgets(self):
        self._scan_data_files_widgets["files_list"].value = ""
        self._scan_progress_widgets["merging_pbar"].value = 0
        self._scan_progress_widgets["merging_pbar"].style.bar_color = jwh.device_state_to_widget_color(tango.DevState.UNKNOWN)
        self._scan_progress_widgets["merging_pval"].value = "0%"

    # -----------------------------------
    def __update_monitored_devices_widgets(self):
        try:
            dev_names = self._fss_data.value(FssAttributes.MON_DEVS_NAME)
            dev_states = self._fss_data.value(FssAttributes.MON_DEVS_STATE)
            if len(dev_names) != len(self._monitored_devices_widgets):
                self.__setup_monitored_devices_widgets()
                return
        except Exception as e:
            self.__set_monitored_devices_widgets_state(tango.DevState.UNKNOWN)
            return
        for n, s in zip(dev_names, dev_states):
            try:
                b = self._monitored_devices_widgets[n]
                jwh.device_state_to_button_color(b, s)
            except KeyError:
                try:
                    self.__setup_monitored_devices_widgets()
                    return
                except Exception as e:
                    self._logger.error(e)
            except:
                pass

    # -----------------------------------
    def __set_monitored_devices_widgets_state(self, state):
        for b in self._monitored_devices_widgets.values():
            jwh.device_state_to_button_color(b, state)

    # -----------------------------------
    def __update_scan_progress_widgets(self):
        failed = False
        try:
            txt = "flyScan Server: {} - {} / {} - ET: {} - RT: {}".format(
                self._fss_data.value(FssAttributes.CURRENT_CFG),
                self._fss_data.value(FssAttributes.SCAN_POS),
                self._fss_data.value(FssAttributes.SCAN_DIMS),
                self._fss_data.value(FssAttributes.ELAPSED_TIME),
                self._fss_data.value(FssAttributes.REMAINING_TIME)
            )
            val = self._fss_data.value(FssAttributes.SCAN_PROGRESS)
            stt = self._fss_data.value(FssAttributes.STATE)
            sta = self._fss_data.value(FssAttributes.STATUS)
            ista = self._fss_data.value(FssAttributes.INTERNAL_STATE)
        except Exception as e:
            failed = True
            txt = "flyScan Server: ?? - [??]/[??] - ET: ??s - RT: ??s - PG: ??%"
            val = 0.
            stt = tango.DevState.UNKNOWN
            sta = "failed to obtain flyScan status!"
            ista = 'UNKNOWN'
        if stt == tango.DevState.ON or stt == tango.DevState.OFF:
            val = 0.
        sc = jwh.device_state_to_widget_color(stt)
        bt = self._scan_progress_widgets["fss_but"]
        bt.description = txt
        bt.style.button_color = sc
        if not failed:
            _, sc, err = self.__decode_fss_internal_state(ista)
        else:
            sc = jwh.device_state_to_widget_color(tango.DevState.UNKNOWN)
            err = True
        pb = self._scan_progress_widgets["scan_pbar"]
        pb.value = int(val) if not err else 100
        pb.style.bar_color = sc
        bt = self._scan_progress_widgets["fss_istate_but"]
        bt.description = ista
        bt.style.button_color = sc
        self._scan_progress_widgets["scan_pval"].value = "{0:.2f}%".format(val)
        self._scan_progress_widgets["status_txt"].value = sta
        if not failed:
            started_from_self = self._scan_started_from_this_interface
            aborted_from_self = self._scan_aborted_from_this_interface
            if started_from_self and not aborted_from_self and not self._fss.is_scanning:
                self.__handle_end_of_scan()

    # -----------------------------------
    def __handle_end_of_scan(self):
        self._scan_started_from_this_interface = False
        self._logger.print_in_cell("> end of scan detected...")
        try:
            txt, _, _ = self.__decode_fss_internal_state(self._fss.proxy.internalState)
            self._logger.print_in_cell("> end of scan detected...\n{}".format(txt))
        except:
            txt = "oops, couldn't contact the flyScan server [check device '{}']".format(self._fss.name)
            self._logger.print_in_cell("> end of scan detected...\n{}".format(txt))
        finally:
            try:
                if self._action_handler['eos']:
                    self._action_handler['eos']()
            except:
                pass

    # -----------------------------------
    @classmethod
    def __decode_fss_internal_state(cls, istate):
        if istate == 'ABORTED BY USER':
            txt = "scan has been aborted on user (i.e. external) request"
            clr = jwh.device_state_to_widget_color(tango.DevState.STANDBY)
            err = False
        elif istate == 'ABORTED ON ERROR' or istate == 'ABORTING ON ERROR':
            txt = "scan has been aborted on error!"
            clr = jwh.device_state_to_widget_color(tango.DevState.FAULT)
            err = True
        elif istate == 'INVALID CONFIG':
            txt = "failed to load the scan configuration on server side!\n"
            txt += "this might be due to: an invalid configuration, an easy-configuration error, a device error\n"
            txt += "check flyScan server log for details"
            clr = jwh.device_state_to_widget_color(tango.DevState.FAULT)
            err = True
        elif istate == 'INIT STEP' or istate == 'STEP PROGRESS':
            txt = "scan in progress"
            clr = jwh.device_state_to_widget_color(tango.DevState.RUNNING)
            err = False
        elif istate == 'SCAN DONE':
            txt = "scan completed successfully"
            clr = jwh.device_state_to_widget_color(tango.DevState.STANDBY)
            err = False
        else:
            txt = "scan seems to be done but the flyScan server has an unexpected internal state: '{}'".format(istate)
            clr = jwh.device_state_to_widget_color(tango.DevState.STANDBY)
            err = False
        return txt, clr, err

    # -----------------------------------
    def on_start_clicked(self, b):
        if self._confirm_abort_pending:
            self.__switch_abort_pending_off()
        else:
            if not self._standalone_mode and self._action_handler['start']:
                try:
                    self._action_handler['start']()
                    self.__reset_merger_progress_widgets()
                    self._scan_started_from_this_interface = True
                    self._scan_aborted_from_this_interface = False
                except:
                    pass

    # -----------------------------------
    def on_stop_clicked(self, b):
        if self._confirm_abort_pending:
            self.__switch_abort_pending_off()
            self._scan_aborted_from_this_interface = True
            if self._action_handler['stop']:
                try:
                    self._action_handler['stop']()
                except:
                    pass
        else:
            self.__switch_abort_pending_on()

    # -----------------------------------
    def on_save_clicked(self, b):
        try:
            self._action_handler['save']()
        except:
            pass
            
    # -----------------------------------
    def on_close_interface_clicked(self, b):
        try:
            self._action_handler['close_fsi']()
        except:
            pass

    # -----------------------------------
    def on_open_close_plt_clicked(self, b):
        try:
            self._action_handler['open_close_plt']()
        except:
            pass

    # -----------------------------------
    def __switch_abort_pending_on(self):
        self._confirm_abort_pending = True
        self._start_button.icon = ""
        self._start_button.description = "Continue"
        self._stop_button.icon = ""
        self._stop_button.description = "Confirm Abort?"

    # -----------------------------------
    def __switch_abort_pending_off(self):
        self._confirm_abort_pending = False
        self._start_button.icon = "play" if not self._standalone_mode else ""
        self._start_button.description = "" if not self._standalone_mode else "-"
        self._stop_button.icon = "stop"
        self._stop_button.description = ""

    # -----------------------------------
    def __setup_main_layout(self):
        vb = ipywidgets.VBox([self._monitored_devices_layout, self._fss_status_layout], layout=jwh.l11a())
        if self._standalone_mode:
            tab = ipywidgets.Tab([vb], layout=jwh.l11a())
            tab.set_title(0, "Scan Monitor")
            vb = ipywidgets.VBox([tab, self._scan_controls_layout], layout=jwh.l11a())
        self._main_layout = vb


# -----------------------------------
def im_plot_perf():
    for m in live_monitors.values():
        if isinstance(m, ImagesMonitor):
            m.plot_perf()


# -----------------------------------
def im_reset_perf():
    for m in live_monitors.values():
        if isinstance(m, ImagesMonitor):
            m.reset_perf()


# -----------------------------------
def im_reads():
    for m in live_monitors.values():
        if isinstance(m, ImagesMonitor):
            m.print_reads_per_channels()


# -----------------------------------
class Tmp(object): #TODO: restore image perf feature

    def __perf_data(self):
        return dict(x=list(self._perf_npx), drt=list(self._perf_drt), prt=list(self._perf_prt))

    def __update_perf(self):
        if self._perf_hdl:
            self._perf_cds.data.update(self.__perf_data())
            push_notebook(handle=self._perf_hdl)

    def plot_perf(self):
        redirect_bokeh_output()
        self._perf_cds = ColumnDataSource(self.__perf_data())
        kwargs = dict()
        kwargs['webgl'] = True
        kwargs['plot_width'] = 950
        kwargs['plot_height'] = 250
        kwargs['toolbar_location'] = 'above'
        kwargs['tools'] = 'pan,box_zoom,wheel_zoom,box_select,resize,reset,hover'
        plt = figure(**kwargs)
        plt.toolbar.active_drag = None
        plt.toolbar.active_scroll = None
        plt.toolbar.logo = None
        kwargs = dict()
        kwargs['x'] = 'x'
        kwargs['y'] = 'drt'
        kwargs['source'] = self._perf_cds
        kwargs['line_color'] = 'blue'
        kwargs['legend'] = 'drt '
        plt.circle(**kwargs)
        kwargs = dict()
        kwargs['x'] = 'x'
        kwargs['y'] = 'prt'
        kwargs['source'] = self._perf_cds
        kwargs['line_color'] = 'red'
        kwargs['legend'] = 'prt '
        plt.legend.location = "top_left"
        plt.circle(**kwargs)
        self._perf_hdl = show(plt, notebook_handle=True)

    def print_reads_per_channels(self):
        for cn, ch in iteritems(self._channels):
            print("{0:10} : {1:1d}".format(cn, ch.rds))

    def reset_perf(self):
        self._perf_drt.clear()
        self._perf_prt.clear()
        self._perf_npx.clear()
        self._perf_npt = 0
