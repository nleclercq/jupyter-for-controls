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

import datetime
from collections import OrderedDict, deque
from math import ceil, pi
import six

import math as mt
import numpy as np

from IPython.display import display

import ipywidgets as ipw

from bokeh.layouts import row, column, layout, gridplot
from bokeh.models import ColumnDataSource, CustomJS, DatetimeTickFormatter, Label
from bokeh.models import widgets as BokehWidgets
from bokeh.models.glyphs import Rect
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.ranges import Range1d
from bokeh.models.tools import BoxSelectTool, HoverTool, CrosshairTool
from bokeh.models.tools import ResetTool, PanTool, BoxZoomTool
from bokeh.models.tools import WheelZoomTool, SaveTool
from bokeh.models.widgets.markups import Div
from bokeh.palettes import Plasma256
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
import bokeh.events


from common.tools import *
from common.datasource import *
from common.session import BokehSession
        
from skimage.transform import rescale

plots_module_logger_name = "fs.client.jupyter.plots"

# ------------------------------------------------------------------------------
class Children(OrderedDict):
    def __init__(self, owner, obj_class):
        OrderedDict.__init__(self)
        self._owner = owner
        self._obj_class = obj_class
        self._add_callbacks = OrderedDict()

    def register_add_callback(self, cb):
        if cb and hasattr(cb, '__call__'):
            l = len(self._add_callbacks)
            self._add_callbacks[l + 1] = cb

    def call_add_callbacks(self, child):
        for cb in self._add_callbacks.values():
            try:
                cb(child)
            except:
                pass

    def add(self, children):
        if isinstance(children, (list, tuple)):
            for c in children:
                self.__add_child(c)
        else:
            self.__add_child(children)

    def __add_child(self, child):
        if child is None:
            return
        if child is self._owner:
            err = "invalid argument: can't add 'self' to children"
            raise ValueError(err)
        if isinstance(child, self._obj_class):
            if not len(self):
                self._master_child = child
            self[child.name] = child
            self.call_add_callbacks(child)
        else:
            ic = child.__class__
            ec = self._obj_class.__name__
            err = "invalid argument: expected an iterable collection or a single instance of {} - got {}".format(ec, ic)
            raise ValueError(err)


# ------------------------------------------------------------------------------
class DataStreamEvent(object):
    """Data stream event"""

    Type = enum(
        'ERROR',
        'RECOVER',
        'MODEL_CHANGED',
        'EOS',
        'UNKNOWN'
    )

    def __init__(self, event_type=Type.UNKNOWN, emitter=None, data=None, error=None, exception=None):
        # evt. type
        self.type = event_type
        # uuid of the evt. emitter
        self.emitter = emitter
        # abstract event data
        self.data = data
        # error text
        self.error = error
        # exception
        self.exception = exception


# ------------------------------------------------------------------------------
class DataStreamEventHandler(object):
    """Data stream event handler"""

    supported_events = [
        DataStreamEvent.Type.ERROR,
        DataStreamEvent.Type.RECOVER,
        DataStreamEvent.Type.MODEL_CHANGED,
        DataStreamEvent.Type.EOS
    ]

    def __init__(self, name):
        self._name = name
        # callbacks
        self._callbacks = dict()
        for event_type in self.supported_events:
            self._callbacks[event_type] = list()

    @property
    def name(self):
        return self._name

    def register_event_handler(self, event_handler, events):
        assert (isinstance(events, (list, tuple)))
        assert (isinstance(event_handler, DataStreamEventHandler))
        for event in events:
            if event in self.supported_events:
                # print("{}: registering event handler {} for event {}".format(self.name, event_handler.name, event))
                self._callbacks[event].append(event_handler)

    def emit(self, event):
        assert (isinstance(event, DataStreamEvent))
        if event.type in self.supported_events:
            for event_handler in self._callbacks[event.type]:
                try:
                    # print("{}: emitting event {} towards {}".format(self.name, event.type, event_handler.name))
                    event_handler.__handle_stream_event(event)
                except Exception as e:
                    print(e)
                    raise

    def __handle_stream_event(self, event):
        try:
            self.handle_stream_event(event)
        except:
            pass
        finally:
            self.__propagate(event)

    def __propagate(self, event):
        assert (isinstance(event, DataStreamEvent))
        # print("{}: propagating event {} ".format(self.name, event.type))
        self.emit(event)

    def emit_error(self, sd):
        evt = DataStreamEvent(DataStreamEvent.Type.ERROR, self.uid, None, sd.error, sd.exception)
        self.emit(evt)

    def emit_recover(self):
        evt = DataStreamEvent(DataStreamEvent.Type.RECOVER, self.uid)
        self.emit(evt)

    def emit_model_changed(self, model):
        evt = DataStreamEvent(DataStreamEvent.Type.MODEL_CHANGED, self.uid, model)
        # print("{}: emitting model changed evt".format(self.name))
        self.emit(evt)

    def handle_stream_event(self, event):
        pass
        

# ------------------------------------------------------------------------------
class Channel(NotebookCellContent, DataStreamEventHandler):
    """single data stream channel"""

    def __init__(self, name, data_sources=None, model_properties=None):
        NotebookCellContent.__init__(self, name, logger=logging.getLogger(plots_module_logger_name))
        DataStreamEventHandler.__init__(self, name)
        # associated bokeh session
        self._session = None
        # data sources
        self._bad_source_cnt = 0
        self._data_sources = Children(self, DataSource)
        self.add_data_sources(data_sources)
        # model properties
        self._model_props = dict() if model_properties is None else model_properties
        # tmp label
        self._msg_label = None

    def handle_stream_event(self, event):
        assert (isinstance(event, DataStreamEvent))
        pass

    @property
    def title(self):
        return self._model_props.get('channel_title', None)

    @property
    def show_title(self):
        return self._model_props.get('show_channel_title', False)

    @property
    def data_source(self):
        """returns the 'first' (and sometimes 'unique') data source"""
        for ds in self._data_sources.values():
            return ds
        return None

    @property
    def data_sources(self):
        """returns the dict of data sources"""
        return self._data_sources

    def set_data_source(self, ds):
        """set the channel unique data source"""
        if ds is not None:
            assert (isinstance(ds, DataSource))
            self._data_sources.clear()
            self.add_data_source(ds)

    def add_data_source(self, ds):
        """add the specified data source to the channel"""
        if ds is not None:
            assert (isinstance(ds, DataSource))
            self._data_sources[ds.name] = ds

    def add_data_sources(self, ds):
        """add the specified data source to the channel"""
        if ds is not None:
            assert (isinstance(ds, (list, tuple)))
            for s in ds:
                self.add_data_source(s)

    def get_data(self):
        """returns a dict containing the data of each data source"""
        data = dict()
        for dsn, dsi in six.iteritems(self._data_sources):
            data[dsn] = dsi.pull_data()
        return data

    def cleanup(self):
        """cleanup data sources"""
        for dsn, dsi in six.iteritems(self._data_sources):
            try:
                self.info("DataStream channel: cleaning up DataSource {}".format(dsn))
                dsi.cleanup()
            except Exception as e:
                self.error(e)

    @property
    def bokeh_session(self):
        """returns the dict of model properties"""
        return self._session

    @bokeh_session.setter
    def bokeh_session(self, bks):
        """bokeh_session"""
        assert (isinstance(bks, BokehSession))
        self._session = bks

    @property
    def model_properties(self):
        """returns the dict of model properties"""
        return self._model_props

    @model_properties.setter
    def model_properties(self, mp):
        """set the dict of model properties"""
        self._model_props = mp

    @staticmethod
    def _merge_properties(mp1, mp2, overwrite=False):
        if mp1 is None:
            props = mp2 if mp2 is not None else dict()
        elif mp2 is None:
            props = mp1
        else:
            props = mp1
            for k, v in six.iteritems(mp2):
                if overwrite or k not in mp1:
                    props[k] = v
        return props

    def setup_model(self, **kwargs):
        """asks the channel to setup then return its Bokeh associated model - returns None if no model"""
        return None

    def _show_msg_label(self, bkh_figure, x=70, y=70, text='Waiting for data'):
        self._msg_text = text
        self._msg_cnt = 0
        self._msg_label = Label(
            x=70,
            y=70,
            x_units='screen',
            y_units='screen',
            text=self._msg_text,
            background_fill_alpha=0.0,
            text_font='helvetica',
            text_font_style='italic',
            text_font_size='11pt',
            text_color='black'
        )
        bkh_figure.add_layout(self._msg_label)

    def _animate_msg_label(self):
        try:
            if self._msg_label:
                self._msg_label.text = self._msg_text + "." * self._msg_cnt
                self._msg_cnt = (self._msg_cnt + 1) % 4
        except Exception as e:
            pass

    def _hide_msg_label(self):
        if self._msg_label:
            self._msg_label.visible = False

    def get_model(self):
        """returns the Bokeh model (figure, layout, ...) associated with the Channel or None if no model"""
        return None

    def update(self):
        """gives the Channel a chance to update itself"""
        pass


# ------------------------------------------------------------------------------
class BoxSelectionManager(NotebookCellContent):
    """BoxSelectTool manager"""

    repository = dict()

    def __init__(self, selection_callback=None, reset_callback=None):
        self._uid = uuid4().hex
        NotebookCellContent.__init__(self, str(self._uid), logger=logging.getLogger(plots_module_logger_name))
        BoxSelectionManager.repository[self._uid] = self
        self._selection_callback = selection_callback
        self._reset_callback = reset_callback
        self._selection_cds = self.__setup_selection_data_source()

    def __del__(self):
        try:
            del BoxSelectionManager.repository[self._uid]
        except KeyError:
            pass

    def __setup_selection_data_source(self):
        cds = ColumnDataSource(data=dict(x0=[0], y0=[0], width=[0], height=[0]))
        cds.tags = [self._uid]
        return cds

    @property
    def selection_callback(self):
        return self._selection_callback

    @selection_callback.setter
    def selection_callback(self, scb):
        self._selection_callback = scb

    @property
    def reset_callback(self):
        return self._reset_callback

    @reset_callback.setter
    def reset_callback(self, rcb):
        self._reset_callback = rcb

    def __selection_glyph(self):
        kwargs = dict()
        kwargs['x'] = 'x0'
        kwargs['y'] = 'y0'
        kwargs['width'] = 'width'
        kwargs['height'] = 'height'
        kwargs['fill_alpha'] = 0.1
        kwargs['fill_color'] = '#009933'
        kwargs['line_color'] = 'white'
        kwargs['line_dash'] = 'dotdash'
        kwargs['line_width'] = 2
        return Rect(**kwargs)

    def register_figure(self, bkh_figure):
        try:
            bkh_figure.js_on_event(bokeh.events.SelectionGeometry, self.__box_selection_callback())
            bkh_figure.on_event(bokeh.events.SelectionGeometry, self.__print_event(attributes=['geometry', 'final']))
        except Exception as e:
            self.error(e)
        try:
            bkh_figure.js_on_event(bokeh.events.Reset, self.__reset_callback())
            bkh_figure.on_event(bokeh.events.Reset, self.__print_event())
        except Exception as e:
            self.error(e)
        rect = self.__selection_glyph()
        bkh_figure.add_glyph(self._selection_cds, glyph=rect, selection_glyph=rect, nonselection_glyph=rect)

    def __print_event(self, attributes=list()):
        def python_callback(event):
            cls_name = event.__class__.__name__
            attrs = ', '.join(['{attr}={val}'.format(attr=attr, val=event.__dict__[attr]) for attr in attributes])
            self.error('{cls_name}({attrs})'.format(cls_name=cls_name, attrs=attrs))
        return python_callback

    def __box_selection_callback(self):
        return CustomJS(args=dict(cds=self._selection_cds), code="""
            function handle_output(data) {
                console.log(data)
            }
            var callbacks = {
                    iopub : {
                        output : handle_output,
                }
            }
            var geometry = cb_obj['geometry']
            var width = geometry['x1'] - geometry['x0']
            var height = geometry['y1'] - geometry['y0']
            var x0 = geometry['x0'] + width / 2
            var y0 = geometry['y0'] + height / 2
            cds.data['x0'][0] = x0
            cds.data['y0'][0] = y0
            cds.data['width'][0] = width
            cds.data['height'][0] = height
            cds.change.emit()
            var imp = "from fs.client.jupyter.plots import BoxSelectionManager;"
            var pfx = "BoxSelectionManager.repository['".concat(cds.tags[0], "'].on_selection_change(")
            var arg = JSON.stringify({'x0':[x0], 'y0':[y0], 'width':[width], 'height':[height]})
            var sfx = ")"
            var cmd  = imp.concat(pfx, arg, sfx)
            console.log(cmd)
            IPython.notebook.kernel.execute(cmd, callbacks)
        """)

    def __reset_callback(self):
        return CustomJS(args=dict(cds=self._selection_cds), code="""
            cds.data['x0'][0] = 0
            cds.data['y0'][0] = 0
            cds.data['width'][0] = 0
            cds.data['height'][0] = 0
            cds.change.emit()
            var imp = "from fs.client.jupyter.plots import BoxSelectionManager;"
            var rst = "BoxSelectionManager.repository['".concat(cds.tags[0],"'].on_selection_reset()")
            var cmd  = imp.concat(rst)
            console.log(cmd)
            Jupyter.notebook.kernel.execute(cmd)
        """)

    @staticmethod
    def __selection_range(selection):
        w = selection['width'][0]
        h = selection['height'][0]
        x0 = selection['x0'][0] - w / 2.
        x1 = selection['x0'][0] + w / 2.
        y0 = selection['y0'][0] - h / 2.
        y1 = selection['y0'][0] + h / 2.
        return {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'width': w, 'height': h}

    def on_selection_change(self, selection):
        try:
            if self._selection_callback:
                self._selection_callback(self.__selection_range(selection))
        except Exception as e:
            self.error(e)

    def on_selection_reset(self):
        if self._reset_callback:
            self._reset_callback()


# ------------------------------------------------------------------------------
class InteractionsManager(object):
    def __init__(self):
        self._session = None
        self._callback = None
        self._range_change_notified = False

    def setup(self, session, figure, callback):
        assert (isinstance(session, BokehSession))
        self._session = session
        self._callback = callback
        # figure.on_event(bokeh.events.Reset, self.__on_reset)
        figure.x_range.on_change('start', self.__on_range_change)
        figure.x_range.on_change('end', self.__on_range_change)
        figure.y_range.on_change('start', self.__on_range_change)
        figure.y_range.on_change('end', self.__on_range_change)

    def __on_reset(self, event):
        self.__notify_range_change()

    def __on_range_change(self, attr, old, new):
        self.__notify_range_change()

    def __notify_range_change(self):
        if not self._range_change_notified and self._callback:
            self._range_change_notified = True
            try:
                # -----------------------------------------------------------------------------
                # InteractionsManager.__on_range_change is called with 'document' locked
                # we consequently have to call the owner's handler asynchrounously so that
                # it will be able to update the plot
                # -----------------------------------------------------------------------------
                # nb: this a tmp impl - we are waiting for the bokeh events to improve it
                # -----------------------------------------------------------------------------
                self._session.timeout_callback(self._callback, 0.25)
            except Exception as e:
                print(e)

    def range_change_handled(self):
        self._range_change_notified = False


# ------------------------------------------------------------------------------
ScaleType = enum(
    'INDEXES',
    'RANGE',
    'CHANNEL'
)


# ------------------------------------------------------------------------------
class Scale(object):
    """a scale"""

    def __init__(self, **kwargs):
        self._start = kwargs.get('start', None)
        self._end = kwargs.get('end', None)
        self._num_points = kwargs.get('num_points', None)
        self._label = kwargs.get('label', None)
        self._unit = kwargs.get('unit', None)
        self._channel = kwargs.get('channel', None)
        if self._channel is not None and len(self._channel):
            self._type = ScaleType.CHANNEL
        elif self._start is None or self._end is None or self._num_points is None:
            self._type = ScaleType.INDEXES
        else:
            self._type = ScaleType.RANGE
        self._array, self._step = self.__compute_linear_space()
        # the following will be used for linear interpolation (point coords -> pixel index)
        self._ix = self._array
        self._iy = np.linspace(0, self._array.shape[0] - 1, num=self._array.shape[0], dtype=int)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, t):
        raise Exception("Scale.type is immutable - can't change its value")

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, s):
        raise Exception("Scale.start is immutable - can't change its value")

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, e):
        raise Exception("Scale.end is immutable - can't change its value")

    @property
    def num_points(self):
        return self._num_points

    @num_points.setter
    def num_points(self, np):
        raise Exception("Scale.num_points is immutable - can't change its value")

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, s):
        raise Exception("Scale.step is immutable - can't change its value")

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        self._unit = unit

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, c):
        raise Exception("Scale.channel is immutable - can't change its value")

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, a):
        raise Exception("Scale.array is immutable - can't change its value")

    @property
    def range(self):
        return tuple([self._start, self._end, self._num_points])

    @property
    def bokeh_range(self):
        return Range1d(self._start, self._end) if self.__validate_range() else Range1d(-1, 1)

    def range_coords_to_indexes(self, start_p, end_p):
        start_i = mt.floor(np.interp(start_p, self._ix, self._iy))
        end_i = mt.ceil(np.interp(end_p, self._ix, self._iy)) + 1
        return start_i, end_i

    def validate(self):
        if self._type != ScaleType.INDEXES:
            self.__validate_range()
            self.__validate_num_points()

    def __validate_range(self):
        if self._start is not None and self._end is not None and self._start == self._end:
            raise ValueError("invalid axis scale: the specified 'range' is empty")

    def __validate_num_points(self):
        if self._start is not None and self._end is not None and self._num_points is not None and self._num_points < 1:
            raise ValueError("invalid axis scale: the specified 'num_points' is invalid")

    def __compute_linear_space(self):
        try:
            array, step = np.linspace(float(self._start),
                                      float(self._end),
                                      int(self._num_points),
                                      endpoint=True,
                                      retstep=True)
        except:
            array, step = np.zeros((0,)), 0.
        return array, step

    def has_valid_scale(self):
        valid_range = self._start is not None and self._end is not None and self._start != self._end
        return valid_range and self._num_points is not None and self._num_points >= 1

    def axis_label(self):
        label = self._label
        unit = self._unit
        axis_label = ''
        if label:
            axis_label = label
        if unit:
            axis_label += ' [' if len(label) else ''
            axis_label += unit
            axis_label += ']' if len(label) else ''
        return None if not len(axis_label) else axis_label

    def __repr__(self):
        return "scale:{} type:{} start:{} end:{} np:{} step:{} unit:{} channel:{}".format(self.label,
                                                                                          self.type,
                                                                                          self.start,
                                                                                          self.end,
                                                                                          self.num_points,
                                                                                          self.step,
                                                                                          self.unit,
                                                                                          self.channel)


# ------------------------------------------------------------------------------
class ModelHelper(object):
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

    @staticmethod
    def line_color(index):
        i = index % 10
        return ModelHelper.line_colors[i]

    @staticmethod
    def plot_style(instance, index):
        assert (isinstance(instance, Figure))
        i = index % 3
        if i == 0:
            return instance.circle
        if i == 1:
            return instance.square
        if i == 2:
            return instance.diamond
        return instance.square


# ------------------------------------------------------------------------------
class ScalarChannel(Channel):
    """scalar data source channel"""

    def __init__(self, name, data_sources=None, model_properties=None):
        Channel.__init__(self, name, data_sources=data_sources, model_properties=model_properties)
        self.__reinitialize()

    def __reinitialize(self):
        self._cds = None  # column data source
        self._mdl = None  # model
        self._lrdr = dict()  # renderers (i.e. y line glyphs)
        self._crdr = dict()  # renderers (i.e. y circle glyphs)

    def get_model(self):
        """returns the Bokeh model (figure, layout, ...) associated with the Channel or None if no model"""
        return self._mdl

    def __instanciate_data_source(self):
        columns = OrderedDict()
        # add an entry for timestamp
        columns['_@time@_'] = np.zeros(1)
        # add an entry for each child
        for cn, ci in six.iteritems(self.data_sources):
            columns[cn] = np.zeros(1)
        return ColumnDataSource(data=columns)

    def __setup_legend(self, figure):
        figure.legend.location = 'top_left'
        figure.legend.click_policy = 'hide'

    def __setup_toolbar(self, figure):
        htt = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)")
        ]
        figure.add_tools(PanTool())
        figure.add_tools(BoxZoomTool())
        figure.add_tools(WheelZoomTool())
        figure.add_tools(ResetTool())
        figure.add_tools(SaveTool())
        figure.add_tools(HoverTool(tooltips=htt))
        figure.add_tools(CrosshairTool())
        figure.toolbar.logo = None
        figure.toolbar.active_drag = None
        figure.toolbar.active_scroll = None
        figure.toolbar.active_tap = None

    def __setup_figure(self, **kwargs):
        fkwargs = dict()
        fkwargs['output_backend'] = 'webgl'
        fkwargs['plot_width'] = kwargs.get('width', 950)
        fkwargs['plot_height'] = kwargs.get('height', 250)
        fkwargs['toolbar_location'] = 'above'
        fkwargs['tools'] = ''
        fkwargs['x_axis_type'] = 'datetime'
        fkwargs['name'] = str(kwargs.get('uid', self.uid))
        f = figure(**fkwargs)
        dtf = DatetimeTickFormatter()
        dtf.milliseconds = "%M:%S:%3N"
        dtf.seconds = "%H:%M:%S"
        dtf.minutes = "%H:%M:%S"
        dtf.hours = "%H:%M:%S"
        dtf.days = "%d:%H:%M"
        f.xaxis.formatter = dtf
        f.xaxis.major_label_orientation = pi / 4
        layout = kwargs.get('layout', 'column')
        if kwargs['show_title'] and layout != 'tabs':
            f.title.text = self.name
        return f

    def __setup_glyph(self, figure, data_source, show_legend=True):
        kwargs = dict()
        kwargs['x'] = '_@time@_'
        kwargs['y'] = data_source
        kwargs['source'] = self._cds
        kwargs['line_color'] = ModelHelper.line_color(len(self._lrdr))
        self._lrdr[data_source] = figure.line(**kwargs)
        kwargs['size'] = 3
        kwargs['line_color'] = ModelHelper.line_color(len(self._crdr))
        kwargs['legend'] = None if not show_legend else data_source + ' '
        self._crdr[data_source] = figure.circle(**kwargs)

    @tracer
    def setup_model(self, **kwargs):
        """asks the channel to setup then return its Bokeh associated model - returns None if no model"""
        try:
            self._mdl = None
            props = self._merge_properties(self.model_properties, kwargs)
            # instanciate the ColumnDataSource
            self._cds = self.__instanciate_data_source()
            # setup figure
            show_title = True if len(self.data_sources) == 1 else False
            show_title = props.get('show_title', show_title)
            props['show_title'] = show_title
            f = self.__setup_figure(**props)
            # setup glyphs
            show_legend = False if len(self.data_sources) == 1 else True
            show_legend = props.get('show_legend', show_legend)
            for data_source in self.data_sources:
                self.__setup_glyph(f, data_source, show_legend)
            # setup the legend
            if show_legend:
                self.__setup_legend(f)
            # show tmp label
            self._show_msg_label(f)
            # setup the toolbar
            self.__setup_toolbar(f)
            # store figure
            self._mdl = f
        except Exception as e:
            self.error(e)
        return self._mdl

    def update(self):
        """gives each Channel a chance to update itself (e.g. to update the ColumDataSources)"""
        # print('scalar channel update')
        try:
            # get data from each channel
            min_len = 2 ** 32 - 1
            data = dict()
            previous_bad_source_cnt = self._bad_source_cnt
            self._bad_source_cnt = 0
            for sn, si in six.iteritems(self.data_sources):
                # print("pulling data from {}...".format(sn))
                data[sn] = sd = si.pull_data()
                if sd.has_failed or sd.buffer is None:
                    self._bad_source_cnt += 1
                    self._animate_msg_label()
                    # print("emitting error...")
                    self.emit_error(sd)
                else:
                    min_len = min(min_len, sd.buffer.shape[0])
                    self._hide_msg_label()
            if not self._bad_source_cnt and previous_bad_source_cnt:
                # print("emitting recover...")
                self.emit_recover()
            updated_data = dict()
            time_scale_set = False
            for cn, ci in six.iteritems(self.data_sources):
                try:
                    if not time_scale_set:
                        updated_data['_@time@_'] = data[cn].time_buffer[-min_len:]
                        time_scale_set = True
                    updated_data[cn] = data[cn].buffer[-min_len:]
                    self._lrdr[cn].visible = True
                    self._crdr[cn].visible = True
                except Exception:
                    updated_data['_@time@_'] = np.zeros((min_len,), dtype=datetime.datetime)
                    updated_data[cn] = np.zeros((min_len,), np.float)
                    self._lrdr[cn].visible = False
                    self._crdr[cn].visible = False
            self._cds.data.update(updated_data)
        except Exception as e:
            raise

    def cleanup(self):
        self.__reinitialize()
        super(ScalarChannel, self).cleanup()


# ------------------------------------------------------------------------------
class SpectrumChannel(Channel):
    """spectrum data source channel"""

    def __init__(self, name, data_sources=None, model_properties=None):
        Channel.__init__(self, name, data_sources=data_sources, model_properties=model_properties)
        self.__reinitialize()

    def __reinitialize(self):
        self._cds = None  # column data source
        self._xsn = None  # x scale name #TODO: inject name into scale class
        self._xsc = None  # x scale
        self._ysc = None  # y scale
        self._mdl = None  # model
        self._rdr = dict()  # renderers (i.e. y glyphs)

    def get_model(self):
        """returns the Bokeh model (figure, layout, ...) associated with the Channel or None if no model"""
        return self._mdl

    def __instanciate_data_source(self):
        columns = OrderedDict()
        # add an entry for x scale data (for indexes or range scales)
        columns[self._xsn] = np.zeros(1)
        # add an entry for each child
        for cn, ci in six.iteritems(self.data_sources):
            columns[cn] = np.zeros(1)
        return ColumnDataSource(data=columns)

    def __validate_x_channel(self):
        xsn = self._xsc.channel
        if self._xsc.type == ScaleType.CHANNEL and (xsn is None or not xsn in self.data_sources):
            err = "invalid SpectrumChannel configuration: the specified 'x' scale channel" \
                  " '{}' must be one of the {} data sources".format(xsn, self.name)
            raise Exception(err)
        if xsn is None or not len(xsn):
            xsn = self.__generate_x_channel_name()
        return xsn

    def __generate_x_channel_name(self):
        x_channel_name = 'x'
        while x_channel_name in self.data_sources:
            x_channel_name += 'x'
        return x_channel_name

    def __setup_legend(self, bkh_figure):
        bkh_figure.legend.location = 'top_left'
        bkh_figure.legend.click_policy = 'hide'

    def __setup_toolbar(self, bkh_figure):
        htt = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)")
        ]
        bkh_figure.add_tools(PanTool())
        bkh_figure.add_tools(BoxZoomTool())
        bkh_figure.add_tools(WheelZoomTool())
        bkh_figure.add_tools(ResetTool())
        bkh_figure.add_tools(SaveTool())
        bkh_figure.add_tools(HoverTool(tooltips=htt))
        bkh_figure.add_tools(CrosshairTool())
        bkh_figure.toolbar.logo = None
        bkh_figure.toolbar.active_drag = None
        bkh_figure.toolbar.active_scroll = None
        bkh_figure.toolbar.active_tap = None

    def __setup_figure(self, **kwargs):
        fkwargs = dict()
        fkwargs['output_backend'] = 'webgl'
        fkwargs['x_range'] = self._xsc.bokeh_range
        fkwargs['plot_width'] = kwargs.get('width', 950)
        fkwargs['plot_height'] = kwargs.get('height', 250)
        fkwargs['toolbar_location'] = 'above'
        fkwargs['tools'] = ''
        fkwargs['name'] = str(kwargs.get('uid', self.uid))
        f = figure(**fkwargs)
        x_label = None if self._xsc is None else self._xsc.axis_label()
        if x_label is not None and x_label:
            f.xaxis.axis_label = x_label
        y_label = None if self._ysc is None else self._ysc.axis_label()
        if y_label is not None and y_label:
            f.yaxis.axis_label = y_label
        layout = kwargs.get('layout', 'column')
        if kwargs['show_title'] and layout != 'tabs':
            f.title.text = self.name if layout != 'tabs' else " "
        return f

    def __setup_glyph(self, bkh_figure, y_column, show_legend=True):
        kwargs = dict()
        kwargs['x'] = self._xsn
        kwargs['y'] = y_column
        kwargs['source'] = self._cds
        kwargs['line_color'] = ModelHelper.line_color(len(self._rdr))
        kwargs['legend'] = None if not show_legend else y_column + ' '
        self._rdr[y_column] = bkh_figure.line(**kwargs)

    @tracer
    def setup_model(self, **kwargs):
        try:
            """asks the channel to setup then return its Bokeh associated model - returns None if no model"""
            self._mdl = None
            props = self._merge_properties(self.model_properties, kwargs)
            # x scale parameters
            self._xsc = props.get('x_scale', Scale())
            self._xsc.validate()
            # y scale parameters
            self._ysc = props.get('y_scale', Scale())
            self._ysc.validate()
            # if specified, x_channel must be one of our children
            self._xsn = self.__validate_x_channel()
            # instanciate the ColumnDataSource
            self._cds = self.__instanciate_data_source()
            # setup figure
            show_title = True if len(self.data_sources) == 1 else False
            show_title = props.get('show_title', show_title)
            props['show_title'] = show_title
            f = self.__setup_figure(**props)
            # setup glyphs
            show_legend = False if len(self.data_sources) == 1 else True
            show_legend = props.get('show_legend', show_legend)
            for data_source in self.data_sources:
                if data_source != self._xsn:
                    self.__setup_glyph(f, data_source, show_legend)
            # setup the legend
            if show_legend:
                self.__setup_legend(f)
            # show tmp label
            self._show_msg_label(f)
            # setup the toolbar
            self.__setup_toolbar(f)
            # store figure
            self._mdl = f
        except Exception as e:
            self.error(e)
        return self._mdl

    def update(self):
        """gives each Channel a chance to update itself (e.g. to update the ColumDataSources)"""
        # print('spectrum channel update')
        if not self._mdl:
            return
        try:
            # get data from each channel
            min_len = 2 ** 32 - 1
            data = dict()
            previous_bad_source_cnt = self._bad_source_cnt
            self._bad_source_cnt = 0
            for sn, si in six.iteritems(self.data_sources):
                # print("pulling data from {}...".format(sn))
                data[sn] = sd = si.pull_data()
                if sd.has_failed or sd.buffer is None:
                    self._bad_source_cnt += 1
                    self._animate_msg_label()
                    # print("emitting error...")
                    self.emit_error(sd)
                else:
                    min_len = min(min_len, sd.buffer.shape[0])
                    self._hide_msg_label()
            if not self._bad_source_cnt and previous_bad_source_cnt:
                # print("emitting recover...")
                self.emit_recover()
            updated_data = dict()
            if self._bad_source_cnt:
                min_len = 3
                updated_data[self._xsn] = np.linspace(-1, 1, min_len)
                self._mdl.x_range.update(start=-1, end=1)
            elif self._xsc.type == ScaleType.INDEXES:
                updated_data[self._xsn] = np.linspace(0, min_len - 1, min_len)
                self._mdl.x_range.update(start=0, end=min_len - 1)
            elif self._xsc.type == ScaleType.RANGE:
                end_point = self._xsc.start + (min_len - 1) * self._xsc.step
                x_scale_data = np.linspace(self._xsc.start, end_point, min_len)
                updated_data[self._xsn] = x_scale_data[:min_len]
                self._mdl.x_range.update(start=self._xsc.start, end=end_point)
            else:
                try:
                    if self._bad_source_cnt:
                        raise Exception('at least one source failed!')
                    x_scale_data = data[self._xsn].buffer[:min_len]
                    updated_data[self._xsn] = x_scale_data
                    self._mdl.x_range.update(start=x_scale_data[0], end=x_scale_data[min_len - 1])
                except Exception:
                    updated_data[self._xsn] = np.zeros((min_len,), np.float)
                    self._mdl.x_range.update(start=0, end=0)
            for cn, ci in six.iteritems(self.data_sources):
                try:
                    if cn != self._xsn:
                        updated_data[cn] = data[cn].buffer[:min_len]
                        self._rdr[cn].visible = True
                except Exception:
                    updated_data[cn] = np.zeros((min_len,), np.float)
                    self._rdr[cn].visible = False
            self._cds.data.update(updated_data)
        except Exception as e:
            self.error(e)

    def cleanup(self):
        self.__reinitialize()
        super(SpectrumChannel, self).cleanup()


# ------------------------------------------------------------------------------
class ImageChannel(Channel):
    """image data source channel"""

    def __init__(self, name, data_source=None, model_properties=None):
        Channel.__init__(self, name, data_sources=[data_source], model_properties=model_properties)
        self.__reinitialize()

    def __reinitialize(self):
        self._sd = None  # last data receive from the associated source
        self._cds = None  # column data source
        self._mdl = None  # model
        self._xsc = None  # x scale
        self._ysc = None  # y scale
        self._ird = None  # image renderer
        self._rrd = None  # rect renderer for hover trick
        self._itm = InteractionsManager()  # an InteractionsManager
        self._expected_image_shape = None
        self._current_image_shape = None
        self._images_size_threshold = 100000

    def __instanciate_data_source(self):
        columns = dict()
        data = np.empty((2, 2))
        data.fill(np.nan)
        columns['image'] = [data]
        columns['image_width'] = [0]
        columns['image_height'] = [0]
        columns['x_hover'] = [0]
        columns['y_hover'] = [0]
        columns['z_hover'] = [0]
        columns['initial_x_range'] = [[-1, 1]]
        columns['initial_y_range'] = [[-1, 1]]
        columns['image_shape_changed'] = [0]
        return ColumnDataSource(data=columns)

    def __hover_callback(self):
        return CustomJS(args=dict(cds=self._cds), code="""
            var pxc = cb_data['geometry'].x
            var pyc = cb_data['geometry'].y
            var plt = cb_obj.document._all_models[cds.tags[0]]
            var xrg = plt.x_range
            var yrg = plt.y_range
            var img = cds.data['image'][0]
            var imw = cds.data['image_width'][0]
            var imh = cds.data['image_height'][0]
            var xst = Math.abs(plt.x_range.end - plt.x_range.start) / imw
            var yst = Math.abs(plt.y_range.end - plt.y_range.start) / imh
            var pxi = Math.floor(Math.abs(pxc - plt.x_range.start) / xst)
            var pyi = Math.floor(Math.abs(pyc - plt.y_range.start) / yst)
            var isc = cds.data['image_shape_changed'][0]
            if (isc != 0) {
                var ixrg = cds.data['initial_x_range'][0]
                //console.log(ixrg)
                xrg._initial_start = ixrg[0]
                xrg._initial_end = ixrg[1]
                var iyrg = cds.data['initial_y_range'][0]
                //console.log(iyrg)
                yrg._initial_start = iyrg[0]
                yrg._initial_end = iyrg[1]
            }
            var flatten_pti = pxi + pyi * imw
            var flatten_array_len = pxi + pyi * imw
            //console.log(xrg)
            //console.log(yrg)
            //console.log('x-step = ', xst)
            //console.log('y-step = ', yst)
            //console.log('img.dims = (%d, %d)', imw, imh)
            //console.log('flatten img. dims. = %d', imw * imh)
            //console.log('flatten img. len = %d', img.length)
            //console.log('flatten point index = %d', flatten_pti)
            cds.data['x_hover'][0] = pxc
            cds.data['y_hover'][0] = pyc
            if (flatten_pti < img.length) {
                cds.data['z_hover'][0] = img[Math.floor(pxi + pyi * imw)]
            }
            else {
                cds.data['z_hover'][0] = NaN
            }
            //console.log('x, y, z = %f, %f, %f', pxc, pxi, cds.data['z_hover'][0])
            cds.change.emit()
        """)

    def __setup_toolbar(self, figure, w=0, h=0):
        hrd = [self._rrd]
        hcb = self.__hover_callback()
        htt = [('x,y,z:', '@x_hover{0.00},@y_hover{0.00},@z_hover{0.00}')]
        hpp = 'follow_mouse'
        # figure.add_tools(PanTool())
        figure.add_tools(BoxZoomTool())
        # figure.add_tools(WheelZoomTool())
        figure.add_tools(BoxSelectTool())
        figure.add_tools(ResetTool())
        figure.add_tools(SaveTool())
        figure.add_tools(HoverTool(tooltips=htt, renderers=hrd, point_policy=hpp, callback=hcb))
        figure.add_tools(CrosshairTool())
        figure.toolbar.logo = None
        figure.toolbar.active_drag = None
        figure.toolbar.active_scroll = None
        figure.toolbar.active_tap = None
        figure.toolbar_location = 'right' if w >= h else 'above'

    def get_model(self):
        """returns the Bokeh model (figure, layout, ...) associated with the Channel or None if no model"""
        return self._mdl

    @tracer
    def setup_model(self, **kwargs):
        """asks the channel to setup then return its Bokeh associated model - returns None if no model"""
        try:
            self._mdl = None
            props = self._merge_properties(self.model_properties, kwargs)
            self._cds = self.__instanciate_data_source()
            self._xsc = props.get('x_scale', Scale())
            self._xsc.validate()
            self._ysc = props.get('y_scale', Scale())
            self._ysc.validate()
            self._images_size_threshold = self.model_properties.get('images_size_threshold', self._images_size_threshold)
            # print('ImageChannel.setup_model.images_size_threshold: {:.00f}'.format(self._images_size_threshold))
            self._expected_image_shape = self.model_properties.get('full_frame_shape', self._expected_image_shape)
            # print('ImageChannel.setup_model.expected_image_shape: {}'.format(self._expected_image_shape))
            self.__setup_undefined_scales(self._expected_image_shape)
            fkwargs = dict()
            fkwargs['name'] = str(kwargs.get('uid', self.uid))
            fkwargs['output_backend'] = 'webgl'
            xrg = Range1d()  # self._xsc.bokeh_range
            yrg = Range1d()  # self._ysc.bokeh_range
            # print("ImageChannel.{}:set initial x-range to ({:.04f}, {:.04f})".format(self.name, xrg.start, xrg.end))
            # print("ImageChannel.{}:set initial y-range to ({:.04f}, {:.04f})".format(self.name, yrg.start, yrg.end))
            fkwargs['x_range'] = xrg
            fkwargs['y_range'] = yrg
            fkwargs['width'] = props.get('width', 320)
            fkwargs['height'] = props.get('height', 320)
            fkwargs['tools'] = ""
            f = figure(**fkwargs)
            self._cds.tags = [f.ref['id']]
            f.xaxis.axis_label = None if not self._xsc else self._xsc.axis_label()
            f.yaxis.axis_label = None if not self._ysc else self._ysc.axis_label()
            layout = kwargs.get('layout', 'column')
            if layout != 'tabs':
                f.title.text = self.name
            ikwargs = dict()
            ikwargs['x'] = 0
            ikwargs['y'] = 1
            ikwargs['dw'] = 1
            ikwargs['dh'] = 1
            ikwargs['image'] = 'image'
            ikwargs['source'] = self._cds
            ikwargs['color_mapper'] = LinearColorMapper(palette=props.get('palette', Plasma256))
            self._ird = f.image(**ikwargs)
            rkwargs = dict()
            rkwargs['x'] = 0
            rkwargs['y'] = 0
            rkwargs['width'] = 0
            rkwargs['height'] = 0
            rkwargs['fill_alpha'] = 0
            rkwargs['line_alpha'] = 0
            rkwargs['source'] = self._cds
            self._rrd = f.rect(**rkwargs)
            f.xgrid.grid_line_color = None
            f.ygrid.grid_line_color = None
            self._show_msg_label(f)
            self.__setup_toolbar(f, fkwargs['width'], fkwargs['height'])
            self._mdl = f
            bsm = props.get('selection_manager', None)
            if bsm:
                bsm.register_figure(f)
            self._itm.setup(self.bokeh_session, self._mdl, self.__handle_range_change)
        except Exception as e:
            self.error(e)
        return self._mdl

    def __setup_undefined_scales(self, img_shape):
        # print("__setup_undefined_scales.img_shape: {}".format(img_shape))
        if img_shape is None:
            return
        # print("__setup_undefined_scales._xsc.has_valid_scale: {}".format(self._xsc.has_valid_scale()))
        if not self._xsc.has_valid_scale():
            skwargs = dict()
            skwargs['start'] = 0
            skwargs['end'] = img_shape[1] - 1
            skwargs['num_points'] = img_shape[1]
            self._xsc = Scale(**skwargs)
        # print("__setup_undefined_scales._ysc.has_valid_scale: {}".format(self._ysc.has_valid_scale()))
        if not self._ysc.has_valid_scale():
            skwargs = dict()
            skwargs['start'] = 0
            skwargs['end'] = img_shape[0] - 1
            skwargs['num_points'] = img_shape[0]
            self._ysc = Scale(**skwargs)

    def __handle_range_change(self):
        try:
            sd = self._sd
            if not sd or sd.has_failed or not sum(sd.buffer.shape):
                return
            # print("ImageChannel.{}:handle_range_change: x-range changed to ({:.04f}, {:.04f})".format(self.name, self._mdl.x_range.start, self._mdl.x_range.end))
            # print("ImageChannel.{}:handle_range_change: y-range changed to ({:.04f}, {:.04f})".format(self.name, self._mdl.y_range.start, self._mdl.y_range.end))
            image = self.__extract_image_for_current_ranges(sd.buffer)
            new_data = dict()
            new_data['image'] = [image]
            new_data['image_width'] = [image.shape[1]]
            new_data['image_height'] = [image.shape[0]]
            self._cds.data.update(new_data)
            self._ird.glyph.update(x=self._mdl.x_range.start,
                                   y=self._mdl.y_range.start,
                                   dw=abs(self._mdl.x_range.end - self._mdl.x_range.start),
                                   dh=abs(self._mdl.y_range.end - self._mdl.y_range.start))
        except Exception as e:
            self.error(e)
        finally:
            self._itm.range_change_handled()

    def __image_shape_changed(self, image_shape):
        return self._current_image_shape != image_shape

    def __extract_image_for_current_ranges(self, image):
        xsc = self._mdl.x_range.start
        xec = self._mdl.x_range.end
        xx = np.linspace(self._xsc.start, self._xsc.end, num=image.shape[1], dtype=float)
        xy = np.linspace(0, image.shape[1] - 1, num=image.shape[1], dtype=int)
        # print("__extract_image_for_current_ranges.2")
        ysc = self._mdl.y_range.start
        yec = self._mdl.y_range.end
        # print("extract_image_for_current_ranges: x:({:.04f}, {:.04f}) - y:({:.04f} -> {:.04f})".format(xsc, xec, ysc, yec))
        yx = np.linspace(self._ysc.start, self._ysc.end, num=image.shape[0], dtype=float)
        yy = np.linspace(0, image.shape[0] - 1, num=image.shape[0], dtype=int)
        xsi = int(mt.floor(np.interp(xsc, xx, xy)))
        xei = int(mt.ceil(np.interp(xec, xx, xy)) + 1)
        ysi = int(mt.floor(np.interp(ysc, yx, yy)))
        yei = int(mt.ceil(np.interp(yec, yx, yy)) + 1)
        # print("extract_image_for_current_ranges: x:[{:.00f} -> {:.00f}] - y:[{:.00f} -> {:.00f}]".format(xsi, xei, ysi, yei))
        image = image[ysi:yei, xsi:xei]
        # print("extract_image_for_current_ranges.sub_image.shape: {}".format(image.shape))
        need_rescale, rescaling_factor = self.__compute_rescaling_factor(image)
        if need_rescale:
            image = self.__rescale_image(image, rescaling_factor)
            # print("extract_image_for_current_ranges.sub_image.rescaled to {}".format(image.shape))
        return image

    def __compute_rescaling_factor(self, image):
        try:
            rescaling_factor = 1.0
            initial_image_size = image_size = image.shape[0] * image.shape[1]
            # print("compute_rescaling_factor: size: {:.04f} - threshold: {:.04f}".format(image_size, image_size_threshold))
            if image_size <= self._images_size_threshold:
                # print("compute_rescaling_factor: no rescaling required")
                return False, rescaling_factor
            for inc in [0.1, 0.01, 0.001, 0.0001]:
                while image_size > self._images_size_threshold:
                    rescaling_factor -= inc
                    image_size = int(initial_image_size * rescaling_factor)
                rescaling_factor += inc
                image_size = int(initial_image_size * rescaling_factor)
            rescaling_factor = mt.sqrt(rescaling_factor)
            # print("compute_rescaling_factor.rescaling factor: {:.04f}".format(rescaling_factor))
            return True, rescaling_factor
        except Exception as e:
            print(e)

    def __rescale_image(self, in_img, rescaling_factor):
        # print('rescale-image: in shape {}'.format(in_img.shape))
        out_img = rescale(in_img, rescaling_factor, mode='constant', cval=np.nan)
        # print('rescale-image: out shape {}'.format(out_img.shape))
        return out_img

    def update(self, update_image=True):
        """gives each Channel a chance to update itself (e.g. to update the ColumnDataSources)"""
        try:
            ds = self.data_source
            if ds is None:
                return
            if update_image:
                self._sd = ds.pull_data()
            sd = self._sd
            previous_bad_source_cnt = self._bad_source_cnt
            if sd.has_failed:
                self._bad_source_cnt = 1
                self.emit_error(sd)
            elif previous_bad_source_cnt:
                self._bad_source_cnt = 0
                self.emit_recover()
            empty_buffer = sd.has_failed or not all(sd.buffer.shape)
            nan_buffer = None
            if empty_buffer:
                nan_buffer = np.empty((2, 2))
                nan_buffer.fill(np.nan)
                self._animate_msg_label()
            else:
                self._hide_msg_label()
            incoming_image = sd.buffer if not empty_buffer else nan_buffer
            image_shape_changed = self.__image_shape_changed(incoming_image.shape)
            self._current_image_shape = incoming_image.shape
            if not empty_buffer:
                self.__setup_undefined_scales(sd.buffer.shape)
            if empty_buffer:
                xss = -1.
                xse = 1.
                xst = 1.
                xpn = 3
            elif self._xsc.type != ScaleType.INDEXES:
                xss = self._xsc.start
                if self._expected_image_shape is None:
                    xse = self._xsc.start + ((sd.buffer.shape[1] - 1) * self._xsc.step)
                else:
                    xse = self._xsc.end
                xst = self._xsc.step
                xpn = sd.buffer.shape[1]
            else:
                xss = 0.
                xse = sd.buffer.shape[1]
                xst = 1.
                xpn = sd.buffer.shape[1]
            if empty_buffer:
                yss = -1.
                yse = 1.
                yst = 1.
                ypn = 3
            elif self._ysc.type != ScaleType.INDEXES:
                yss = self._ysc.start
                if self._expected_image_shape is None:
                    yse = self._ysc.start + ((sd.buffer.shape[0] - 1) * self._ysc.step)
                else:
                    yse = self._ysc.end
                yst = self._ysc.step
                ypn = sd.buffer.shape[0]
            else:
                yss = 0.
                yse = sd.buffer.shape[0]
                yst = 1.
                ypn = sd.buffer.shape[0]
            w = abs(xse - xss)
            h = abs(yse - yss)
            if not w:
                xss = -1.
                xse = 1.
            if not h:
                yss = -1.
                yse = 1.
            if image_shape_changed and not empty_buffer:  # TODO: remove 'and not empty_buffer'
                # print("ImageChannel.{}:changing x-range to ({:.04f}, {:.04f})".format(self.name, xss, xse))
                # print("ImageChannel.{}:changing y-range to ({:.04f}, {:.04f})".format(self.name, yss, yse))
                self._mdl.x_range.update(start=xss, end=xse)
                self._mdl.y_range.update(start=yss, end=yse)
                self._ird.glyph.update(x=xss, y=yss, dw=w, dh=h)
                self._rrd.glyph.update(x=xss + w / 2, y=yss + h / 2, width=w, height=h)
            else:
                x = self._mdl.x_range.start
                y = self._mdl.y_range.start
                dw = abs(self._mdl.x_range.end - self._mdl.x_range.start)
                dh = abs(self._mdl.y_range.end - self._mdl.y_range.start)
                self._ird.glyph.update(x=x, y=y, dw=dw, dh=dh)
                self._rrd.glyph.update(x=x + dw / 2, y=y + dh / 2, width=dw, height=dh)
            if not empty_buffer:
                image = self.__extract_image_for_current_ranges(sd.buffer)
            else:
                image = nan_buffer
            new_data = dict()
            new_data['image'] = [image]
            new_data['image_width'] = [image.shape[1]]
            new_data['image_height'] = [image.shape[0]]
            if image_shape_changed:
                new_data['image_shape_changed'] = [1]
                new_data['initial_x_range'] = [[xss, xse]]
                new_data['initial_y_range'] = [[yss, yse]]
            else:
                new_data['image_shape_changed'] = [0]
            self._cds.data.update(new_data)
        except Exception as e:
            self.error(e)

    def cleanup(self):
        self.__reinitialize()
        super(ImageChannel, self).cleanup()


# ------------------------------------------------------------------------------
class GenericChannel(Channel):
    """this is not supposed to be instanciated directly"""

    def __init__(self, name, data_source=None, model_properties=dict()):
        Channel.__init__(self, name, data_sources=[data_source], model_properties=model_properties)
        self._delegate = None
        self._delegate_model_id = str(uuid4().hex)
        self._delegate_model = None

    def get_model(self):
        """returns the Bokeh model (figure, layout, ...) associated with the Channel or None if no model"""
        return self._delegate_model

    @tracer
    def setup_model(self, **kwargs):
        """asks the channel to setup then return its Bokeh associated model - returns None if no model"""
        try:
            self.update()
        finally:
            return self._delegate_model

    def update(self):
        """gives each Channel a chance to update itself (e.g. to update the ColumDataSources)"""
        if self._delegate:
            self._delegate.update()
            return
        try:
            ds = self.data_source
            if ds is None:
                return
            sd = ds.pull_data()
            previous_bad_source_cnt = self._bad_source_cnt
            if sd.has_failed:
                self._bad_source_cnt = 1
                self.emit_error(sd)
                return
            elif previous_bad_source_cnt:
                self._bad_source_cnt = 0
                self.emit_recover()
            self.model_properties['uid'] = self.uid
            if sd.format == ChannelData.Format.SCALAR:
                #self.info("GenericChannel.update.instanciating SCALAR channel")
                self._delegate = ScalarChannel(name=self.name,
                                               data_sources=[self.data_source],
                                               model_properties=self.model_properties)
            elif sd.format == ChannelData.Format.SPECTRUM:
                #self.info("GenericChannel.update.instanciating SPECTRUM channel")
                self._delegate = SpectrumChannel(name=self.name,
                                                 data_sources=[self.data_source],
                                                 model_properties=self.model_properties)
            elif sd.format == ChannelData.Format.IMAGE:
                #self.info("GenericChannel.update.instanciating IMAGE channel")
                self._delegate = ImageChannel(name=self.name,
                                              data_source=self.data_source,
                                              model_properties=self.model_properties)
            if self._delegate:
                self._delegate.output = self.output
                events = [DataStreamEvent.Type.ERROR, DataStreamEvent.Type.RECOVER, DataStreamEvent.Type.EOS]
                self._delegate.register_event_handler(self, events)
                self._delegate.bokeh_session = self.bokeh_session
                self._delegate_model = self._delegate.setup_model()
                self.emit_model_changed(self._delegate_model)
        except Exception as e:
            self.error(e)

    def cleanup(self):
        self._delegate = None
        super(GenericChannel, self).cleanup()


# ------------------------------------------------------------------------------
class LayoutChannel(Channel):
    """simulates a data monitor handling several image data sources"""

    def __init__(self, name, channels=None, model_properties=None):
        Channel.__init__(self, name, model_properties=model_properties)
        # model
        self._mdl = None
        # tabs widget (layout option)
        self._tabs_widget = None
        # sub-channels
        self._channels = Children(self, Channel)
        self._channels.register_add_callback(self.__on_add_channel)
        self.add(channels)

    @Channel.bokeh_session.setter
    def bokeh_session(self, bks):
        """overwrites Channel.bokeh_session.setter"""
        assert (isinstance(bks, BokehSession))
        self._session = bks
        for channel in self._channels.values():
            channel.bokeh_session = bks

    @NotebookCellContent.output.setter
    def output(self, new_output):
        """overwrites NotebookCellContent.output.setter"""
        self._output = new_output
        for channel in self._channels.values():
            channel.output = new_output

    def add(self, channels):
        """add the specified (sub)channels"""
        self._channels.add(channels)

    def __on_add_channel(self, channel):
        """called when a sub-channel is added to this channel"""
        if channel is not self:
            channel.output = self.output
            events = [DataStreamEvent.Type.ERROR, DataStreamEvent.Type.RECOVER, DataStreamEvent.Type.MODEL_CHANGED]
            channel.register_event_handler(self, events)

    def handle_stream_event(self, event):
        assert (isinstance(event, DataStreamEvent))
        if event.type == DataStreamEvent.Type.MODEL_CHANGED:
            self.__on_model_changed(event)

    def __on_model_changed(self, event):
        """
        if self._layout and self._layout.children and event.emitter and event.data:
            for c in self._layout.children:
                if c.name == event.emitter:
                    self._layout.children.remove(c)
                    self._layout.children.append(event.data)
        """
        # TODO: handle DataStreamEvent.Type.MODEL_CHANGED
        pass

    @property
    def model(self):
        """returns the Bokeh model (figure, layout, ...) associated with the Channel or None if no model"""
        return self._mdl if self._mdl else self.setup_model()

    @classmethod
    def __model_width(cls, model):
        if hasattr(model, 'plot_width'):
            # print("model_width:plot_width={}".format(model.plot_width))
            w = model.plot_width
        elif hasattr(model, 'width'):
            # print("model_width:width={}".format(model.width))
            w = model.width
        else:
            # print("model_width:width=six.MAXSIZE={}".format(six.MAXSIZE))
            w = six.MAXSIZE
        return w

    @classmethod
    def __model_height(cls, model):
        if hasattr(model, 'plot_height'):
            # print("model_width:plot_height={}".format(model.plot_height))
            h = model.plot_height
        elif hasattr(model, 'height'):
            # print("model_width:height={}".format(model.height))
            h = model.height
        else:
            # print("model_width:height=six.MAXSIZE={}".format(six.MAXSIZE))
            h = six.MAXSIZE
        return h

    def __setup_layout(self, children, **kwargs):
        """setup the layout"""
        layout = kwargs.get('layout', 'column')
        if layout == 'tabs':
            return self.__setup_tabs_layout(children, **kwargs)
        elif layout == 'grid':
            return self.__setup_grid_layout(children, **kwargs)
        else:
            return self.__setup_column_layout(children, **kwargs)

    def __setup_grid_layout(self, children, **kwargs):
        """spread the children in rows"""
        try:
            if len(children) > 1:
                num_columns, width_sum = 1, 0
                # print("setup_grid_layout: images layout contains {} children".format(len(children)))
                for c in children.values():
                    width_sum += self.__model_width(c)
                    if width_sum <= 600:
                        num_columns += 1
                    else:
                        break
                num_rows = int(ceil(float(len(children)) / float(num_columns)))
                # print("num_columns={} - num_rows={}".format(num_columns, num_rows))
                rl = list()
                for i in range(num_rows):
                    rl.append([None for i in range(num_columns)])
                ri, fi = 0, 0
                for c in children.values():
                    rl[ri][fi] = c
                    fi = (fi + 1) % num_columns
                    if not fi:
                        ri += 1
                merge_tools = kwargs.get('merge_tools', True)
                if merge_tools:
                    tbo = dict()
                    tbo['merge_tools'] = True
                    tbo['logo'] = None
                    tbo['toolbar_location'] = 'above'
                    return gridplot(children=rl, toolbar_options=tbo)
            else:
                return layout(children=children.values())
        except Exception as e:
            raise

    def __setup_column_layout(self, children, **kwargs):
        """spread the children in rows"""
        ml = list()
        for c in children.values():
            ml.append(c)
        return column(name=str(self.uid), children=ml, responsive=True)

    def __setup_tabs_layout(self, children, **kwargs):
        """spread the children in tabs"""
        tl = list()
        for cn, ci in six.iteritems(children):
            tl.append(BokehWidgets.Panel(child=ci, title=cn))
        self._tabs_widget = BokehWidgets.Tabs(tabs=tl)
        self._tabs_widget.on_change("active", self.on_tabs_selection_change)
        return column(name=str(self.uid), children=[self._tabs_widget], responsive=True)

    def get_model(self):
        """returns the Bokeh model (figure, layout, ...) associated with the Channel or None if no model"""
        return self._mdl

    @tracer
    def setup_model(self, **kwargs):
        """asks the channel to setup then return its Bokeh associated model - returns None if no model"""
        try:
            self._mdl = None
            props = self._merge_properties(self.model_properties, kwargs)
            cd = OrderedDict()
            for cn, ci in six.iteritems(self._channels):
                cd[cn] = ci.setup_model(**props)
            self._mdl = self.__setup_layout(cd, **props)
        except Exception as e:
            self.error(e)
        return self._mdl

    def on_tabs_selection_change(self, attr, old, new):
        self.__update_tabs_selection()

    def __update_tabs_selection(self):
        # TODO: we might face a race condition between server periodic callback and user action: mutex required?
        at = self._tabs_widget.active
        cn = self._tabs_widget.tabs[at].title
        self._channels[cn].update()

    def update(self):
        try:
            if self._tabs_widget:
                self.__update_tabs_selection()
            else:
                for c in self._channels.values():
                    c.update()
        except Exception as e:
            self.error(e)


# ------------------------------------------------------------------------------
class DataStream(NotebookCellContent, DataStreamEventHandler):
    """data stream interface"""

    def __init__(self, name, channels=None):
        NotebookCellContent.__init__(self, name, logger=logging.getLogger(plots_module_logger_name))
        DataStreamEventHandler.__init__(self, name)
        # bokeh session
        self._session = None
        # channels
        self._channels = Children(self, Channel)
        self._channels.register_add_callback(self._on_add_channel)
        self.add(channels)

    @property
    def bokeh_session(self):
        """return the associated bokeh session"""
        return self._session

    @bokeh_session.setter
    def bokeh_session(self, bks):
        """set the associated bokeh session"""
        assert (isinstance(bks, BokehSession))
        self._session = bks
        for channel in self._channels.values():
            channel.bokeh_session = bks

    @NotebookCellContent.output.setter
    def output(self, new_output):
        """overwrites NotebookCellContent.output.setter"""
        self._output = new_output
        for channel in self._channels.values():
            channel.output = new_output

    def add(self, channels):
        """add the specified channels"""
        self._channels.add(channels)

    def _on_add_channel(self, channel):
        """called when a new channel is added to the data stream"""
        channel.output = self.output
        events = [DataStreamEvent.Type.ERROR, DataStreamEvent.Type.RECOVER, DataStreamEvent.Type.MODEL_CHANGED]
        channel.register_event_handler(self, events)

    def handle_stream_event(self, event):
        assert (isinstance(event, DataStreamEvent))
        pass

    def get_models(self):
        """returns the Bokeh model (figure, layout, ...)s associated with the DataStream"""
        return [channel.get_model() for channel in self._channels.values()]

    def setup_models(self):
        """returns the Bokeh model (figure, layout, ...)s associated with the DataStream"""
        models = list()
        for channel in self._channels.values():
            model = channel.setup_model()
            #if channel.show_title and channel.title is not None:
            #    div_txt = "<b>{}</b>".format(channel.title)
            #    models.append(Div(text=div_txt))
            if model:
                models.append(model)
        return models

    def update(self):
        """gives each Channel a chance to update itself (e.g. to update the ColumDataSources)"""
        # print("data stream: {} update".format(self.name))
        for channel in self._channels.values():
            try:
                channel.update()
            except Exception as e:
                self.error(e)

    def cleanup(self):
        """asks each Channel to cleanup itself (e.g. release resources)"""
        for channel in self._channels.values():
            try:
                self.info("DataStream: cleaning up Channel {}".format(channel.name))
                channel.cleanup()
            except Exception as e:
                self.error(e)

                
# ------------------------------------------------------------------------------
class DataStreamer(NotebookCellContent, DataStreamEventHandler, BokehSession):
    """a data stream manager embedded a bokeh server"""

    def __init__(self, name, data_streams, update_period=None, auto_start=False, start_delay=0., output=None):
        # route output to current cell
        NotebookCellContent.__init__(self,
                                     name,
                                     logger=logging.getLogger(plots_module_logger_name),
                                     output=output)
        DataStreamEventHandler.__init__(self, name)
        BokehSession.__init__(self)
        # a FIFO to store incoming DataStreamEvent
        self._events = deque()
        # update period in seconds
        self.callback_period = update_period
        # the data streams
        self._data_streams = list()
        self.add(data_streams)
        # auto start
        self._auto_start = auto_start
        # start delay
        self._start_delay = start_delay
        # open the session
        if auto_start:
            self.open()

    @property
    def bokeh_session(self):
        """return the associated bokeh session"""
        return self

    @NotebookCellContent.output.setter
    def output(self, new_output):
        """overwrites NotebookCellContent.output.setter"""
        self._output = new_output
        for ds in self._data_streams:
            ds.output = new_output
    
    def add(self, ds):
        events = [DataStreamEvent.Type.ERROR, DataStreamEvent.Type.RECOVER, DataStreamEvent.Type.MODEL_CHANGED]
        if isinstance(ds, DataStream):
            ds.bokeh_session = self
            ds.output = self.output
            ds.register_event_handler(self, events)
            self._data_streams.append(ds)
        elif isinstance(ds, (list, tuple)):
            for s in ds:
                if not isinstance(s, DataStream):
                    raise ValueError("invalid argument: expected a list, a tuple or a single instance of DataStream")
                s.bokeh_session = self
                s.output = self.output
                s.register_event_handler(self, events)
                self._data_streams.append(s)
        else:
            raise ValueError("invalid argument: expected a list, a tuple or a single instance of DataStream")

    @tracer
    def open(self):
        """open the session and optionally start it """
        with self.output:
            super(DataStreamer, self).open()

    @tracer
    def close(self):
        """close the session"""
        # suspend periodic callback
        self.pause()
        # the underlying actions will be performed under critical section
        self.safe_document_modifications(self.__close)
        
    @tracer
    def __close(self):
        """close/cleanup everything"""
        # cleanup each data stream
        for ds in self._data_streams:
            try:
                self.debug("DataStreamer: cleaning up DataStream {}".format(ds.name))
                ds.cleanup()
            except Exception as e:
                self.error(e)
        self.debug("DataStreamer: closing Bokeh session...")
        # delegate the remaining actions to our super class (this is mandatory)
        try:
            super(DataStreamer, self).cleanup(async=False)
            self.debug("DataStreamer: Bokeh session closed")
        except Exception as e:
            self.error(e)
        
    @tracer
    def start(self, delay=0.):
        """start periodic activity"""
        if not self.ready:
            self.debug("DataStreamer.start:session not ready:set auto_start to True")
            self._auto_start = True
        else:
            actual_tmo = delay if delay > 0. else self._start_delay
            if actual_tmo > 0.:
                self.debug('DataStreamer.start: actual start in {} seconds'.format(actual_tmo))
                self._start_delay = 0.
                self.timeout_callback(self.start, actual_tmo)
            else:
                self.debug("DataStreamer.start: session ready, no delay, resuming...")
                self.resume()

    @tracer
    def stop(self):
        """stop periodic activity"""
        if not self.ready:
            self._auto_start = False
        else:
            self.pause()

    @tracer
    def setup_document(self):
        """add the data stream models to the bokeh document"""
        try:
            models = list()
            for ds in self._data_streams:
                try:
                    models.extend(ds.setup_models())
                except Exception as e:
                    self.error(e)
            try:
                self.periodic_callback()
            except:
                pass
            try:
                for model in models:
                    self.document.add_root(model, setter=self.bokeh_session_id)
                if self._auto_start:
                    self.start(self._start_delay)
            except Exception as e:
                self.error(e)
        except Exception as e:
            self.error(e)
            raise

    def handle_stream_event(self, event):
        assert (isinstance(event, DataStreamEvent))
        if event.type == DataStreamEvent.Type.MODEL_CHANGED:
            self._events.appendleft(event)
            self.safe_document_modifications(self.__on_model_changed)

    def __on_model_changed(self):
        event = self._events.pop()
        if event.emitter and event.data:
            # print("handling DataStreamEvent.Type.MODEL_CHANGED")
            if len(self.document.roots):
                for root in self.document.roots:
                    if root.name == str(event.emitter):
                        # print("removing figure {}".format(root.name))
                        self.document.remove_root(root)
            try:
                # print("adding new root {}:{} to document".format(event.data, event.data.name))
                self.document.add_root(event.data, setter=self.bokeh_session_id)
            except Exception as e:
                self.error(e)
                # print("DataStreamEvent.Type.MODEL_CHANGED successfully handled")

    @property
    def update_period(self):
        """returns the update period (in seconds)"""
        return self.callback_period

    @update_period.setter
    def update_period(self, up):
        """set the update period (in seconds)"""
        self.update_callback_period(up)

    def periodic_callback(self):
        """the session periodic callback"""
        for ds in self._data_streams:
            try:
                ds.update()
            except Exception as e:
                self.error(e)


# ------------------------------------------------------------------------------
class DataStreamerController(NotebookCellContent, DataStreamEventHandler):
    """a DataStreamer controller"""

    def __init__(self, name, data_streamer=None, **kwargs):
        try:
            # check input parameters
            assert(isinstance(data_streamer, DataStreamer))
            # route output to current cell
            NotebookCellContent.__init__(self,
                                         name,
                                         output=kwargs.get('output', None),
                                         logger=logging.getLogger(plots_module_logger_name))
            DataStreamEventHandler.__init__(self, name)
            # start/stop/close button
            self.__setup_controls(data_streamer, **kwargs)
            # data streamer
            self.data_streamer = data_streamer
            # function called when the close button is clicked
            self._close_callbacks = list()
            # auto-start
            self._running = False
            auto_start = kwargs.get('auto_start', True)
            if auto_start:
                self._data_streamer.open()
                self._data_streamer.start()
                self.__on_freeze_unfreeze_clicked()
        except Exception as e:
            print(e)

    @staticmethod
    def l01a(width='auto', *args, **kwargs):
        return ipw.Layout(flex='0 1 auto', width=width, *args, **kwargs)

    @staticmethod
    def l11a(width='auto', *args, **kwargs):
        return ipw.Layout(flex='1 1 auto', width=width, *args, **kwargs)

    def __setup_update_period_slider(self, data_streamer, **kwargs):
        return ipw.FloatSlider(
            value=data_streamer.update_period,
            min=kwargs.get('min_refresh_period', 0.25),
            max=kwargs.get('max_refresh_period', 5.0),
            step=kwargs.get('step_refresh_period', 0.25),
            description='Update period (s)',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            style={'description_width':'initial'}
        )

    def __setup_controls(self, data_streamer, **kwargs):
        self._error_area = None
        self._error_area_enabled = kwargs.get('error_area_enabled', True)
        if kwargs.get('up_slider_enabled', True):
            self._up_slider = self.__setup_update_period_slider(data_streamer, **kwargs)
            self._up_slider.observe(self.__on_refresh_period_changed, names='value')
        else:
            self._up_slider = None
        bd = "Freeze" if kwargs.get('auto_start', True) else "Unfreeze"
        self._freeze_unfreeze_button = ipw.Button(description=bd, layout=self.l01a(width="100px"))
        self._freeze_unfreeze_button.on_click(self.__on_freeze_unfreeze_clicked)
        self._close_button = ipw.Button(description="Close", layout=self.l01a(width="100px"))
        self._close_button.on_click(self.__on_close_clicked)
        self._switch_buttons_to_valid_state()
        widgets_list = list()
        title = kwargs.get('title', None)
        if title:
            div = ipw.HTML(value="<b>{}</b>:".format(title))
            widgets_list.append(div)
            lbl = ipw.HTML(value="", layout=self.l01a(width="40px"))
            widgets_list.append(lbl)
        if self._up_slider:
            widgets_list.append(self._up_slider)
        widgets_list.extend([self._freeze_unfreeze_button, self._close_button])
        main_controls = ipw.HBox(widgets_list, layout=self.l01a())
        self._ea_output = ipw.Output(layout=self.l11a())
        self._ea_output.layout.border = "1px solid grey"
        self._ds_output = ipw.Output(layout=self.l11a())
        #self._ds_output.layout.border = "1px solid red"
        self._controls = ipw.VBox([main_controls, self._ea_output, self._ds_output], layout=self.l01a())
        self.display(self._controls)

    def __on_refresh_period_changed(self, event):
        try:
            self.data_streamer.update_period = event['new']
        except Exception as e:
            self.error(e)

    def __on_freeze_unfreeze_clicked(self, b=None):
        if self._running:
            self._data_streamer.stop()
            self._freeze_unfreeze_button.description = "Unfreeze"
        else:
            self._data_streamer.start()
            self._freeze_unfreeze_button.description = "Freeze"
        self._running = not self._running
        if self._running and self._up_slider is not None:
            self._up_slider.value = self.data_streamer.update_period

    def __on_close_clicked(self, b=None):
        self.close()

    def start(self):
        try:
            self.info("DataStreamerController : starting DataStreamer {}".format(self._data_streamer.name))
            self._data_streamer.start()
        except Exception as e:
            self.error(e)

    def close(self):
        try:
            self.info("DataStreamerController : closing DataStreamer {}".format(self._data_streamer.name))
            self._data_streamer.close()
        except Exception as e:
            self.error(e)
        self._ea_output.clear_output()
        self._ea_output.close()
        self._ds_output.clear_output()
        self._ds_output.close()
        self._controls.close()

    def register_close_callback(self, cb, kwargs=None):
        assert (hasattr(cb, '__call__'))
        self._close_callbacks.append({'func':cb, 'args':kwargs})

    def __call_close_callbacks(self):
        for cb in self._close_callbacks:
            try:
                cb['func'](cb['args'])
            except Exception as e:
                print(e)

    def handle_stream_event(self, event):
        assert (isinstance(event, DataStreamEvent))
        if event.type == DataStreamEvent.Type.ERROR:
            self.__on_stream_error(event)
        elif event.type == DataStreamEvent.Type.RECOVER:
            self.__on_stream_recover(event)
        elif event.type == DataStreamEvent.Type.EOS:
            self.__on_end_of_stream(event)

    def __on_stream_error(self, event):
        self._switch_buttons_to_invalid_state()
        self._show_error(event.error)

    def __on_stream_recover(self, event):
        self._switch_buttons_to_valid_state()
        self._hide_error()

    def __on_end_of_stream(self, event):
        self.__on_freeze_unfreeze_clicked()

    def _switch_buttons_to_valid_state(self):
        self._close_button.style.button_color = '#00FF00'
        self._freeze_unfreeze_button.style.button_color = '#00FF00'

    def _switch_buttons_to_invalid_state(self):
        self._close_button.style.button_color = '#FF0000'
        self._freeze_unfreeze_button.style.button_color = '#FF0000'

    def _show_error(self, err_desc):
        if not self._error_area_enabled:
            return
        err = "Oops, the following error occurred:\n"
        err += err_desc
        if self._error_area is None:
            self._error_area = ipw.Textarea(value=err, rows=3, layout=self.l11a()) 
            with self._ea_output:
                display(self._error_area)
        else:
            self._error_area.value = err
            self._error_area.rows = 3
            self._error_area.disabled = False
        self._ea_output.layout.border = ""

    def _hide_error(self):
        if self._error_area:
            self._error_area.close()
            self._error_area = None
        self._ea_output.clear_output()
        self._ea_output.layout.border = "1px solid grey"

    @property
    def data_streamer(self):
        return self._data_streamer

    @data_streamer.setter
    def data_streamer(self, ds):
        # check input parameter
        assert (isinstance(ds, DataStreamer))
        # data streamer
        self._data_streamer = ds
        # route data streamer output to current cell
        self._data_streamer.output = self._ds_output
        # register event handler
        events = [DataStreamEvent.Type.ERROR, DataStreamEvent.Type.RECOVER, DataStreamEvent.Type.EOS]
        self._data_streamer.register_event_handler(self, events)

