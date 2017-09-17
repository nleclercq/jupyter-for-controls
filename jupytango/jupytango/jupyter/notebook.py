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

"""jupytango"""

from __future__ import print_function
import datetime

# bokeh modules
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.tools import HoverTool
from bokeh.palettes import Plasma256
from bokeh.io import show, output_notebook, reset_output
from bokeh.plotting import figure
from bokeh.resources import INLINE

bokeh_redirected = False

# tango module
try:
    import tango
except:
    import PyTango as tango


# ------------------------------------------------------------------------------
def load_ipython_extension(shell):
    pass


# ------------------------------------------------------------------------------
def unload_ipython_extension(shell):
    global bokeh_redirected
    bokeh_redirected = False


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
def plot_tango_attribute(ns):
    redirect_bokeh_output()
    n = ns.attr.count("/")
    if not n:
        ap = tango.AttributeProxy(ns.attr)
        av = ap.read()
        fqan = ap.get_device_proxy().name() + "/" + ap.name()
    elif n == 3:
        dn, _, an = ns.attr.rpartition("/")
        dp = tango.DeviceProxy(dn)
        av = dp.read_attribute(an)
        fqan = ns.attr
    else:
        raise Exception("invalid attribute name specified - expected an alias or something like 'fully/qualified/attribute/name'")
    kwargs = dict()
    kwargs['tools'] = 'pan,box_zoom,wheel_zoom,reset,hover'
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
        show(plot, notebook_handle=True)

