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
import re
import logging

try:
  from traitlets.config import SingletonConfigurable 
  from traitlets import CUnicode, Enum, Instance
except:
  from IPython.config import SingletonConfigurable
  from IPython.utils.traitlets import CUnicode, Enum, Instance

from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from jupytango.utils import ColoredFormatter, register_alias, spawn_atk, spawn_jive, spawn_astor

from jupytango.jupyter.monitors import open_tango_monitor
from jupytango.jupyter.notebook import plot_tango_attribute

try:
    # try to import tango modules
    import tango
except:
    import PyTango as tango

from itango.itango import __DeviceProxy_completer as complete_device_proxy
from itango.itango import __AttributeProxy_completer as complete_attribute_proxy

fs_logger = logging.getLogger('jupytango.magics')

# =====
# Tools
# =====

def attribute_key(name):
    """compute a regex key matching an attribute setting"""
    return r".*\.{}\s*=".format(re.escape(name))


def magic_key(name):
    """compute a regex key matching a magic call"""
    return '%?{}'.format(re.escape(name))


def str_param_key(name):
    """compute a regex key matching a function call parameter which is expected to be a string"""
    return r'''.*{}\(['"]\w*$'''.format(re.escape(name))

# ================
# Extension Magics
# ================

@magics_class
class JupytangoMagics(Magics):
    """This class defines the magics related to the Jupytango extension"""

    # ------------------
    # static definitions
    # ------------------

    aliases = {
        'plot_tango_attribute': 'pta',
        'tango_monitor': 'tm',
        'open_atkpanel': 'atk',
        'open_jive': 'jive',
        'open_astor': 'astor'
    }

    completers = {
        'plot_tango_attribute': complete_attribute_proxy,
        'tango_monitor': complete_attribute_proxy,
        'open_atkpanel': complete_device_proxy
    }

    # logger
    _logger = fs_logger

    @classmethod
    def iter_hooks(cls):
        """return hooks that must be register as 'complete_command'"""
        # magic completion
        for magic_name, completer in cls.completers.items():
            yield magic_key(magic_name), completer
            try:
                yield magic_key(cls.aliases[magic_name]), completer
            except KeyError:
                pass
            except Exception as e:
                print(e)

    # -----------------
    # magics definition
    # -----------------
    # NOTE missing 'reload', 'reinitialize', 'set_cfg', 'start', 'pause', 'resume'

    @magic_arguments()
    @argument('attr', type=str, help='fully qualified attribute name or alias')
    @argument('-w', '--width', type=int, help='optional plot width in pixels')
    @argument('-h', '--height', type=int,  help='optional plot height in pixels')
    @line_magic
    def plot_tango_attribute(self, line):
        """Plot the specified Tango device attribute"""
        ns = parse_argstring(self.plot_tango_attribute, line)
        plot_tango_attribute(ns)

    @magic_arguments()
    @argument('attr', type=str, help='fully qualified attribute name or alias')
    @argument('-w', '--width', type=int, help='optional plot width in pixels')
    @argument('-h', '--height', type=int,  help='optional plot height in pixels')
    @argument('-p', '--period',  type=float, help='plot refresh period in [0.1, 5] seconds - defaults to 1s')
    @argument('-d', '--depth',  type=float, help='history depth in [1, 3600] seconds - defaults to 900s')
    @line_magic
    def tango_monitor(self, line):
        """Monitor the specified Tango device attribute"""
        ns = parse_argstring(self.tango_monitor, line)
        kwargs = dict()
        kwargs['attribute'] = ns.attr
        if ns.period is not None:
            kwargs['refresh_period'] = ns.period
        if ns.depth is not None:
            kwargs['history_buffer_depth'] = ns.depth
        if ns.width is not None:
            kwargs['width'] = ns.width
        if ns.height is not None:
            kwargs['height'] = ns.height
        open_tango_monitor(**kwargs)

    @magic_arguments()
    @argument('name', help='device name')
    @line_magic
    def open_atkpanel(self, line):
        """Open an AtkPanel for the specified Tango device"""
        ns = parse_argstring(self.open_atkpanel, line)
        spawn_atk(ns.name)

    @line_magic
    def open_jive(self, line):
        """Open Jive"""
        spawn_jive()

    @line_magic
    def open_astor(self, line):
        """Open Astor"""
        spawn_astor()

# =========
# Extension
# =========

def load_ipython_extension(shell):
    print("Welcome to the JupyTango.")
    # configure some logging if no handler is specified for root or fs
    if not logging.root.handlers and not fs_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter("%(levelname)s [%(name)s] %(message)s"))
        fs_logger.addHandler(handler)
    # register magics
    shell.register_magics(JupytangoMagics)
    # register magics aliases
    for magic_name, alias_name in JupytangoMagics.aliases.items():
        register_alias(shell, alias_name, magic_name)
    # register completers
    for key, completer in JupytangoMagics.iter_hooks():
        shell.set_hook("complete_command", completer, re_key=key)
    shell.set_hook('complete_command', complete_device_proxy, re_key = ".*DeviceProxy[^\w\.]+")
    shell.set_hook('complete_command', complete_device_proxy, re_key = ".*Device[^\w\.]+")
    shell.set_hook('complete_command', complete_attribute_proxy, re_key = ".*AttributeProxy[^\w\.]+")
    shell.set_hook('complete_command', complete_attribute_proxy, re_key = ".*Attribute[^\w\.]+")
    print("JupyTango magics registered. Have a nice session...")


def unload_ipython_extension(shell):
    print('Leaving the JupyTango...')
    print('Bye!')

