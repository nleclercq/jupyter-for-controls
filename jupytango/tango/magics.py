from __future__ import print_function
import re
import logging

from IPython.display import display, clear_output
from IPython.core.page import page
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring, argument_group, UsageError

try:
    # just in case tango stuffs are not installed
    import PyTango as tango
except:
    pass

from jupytango.tools import silent_catch
from jupytango.tango.monitor import open_tango_monitor, plot_tango_attribute

try:
    # try to import itango modules
    from itango.itango import __DeviceProxy_completer as complete_device_proxy
    from itango.itango import __AttributeProxy_completer as complete_attribute_proxy
    from itango.itango import __get_device_proxy as get_device_proxy
except:
    def complete_attribute_proxy(shell, evt):
        del shell, evt  # unused
        raise Exception("complete_attribute_proxy is not defined")
    def complete_device_proxy(shell, evt):
        del shell, evt  # unused
        raise Exception("complete_device_proxy is not defined") 
    def get_device_proxy(shell, evt):
        del shell, evt  # unused
        raise Exception("get_device_proxy is not defined")


module_logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
tango_db = tango.Database()

# ------------------------------------------------------------------------------
def attribute_key(name):
    """compute a regex key matching an attribute setting"""
    return r".*\.{}\s*=".format(re.escape(name))

def magic_key(name):
    """compute a regex key matching a magic call"""
    return '%?{}'.format(re.escape(name))

def str_param_key(name):
    """compute a regex key matching a function call parameter which is expected to be a string"""
    return r'''.*{}\(['"]\w*$'''.format(re.escape(name))

# ------------------------------------------------------------------------------
@magics_class
class JupytangoMagics(Magics):
    """This class defines the magics related to the flyScan extension"""

    aliases = {
        'plot_tango_attribute': 'pta',
        'tango_monitor': 'tgm'
    }

    completers = {
        'plot_tango_attribute': complete_attribute_proxy,
        'tango_monitor': complete_attribute_proxy
    }

    # logger
    _logger = module_logger

    @classmethod
    def iter_hooks(cls):
        """return hooks that must be register as 'complete_command'"""
        # magic completion
        for magic_name, completer in cls.completers.items():
            yield magic_key(magic_name), completer
            with silent_catch(KeyError):
                yield magic_key(cls.aliases[magic_name]), completer

    # -----------------
    # magics definition

    @magic_arguments()
    @argument('attr', type=str, help='fully qualified attribute name or alias')
    @argument('-w', '--width', type=int, help='optional plot width in pixels')
    @argument('-h', '--height', type=int,  help='optional plot height in pixels')
    @argument('-u', '--upsidedown', action='store_true', help='display image upsidedown')
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
    @argument('-u', '--upsidedown', action='store_true', help='display image upsidedown')
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
        if ns.upsidedown is not None:
            kwargs['upsidedown'] = ns.upsidedown
        open_tango_monitor(**kwargs)
        

# ------------------------------------------------------------------------------
def load_ipython_extension(shell):
    # ----------------------------------------
    def register_alias(shell, alias_name, magic_name):
        """register an IPython alias (support old versions of IPython)"""
        mm = shell.magics_manager
        if hasattr(mm, 'register_alias'):
            return mm.register_alias(alias_name, magic_name)
        wrapper = lambda line: mm.magics['line'][magic_name](line)
        wrapper.__name__ = alias_name
        wrapper.__doc__ = "Alias for `%{}`".format(magic_name)
        mm.register_function(wrapper)
    # ----------------------------------------
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
    print("jupytango magics registered - type '%fsh' for help. Have a nice session...")

    
# ------------------------------------------------------------------------------
def unload_ipython_extension(shell):
    pass

