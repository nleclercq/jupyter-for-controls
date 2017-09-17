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
import os
import sys
import signal
import logging
import subprocess

from IPython.utils.io import ask_yes_no
from IPython.core.interactiveshell import InteractiveShell

from jupytango.tools.display import pretty, Ansi

class ColoredFormatter(logging.Formatter):

    modes = {
        logging.INFO: dict(fg='green'),
        logging.WARNING: dict(fg='red'),
        logging.ERROR: dict(fg='red'),
        logging.FATAL: dict(fg='red'),
    }

    def format(self, record):
        string = logging.Formatter.format(self, record)
        kwargs = self.modes.get(record.levelno, {})
        return Ansi.graphics(string, **kwargs)


def ask_yn(question, default=None, interrupt=None):
    """alias for IPython.utils.io.ask_yes_no with smart prompt options"""
    y_choice = '[y]' if default == 'y' else 'y'
    n_choice = '[n]' if default == 'n' else 'n'
    prompt = '{} ({}/{})?'.format(question, y_choice, n_choice)
    return ask_yes_no(prompt, default)


def running_in_jupyter_notebook():
    return 'ipykernel' in sys.modules


@pretty
class String(str):
    """
    Simple wrapper that lets IPython display strings like 'print' instead of 'repr'
    On Windows, it also provides ANSI sequence support
    """
    pass


def ignore_signals():
    """preexec function called in the process that let it ignore KeyboardInterrupt signals"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def program_name(name):
    return name if sys.platform != 'win32' else "{}.bat".format(name)


class Spawn(object):
    """Callable object that wraps a call to a system command"""

    def __init__(self, program_name):
        self.program_name = str(program_name)

    def __call__(self, *args, **kwargs):
        """
        spawns the program
        - args should contain arguments (as strings) to pass on the command line
        - kwargs are forwarded to the subprocess.Popen
        default kwargs discard standard output/error and let signals be ignored (avoiding SIGINT forwarding)
        """
        full_kwargs = dict(stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
        if sys.platform != "win32":  # preexec function is not supported on Windows
            full_kwargs['preexec_fn'] = ignore_signals
        full_kwargs.update(kwargs)
        return subprocess.Popen([self.program_name] + list(args), **full_kwargs)


spawn_atk = Spawn(program_name('atkpanel'))
spawn_jive = Spawn(program_name('jive-rw'))
spawn_astor = Spawn(program_name('astor'))


def register_alias(shell, alias_name, magic_name):
    """register an IPython alias (support old versions of IPython)"""
    mm = shell.magics_manager
    if hasattr(mm, 'register_alias'):
        return mm.register_alias(alias_name, magic_name)
    wrapper = lambda line: mm.magics['line'][magic_name](line)
    wrapper.__name__ = alias_name
    wrapper.__doc__ = "Alias for `%{}`".format(magic_name)
    mm.register_function(wrapper)


def get_ipython():
    if not InteractiveShell.initialized():
        raise Exception('no shell initialized')
    return InteractiveShell.instance()


def get_doc(fn):
    """returns docstring of a wrapped function"""
    # fn = getattr(fn, 'undecorated', fn)
    fn = getattr(fn, "im_func", fn)
    fn = getattr(fn, "__wrapped__", fn)
    return fn.__doc__
