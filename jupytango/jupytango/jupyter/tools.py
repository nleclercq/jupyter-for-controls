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
import logging
import time
import sys
from contextlib import wraps, contextmanager
from uuid import uuid4
from IPython import get_ipython
from IPython.display import display, clear_output


# ------------------------------------------------------------------------------
def is_iterable(some_object):
    try:
        some_object_iterator = iter(some_object)
        return True
    except:
        return False


# ------------------------------------------------------------------------------
def enum(*sequential):
    enums = dict(zip(sequential, range(len(sequential))))
    enums['len'] = len(sequential)
    return type('Enum', (), enums)


# ------------------------------------------------------------------------------
def tracer(fn):
    def debug_trace(obj, message):
        trace(obj.logger.debug, message)

    def exception_trace(obj, exception):
        trace(obj.logger.exception, exception)

    def trace(function, message):
        try:
            function(message)
        except Exception:
            print(message)

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        name = self.name if hasattr(self, 'name') else ''
        if len(name) > 5:
            name = name[-5:]
        qualified_name = "{}.{}.{}".format(self.__class__.__name__, fn.__name__, name)
        debug_trace(self, "{} <<in".format(qualified_name))
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            exception_trace(self, e)
        finally:
            dt = 1000 * (time.time() - t0)
            debug_trace(self, "{} out>> [took: {:.2f} ms]".format(qualified_name, dt))
    return wrapper


# ------------------------------------------------------------------------------
class CellContext(object):
    
    def __init__(self):
        k = get_ipython().kernel
        self._ident = k._parent_ident
        self._header = k._parent_header
        
    def __call__(self):
        return (self._ident, self._header)


# ------------------------------------------------------------------------------
@contextmanager
def cell_context(context):
    try:
        assert(isinstance(context, CellContext))
        kernel = get_ipython().kernel
        save_context = (kernel._parent_ident, kernel._parent_header)
        sys.stdout.flush()
        sys.stderr.flush()
        kernel.set_parent(*context())
    except Exception as e:
        print(e)
        return
    try:
        yield
    except:
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        kernel.set_parent(*save_context)


# ------------------------------------------------------------------------------
class NotebookCellContent(object):

    default_logger = "jupytango.jupyter"

    """
    try:
        h = logging.getLogger(default_logger).handlers[0]
    except IndexError:
        logging.basicConfig(format="[%(asctime)-15s] %(name)s: %(message)s", level=logging.ERROR)
    """

    def __init__(self, name, logger_name=None):
        self._name = name
        self._uid = uuid4().int
        self._context = CellContext()
        self._logger = logging.getLogger(NotebookCellContent.default_logger if logger_name is None else logger_name)

    @property
    def name(self):
        return self._name

    @property
    def uid(self):
        return self._uid

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, new_context):
        self._context = new_context

    def display(self, widgets_layout):
        with cell_context(self._context):
            display(widgets_layout)
       
    def clear_output(self):
        with cell_context(self._context):
            clear_output()

    @property
    def logger(self):
        return self._logger

    def set_logging_level(self, level):
        self._logger.setLevel(level)

    def print(self, *args):
        with cell_context(self._context):
            print(*args)
            
    def debug(self, msg, *args, **kwargs):
        with cell_context(self._context):
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        with cell_context(self._context):
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        with cell_context(self._context):
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        with cell_context(self._context):
            self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        with cell_context(self._context):
            self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        with cell_context(self._context):
            self._logger.exception(msg, *args, **kwargs)