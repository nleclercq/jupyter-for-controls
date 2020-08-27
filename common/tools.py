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

import os
import sys
import time
import logging
import traceback
from contextlib import wraps, contextmanager
from uuid import uuid4
from IPython import get_ipython
from IPython.display import display, clear_output

try:
    logging.getLogger().handlers[0]
except IndexError:
    #logging.basicConfig(format="[%(asctime)-15s] %(name)s: %(message)s", level=logging.ERROR)
    pass
except:
    pass

# ------------------------------------------------------------------------------
def enum(*sequential):
    enums = dict(zip(sequential, range(len(sequential))))
    enums['len'] = len(sequential)
    return type('Enum', (), enums)


# ------------------------------------------------------------------------------
JupyterContext = enum('LAB', 'NOTEBOOK', 'CONSOLE')


# ------------------------------------------------------------------------------
def get_ioloop():
    import IPython, zmq
    ipython = IPython.get_ipython()
    if ipython and hasattr(ipython, 'kernel'):
        return zmq.eventloop.ioloop.IOLoop.instance()
   

# ------------------------------------------------------------------------------
def get_jupyter_context():
    try:
        jcv = os.environ['JUPYTER_CONTEXT']
        if jcv.upper() not in ['LAB', 'NOTEBOOK']:
            raise KeyError()
        jc = JupyterContext.LAB if jcv.upper() == 'LAB' else JupyterContext.NOTEBOOK
    except KeyError:
        print("warning: JUPYTER_CONTEXT env. var. not set or has invalid value!")
        print("warning: using default JUPYTER_CONTEXT value 'LAB'  [possible values: LAB, NOTEBOOK]")
        jc = JupyterContext.LAB
    return jc


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
class timer(object):

    op_id = 0

    def __init__(self, name=None, logger=logging.getLogger(__name__)):
        self.logger = logger
        timer.op_id += 1 if name is None else 0
        self.name = name if name is not None else str(timer.op_id)

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, type, value, traceback):
        dt = 1000 * (time.time() - self.t0)
        self.logger.debug("op.{} took: {:.2f} ms".format(self.name, dt))


# ------------------------------------------------------------------------------
def tracer(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        name = '{}'.format(self.name) if hasattr(self, 'name') else ''
        if len(name) > 5:
            name = name[-5:]
        qualified_name = "{}.{}.{}".format(self.__class__.__name__, fn.__name__, name)
        self.debug("{} <<in".format(qualified_name))
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            self.error(e)
        finally:
            dt = 1000 * (time.time() - t0)
            self.debug("{} out>> [took: {:.2f} ms]".format(qualified_name, dt))
    return wrapper


# ------------------------------------------------------------------------------
class CellOutput(object):
    def __init__(self):
        k = get_ipython().kernel
        self._ident = k._parent_ident
        self._header = k._parent_header
        self._save_context = None

    def __enter__(self):
        kernel = get_ipython().kernel
        self._save_context = (kernel._parent_ident, kernel._parent_header)
        sys.stdout.flush()
        sys.stderr.flush()
        kernel.set_parent(self._ident, self._header)

    def __exit__(self, etype, evalue, tb):
        sys.stdout.flush()
        sys.stderr.flush()
        kernel = get_ipython().kernel
        kernel.set_parent(*self._save_context)
        return False

    def clear_output(self):
        with self:
            clear_output()

    def close(self):
        pass


# ------------------------------------------------------------------------------
class NotebookCellContent(object):
    default_logger = "fs.client.jupyter"

    def __init__(self, name=None, logger=None, output=None):
        uuid = uuid4().hex
        self._uid = uuid
        self._name = name if name is not None else str(uuid)
        self._output = CellOutput() if output is None else output
        self._logger = logger if logger else logging.getLogger(NotebookCellContent.default_logger)
        try:
            # no default handler workaround 
            h = self._logger.handlers[0]
        except IndexError:
            logging.basicConfig(format="[%(asctime)-15s] %(name)s: %(message)s", level=logging.ERROR)
            
    @property
    def name(self):
        return self._name

    @property
    def uid(self):
        return self._uid
        
    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, l):
        self._logger = l

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, output):
        self._output = output
        
    def display(self, widgets):
        with self._output:
            display(widgets)

    def clear_output(self):
        self._output.clear_output()

    def close_output(self):
        self._output.close()
        
    def set_logging_level(self, level):
        self._logger.setLevel(level)

    def print_to_cell(self, *args):
        self.print(*args)

    def print(self, *args, **kwargs):
        with self._output:
            try:
                self._logger.print(*args, **kwargs)
            except:
                print(*args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        with self._output:
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        with self._output:
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        with self._output:
            self._logger.warning(msg, *args, **kwargs)

    def print_error(self, msg, *args, **kwargs):
        with self._output:
            self._logger.error(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        with self._output:
            self._logger.error(msg, *args, **kwargs)
            exc_info = sys.exc_info()
            if exc_info is not None:
                traceback.print_exception(*exc_info)
                del exc_info

    def critical(self, msg, *args, **kwargs):
        with self._output:
            self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)
