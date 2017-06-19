from __future__ import print_function
import logging
import time
import sys
from contextlib import wraps, contextmanager
from uuid import uuid4
from IPython import get_ipython
from IPython.display import display, clear_output

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
@contextmanager
def cell_context(ident, header):
    """
    a context manager for routing output to a particular cell
    see https://nbviewer.jupyter.org/gist/minrk/049545c1edcf20415bb3d68f16047628
    """
    try:
        kernel = get_ipython().kernel
        save_parent = (kernel._parent_ident, kernel._parent_header)
        kernel.set_parent(ident, header)
    except Exception as e:
        print(e)
        return
    try:
        yield
    except:
        pass
    finally:
        kernel.set_parent(*save_parent)


# ------------------------------------------------------------------------------
class NotebookCellContext(object):
    """
    this is used to route outputs to a particular cell
    see https://nbviewer.jupyter.org/gist/minrk/049545c1edcf20415bb3d68f16047628
    """

    def __init__(self, name):
        k = get_ipython().kernel
        self._cell_context = (k._parent_ident, k._parent_header)
        self._save_publish_status = k._publish_status
        k._publish_status = self.__disable_once
        self._logger = logging.getLogger('fs.client.jupyter')
        try:
            #TODO: understand why we need this workaround, move it to another place (very first log don't appear)
            s = self._logger.handlers[0].stream
        except IndexError:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
            self._logger.addHandler(handler)

    def __disable_once(self, *args, **kw):
        get_ipython().kernel._publish_status = self._save_publish_status

    @property
    def context(self):
        return self._cell_context

    def clear_output(self):
        with cell_context(*self._cell_context):
            clear_output()

    def display(self, widgets_layout):
        with cell_context(*self._cell_context):
            display(widgets_layout)

    @property
    def logger(self):
        return self._logger

    def set_logging_level(self, level):
        self._logger.setLevel(level)

    def debug(self, msg, *args, **kwargs):
        with cell_context(*self._cell_context):
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        with cell_context(*self._cell_context):
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        with cell_context(*self._cell_context):
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        with cell_context(*self._cell_context):
            self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        with cell_context(*self._cell_context):
            self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        with cell_context(*self._cell_context):
            self._logger.exception(msg, *args, **kwargs)


# ------------------------------------------------------------------------------
class NotebookCellContent(object):
    """holds a NotebookCellContext in order to be able to route outputs to a particular cell"""

    def __init__(self, name, notebook_cell=None):
        self._name = name
        self._uid = uuid4().int
        self._cell_context = notebook_cell if notebook_cell else NotebookCellContext(name)

    @property
    def name(self):
        return self._name

    @property
    def uid(self):
        return self._uid

    @property
    def cell_context(self):
        return self._cell_context

    @cell_context.setter
    def cell_context(self, cell_context):
        assert (isinstance(cell_context, NotebookCellContext))
        self._cell_context = cell_context

    def clear_output(self):
        self._cell_context.clear_output()

    def display(self, widgets_layout):
        self._cell_context.display(widgets_layout)

    @property
    def logger(self):
        return self._cell_context.logger

    def set_logging_level(self, level):
        self._cell_context.set_logging_level(level)

    def debug(self, msg, *args, **kwargs):
        self._cell_context.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._cell_context.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._cell_context.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._cell_context.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._cell_context.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._cell_context.exception(msg, *args, **kwargs)
