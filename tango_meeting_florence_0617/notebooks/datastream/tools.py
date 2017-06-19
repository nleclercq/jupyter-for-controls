from __future__ import print_function
import logging
import time
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
def parent_cell_context(parent_ident, parent_header):
    """
    a context manager for routing output to a particular cell
    see https://nbviewer.jupyter.org/gist/minrk/049545c1edcf20415bb3d68f16047628
    """
    try:
        kernel = get_ipython().kernel
        save_parent = (kernel._parent_ident, kernel._parent_header)
        kernel.set_parent(parent_ident, parent_header)
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
@contextmanager
def parent_cell_context_no_effect(parent_ident, parent_header):
    try:
        yield
    except:
        pass

# ------------------------------------------------------------------------------
class NotebookCell(object):
    """
    this is used to route outputs to a particular cell
    see https://nbviewer.jupyter.org/gist/minrk/049545c1edcf20415bb3d68f16047628
    """

    def __init__(self, name):
        k = get_ipython().kernel
        self._save_parent = (k._parent_ident, k._parent_header)
        self._save_publish_status = k._publish_status
        k._publish_status = self.__disable_once
        self._logger = logging.getLogger('fs.client.jupyter')
        self._logger.setLevel(logging.ERROR)

    def __disable_once(self, *args, **kw):
        get_ipython().kernel._publish_status = self._save_publish_status

    def parent(self):
        return self._save_parent

    def cell(self):
        return self.parent()

    def clear_output(self):
        with parent_cell_context(*self._save_parent):
            clear_output()

    def display(self, widgets_layout):
        with parent_cell_context(*self._save_parent):
            display(widgets_layout)

    @property
    def logger(self):
        return self._logger

    def set_logging_level(self, level):
        self._logger.setLevel(level)

    def debug(self, msg, *args, **kwargs):
        with parent_cell_context(*self._save_parent):
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        with parent_cell_context(*self._save_parent):
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        with parent_cell_context(*self._save_parent):
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        with parent_cell_context(*self._save_parent):
            self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        with parent_cell_context(*self._save_parent):
            self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        with parent_cell_context(*self._save_parent):
            self._logger.exception(msg, *args, **kwargs)


# ------------------------------------------------------------------------------
class CellChild(object):
    """holds a NotebookCell in order to be able to route outputs to a particular cell"""

    def __init__(self, name, notebook_cell=None):
        self._name = name
        self._uid = uuid4().int
        self._parent = notebook_cell if notebook_cell else NotebookCell(name)

    @property
    def name(self):
        return self._name

    @property
    def uid(self):
        return self._uid

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        assert (isinstance(parent, NotebookCell))
        self._parent = parent if parent else NotebookCell(self._name)

    def clear_output(self):
        self._parent.clear_output()

    def display(self, widgets_layout):
        self._parent.display(widgets_layout)

    @property
    def logger(self):
        return self._parent.logger

    def set_logging_level(self, level):
        self._parent.set_logging_level(level)

    def debug(self, msg, *args, **kwargs):
        self._parent.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._parent.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._parent.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._parent.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._parent.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._parent.exception(msg, *args, **kwargs)
