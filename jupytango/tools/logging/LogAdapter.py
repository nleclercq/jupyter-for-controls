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
import sys
import time
from contextlib import wraps, contextmanager
from IPython import get_ipython

def tracer(fn):

    def printer(obj, message):
        try:
            obj.logger.debug(message)
        except Exception:
            print(message)

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        qualified_name = "{}.{}".format(self.__class__.__name__, fn.__name__)
        printer(self, "{} <<in".format(qualified_name))
        try:
            return fn(self, *args, **kwargs)
        finally:
            dt = 1000 * (time.time() - t0)
            printer(self, "{} out>> [took: {:.2f} ms]".format(qualified_name, dt))

    return wrapper


class LogAdapter(object):

    def __init__(self, logger=None):
        self.logger = logger

    @classmethod
    def __log(cls, stream, txt, *args):
        try:
            stream(txt, *args)
        except Exception:
            print(txt)

    def debug(self, txt, *args):
        self.__log(self.logger.debug, txt, *args)

    def info(self, txt, *args):
        self.__log(self.logger.info, txt, *args)

    def warning(self, txt, *args):
        self.__log(self.logger.warning, txt, *args)

    def deprecated(self, txt, *args):
        self.__log(self.logger.deprecated, txt, *args)

    def error(self, txt, *args):
        self.__log(self.logger.error, txt, *args)

    def fatal(self, txt, *args):
        self.__log(self.logger.fatal, txt, *args)

    def critical(self, txt, *args):
        self.__log(self.logger.critical, txt, *args)

    def exception(self, txt, *args):
        self.__log(self.logger.exception, txt, *args)


class DeviceLogAdapter(object):
    """A logger forwarding log to the tango device logging streams"""

    def __init__(self, host_device=None):
        self.host_device = host_device

    @property
    def logger(self):
        return self

    @classmethod
    def __log(cls, stream, *args):
        try:
            stream(*args)
        except Exception:
            print(*args)

    def debug(self, msg, *args):
        self.__log(self.host_device.debug_stream, msg, *args)

    def info(self, msg, *args):
        self.__log(self.host_device.info_stream, msg, *args)

    def deprecated(self, msg, *args):
        self.__log(self.host_device.info_stream, msg, *args)

    def warning(self, msg, *args):
        self.__log(self.host_device.warn_stream, msg, *args)

    def error(self, msg, *args):
        self.__log(self.host_device.error_stream, msg, *args)

    def critical(self, msg, *args):
        self.__log(self.host_device.fatal_stream, msg, *args)

    def fatal(self, msg, *args):
        self.__log(self.host_device.fatal_stream, msg, *args)

    def exception(self, msg, *args):
        self.__log(self.host_device.error_stream, msg, *args)


@contextmanager
def temp_parent(parent_ident, parent_header):
    """a context manager temporarily setting the parent header for routing output to a particular cell
       see https://nbviewer.jupyter.org/gist/minrk/049545c1edcf20415bb3d68f16047628"""
    try:
        kernel = get_ipython().kernel
        save_parent = (kernel._parent_ident, kernel._parent_header)
        sys.stdout.flush()
        sys.stderr.flush()
        kernel.set_parent(parent_ident, parent_header)
    except Exception as e:
        print(e)
        return
    try:
        yield
    except:
        pass
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        kernel.set_parent(*save_parent)


class NotebookCellLogger(object):
    """a per cell logger for the ipython notebook
       see https://nbviewer.jupyter.org/gist/minrk/049545c1edcf20415bb3d68f16047628"""

    def __init__(self, delegate):
        self._logger = delegate
        k = get_ipython().kernel
        self._save_parent = (k._parent_ident, k._parent_header)
        self._save_publish_status = k._publish_status
        k._publish_status = self._disable_once

    def _disable_once(self, *args, **kw):
        get_ipython().kernel._publish_status = self._save_publish_status

    def debug(self, msg, *args, **kwargs):
        with temp_parent(*self._save_parent):
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        with temp_parent(*self._save_parent):
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        with temp_parent(*self._save_parent):
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        with temp_parent(*self._save_parent):
            self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        with temp_parent(*self._save_parent):
            self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        with temp_parent(*self._save_parent):
            self._logger.exception(msg, *args, **kwargs)

    def print_in_cell(self, msg):
        with temp_parent(*self._save_parent):
            print('{}'.format(msg), end='')

