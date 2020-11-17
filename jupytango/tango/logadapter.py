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

