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
import traceback
from contextlib import contextmanager

from jupytango.tools.display import TreeNode


# =====
# Tools
# =====

def get_error_stack():
    """Return current exception stack as a string"""
    return "".join(traceback.format_exception(*sys.exc_info()))


@contextmanager
def silent_catch(*args):
    """Silently catch all specified exception types. If empty, all BaseExceptions are caught"""
    types = args or BaseException
    try:
        yield
    except types:
        pass


# =============
# ExceptionTree
# =============

class ExceptionNode(object):

    include_tb = True

    @classmethod
    @contextmanager
    def tb_policy(cls, include_tb):
        last, cls.include_tb = cls.include_tb, include_tb
        try:
            yield
        finally:
            cls.include_tb = last

    def __init__(self, exc_type, exc_val, exc_tb):
        self.exc_type = exc_type
        self.exc_val = exc_val
        self.exc_tb = exc_tb

    def __str__(self):
        if self.include_tb:
            lines = traceback.format_exception(self.exc_type, self.exc_val, self.exc_tb)  # format traceback lines
            tb_list = traceback.extract_tb(self.exc_tb)  # get traceback info as [(file, line, fun, code)]
            if tb_list[0][2] == 'grab':  # check if the top-level call is ExceptionTree.grab ...
                lines.pop(1)  # ... then remove it from the formatted lines (first one after header)
        else:
            lines = traceback.format_exception_only(self.exc_type, self.exc_val)
        return "".join(lines)


class ExceptionTree(TreeNode, Exception):
    """This exception is used to group multiple exceptions thrown in a context and pretty print it"""

    def __init__(self, data=None):
        TreeNode.__init__(self, data)
        Exception.__init__(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            child = exc_val if isinstance(exc_val, ExceptionTree) else ExceptionNode(exc_type, exc_val, exc_tb)
            self.add_child(child)
        if self.children:
            raise self

    @contextmanager
    def grab(self, *args):
        try:
            with ExceptionTree(*args) as et:
                yield et
        except ExceptionTree as err:
            self.add_child(err)


# ==========
# Exceptions
# ==========

class Error(Exception):
    """Base class of any error."""

    def __init__(self, msg="unknown error"):
        self._message = str(msg)
        Exception.__init__(self, msg)

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, msg):
        self._message = msg

    def append(self, msg):
        self._message = '{}\n{}'.format(msg, self._message)

    def dump(self):
        print("\033[91m{}\033[0m".format(self))

    def __str__(self):
        return self._message


class GenericError(Error):
    """Generic error"""
    def __init__(self, msg=""):
        Error.__init__(self, (msg or "") + get_error_stack())
