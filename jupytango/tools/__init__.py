# ===========================================================================
#  This file is part of the Flyscan Ecosystem
#
#  Copyright 2014-EOT Synchrotron SOLEIL, St.Aubin, France
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

"""This package provides some utilities classes for the Flyscan project"""

from __future__ import print_function
from six import with_metaclass, iteritems
from collections import MutableMapping
import imp
import os.path as osp
from os import listdir

from fs.utils.display import mapping_tree, pretty

from six import PY2
if not PY2:
    from functools import reduce as reduce

# TODO put in a specific module


# ==============
# Module Finders
# ==============

_modules_suffixes = {s for s, m, t in imp.get_suffixes()}


def find_modules(*plugins_paths):
    """
    Recursively looks for python modules in the given paths returning a dict {name: directory}
    NOTE: concerning duplicates, only the last file processed is selected
    """
    files = {}
    for path in plugins_paths:
        # if path is a directory, lookup modules in it
        if osp.isdir(path):
            sub_plugins_paths = [osp.join(path, child) for child in listdir(path) if child != '__pycache__']
            files.update(find_modules(*sub_plugins_paths))
        # if path is a regular file, check for python extension
        # and override the base name of the file as the name of the module
        elif osp.isfile(path):
            dirname, basename = osp.dirname(path), osp.basename(path)
            name, ext = osp.splitext(basename)
            if ext in _modules_suffixes:
                files[name] = dirname
    return files


def load_module(name, *dirnames):
    f = None
    try:
        f, p, d = imp.find_module(name, list(dirnames))
        return imp.load_module(name, f, p, d)
    finally:
        if f is not None:
            f.close()


# =============
# miscellaneous
# =============

# simple object used as default value in getter
# meaning that an exception should be raised instead of being returned
# this object is meant to be shared so that it can be forwarded to other getters
no_default = object()


class AttributeDelegate(object):
    """Descriptor that maps object attributes of any depth"""

    def __init__(self, *attributes):
        if not attributes:
            raise ValueError("attributes can't be empty")
        self.attributes = attributes

    def __get__(self, instance, owner):
        del owner  # unused
        return reduce(getattr, self.attributes, instance)

    def __set__(self, instance, value):
        setattr(reduce(getattr, self.attributes[:-1], instance), self.attributes[-1], value)

    def __delete__(self, instance):
        delattr(reduce(getattr, self.attributes[:-1], instance), self.attributes[-1])


# ==========
# KeyMapping
# ==========

class SharedKeyError(LookupError):
    pass


class KeyMapping(MutableMapping):
    """Mapping where key is based on values"""

    def __init__(self, key):
        # note that we don't need MutableMapping.__init__(self)
        self.key = key  # function value -> key
        self._values = []  # internal storage

    def add(self, value):
        """
        add a value in this mapping, unlike a dict, if the key is already mapped,
        it raises a SharedKeyError instead of overriding value
        """
        key = self.key(value)
        if key in self:
            raise SharedKeyError(key)
        self._values.append(value)

    def setdefault(self, key, default=None):
        # overridden to avoid using __setitem__
        try:
            return self[key]
        except KeyError:
            self._values.append(default)
            return default

    def __contains__(self, key):
        # overridden to avoid raising SharedKeyError
        return any(self.key(v) == key for v in self._values)

    def __getitem__(self, key):
        values = [v for v in self._values if self.key(v) == key]
        if not values:
            raise KeyError(key)
        if len(values) > 1:
            raise SharedKeyError(key)
        return values.pop()

    def __setitem__(self, key, value):
        raise AttributeError("unable to set item of a value-based mapping")

    def __delitem__(self, key):
        filtered_values = [v for v in self._values if self.key(v) != key]
        if len(filtered_values) == len(self._values):
            raise KeyError(key)
        self._values[:] = filtered_values

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self.key(v) for v in self._values)

    def __repr__(self):
        return '{%s}' % ', '.join('{!r}: {!r}'.format(k, v) for k, v in iteritems(self))


# ====
# View
# ====

@pretty
class View(object):
    """
    A view is an object built from a mapping that turns keys into attributes
    It can be useful for command-line interactions using IPython for example
    """

    def __init__(self, other=None, **kwargs):
        self.__dict__.update(other, **kwargs)

    def __unicode__(self):
        return unicode(mapping_tree(self.__class__.__name__, vars(self), fmt=repr))

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, vars(self))


# ======
# Hidden
# ======

class HideType(type):
    """Metaclass overriding __dir__ in order to discard some names from the list of available names"""

    def __dir__(cls):
        names = set.union(*[set(vars(k)) for k in cls.mro()])
        return sorted(names - set(cls.__type_hidden__()))


class Hidden(with_metaclass(HideType,object)):
    """Interface of the HideType which is able to discard instance names"""

    @classmethod
    def __type_hidden__(cls):
        return []

    def __hidden__(self):
        return []

    def __dir__(self):
        names = set(vars(self)) | set(dir(type(self)))
        return sorted(names - set(self.__hidden__()))
