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

"""This package defines multiple tools dedicated to (screen) output features"""

from __future__ import print_function
from six import text_type, string_types, iteritems
import re
import os
import sys
from difflib import ndiff
import cgi
import h5py

from prettytable import PrettyTable


def print_over(text):
    """Write some text over the last output line"""
    print('\r\033[K{}'.format(text), end='')  # vs print('\r{}'.format(text.ljust(79)), end='')
    sys.stdout.flush()  # just in case


def print_now(text):
    """Write some text over the last output line"""
    print('{}'.format(text), end='')  # vs print('\r{}'.format(text.ljust(79)), end='')
    sys.stdout.flush()  # just in case

# ======================
# Pretty Print (IPython)
# ======================

def pretty(cls):
    """
    pretty is a decorator that implements _repr_pretty_ for the given class.
    We will prefer this syntax over the inheritance to avoid an annoying bug
    in IPython 0.13 where _repr_pretty_ is not accessible through subclasses
    """
    # pylint: disable=W0212
    def prettify(self, p, cycle):
        del cycle  # unused
        return p.text(text_type(self) if len(p.stack) == 1 else repr(self))
    cls._repr_pretty_ = prettify
    return cls


# =====
# Table
# =====

@pretty
class Table(PrettyTable):
    """
    Get a table representation of the data provided.
    * data is expected to be a 2D array (supports numpy)
    * labels is the column labels (None won't display any header)
    """

    @staticmethod
    def tab2spaces(item):
        if isinstance(item, string_types):
            return item.expandtabs(4)
        return item

    def __init__(self, labels=None, data=None):
        PrettyTable.__init__(self, labels, header=labels is not None)
        for row in data or []:
            self.add_row(row)

    def add_row(self, row):
        """overriding add_row in order to prevent \t in string_types inputs
        NOTE: very inefficient
        NOTE: won't expand tabs for non-string_types input "stringified" after
        """
        PrettyTable.add_row(self, [self.tab2spaces(i) for i in row])

    def _repr_html_(self):
        return Ansi.to_html(self.get_html_string())


# ========
# TreeNode
# ========

class TreeEntry(object):
    """A tree entry is a pair of strings which is designed to be aligned when used within a TreeNode"""

    @classmethod
    def adjust(cls, entries):
        entry_list = list(entries)
        if entry_list:
            width = max(len(k.key) for k in entry_list) + 3
            for entry in entry_list:
                entry.width = width

    def __init__(self, key, value):
        self.key = u"{} ".format(key)
        self.value = text_type(value)
        self.width = 0

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        return u"{} {}".format(self.key.ljust(self.width, '.'), self.value)


@pretty
class TreeNode(object):
    """
    A tree node is an object that can be represented as a tree in ASCII
    note that children can be any type that can be converted to text_type

    WARNING: text is not wrapped
    WARNING: the algorithm does not check for recursive input

    example:

        def class_tree(cls):
            node = TreeNode(cls.__name__)
            for subcls in cls.__subclasses__():
                node.children.append(class_tree(subcls))
            return node

    """

    vertical = u' |'  # vertical link of nodes
    link = u' +- '  # horizontal header for children
    vspacing = 0  # number of lines between two siblings

    # automatic
    link_empty = u' ' * len(link)
    link_fill = vertical.ljust(len(link))
    vsep = vertical.join(u'\n' * (vspacing + 1))

    @classmethod
    def linked(cls, data, last):
        lines = text_type(data).splitlines()
        filler = cls.link_empty if last else cls.link_fill
        padding = [cls.link] + [filler] * (len(lines) - 1)
        return u"\n".join(a + b for a, b in zip(padding, lines))

    def __init__(self, data=None, parent=None):
        self.data = data
        self.children = []
        if parent is not None:
            parent.add_child(self)

    def add_entry(self, key, value):
        self.add_child(TreeEntry(key, value))

    def add_child(self, child):
        """Add a new node/leaf to this children. If you insert a tree which data is None, the children are merged"""
        extend = isinstance(child, TreeNode) and child.data is None
        self.children += child.children if extend else [child]

    def _tag_li(self):
        lis = []
        for child in self.children:
            node = child if isinstance(child, TreeNode) else TreeNode(child)  # dummy node from child to get _tag_li
            lis.append(node._tag_li())
        text = u"" if self.data is None else cgi.escape(text_type(self.data), True)
        if lis:
            text += u"<ul>{}</ul>".format(u"".join(lis))
        return u"<li>{}</li>".format(text)

    def _repr_html_(self):
        return u"<ul>{}</ul>".format(self._tag_li())

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        TreeEntry.adjust(k for k in self.children if isinstance(k, TreeEntry))
        tree = [] if self.data is None else [text_type(self.data)]
        tree += (self.linked(child, False) for child in self.children[:-1])
        tree += (self.linked(child, True) for child in self.children[-1:])
        return self.vsep.join(tree)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.data)


def mapping_tree(title, mapping, sort=True, fmt=None):
    node = TreeNode(title)
    items = sorted(mapping.items()) if sort else mapping.items()
    if not items:
        node.add_child(u"\u2205")
    for k, v in items:
        node.add_entry(k, v if fmt is None else fmt(v))
    return node


def directory_tree(path, no_files=False):
    node = TreeNode()
    is_dir = os.path.isdir(path)
    if is_dir or not no_files:
        node.data = os.path.basename(path)
    if is_dir:
        for child in sorted(os.listdir(path)):
            child_path = os.path.join(path, child)
            node.add_child(directory_tree(child_path, no_files))
    return node


# ====
# hdf5
# ====

def hdf5_tree(value):
    if isinstance(value, string_types):
        with h5py.File(value, 'r') as f:
            return hdf5_node_tree(f, value)
    return hdf5_node_tree(value, value.name)


def hdf5_node_tree(value, name):
    if hasattr(value, 'shape'):
        name = "{} {}".format(name, value.shape)
    node = TreeNode(name)
    attrs = dict(getattr(value, 'attrs', {}))
    if attrs:
        attrs_node = TreeNode("<attributes>", parent=node)
        for k, v in attrs.items():
            attrs_node.add_entry(k, v)
    if hasattr(value, 'iteritems'):
        for child_name, child in iteritems(value):
            node.children.append(hdf5_node_tree(child, child_name))
    return node


# ========
# Coloring
# ========

class Ansi(object):
    """
    Tools related to ANSI escape sequences
    see http://ascii-table.com/ansi-escape-sequences.php
    """

    activated = True

    # Select Graphic Rendition CSI Pattern
    sgr_pattern = re.compile(r"\033\[([\d;]*)m")

    reset = '0'

    attributes = {
        'bold': '1',
        'underscore': '4',
        'blink': '5',
        'reversed': '7',
    }

    foregrounds = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37',
    }

    backgrounds = {
        'black': '40',
        'red': '41',
        'green': '42',
        'yellow': '43',
        'blue': '44',
        'magenta': '45',
        'cyan': '46',
        'white': '47',
    }

    html_classes = {
        '1': 'ansibold',
        '30': 'ansiblack',
        '31': 'ansired',
        '32': 'ansigreen',
        '33': 'ansiyellow',
        '34': 'ansiblue',
        '35': 'ansipurple',
        '36': 'ansicyan',
        '37': 'ansigrey',
    }

    @classmethod
    def to_html(cls, text):
        """Convert ansi colors to html colors.
        (inspired by IPython.nbconvert.filters.ansi.ansi2html)"""
        opened = False  # True if a span tag has been opened
        while True:
            match = cls.sgr_pattern.search(text)
            # break if there are no more sequences
            if match is None:
                break
            # get html classes specified by the current sequence
            codes = set(match.groups()[0].split(';'))
            classes = [cls.html_classes[code] for code in codes if code in cls.html_classes]
            # make tags
            closer = '</span>' if opened else ''
            opener = '<span class="%s">' % ' '.join(classes) if classes else ''
            # substitute sequence by tags
            text = cls.sgr_pattern.sub(closer + opener, text, 1)
            opened = bool(classes)
        if opened:
            text += '</span>'
        return text

    @classmethod
    def clean(cls, string):
        """Remove all escape sequences from the string"""
        return cls.sgr_pattern.sub("", string)

    @classmethod
    def graphics(cls, string, **kwargs):
        """
        Accepted kwargs:
        * fg (foreground) bg (background) as the name of the color
        * any known attributes as booleans: bold, underscore, blink, reversed
        """
        # get attributes
        codes = [v for k, v in cls.attributes.items() if kwargs.pop(k, False)]
        # get foreground color
        fg = str(kwargs.pop('fg', '')).lower()
        if fg in cls.foregrounds:
            codes.append(cls.foregrounds[fg])
        # get background color
        bg = str(kwargs.pop('bg', '')).lower()
        if bg in cls.backgrounds:
            codes.append(cls.backgrounds[bg])
        # check unknown kwargs
        if kwargs:
            raise ValueError('unknown keywords arguments %s' % list(kwargs))
        # format and returns (must be activated)
        # string is surrounded by prefix & postfix keeping its original type
        if cls.activated and codes and string:
            return string.join((cls._format(cls.reset, *codes), cls._format(cls.reset)))
        return string

    @staticmethod
    def _format(*args):
        return '\033[%sm' % ";".join(args)


# ====
# Diff
# ====

@pretty
class Diff(object):
    """A diff contains lines compared via 'ndiff' and is pretty-printable"""

    @classmethod
    def compare(cls, lhs, rhs):
        return cls(ndiff(list(lhs), list(rhs)))

    def __init__(self, iterable=None):
        self.lines = []
        if iterable is not None:
            self.lines.extend(iterable)

    @classmethod
    def format_line(cls, line):
        if line[0] == '-':
            return Ansi.graphics(line[2:], fg='red', bold=True)
        elif line[0] == '+':
            return Ansi.graphics(line[2:], fg='green', bold=True)
        elif line[0] == '?':
            return Ansi.graphics(line[2:], fg='cyan', bold=True)
        return line[2:]

    def __str__(self):
        return "\n".join(self.format_line(line) for line in self.lines)
