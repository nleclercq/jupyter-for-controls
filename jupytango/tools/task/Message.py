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

"""This module defines the pyYat.Message class"""

from __future__ import print_function
import threading


# ===========================================================================
class MessageInvalidIdentifier(Exception):
    """Invalid message identifier specified (must be > 0)"""
    pass


# ===========================================================================
class MessageIsNotWaitable(Exception):
    """Message is not waitable (this is a programming error)"""
    pass


# ===========================================================================
class MessageProcessingTimeout(Exception):
    """Timeout expired while waiting for the message to be processed"""
    pass


# ===========================================================================
class Message(object):
    """
    The pyYat message object.
    """
    #  some predefined msd IDs (for Task's internal cooking)
    MSG_ID_INIT = -1
    MSG_ID_PERIODIC = -2
    MSG_ID_EXIT = -3

    #  msg priority range
    MSG_LOWEST_PRIORITY = 65536
    MSG_HIGHEST_PRIORITY = 0
    MSG_DEFAULT_PRIORITY = MSG_LOWEST_PRIORITY

    def __init__(self, msg_id, msg_data=None, msg_priority=MSG_DEFAULT_PRIORITY, msg_is_waitable=False):
        """
        Constructor.
        @param msg_id the message identifier (must be >= 0 - negative IDs are reserved)
        @param msg_data the optional message data (defaults to None)
        @param msg_priority the optional message priority (defaults to Message.MSG_DEFAULT_PRIORITY)
        @param msg_is_waitable if set to True, the msg is processed synchronously (waiter notified once msg processed)
        """
        self._id = int(msg_id)
        self._data = msg_data
        self._processed = False
        self._error = None
        self._condition = threading.Condition() if msg_is_waitable else None

        if msg_priority is None:
            self._priority = Message.MSG_DEFAULT_PRIORITY
        else:
            if msg_priority > Message.MSG_LOWEST_PRIORITY:
                self._priority = Message.MSG_LOWEST_PRIORITY
            elif msg_priority < Message.MSG_HIGHEST_PRIORITY:
                self._priority = Message.MSG_HIGHEST_PRIORITY
            else:
                self._priority = int(msg_priority)

    @property
    def identifier(self):
        """
        Return the message identifier (a python int value).
        """
        return self._id

    @property
    def priority(self):
        """
        Return the message's priority (a python int value).
        """
        return self._priority

    @priority.setter
    def priority(self, priority):
        """
        Set the message's priority.
        """
        self._priority = priority

    @property
    def data(self):
        """
        Returns the message data.
        """
        return self._data

    @data.setter
    def data(self, data):
        """
        Sets the message data.
        """
        self._data = data

    def has_data(self):
        """
        Returns True if this message has attached data, returns False otherwise.
        """
        return self._data is not None

    def processed(self):
        """
        Marks the message's as "processed". For <waitable> message, also notify the waiters.
        """
        if self.is_waitable():
            with self._condition:
                self._processed = True
                self._condition.notify_all()
        else:
            self._processed = True

    def has_been_processed(self):
        """
        Returns True if this message has been processed, returns False otherwise.
        """
        return self._processed

    def is_ctrl_message(self):
        """
        Is the message a Task control message?
        """
        return self._id < 0

    def is_waitable(self):
        """
        Is the message 'waitable'?
        """
        return self._condition is not None

    def has_failed(self):
        """
        Did the message processing failed? Useless if the message is not waitable.
        """
        return self._error is not None

    @property
    def error(self):
        """
        Returns the exception raised during message processing or None if no error.
        """
        return self._error

    @error.setter
    def error(self, exception=None):
        """
        Attaches the exception raised during message processing.
        """
        self._error = exception

    def make_waitable(self):
        """
        Makes the message 'waitable'.
        """
        if not self.is_waitable():
            self._condition = threading.Condition()

    def wait_processed(self, timeout=3.0):
        """
        Waits for the message to be processed (synchronous message processing).
        @param timeout The amount of time the caller accepts to wait for the message to be processed (in seconds).
        """
        if not self.is_waitable():
            raise MessageIsNotWaitable()
        with self._condition:
            #  it seems that <Condition.wait> doesn't raise any exception!
            #  so we do it manually by testing the <Message._processed> flag...
            if not self.has_been_processed():
                self._condition.wait(timeout)
                if not self.has_been_processed():
                    raise MessageProcessingTimeout()
        if self.has_failed():
            raise self._error
