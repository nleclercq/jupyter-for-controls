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

"""This module defines the Task class: a thread to which you can post messages"""

from __future__ import print_function

from six import PY2
if PY2:
    import Queue
else:
    import queue as Queue

import threading
import time
import ctypes

from fs.utils.task.Message import Message, MessageInvalidIdentifier, MessageIsNotWaitable
from fs.utils.errors import silent_catch


# ===========================================================================
class FsTaskKilled(Exception):
    pass


# ===========================================================================
class Task(threading.Thread):
    """
    The pyYat active object.
    This class is supposed to be derived (useless otherwise).
    The child class should implement the following member functions:
    * on_init()
    * on_exit()
    * handle_message(fs.utils.task.Message.Message)
    * handle_periodic_message()
    """

    def __init__(self, task_name, msgq_wm=1024):
        """
        Constructor.
        @param task_name the task's name.
        @param msgq_wm message queue water mark (i.e. messages queue depth - defaults to 1024 Messages).
        """
        threading.Thread.__init__(self)
        # task name
        self._task_name = task_name
        # initialize the associated message queue
        self._msgq = Queue.PriorityQueue(msgq_wm)
        # periodic message period
        self._periodic_msg_enabled = False
        self._periodic_msg_period_in_secs = 1.0
        # pending message (used to avoid periodic message starvation)
        self._pending_msg = None
        # was the last processed msg a periodic message?
        self._last_msg_was_a_periodic_msg = False
        # last periodic message timestamp
        self._last_periodic_msg_timestamp = time.time()
        # running flag
        self._has_been_successfully_started = False
        # thread uid
        self._thread_uid = 0

    @property
    def task_name(self):
        return self._task_name

    @property
    def thread_name(self):
        return self.name

    def handle_message(self, msg):
        raise NotImplementedError

    def enable_periodic_message(self, period_in_sec=1.0):
        """
        Enables the periodic messages.
        The child class MUST implement the <handle_periodic_message> function member.
        @param period_in_sec The periodic message  period in seconds (defaults to 1)
        """
        if float(period_in_sec) > 0.:
            self._periodic_msg_enabled = True
            self._periodic_msg_period_in_secs = period_in_sec

    def disable_periodic_message(self):
        """
        Disables the periodic messages.
        """
        self._periodic_msg_enabled = False

    def periodic_message_enabled(self):
        """
        Returns True if the periodic message mechanism is enabled, returns False otherwise
        """
        return self._periodic_msg_enabled

    @property
    def periodic_message_period(self):
        """
        Returns the periodic message period in seconds
        """
        return self._periodic_msg_period_in_secs

    @periodic_message_period.setter
    def periodic_message_period(self, period_in_sec):
        """
        Sets the periodic message period in seconds
        """
        self.enable_periodic_message(period_in_sec)

    def start(self):
        """
        Starts the Task synchronously (i.e. waits for the 'on_init' function to return).
        Must be there in order to override threading.Thread.Start properly.
        """
        self.start_synchronously()

    def start_synchronously(self, tmo_sec=5.0):
        """
        Starts the Task synchronously (i.e. waits for the 'on_init' function to return).
        """
        threading.Thread.start(self)
        if self.is_alive():
            self._has_been_successfully_started = True
        self.__wait_message_processed(Message.MSG_ID_INIT, timeout=tmo_sec)

    def start_asynchronously(self):
        """
        Starts the Task asynchronously (i.e. does NOT wait for the 'on_init' function to return).
        """
        threading.Thread.start(self)
        if self.is_alive():
            self._has_been_successfully_started = True
        self.__post(Message.MSG_ID_INIT)

    def kill(self):
        """
        Tries to properly kill the underlying thread
        """
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(self._thread_uid, ctypes.py_object(FsTaskKilled))
        if res == 0:
            raise ValueError("invalid thread identifier specified!")
        elif res != 1:
            # oops! we're are in trouble,
            # we have to to revert the effect of the first call to 'PyThreadState_SetAsyncExc'
            ctypes.pythonapi.PyThreadState_SetAsyncExc(self._thread_uid, 0)
            raise RuntimeError("ctypes.pythonapi.PyThreadState_SetAsyncExc failed!")

    def exit(self):
        """
        Asks the Task to exit then join with the underlying thread.
        """
        if self._has_been_successfully_started:
            self.__post(Message.MSG_ID_EXIT)
            self.join()
            self._has_been_successfully_started = False

    def post(self, msg_id, msg_data=None, msg_priority=Message.MSG_DEFAULT_PRIORITY):
        """
        Posts a Message to the Task.
        @param msg_id the identifier of the Message to be posted.
        @param msg_data the data to be attached to the Message to be posted.
        @param msg_priority the Message priority (see Message for details).
        """
        # negative msg. id. are reserved for task's internal cooking
        if msg_id < int(0):
            raise MessageInvalidIdentifier()
        self.__post(msg_id, msg_data, msg_priority)

    def post_message(self, msg):
        """
        Posts the specified Message to the Task.
        @param msg of the Message to be posted.
        """
        # insert the message into the queue
        self._msgq.put((msg.priority, msg))
        # yield execution to another thread (in the hope that it will be the consumer)
        time.sleep(.000001)

    def wait_message_processed(self, msg_id, msg_data=None, msg_priority=Message.MSG_DEFAULT_PRIORITY, timeout=3.0):
        """
        Posts a Message to the Task then wait up to <timeout> seconds for the message to be processed.
        @param msg_id the identifier of the Message to be posted.
        @param msg_data the data to be attached to the Message to be posted.
        @param msg_priority the Message priority (see Message for details).
        @param timeout the amount of time the caller accepts to wait for the message to be processed (in seconds)
        """
        # negative msg. id. are reserved for task's internal cooking
        if msg_id < int(0):
            raise MessageInvalidIdentifier()
        # wait for the message to be processed (might raise an exception in case tmo expires)
        self.__wait_message_processed(msg_id, msg_data, msg_priority, timeout)

    def wait_message_response(self, msg, timeout=3.0):
        """
        Posts a Message to the Task then wait up to <timeout> seconds for the a response.
        @param msg of the Message to be posted.
        @param timeout the amount of time the caller accepts to wait for the message to be processed (in seconds)
        return msg.data (might be None)
        """
        # be sure the message is waitable
        if not msg.is_waitable():
            raise MessageIsNotWaitable()
        # insert the message into the queue
        self._msgq.put((msg.priority, msg))
        # wait for the message to be processed (might raise an exception in case tmo expires)
        msg.wait_processed(timeout)
        # return data attached to the message (might be None)
        return msg.data

    def run(self):
        """
        The Task's entry point. Not supposed to be called directly (see threading.Thread.start).
        """
        # almost infinite thread loop control flag
        go_on = True
        # thread uid
        self._thread_uid = threading.current_thread().ident
        # enter main loop...
        while go_on:
            try:
                if self._pending_msg is None:
                    tmo = 1.0
                    if self._periodic_msg_enabled:
                        tmo = self._periodic_msg_period_in_secs
                    while self._pending_msg is None:
                        self._pending_msg = self._msgq.get(block=True, timeout=tmo)[1]
                if not self._pending_msg.is_ctrl_message() and self._its_time_to_process_a_periodic_msg():
                    with silent_catch():
                        # call user defined <handle_periodic_message>
                        self.handle_periodic_message()
                    self.__periodic_msg_processed()
                else:
                    try:
                        if not self._pending_msg.is_ctrl_message():
                            self.handle_message(self._pending_msg)
                        elif self._pending_msg.identifier == Message.MSG_ID_INIT:
                            self.on_init()
                        elif self._pending_msg.identifier == Message.MSG_ID_EXIT:
                            go_on = False
                            self.on_exit()
                    except BaseException as e:
                        self._pending_msg.error = e
                    finally:
                        self.__pending_msg_processed()
                        self._pending_msg = None
            except Queue.Empty:
                # call to <self._msgq.get> timed out
                if self._periodic_msg_enabled:
                    with silent_catch():
                        # call user defined <handle_periodic_message>
                        self.handle_periodic_message()
                    self.__periodic_msg_processed()
            except BaseException:
                pass

    def on_init(self):
        """ Task's <on_init> default (i.e. empty) implementation"""
        print("Task::{} warning default <on_init> implementation called!".format(self.name))

    def on_exit(self):
        """ Task's <on_exit> default (i.e. empty) implementation"""
        print("Task::{} warning default <on_exit> implementation called!".format(self.name))

    def handle_periodic_message(self):
        """ Task's <handle_periodic_message> default (i.e. empty) implementation"""
        print("Task::{} warning default <handle_periodic_message> implementation called!".format(self.name))

    def _its_time_to_process_a_periodic_msg(self):
        """ This is private and undocumented! """
        if self._periodic_msg_enabled and not self._last_msg_was_a_periodic_msg:
            dt = time.time() - self._last_periodic_msg_timestamp
            if dt >= self._periodic_msg_period_in_secs:
                return True
        return False

    def __post(self, msg_id, msg_data=None, msg_priority=Message.MSG_DEFAULT_PRIORITY):
        # instantiate the message
        msg = Message(int(msg_id), msg_data, msg_priority)
        # insert the message into the queue
        self._msgq.put((int(msg.priority), msg))
        # yield execution to another thread (in the hope that it will be the consumer)
        time.sleep(.000001)

    def __wait_message_processed(self, msg_id, msg_data=None, msg_priority=Message.MSG_DEFAULT_PRIORITY, timeout=3.0):
        # instantiate the message
        msg = Message(msg_id, msg_data, msg_priority, True)
        # insert the message into the queue
        self._msgq.put((msg.priority, msg))
        # wait for the message to be processed (might raise an exception in case tmo expires)
        msg.wait_processed(timeout)

    def __pending_msg_processed(self):
        """ This is private and undocumented! """
        self._pending_msg.processed()
        self._pending_msg = None
        self._last_msg_was_a_periodic_msg = False
        self._msgq.task_done()

    def __periodic_msg_processed(self):
        self._last_msg_was_a_periodic_msg = True
        self._last_periodic_msg_timestamp = time.time()
