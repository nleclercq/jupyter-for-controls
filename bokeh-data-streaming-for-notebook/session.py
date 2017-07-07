from __future__ import print_function

import socket
import logging
from threading import Lock, Condition
from collections import deque

from IPython.display import HTML, clear_output, display

from tornado.ioloop import IOLoop

from bokeh.io import output_notebook
from bokeh.resources import INLINE
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import Handler, FunctionHandler
from bokeh.embed import autoload_server

module_logger_name = "fs.client.jupyter.session"

# ------------------------------------------------------------------------------
class BokehSessionHandler(Handler):

    def on_server_loaded(self, server_context):
        pass

    def on_server_unloaded(self, server_context):
        pass

    def on_session_created(self, session_context):
        pass

    def on_session_destroyed(self, session_context):
        pass


# ------------------------------------------------------------------------------
class BokehSession(object):
    
    def __init__(self):
        # the associated bokeh document (for experts only)
        self._doc = None
        # periodic callback period in seconds - defaults to None (i.e. disabled)
        self._callback_period = None
        # periodic activity enabled?
        self._suspended = True

    def _on_session_created(self, doc):
        self._doc = doc
        self.setup_document()

    def _on_session_destroyed(self):
        pass

    @property
    def ready(self):
        return self._doc is not None

    @property
    def document(self):
        return self._doc

    @property
    def id(self):
        return self._doc.session_context.id if self._doc else None

    @property
    def suspended(self):
        return self._suspended

    @property 
    def callback_period(self):
        """return the (periodic) callback period in seconds or None (i.e. disabled)"""
        return self._callback_period

    @callback_period.setter 
    def callback_period(self, ucbp):
        """set the (periodic) callback period in seconds or None to disable the callback"""
        self._callback_period = max(0.1, ucbp) if ucbp is not None else None

    def open(self):
        """open the session"""
        BokehServer.open_session(self)

    def close(self):
        """close the session"""
        if self._doc:
            self._doc.clear()
        BokehServer.close_session(self)
        
    def setup_document(self):
        """give the session a chance to setup the freshy created bokeh document"""
        pass

    def periodic_callback(self):
        """periodic callback (default impl. does nothing)"""
        pass
  
    def pause(self):
        """suspend the (periodic) callback"""
        BokehServer.update_callback_period(self, None)
        self._suspended = True
    
    def resume(self):
        """resume the (periodic) callback"""
        BokehServer.update_callback_period(self, self.callback_period)
        self._suspended = False

    def update_callback_period(self, ucbp):
        """update the (periodic) callback"""
        self.callback_period = ucbp
        BokehServer.update_callback_period(self, self.callback_period)

    def safe_document_modifications(self, cb):
        """call the specified callback in the a context in which the session document is locked"""
        if self._doc:
            self._doc.add_next_tick_callback(cb)


# ------------------------------------------------------------------------------
class BokehServer(object):

    __bkh_srv__ = None
    __srv_url__ = None

    __sessions__ = deque()

    __logger__ = logging.getLogger(module_logger_name)
    __logger__.setLevel(logging.ERROR)

    @staticmethod
    def __start_server():
        output_notebook(resources=INLINE, hide_banner=True)
        app = Application(FunctionHandler(BokehServer.__session_entry_point))
        app.add(BokehSessionHandler())
        srv = Server(
            {'/': app},
            io_loop=IOLoop.instance(),
            port=0,
            host='*',
            allow_websocket_origin=['*']
        )
        srv.start()
        srv_addr = srv.address if srv.address else socket.gethostbyname(socket.gethostname())
        BokehServer.__bkh_srv__ = srv
        BokehServer.__srv_url__ = 'http://{}:{}'.format(srv_addr, srv.port)
        
    @staticmethod
    def __session_entry_point(doc):
        try:
            #TODO: should we lock BokehServer.__sessions__?
            BokehServer.__logger__.debug('BokehServer.__session_entry_point [doc:{}] <<'.format(id(doc)))
            session = BokehServer.__sessions__.pop()
            session._on_session_created(doc)
            BokehServer.__logger__.debug('BokehServer.__session_entry_point [doc:{}] >>'.format(id(doc)))
        except Exception as e:
            print(e)

    @staticmethod
    def __add_periodic_callback(session, ucbp):
        assert(isinstance(session, BokehSession))
        pcb = session.periodic_callback
        try:
            session.document.remove_periodic_callback(pcb)
        except:
            pass
        if ucbp is not None:
            session.document.add_periodic_callback(pcb, max(100, 1000. * ucbp))
        
    @staticmethod
    def open_session(new_session):
        BokehServer.__logger__.debug("BokehServer.open_session <<")
        assert(isinstance(new_session, BokehSession))
        if not BokehServer.__bkh_srv__:
            BokehServer.__logger__.debug("BokehServer.open_session.starting server")
            BokehServer.__start_server()
            BokehServer.__logger__.debug("BokehServer.open_session.server started")
        #TODO: should we lock BokehServer.__sessions__?
        BokehServer.__sessions__.appendleft(new_session)
        BokehServer.__logger__.debug("BokehServer.open_session.autoload server - url is {}".format(BokehServer.__srv_url__))
        script = autoload_server(model=None, url=BokehServer.__srv_url__)
        html_display = HTML(script)
        display(html_display)
        BokehServer.__logger__.debug("BokehServer.open_session >>")
        
    @staticmethod
    def close_session(session):
        """totally experimental attempt to destroy a session from python!"""
        assert(isinstance(session, BokehSession))
        session_id = session._doc.session_context.id
        bkh_session = BokehServer.__bkh_srv__.get_session('/', session_id)
        bkh_session.destroy()
        session._on_session_destroyed()
        
    @staticmethod
    def update_callback_period(session, ucbp):
        assert(isinstance(session, BokehSession))
        BokehServer.__add_periodic_callback(session, ucbp)
        
    @staticmethod
    def print_info(called_from_session_handler=False):
        if not BokehServer.__bkh_srv__:
            BokehServer.__logger__.debug("no Bokeh server running") 
            return
        try:
            BokehServer.__logger__.debug("Bokeh server URL: {}".format(BokehServer.__srv_url__))
            sessions = BokehServer.__bkh_srv__.get_sessions()
            num_sessions = len(sessions)
            if called_from_session_handler:
                num_sessions += 1
            BokehServer.__logger__.debug("Number of opened sessions: {}".format(num_sessions))
        except Exception as e:
            BokehServer.__logger__.error(e)


