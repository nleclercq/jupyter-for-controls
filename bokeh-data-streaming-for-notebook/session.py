from __future__ import print_function

import socket
import logging
from threading import Lock, Condition
from collections import deque

from IPython.display import HTML, clear_output, display

from tornado.ioloop import IOLoop

from bokeh.io import output_notebook
from bokeh.resources import Resources, INLINE
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import Handler, FunctionHandler
from bokeh.embed import server_document

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
        self.__set_callback_period(None)
        self._suspended = True
    
    def resume(self):
        """resume the (periodic) callback"""
        self.__set_callback_period(self.callback_period)
        self._suspended = False

    def update_callback_period(self, cbp):
        self.callback_period = cbp
        self.__set_callback_period(cbp)

    def __set_callback_period(self, cbp):
        try:
            self.document.remove_periodic_callback(self.periodic_callback)
        except:
            pass
        if cbp is not None:
            self.document.add_periodic_callback(self.periodic_callback, max(100, int(1000. * cbp)))

    def timeout_callback(self, cb, tmo):
        """call the specified callback after the specified timeout (in seconds) expires"""
        if self._doc:
            #print('BokehSession.timeout_callback: installing timeout callback...')
            self._doc.add_timeout_callback(cb, int(1000. * tmo))
            #print('BokehSession.timeout_callback: timeout callback installed: cb {} will be called in {} seconds'.format(cb, tmo))

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
        logging.getLogger('bokeh.server.util').setLevel(logging.ERROR) #TODO: tmp stuff
        output_notebook(Resources(mode='inline', components=["bokeh", "bokeh-gl"]), hide_banner=True)
        app = Application(FunctionHandler(BokehServer.__session_entry_point))
        #TODO the following in broken since bokeh 0.12.7: app.add(BokehSessionHandler())
        srv = Server(
            {'/': app},
            io_loop=IOLoop.instance(),
            port=0,
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
    def open_session(new_session):
        BokehServer.__logger__.debug("BokehServer.open_session <<")
        assert(isinstance(new_session, BokehSession))
        if not BokehServer.__bkh_srv__:
            BokehServer.__logger__.debug("BokehServer.open_session.starting server")
            BokehServer.__start_server()
            BokehServer.__logger__.debug("BokehServer.open_session.server started")
        BokehServer.__sessions__.appendleft(new_session)
        BokehServer.__logger__.debug("BokehServer.open_session.autoload server - url is {}".format(BokehServer.__srv_url__))
        script = server_document(url=BokehServer.__srv_url__)
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


