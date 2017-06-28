from __future__ import print_function

import socket

from collections import deque

import numpy as np

from IPython.display import HTML, clear_output, display

from tornado.ioloop import IOLoop
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import Handler, FunctionHandler
from bokeh.embed import autoload_server
from bokeh.io import reset_output

from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from bokeh.models.glyphs import Rect
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, Button
from bokeh.layouts import row, layout, widgetbox


class BokehSessionHandler(Handler):

    def on_server_loaded(self, server_context):
        print("SessionHandler: on_server_loaded <<")
        print("SessionHandler: on_server_loaded >>")

    def on_server_unloaded(self, server_context):
        print("SessionHandler: on_server_unloaded <<")
        print("SessionHandler: on_server_unloaded >>")

    def on_session_created(self, session_context):
        print("SessionHandler: on_session_created <<")
        print("SessionHandler: on_session_created >>")

    def on_session_destroyed(self, session_context):
        print("SessionHandler: on_server_unloaded <<")
        print("SessionHandler: on_server_unloaded >>")


class BokehSession(object):
    
    def __init__(self):
        """the associated bokeh document (for experts only)"""
        self._doc = None
        """periodic callback period in seconds - defaults to None (i.e. disabled)"""
        self._callback_period = None
            
    def open(self):
        """open the session"""
        BokehServer.open_session(self)
        
    def close(self):
        """close the session"""
        BokehServer.close_session(self)
        
    def setup_model(self):
        """return the session model or None if no model"""
        return None

    def periodic_callback(self):
        """periodic callback (default impl. does nothing)"""
        pass
  
    def pause(self):
        """suspend the (periodic) callback"""
        self._saved_callback_period = self._callback_period
        self.callback_period = None
    
    def resume(self):
        """resume the (periodic) callback"""
        try:
            self.callback_period = self._saved_callback_period
        except:
            pass
        
    @property 
    def callback_period(self):
        """return the (periodic) callback period in seconds or None (i.e. disabled)"""
        return self._callback_period

    @callback_period.setter 
    def callback_period(self, p):
        """set the (periodic) callback period in seconds or None to disable the callback"""
        self._callback_period = p
        if self._doc is not None:
            BokehServer.update_callback_period(self)


class BokehServer(object):

    __bkh_app__ = None
    __bkh_srv__ = None
    __srv_url__ = None
    __sessions__ = deque()
        
    @staticmethod
    def __start_server():
        app = Application(FunctionHandler(BokehServer.__entry_point))
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
        BokehServer.__bkh_app__ = app
        BokehServer.__srv_url__ = 'http://{}:{}'.format(srv_addr, srv.port)
        
    @staticmethod
    def __entry_point(doc):
        try:
            #TODO: should we lock BokehServer.__sessions__? 
            session = BokehServer.__sessions__.pop() 
            session._doc = doc
            model = session.setup_model()
            if model:
                doc.add_root(model)
            BokehServer.__add_periodic_callback(session)
        except Exception as e:
            print(e)
        
    @staticmethod
    def __add_periodic_callback(session):
        assert(isinstance(session, BokehSession))
        pcb = session.periodic_callback
        try:
            session._doc.remove_periodic_callback(pcb)
        except:
            pass
        prd = session.callback_period
        if prd is not None:
            session._doc.add_periodic_callback(pcb, max(1, 1000. * prd))
        
    @staticmethod
    def open_session(new_session):
        assert(isinstance(new_session, BokehSession))
        if not BokehServer.__bkh_srv__:
            BokehServer.__start_server()
        #TODO: should we lock BokehServer.__sessions__? 
        BokehServer.__sessions__.appendleft(new_session) 
        script = autoload_server(model=None, url=BokehServer.__srv_url__)
        html_display = HTML(script)
        display(html_display)
        
    @staticmethod
    def close_session(session):
        """totally experimental attempt to destroy a session from python!"""
        assert(isinstance(session, BokehSession))
        session_id = session._doc.session_context.id
        session = BokehServer.__bkh_srv__.get_session('/', session_id)
        session.destroy()
        
    @staticmethod
    def update_callback_period(session):
        assert(isinstance(session, BokehSession))
        BokehServer.__add_periodic_callback(session)
        
    @staticmethod
    def print_info(called_from_session_handler=False):
        if not BokehServer.__bkh_srv__:
            print("no Bokeh server running") 
            return
        try:
            print("Bokeh server URL: {}".format(BokehServer.__srv_url__))
            sessions = BokehServer.__bkh_srv__.get_sessions()
            num_sessions = len(sessions)
            if called_from_session_handler:
                num_sessions += 1
            print("Number of opened sessions: {}".format(num_sessions))
        except Exception as e:
            print(e)


