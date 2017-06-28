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
        BokehServer.print_info(True)
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
            session._doc.add_periodic_callback(pcb, max(250, 1000. * prd))
        
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


class MySessionExample(BokehSession):
    
    def __init__(self):
        BokehSession.__init__(self)
        self.callback_period = 1.
        self._np = 100
        self._widgets_layout = None
        columns = dict()
        columns['x'] = self._gen_x_scale()
        columns['y'] = self._gen_y_random_data()
        self._cds = ColumnDataSource(data=columns)

    def _gen_x_scale(self):
        """x data"""
        return np.linspace(1, self._np, num=self._np, endpoint=True)
    
    def _gen_y_random_data(self):
        """y data"""
        return np.random.rand(self._np)
    
    def __on_update_period_change(self, attr, old, new):
        """called when the user changes the refresh period using the dedicated slider"""
        self.callback_period = new
        
    def __on_num_points_change(self, attr, old, new):
        """called when the user changes the number of points using the dedicated slider"""
        self._np = int(new)

    def setup_model(self):
        """setup the session model then return it"""
        # a slider to control the update period
        rrs = Slider(start=0.25, 
                     end=2, 
                     step=0.25, 
                     value=self.callback_period, 
                     title="Updt.period [s]",)
        rrs.on_change("value", self.__on_update_period_change)
        # a slider to control the number of points
        nps = Slider(start=0, 
                     end=1000, 
                     step=10, 
                     value=self._np, 
                     title="Num.points")
        nps.on_change("value", self.__on_num_points_change)
        # the figure and its content
        p = figure(plot_width=650, plot_height=200)
        p.toolbar_location = 'above'
        p.line(x='x', y='y', source=self._cds, color="navy", alpha=0.5)
        # widgets are placed into a dedicated layout
        self._widgets_layout = widgetbox(nps, rrs)
        # arrange all items into a layout then return it as the session model
        return layout([[self._widgets_layout, p]])
    
    def periodic_callback(self):
        """periodic activity"""
        self._cds.data.update(x=self._gen_x_scale(), y=self._gen_y_random_data())


class MyExtendedSessionExample(MySessionExample):
    
    def __init__(self):
        MySession.__init__(self)
        self._suspend_resume_button = None

    def setup_model(self):
        model = super(MyExtendedSession, self).setup_model()
        cb = Button(label='close')
        cb.on_click(self.close)
        self._suspend_resume_button = Button(label='suspend')
        self._suspend_resume_button.on_click(self.suspend_resume)
        self._widgets_layout.children.append(self._suspend_resume_button)
        self._widgets_layout.children.append(cb)
        return model
    
    def periodic_callback(self):
        print('MyExtendedSession.periodic_callback <<')
        super(MyExtendedSession, self).periodic_callback()
        print('MyExtendedSession.periodic_callback >>')
   
    def suspend_resume(self): 
        # callback period == None means callback disabled (i.e. suspended)
        if self.callback_period is None:
            self._suspend_resume_button.label = 'suspend'
            self.resume()
        else:
            self._suspend_resume_button.label = 'resume'
            self.pause()
        
    def close(self):  
        """overwrites super.close - tries to cleanup everything properly"""
        try:
            # clear document content (i.e. remove roots)
            self._doc.clear()
        except Exception as e:
            print(e)
        try:  
            # close the session (will remove all callbacks)
            BokehServer.close_session(self)
        except Exception as e:
            print(e) 
        # finally clear cell outputs - e.g. logging (this is an ipython call - not a bokeh one)
        clear_output()

