from __future__ import print_function
import logging
import socket
from threading import Lock
from uuid import uuid4

from IPython.display import HTML, display, publish_display_data

from tornado.ioloop import IOLoop

from bokeh.io.state import curstate
from bokeh.io import output_notebook
from bokeh.resources import Resources
from bokeh.application import Application
from bokeh.application.handlers import Handler, FunctionHandler
from bokeh.embed import server_document
from bokeh.io.notebook import EXEC_MIME_TYPE, HTML_MIME_TYPE
from bokeh.server.server import Server

from jupytango.tools import JupyterContext, get_jupyter_context, NotebookCellContent

module_logger = logging.getLogger(__name__)

output_notebook(Resources(mode='inline', components=["bokeh", "bokeh-gl"]), verbose=False, hide_banner=True)

# ------------------------------------------------------------------------------
class BokehSession(object):
    
    __repo__ = dict()
    __repo_lock__ = Lock()
    
    def __init__(self, uuid=None):
        # session identifier
        self._uuid = uuid if uuid else str(uuid4().hex)
        # logger
        self._session_logger = NotebookCellContent(self._uuid, logger=module_logger) 
        # the session info
        self._server_info = dict()
        # the associated bokeh document (for experts only)
        self._doc = None
        # periodic callback id
        self._callback_id = None
        # periodic callback period in seconds - defaults to None (i.e. disabled)
        self._callback_period = None
        # periodic activity enabled?
        self._suspended = True
        # is this session closed?
        self._closed = False
        # close existing session: this is a way to avoid leaks & resources waste
        if uuid is not None:
            self.__close_existing_session()
        # insert new session into the repo
        with BokehSession.__repo_lock__:
            BokehSession.__repo__[self._uuid] = self
        #BokehSession.print_repository_status()
    
    def __close_existing_session(self):
        with BokehSession.__repo_lock__:
            try:
                session = BokehSession.__repo__[self._uuid]
                session.close()
            except KeyError:
                pass
            except Exception as e:
                self._session_logger.error(e)
            finally:
                try:
                    del BokehSession.__repo__[self._uuid]
                except KeyError:
                    pass
                except Exception as e:
                    self._session_logger.error(e)

    def _on_session_destroyed(self):
        pass

    @property
    def uuid(self):
        return self._uuid

    @property
    def ready(self):
        return self._doc is not None

    @property
    def server(self):
        return self._server_info['server']

    @property
    def document(self):
        return self._doc

    @property
    def bokeh_session(self):
        return self._server_info['server'].get_sessions('/')[0]

    @property
    def bokeh_session_id(self):
        return self._doc.session_context.id if self._doc else None

    @property
    def suspended(self):
        return self._suspended

    @property
    def opened(self):
        return not self._closed

    @property
    def closed(self):
        return self._closed

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
        self.__open()

    def close(self, async_mode=True):
        """close the session"""
        self.cleanup(async_mode)

    def cleanup(self, async_mode=True):
        """cleanup the session"""
        # TODO: async cleanup required but might not be safe!
        self.pause()
        if async_mode:
            self.safe_document_modifications(self.__cleanup)
        else:
            self.__cleanup()
        
    def __cleanup(self):
        """asynchronous close"""
        # TODO: async close required but might not be safe!
        try:
            if self._doc:
                self._doc.clear()
            self.__close()
        except Exception as e:
            self._session_logger.error(e)
        finally:
            self._closed = True
            with BokehSession.__repo_lock__:
                try:
                    del BokehSession.__repo__[self._uuid]
                except:
                    pass
            #BokehSession.print_repository_status()

    def setup_document(self):
        """give the session a chance to setup the freshy created bokeh document"""
        pass

    def periodic_callback_enabled(self):
        """return True if the periodic callback is enabled, return False otherwise"""
        return not self.callback_period == None

    def periodic_callback(self):
        """periodic callback (default impl. does nothing)"""
        pass

    def start(self):
        """start the periodic activity (if any)"""
        self.resume()

    def stop(self):
        """stop the periodic activity (if any)"""
        self.pause()

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
            self.document.remove_periodic_callback(self._callback_id)
        except:
            pass
        finally:
            self._callback_id = None
        if cbp is not None:
            self._callback_id = self.document.add_periodic_callback(self.periodic_callback, max(100, int(1000. * cbp)))

    def timeout_callback(self, cb, tmo):
        """call the specified callback after expiration of the specified timeout (in seconds)"""
        if self.ready:
            self._doc.add_timeout_callback(cb, int(1000. * tmo))

    def safe_document_modifications(self, cb):
        """call the specified callback in the a context in which the session document is locked"""
        if self.ready:
            self._doc.add_next_tick_callback(cb)

    def __repr__(self):
        return "BokehSession:{}:{}".format(self._uuid, ('closed' if self._closed else 'opened'))
    
    def __open(self):
        self._session_logger.debug("BokehSession.__open.spawning server for session {}".format(self.uuid[-5:]))
        self.__spawn_server()
        script = server_document(url=self._server_info['server_url'])
        if get_jupyter_context() == JupyterContext.LAB:
            self._session_logger.debug("BokehSession.open_session:running in JupyterContext.LAB")
            data = {HTML_MIME_TYPE: script, EXEC_MIME_TYPE: ""}
            metadata = {EXEC_MIME_TYPE: {"server_id": self._server_info['server_id']}}
            publish_display_data(data, metadata=metadata)
        else:
            self._session_logger.debug("BokehSession.open_session:running in JupyterContext.NOTEBOOK")
            display(HTML(script))
        self._session_logger.debug("BokehSession.open_session.server spawn for session {}".format(self.uuid[-5:]))
  
    def __spawn_server(self):
        bslg = logging.getLogger('bokeh.server.util')
        bsll = bslg.getEffectiveLevel()
        bslg.setLevel(logging.ERROR) 
        self._server_info['application'] = app = Application(FunctionHandler(self.__entry_point))
        self._server_info['server'] = srv = Server({'/': app}, io_loop=IOLoop.instance(), port=0, allow_websocket_origin=['*'])
        self._server_info['server_id'] = srv_id = uuid4().hex
        curstate().uuid_to_server[srv_id] = srv
        srv_addr = srv.address if srv.address else socket.gethostbyname(socket.gethostname())
        self._server_info['server_url'] = 'http://{}:{}/'.format(srv_addr, srv.port)
        srv.start()
        bslg.setLevel(bsll)
    
    def __entry_point(self, doc):
        try:
            self._session_logger.debug("BokehSession.entry_point << for session {}".format(self.uuid[-5:]))
            self._doc = doc
            self.setup_document()
            self._session_logger.debug("BokehSession.entry_point >> for session {}".format(self.uuid[-5:]))
        except Exception as e:
            self._session_logger.error(e)
        finally:
            return doc
        
    def __close(self):
        # TODO: how to release every single resource associated with the session?
        self._session_logger.debug("BokehSession.closing session {}".format(self.uuid))
        if self.server:
            try:
                self.server.stop()
            except AssertionError as e:
                # ignore bokeh "already stopped" error
                pass
     
    @staticmethod
    def close_all():
        with BokehSession.__repo_lock__:
            for s in BokehSession.__repo__.values():
                try:
                    s.close()
                except Exception as e:
                    pass
                    
    @staticmethod
    def print_repository_status():
        with BokehSession.__repo_lock__:
            if len(BokehSession.__repo__):
                print('BokehSession.repository contains {} session(s):'.format(len(BokehSession.__repo__)))
                for s in BokehSession.__repo__.values():
                    print('- {}'.format(s))
            else:
                print('BokehSession.repository is empty')
