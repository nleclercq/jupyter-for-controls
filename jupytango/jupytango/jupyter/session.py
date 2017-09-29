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

from __future__ import print_function

import imp
import logging
import socket
from threading import Lock, Condition
from collections import deque
from uuid import uuid4

from IPython.display import HTML, display, publish_display_data

from tornado.ioloop import IOLoop
        
from bokeh.io import output_notebook, curstate, show
from bokeh.resources import Resources
from bokeh.application import Application
from bokeh.application.handlers import Handler, FunctionHandler
from bokeh.embed import server_document
from bokeh.util.notebook import EXEC_MIME_TYPE, HTML_MIME_TYPE

from bokeh.server.server import Server
        
try:
    from fs.client.jupyter.tools import JupyterContext, get_jupyter_context, NotebookCellContent
except:
    try: 
        from jupytango.jupyter.tools import JupyterContext, get_jupyter_context, NotebookCellContent
    except:
        from common.tools import JupyterContext, get_jupyter_context, NotebookCellContent
        
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

    def modify_document(self, doc):
        return doc


# ------------------------------------------------------------------------------
class BokehSession(object):
    
    __repo__ = dict()
    __repo_lock__ = Lock()
    
    def __init__(self, uuid=None):
        # session identifier
        self._uuid = uuid if uuid else str(uuid4().hex)
        # logger
        self._logger = NotebookCellContent(self._uuid, logger=logging.getLogger(module_logger_name)) 
        # the session info
        self._server_info = dict()
        # the associated bokeh document (for experts only)
        self._doc = None
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
        BokehSession.print_repository_status()
    
    def __close_existing_session(self):
        with BokehSession.__repo_lock__:
            try:
                session = BokehSession.__repo__[self._uuid]
                session.close()
            except KeyError:
                pass
            except Exception as e:
                self._logger.error(e)
            finally:
                try:
                    del BokehSession.__repo__[self._uuid]
                except KeyError:
                    pass
                except Exception as e:
                    self._logger.error(e)

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

    def close(self, async=True):
        """close the session"""
        self.cleanup(async)

    def cleanup(self, async=True):
        """cleanup the session"""
        # TODO: async cleanup required but might not be safe!
        self.pause()
        if async:
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
            self._logger.error(e)
        finally:
            self._closed = True
            with BokehSession.__repo_lock__:
                try:
                    del BokehSession.__repo__[self._uuid]
                except:
                    pass
            BokehSession.print_repository_status()

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
            self.document.remove_periodic_callback(self.periodic_callback)
        except:
            pass
        if cbp is not None:
            self.document.add_periodic_callback(self.periodic_callback, max(100, int(1000. * cbp)))

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
        self._logger.debug("BokehSession.open_session.spawning server for session {}".format(self.uuid))
        self.__spawn_server()
        script = server_document(url=self._server_info['server_url'])
        if get_jupyter_context() == JupyterContext.LAB:
            self._logger.info("BokehSession.open_session:running in JupyterContext.LAB")
            data = {HTML_MIME_TYPE: script, EXEC_MIME_TYPE: ""}
            metadata = {EXEC_MIME_TYPE: {"server_id": self._server_info['server_id']}}
            publish_display_data(data, metadata=metadata)
        else:
            self._logger.info("BokehSession.open_session:running in JupyterContext.NOTEBOOK")
            display(HTML(script))
            self._logger.debug("BokehSession.open_session.server spawn for session {}".format(self.uuid))
          
    def __spawn_server(self):
        bslg = logging.getLogger('bokeh.server.util')
        bsll = bslg.getEffectiveLevel()
        bslg.setLevel(logging.ERROR) 
        output_notebook(Resources(mode='inline', components=["bokeh", "bokeh-gl"]), hide_banner=True)
        self._server_info['application'] = app = Application(FunctionHandler(self.__entry_point))
        app.add(BokehSessionHandler())
        self._server_info['server'] = srv = Server({'/': app}, io_loop=IOLoop.instance(), port=0, allow_websocket_origin=['*'])
        self._server_info['server_id'] = srv_id = uuid4().hex
        curstate().uuid_to_server[srv_id] = srv
        srv_addr = srv.address if srv.address else socket.gethostbyname(socket.gethostname())
        self._server_info['server_url'] = 'http://{}:{}/'.format(srv_addr, srv.port)
        srv.start()
        bslg.setLevel(bsll)
    
    def __entry_point(self, doc):
        try:
            self._logger.debug("BokehSession.entry_point for session {}".format(self.uuid))
            self._doc = doc
            self.setup_document()
        except Exception as e:
            self._logger.error(e)
        finally:
            return doc
        
    def __close(self):
        # TODO: how to release every single resource associated with the session?
        self._logger.debug("BokehSession.closing session {}".format(self.uuid))
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
                print('BokehSession.repository. contains {} session(s):'.format(len(BokehSession.__repo__)))
                for s in BokehSession.__repo__.values():
                    print('- {}'.format(s))
            else:
                print('BokehSession.repository is empty')