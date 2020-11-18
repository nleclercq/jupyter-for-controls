import logging
import itango
import jupytango.tango.magics

# ------------------------------------------------------------------------------
def load_ipython_extension(app):
    logging.getLogger('jupytango').setLevel(logging.ERROR)
    logging.getLogger('bokeh').setLevel(logging.ERROR)
    logging.getLogger('tornado').setLevel(logging.ERROR)
    try:
        itango.load_ipython_extension(app)
    except Exception as e:
        print("failed to load itango extension: {}".format(e))
        raise
    try:
        jupytango.tango.magics.load_ipython_extension(app)
    except Exception as e:
        print("failed to load jupytango extension: {}".format(e))
        raise
        
# ------------------------------------------------------------------------------   
def unload_ipython_extension(app):
    try:
        itango.unload_ipython_extension(app)
    except Exception as e:
        print(e)
    try:
        jupytango.unload_ipython_extension(app)
    except Exception as e:
        print(e)