import logging
import itango
import jupytango.tango.magics

# ------------------------------------------------------------------------------
def load_ipython_extension(app):
    logging.getLogger('jupytango').setLevel(logging.ERROR)
    logging.getLogger('bokeh').setLevel(logging.ERROR)
    logging.getLogger('tornado').setLevel(logging.ERROR)
    try:
        print("loading itango extension...")
        itango.load_ipython_extension(app)
        print("itango extension successfully loaded.")
    except Exception as e:
        print("failed to load itango extension: {}".format(e))
        raise
    try:
        print("loading jupytango extension...")
        jupytango.tango.magics.load_ipython_extension(app)
        print("jupytango extension successfully loaded.")
    except Exception as e:
        print("failed to load jupytango extension: {}".format(e))
        raise
        
# ------------------------------------------------------------------------------   
def unload_ipython_extension(app):
    try:
        print("unloading itango extension...")
        itango.unload_ipython_extension(app)
        print("itango extension successfully unloaded.")
    except Exception as e:
        print(e)
    try:
        print("unloading jupytango extension...")
        jupytango.tango.magics.unload_ipython_extension(app)
        print("jupytango extension successfully unloaded.")
    except Exception as e:
        print(e)
