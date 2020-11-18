# **jupyter-for-controls**
Small projects, evaluations, POC, presentations, ... related to the Jupyter Ecosystem (and 'embeddable' technologies, e.g. bokeh) for scientific experiments controls.

# **jupyTango on Ubuntu Linux**
### **leveraging pytango, itango and bokeh features in the Jupyter notebook**

## About this notebook

This notebook belongs to a series of small projects which aim is to evaluate the [Jupyter](http://jupyter.org/) ecosystem for science experiments control. The main idea is to use the _Juypter notebook_ as a convergence platform in order to offer a fully featured environment to scientists. 

## About Tango 
[Tango](https://www.tango-controls.org) is a free open source device-oriented controls toolkit for controlling any kind of hardware or software and building SCADA (supervisory control and data acquisition) systems. Tango Controls is operating system independent and supports C++, Java and Python for all the components.

## About bokeh
Experiments control requires both static and dynamic (i.e live) data visualization. Since Jupyter doesn't provide any 'official' data visualization solution, we need to select one. Among the available solutions, [bokeh](http://bokeh.pydata.org/en/latest) presents the highest potential for our application.

## Topic of the day
This notebook is a tutorial on "How to provide amazing Tango releated features in the jupyterlab"

## jupyTango installation
Here is a step by step jupyTango installation procedure.

#### Step-00: (if required) follow the tango installation procedure for [ubuntu](https://tango-controls.readthedocs.io/en/latest/installation/tango-on-linux.html#debian-ubuntu)
- `sudo apt install mariadb-server`
- `sudo apt install tango-db tango-test`

#### Step-01: install [anaconda](https://www.anaconda.com/products/individual)
- create a dedicated jupytango environment (optional but recommended): `conda create -n jupytango python=3.7`
- activate the jupytango environment: `conda activate jupytango`

#### Step-02: install [scikit-image](https://scikit-image.org)
- `conda install -c conda-forge scikit-image`

#### Step-03: install [jupyterlab](https://jupyter.org)
- `conda install -c conda-forge jupyterlab`

#### Step-04: install [nodejs](https://nodejs.org/en/)
- `conda install -c conda-forge nodejs=10`

#### Step-05: install [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
- `conda install -c conda-forge ipywidgets`
- install the ipywidgets extension for jupyterlab: `jupyter labextension install @jupyter-widgets/jupyterlab-manager@2.0`

#### Step-06: install [bokeh](https://docs.bokeh.org/en/latest/)
- `conda install -c conda-forge bokeh`
- install the bokeh extension for jupyterlab: `jupyter labextension install @bokeh/jupyter_bokeh`

#### Step-07: install [pytango](https://pytango.readthedocs.io/en/stable/)
- `conda install -c tango-controls pytango`

#### Step-08: install [itango](https://pythonhosted.org/itango/)
- `conda install -c tango-controls itango`

#### Step-09: create the jupyTango profile (i.e. [ipython](https://ipython.org) profile) 
- create the profile directory: `cp -Rf $HOME/.ipython/profile_default $HOME/.ipython/profile_jupytango`
- in that directory create a file named `ipython_config.py` containing:
```
    config = get_config()
    config.InteractiveShellApp.extensions = ['jupytango']
```

#### Step-10: create the jupyTango (i.e. the [jupyter](https://jupyter.org) kernel) 
- create the kernel directory: `mkdir $HOME/.local/share/jupyter/kernels/jupytango`
- in that directory create a file named `kernel.json` containing the following configuration:
```
    {
         "argv": [
              "$HOME/anaconda/bin/python",
              "-m",
              "ipykernel",
              "-f",
              "{connection_file}",
              "--profile",
              "jupytango"
         ],
         "language": "python",
         "display_name": "jupyTango"
    }
```

## opening a jupyTango notebook
- Open a terminal 
- Specify the jupyTango context by typing: `export JUPYTER_CONTEXT=LAB` (this will be removed in a near future)
- Start jupyterlab by typing: `jupyter-lab` (will spawn a web browser instance)
- From the `Launcher` tab of the web browser `JupyterLab` tab, open a `jupyTango` notebook (click on the `J` icon)
- the `09_jupytango.ipynb` is a good starting point to get started with jupyTango

Enjoy!
