Catalyst back-end
=================
ParaView Catalyst (Catalyst) is an in situ use case library, with an adaptable application programming interface (API), that orchestrates the delicate alliance between simulation and analysis and/or visualization tasks. It brings the renown, scaling capabilities of VTK and ParaView to bear on the in situ use case. The analysis and visualization tasks can be implemented in C++ or Python, and Python scripts can be crafted from scratch or using the ParaView GUI to interactively setup Catalyst scripts.

SENSEI XML Options
------------------
The Catalyst back-end is activated using the :code:`<analysis type="catalyst">`. The supported attributes are:

+-------------------+--------------------------------------------------------+
| attribute         | description                                            |
+-------------------+--------------------------------------------------------+
|  pipeline         | Use "pythonscript", but other fixed pipelines exist.   |
+-------------------+--------------------------------------------------------+
|  filename         | pythonscript filename                                  |
+-------------------+--------------------------------------------------------+
|  enabled          | "1" enables this back-end                              |
+-------------------+--------------------------------------------------------+

Example XML
^^^^^^^^^^^
Catalyst Python script example. This XML configures a Catalyst with a Python script that creates a pipeline(s).

.. code-block:: XML

  <sensei>
  <analysis type="catalyst" pipeline="pythonscript"
            filename="configs/random_2d_64_catalyst.py" enabled="1" />
  </sensei>

Back-end specific configuration
-------------------------------
The easiest way to create a python script for Catalyst: 

#. Load a sample of the data (possibly downsampled) into ParaView, including all the desired fields.
#. Create analysis and visualization pipeline(s) in ParaView by applying successive filters producing subsetted or alternative visual metaphors of data.
#. Define a Catalyst extracts with the menu choice *Catalyst→Define Exports*: this will pop up the *Catalyst Export Inspector* panel.
#. Export the Catalyst Python script using the menu *Catalyst→Export Catalyst Script*.

Example
-------

Reaction rate in situ demo :ref:`catalyst_insitu_demo`.
