Ascent back-end
===============
Ascent is a many-core capable lightweight in-situ visualization and analysis
infrastructure for multi-physics HPC simulations. The SENSEI
AscentAnalysisAdaptor enables simulations instrumented with SENSEI to process
data using Ascent.

SENSEI XML
----------
The ascent back-end is activated using the :code:`<analysis type="ascent">`. The supported attributes are:

+-------------------+--------------------------------------------------------+
| attribute         | description                                            |
+-------------------+--------------------------------------------------------+
|  actions          |  Path to ascent specific JSON file configuring ascent  |
+-------------------+--------------------------------------------------------+
|  options          |  Path to ascent specific JSON file configuring ascent  |
+-------------------+--------------------------------------------------------+


Back-end specific configurarion
-------------------------------
SENSEI uses XML to select the specific back-end, in this case Ascent. The
SENSEI XML will also contain references to Ascent specific configuration files
that tell Ascent what to do. These files are native to Ascent. More information
about configuring Ascent can be found in the Ascent documentation at
https://ascent.readthedocs.io/en/latest/

Examples
--------
Reaction rate in situ demo :ref:`ascent_insitu_demo`.


