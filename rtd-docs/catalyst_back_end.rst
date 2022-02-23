.. _catalyst_back_end:

Catalyst back-end
=================

ParaView Catalyst (Catalyst) is an in situ use case library, with an adaptable
application programming interface (API), that orchestrates the delicate
alliance between simulation and analysis and/or visualization tasks. It brings
the renown, scaling capabilities of VTK and ParaView to bear on the in situ use
case. The analysis and visualization tasks can be implemented in C++ or Python,
and Python scripts can be crafted from scratch or using the ParaView GUI to
interactively setup Catalyst scripts
(see `Catalyst User Guide <https://www.paraview.org/files/catalyst/docs/ParaViewCatalystUsersGuide_v2.pdf>`_).

.. _catalyst_xml_options:

SENSEI XML Options
------------------

The Catalyst back-end is activated using the :code:`<analysis type="catalyst">`.

.. _catalyst_python_script:

Python Script
^^^^^^^^^^^^^

The supported attributes are:

+-------------------+--------------------------------------------------------+
| attribute         | description                                            |
+-------------------+--------------------------------------------------------+
|  pipeline         | Use "pythonscript".                                    |
+-------------------+--------------------------------------------------------+
|  filename         | pythonscript filename.                                 |
+-------------------+--------------------------------------------------------+
|  enabled          | "1" enables this back-end.                             |
+-------------------+--------------------------------------------------------+

Example XML
~~~~~~~~~~~

Catalyst Python script example. This XML configures a Catalyst with a Python script that creates a pipeline(s).

.. code-block:: XML

  <sensei>
  <analysis type="catalyst" pipeline="pythonscript"
            filename="configs/random_2d_64_catalyst.py" enabled="1" />
  </sensei>

Back-end specific configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to create a python script for Catalyst:

#. Load a sample of the data (possibly downsampled) into ParaView, including all the desired fields.
#. Create analysis and visualization pipeline(s) in ParaView by applying successive filters producing subsetted or alternative visual metaphors of data.
#. Define a Catalyst extracts with the menu choice *Catalyst→Define Exports*: this will pop up the *Catalyst Export Inspector* panel.
#. Export the Catalyst Python script using the menu *Catalyst→Export Catalyst Script*.

The Catalyst Export Inspector reference.

.. _catalyst_slice_fixed:

Slice Fixed Pipeline
^^^^^^^^^^^^^^^^^^^^

For the Catalyst slice fixed pipeline the supported attributes are:

+-------------------+--------------------------------------------------------+
| attribute         | description                                            |
+-------------------+--------------------------------------------------------+
|  pipeline         | Use "slice".                                           |
+-------------------+--------------------------------------------------------+
|  mesh             | The name of the mesh to slice.                         |
+-------------------+--------------------------------------------------------+
|  array            | The data array name for coloring.                      |
+-------------------+--------------------------------------------------------+
|  association      | Either "cell" or "point" data.                         |
+-------------------+--------------------------------------------------------+
|  image-filename   | The filename template to write images.                 |
+-------------------+--------------------------------------------------------+
|  image-width      | The image width in pixels.                             |
+-------------------+--------------------------------------------------------+
|  image-height     | The image height in pixels.                            |
+-------------------+--------------------------------------------------------+
|  slice-origin     | The origin to use for slicing (optional).              |
+-------------------+--------------------------------------------------------+
|  slice-normal     | The normal to use for slicing.                         |
+-------------------+--------------------------------------------------------+
|  color-range      | The color range of the array (optional).               |
+-------------------+--------------------------------------------------------+
|  color-log        | Use logarithmic color scale (optional).                |
+-------------------+--------------------------------------------------------+
|  enabled          | "1" enables this back-end                              |
+-------------------+--------------------------------------------------------+

Example XML
~~~~~~~~~~~
This XML configures a C++-based fixed pipeline for a slice using Catalyst.

.. code-block:: XML

  <sensei>
    <analysis type="catalyst"
              pipeline="slice" mesh="mesh" array="data" association="cell"
              image-filename="slice-%ts.png" image-width="1920" image-height="1080"
              slice-normal="0,0,1"
              color-range="0.0001,1.5" color-log="1"
              enabled="1" />
  </sensei>

.. _catalyst_particles_fixed:

Particles Fixed Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^

For the Catalyst particle fixed pipeline the supported attributes are:

+-------------------+--------------------------------------------------------+
| attribute         | description                                            |
+-------------------+--------------------------------------------------------+
|  pipeline         | Use "particle".                                        |
+-------------------+--------------------------------------------------------+
|  mesh             | The name of the mesh to slice.                         |
+-------------------+--------------------------------------------------------+
|  array            | The data array name for coloring.                      |
+-------------------+--------------------------------------------------------+
|  association      | Either "cell" or "point" data.                         |
+-------------------+--------------------------------------------------------+
|  image-filename   | The filename template to write images.                 |
+-------------------+--------------------------------------------------------+
|  image-width      | The image width in pixels.                             |
+-------------------+--------------------------------------------------------+
|  image-height     | The image height in pixels.                            |
+-------------------+--------------------------------------------------------+
|  particle-style   | The representation such as: "Gaussian Blur", "Sphere", |
+-------------------+--------------------------------------------------------+
|                   | "Black-edged circle", "Plain circle", "Triangle", and  |
+-------------------+--------------------------------------------------------+
|                   | "Square Outline".                                      |
+-------------------+--------------------------------------------------------+
|  particle-radius  | The normal to use for slicing.                         |
+-------------------+--------------------------------------------------------+
|  color-range      | The color range of the array (optional).               |
+-------------------+--------------------------------------------------------+
|  camera-position  | The position of the camera (optional).                 |
+-------------------+--------------------------------------------------------+
|  camera-focus     | Where the camera points (optional).                    |
+-------------------+--------------------------------------------------------+
|  enabled          | "1" enables this back-end                              |
+-------------------+--------------------------------------------------------+

Example XML
~~~~~~~~~~~

This XML configures a C++-based fixed pipeline for particles using Catalyst.

.. code-block:: XML

  <sensei>
    <analysis type="catalyst"
       pipeline="particle" mesh="particles" array="data" association="point"
       image-filename="/tmp/particles-%ts.png" image-width="1920" image-height="1080"
       particle-style="Black-edged circle" particle-radius="0.5"
       color-range="0.0,1024.0" color-log="0"
       camera-position="150,150,100" camera-focus="0,0,0"
       enabled="1" />
  </sensei>

.. _catalyst_example:

Example
-------

Reaction rate in situ demo :ref:`catalyst_insitu_demo`.
