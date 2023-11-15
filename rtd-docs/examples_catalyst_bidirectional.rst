.. _catalystbidirectional:

***********************************************
Computational Steering with Catalyst and SENSEI
***********************************************

========
Synopsis
========

In this example, we demonstrate using SENSEI with a bidirectional Catalyst 1.0
Analysis Adaptor to perform computational steering. Here, we employ the
oscillator miniapp. The list of oscillator objects are exposed to Catalyst
where each oscillator can be manipulated. Position of oscillators as well as
oscillator properties can be modified on the fly, changing the behavior of the
simulated domain.

==========
Setting Up
==========

You will need ParaView 5.10 installed, and SENSEI compiled with the options
`SENSEI_ENABLE_CATALYST` and `SENSEI_ENABLE_CATALYST_PYTHON` turned on.

Note, only Catalyst 1.0 scripts are compatible with computational steering in
SENSEI. Unfortunately, those scripts cannot be generated automatically in the
most recent editions of ParaView. The provided example script can be modified
manually for your own purposes.

===================
Running the Example
===================

Open the ParaView application and navigate to `Tools`->`Manage Plugins`. Press
the `Load New...` button, and navigate the file browser to the location of the
`oscillator_catalyst_steering_proxies.xml` file, and select that file.

Next, navigate to `Catalyst`->`Connect` and accept incoming connections on port
22222. This will tell ParaView to begin listening for Catalyst connections on
that port. If a different port number is desired, you will need to edit the
port number in the Catalyst python script `oscillator_catalyst_steering.py` to
the desired port, and then start Catalyst in ParaView with the same desired
port number.

In a terminal, navigate to your desired run directory (this testing directory
is fine to run from), and start the oscillator miniapp with the SENSEI config
xml `oscillator_catalyst_steering.xml`. Oscillator miniapp options can be set
as desired, but a good starting point is:

.. code-block:: bash

    $ mpirun -np 1 /path/to/oscillator -g 1 -t 0.01 -f oscillator_catalyst_steering.xml simple.osc

With the Oscillator miniapp running, ParaView should automatically detect a new Catalyst connection and add several items to the catalsyt server list in the `Pipeline Browser`. Clicking the icon next to `mesh` and `oscillator` will display the data to the 3D viewport, updating as the miniapp progresses.

Click on the `Steering Parameters` item in the `Pipeline Browser`. The `Properties` panel will display several controls over each oscillator which can be modified, manipulating the oscillator parameters as the miniapp executes.

.. figure:: images/pv_catalyst_steering_gui.png
   :width: 30 %
   :align: center

   The Properties panel provides parameters to add or delete oscillators in the domain, and change the parameters of the oscillators independently.


.. figure:: images/pv_catalyst_steering_window.png
   :width: 80 %
   :align: center

   ParaView's GUI contains the Properties panel, where oscillator parameters can be edited, a center 3D Viewport where the oscillators are rendered using ray-traced volume rendering, and a second 3D Viewport where the 5 existing oscillators' locations are visualized with respect to one another.


=======
Results
=======

The key takeaway from this example is that Catalyst and SENSEI can be used to
perform computational steering tasks with in situ visualization. The
oscillators, whose properties and locations can be modified in situ, respond to
the user's modifications. Setting up such a computational steering workflow in
your own simulation code requires exposing desired parameters to SENSEI, and
writing XML instructions for ParaView to generate the GUI for modifying the
parameters.

