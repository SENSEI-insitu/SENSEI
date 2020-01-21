Autocorrelation back-end
========================
As a prototypical time-dependent analysis routine, the Autocorrelation back-end
computes the autocorrelation. Given a signal f(x) and a delay t, we find

.. math::

   \sum_{x}f(x)f(x+t).

Starting with an integer time delay t, we maintain in a circular buffer, for
each grid cell, a window of values of the last t time steps. We also maintain a
window of running correlations for each t′ ≤ t. When called, the analysis
updates the autocorrelations and the circular buffer. When the execution
completes, all processes perform a global reduction to determine the top k
autocorrelations for each delay t′ ≤ t (k is specified by the user). For
periodic oscillators, this reduction identifies the centers of the oscillators.

SENSEI XML
----------
The Autocorrelation back-end is activated using the :code:`<analysis type="autocorrelation">`. The supported attributes are:

+-------------------+--------------------------------------------------------+
| attribute         | description                                            |
+-------------------+--------------------------------------------------------+
|  mesh             | The name of the mesh for autocorrelation.              |
+-------------------+--------------------------------------------------------+
|  array            | The data array name for autocorrelation.               |
+-------------------+--------------------------------------------------------+
|  association      | Either "cell" or "point" data.                         |
+-------------------+--------------------------------------------------------+
|  window           | The delay (t) for f(x).                                |
+-------------------+--------------------------------------------------------+
|  k-max            | The number of strongest autocorrelations to report.    |
+-------------------+--------------------------------------------------------+

Example XML
^^^^^^^^^^^

Autocorrelation example. This XML configures Autocorrelation analysis.

.. code-block:: XML

  <sensei>
    <analysis type="autocorrelation"
      mesh="mesh" array="data" association="cell"
      window="10" k-max="3" enabled="1" />
  </sensei>

Examples
--------
VM Demo reference.
