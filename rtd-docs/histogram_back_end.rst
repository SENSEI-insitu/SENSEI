Histogram back-end
==================
As a simple analysis routine, the Histogram back-end computes the histogram of the data. At any given time step, the processes perform two reductions to determine the minimum and maximum values on the mesh. Each processor divides the range into the prescribed number of bins and fills the histogram of its local data. The histograms are reduced to the root process. The only extra storage required is proportional to the number of bins in the histogram.

SENSEI XML
----------
The Histogram back-end is activated using the :code:`<analysis type="histogram">`. The supported attributes are:

+-------------------+--------------------------------------------------------+
| attribute         | description                                            |
+-------------------+--------------------------------------------------------+
|  mesh             | The name of the mesh for histogram.                    |
+-------------------+--------------------------------------------------------+
|  array            | The data array name for histogram.                     |
+-------------------+--------------------------------------------------------+
|  association      | Either "cell" or "point" data.                         |
+-------------------+--------------------------------------------------------+
|  file             | The filename template to write images.                 |
+-------------------+--------------------------------------------------------+
|  bins             | The number of histogram bins.                          |
+-------------------+--------------------------------------------------------+

Example XML
^^^^^^^^^^^

Histogram example. This XML configures Histogram analysis.

.. code-block:: XML

  <sensei>
    <analysis type="histogram"
      mesh="mesh" array="data" association="cell"
      file="hist.txt" bins="10"
      enabled="1" />
  </sensei>

Back-end specific configurarion
-------------------------------
No special back-end configuration is necessary.

Examples
--------
VM Demo reference.

