.. _miniapps:

********
Miniapps
********

.. _oscillator:

oscillator
----------

The oscillator mini-application computes a sum of damped, decaying, or periodic oscillators, convolved with (unnormalized) Gaussians, on a grid. It could be configured as a proxy for simulation of a chemical reaction on a two-dimensional substrate (see :ref:`reaction_rate_demo`).

+-----------------------------+----------------------------------------------------+
| option                      | description                                        |
+-----------------------------+----------------------------------------------------+
|  -b, --blocks INT           | Number of blocks to use [default: 1].              |
+-----------------------------+----------------------------------------------------+
|  -s, --shape POINT          | Number of cells in the domain [default: 64 64 64]. |
+-----------------------------+----------------------------------------------------+
|  -e, --bounds FLOAT         | Bounds of the Domain [default: {0,-1,0,-1,0,-1]}.  |
+-----------------------------+----------------------------------------------------+
|  -t, --dt FLOAT             | The time step [default: 0.01].                     |
+-----------------------------+----------------------------------------------------+
|  -f, --config STRING        | SENSEI analysis configuration xml (required).      |
+-----------------------------+----------------------------------------------------+
|  -g, --ghost-cells INT      | Number of ghost cells [default: 1].                |
+-----------------------------+----------------------------------------------------+
|  --t-end FLOAT              | Request synchronize after each time step.          |
+-----------------------------+----------------------------------------------------+
|  -j, --jobs INT             | Number of threads [default: 1].                    |
+-----------------------------+----------------------------------------------------+
|  -o, --output STRING        | Prefix for output [default: ""].                   |
+-----------------------------+----------------------------------------------------+
|  -p, --particles INT        | Number of particles [default: 0].                  |
+-----------------------------+----------------------------------------------------+
|  -v, --v-scale FLOAT        | Gradient to Velocity scale factor [default: 50].   |
+-----------------------------+----------------------------------------------------+
|  -r, --seed INT             | Random seed [default: 1].                          |
+-----------------------------+----------------------------------------------------+
|  --sync                     | The end time [default: 10].                        |
+-----------------------------+----------------------------------------------------+
|  -h, --help                 | Show help.                                         |
+-----------------------------+----------------------------------------------------+

The oscillators' locations and parameters are specified in an input file (see `input <https://gitlab.kitware.com/sensei/sensei/tree/master/miniapps/oscillators/inputs>`_ folder for examples). 

.. code-block::

   # type      center      r       omega0      zeta
   damped      32 32 32    10.     3.14        .3
   damped      16 32 16    10.     9.5         .1
   damped      48 32 48    5.      3.14        .1
   decaying    16 32 48    15      3.14
   periodic    48 32 16    15      3.14

Note that the `generate_input <https://gitlab.kitware.com/sensei/sensei/tree/master/miniapps/oscillators/inputs/generate_input>`_ script can generate a set of randomly initialized oscillators.

The simulation code is in `main.cpp <https://gitlab.kitware.com/sensei/sensei/tree/master/miniapps/oscillators/main.cpp>`_ while the computational kernel is in `Oscillator.cpp <https://gitlab.kitware.com/sensei/sensei/tree/master/miniapps/oscillators/Oscillator.cpp>`_.

To run:

.. code-block::

   mpiexec -n 4 oscillator -b 4 -t 0.25 -s 64,64,64 -g 1 -p 0 -f sample.xml sample.osc

There are a number of examples available in the SENSEI repositoory that leverage the oscillator mini-application.

newton
------

mandelbrot
----------
