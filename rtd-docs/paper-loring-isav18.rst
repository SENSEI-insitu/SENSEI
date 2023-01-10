.. _LoringIsav18:

************************************************
Python-based In Situ Analysis and Visualization
************************************************

Burlen Loring, Andrew Myers, David Camp, E. Wes Bethel

============
Full Text
============

Link to the full text `PDF <https://sc18.supercomputing.org/proceedings/workshops/workshop_files/ws_isav107s3-file1.pdf>`_.


========
Abstract
========

This work focuses on enabling the use of Python-based methods for the purpose
of performing in situ analysis and visualization.  This approach facilitates
access to and use of a rapidly growing collection of Python-based, third-party
libraries for analysis and visualization, as well as lowering the barrier to
entry for userwritten Python analysis codes. Beginning with a simulation code
that is instrumented to use the SENSEI in situ interface, we present how to
couple it with a Python-based data consumer, which may be run in situ, and in
parallel at the same concurrency as the simulation.  We present two examples
that demonstrate the new capability. One is an analysis of the reaction rate in
a proxy simulation of a chemical reaction on a 2D substrate, while the other is
a coupling of an AMR simulation to Yt, a parallel visualization and analysis
library written in Python. In the examples, both the simulation and Python in
situ method run in parallel on a large-scale HPC platform.

