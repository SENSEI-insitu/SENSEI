System Overview & Architecture
==============================
SENSEI is a light weight framework for in situ data analysis. SENSEI's data
model and API provide uniform access to and run time selection of a diverse set
of visualization and analysis back ends including VisIt Libsim, ParaView
Catalyst, VTK-m, Ascent, ADIOS, Yt, and Python.

In situ architecture
~~~~~~~~~~~~~~~~~~~~

.. _in_situ_arch:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/sensei_in_situ_arch.png

   SENSEI's in situ architecture enables use of a diverse of back ends which
   can be selected at run time via an XML configuration file

The three major architectural components in SENSEI are *data adaptors* which
present simulation data in SENSEI's data model, *analysis adaptors* which
present the back end data consumers to the simulation, and *bridge code* from
which the simulation manages adaptors and periodically pushes data through the
system. SENSEI comes equipped with a number of analysis adaptors enabling use
of popular analysis and visualization libraries such as VisIt Libsim, ParaView
Catalyst, Python, and ADIOS to name a few. AMReX contains SENSEI data adaptors
and bridge code making it easy to use in AMReX based simulation codes.

SENSEI provides a *configurable analysis adaptor* which uses an XML file to
select and configure one or more back ends at run time. Run time selection of
the back end via XML means one user can access Catalyst, another Libsim, yet
another Python with no changes to the code.  This is depicted in figure
:numref:`in_situ_arch`. On the left side of the figure AMReX produces data, the
bridge code pushes the data through the configurable analysis adaptor to the
back end that was selected at run time.

In transit architecture
~~~~~~~~~~~~~~~~~~~~~~~
.. _in_transit_arch:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/sensei_in_transit_arch.png

   SENSEI's in transit architecture enables decoupling of analysis and simulation.

SENSEI's in transit architecture enables decoupling of analysis and simulation.
In this configuration the simulation runs in one job and the analysis runs in a
second job, optionally on a separate set of compute resources, optionally at a
smaller or larger level of concurrency. The configuration is made possible by a
variety of *transports* who's job is to move and repartitions data. This is
depicted in figure :numref:`in_transit_arch`.

In the in transit configuration, the simulation running in one job uses
SENSEI's *configurable analysis adaptor* to select and configure the write side
of the transport. When the simulation pushes data through the SENSEI API for
analysis the transport deals with presenting and moving data needed for
analysis across the network.  In asynchronous mode the simulation proceeds while
the data is processed.

A second job, running the SENSEI *in transit end-point*, uses the *configurable
analysis adaptor* to select and configure one of the back-ends. A transport
specific data adaptor presents the available data to the analysis. The analysis
can select and request data to be moved across the network for processing.

SENSEI's design enables this configuration to occur with no changes to either
the simulation or analysis back-end. The process is entirely seamless from the
simulations point of view and can be so if desired on the analysis side as
well.  SENSEI supports in transit aware analyses, and provides API's for
yielding control data repartitioning to the analysis.


