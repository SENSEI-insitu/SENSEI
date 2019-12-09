######################
The SENSEI User Guide
######################

.. write_once:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/write_once.png
   :width: 100 %
   :align: center

   SENSEI gives simulations access to a wide array of scalable data analytics
   and visualization solutions through a single API.

************
Introduction
************
Write once run anywhere. SENSEI seamlessly & efficiently enables in situ data
processing with a diverse set of languages, tools & libraries through a simple
API, data model, and run-time configuration mechanism.

A SENSEI instrumented simulation can switch between different analysis
back-ends such as ADIOS, Libsim, Ascent, Catalyst etc, at run time without
modifying code. This flexibility provides options for users, who may have
varying needs and preferences about which tools to use to accomplish a task.

Deploying a back end data consumer in SENSEI makes it usable by any SENSEI
instrumented simulation. SENSEI is focused on being light weight, having low
memory and execution overhead, having a simple API, and minimizing dependencies
for each back-end that we support.

Scientists often want to add their own diagnostics in addition to or in place
of other back-ends. We strive to make this easy to do in Python and C++. We
don't aim to replace or compete against an individual vis/analysis tool, we aim
to make the pie bigger by making their tool and capabilities available to a
broader set of users.

With SENSEI the sum is greater than the parts. For instance for both
simulations and back-end data consumers, which have not been designed for in
transit use, can be run in transit with out modification. Configuring for in
transit run makes use of the same simple configuration mechanism that is used
to select back-end data consumer.

Write once run everywhere. SENSEI provides access to a diverse set of in situ
analysis back-ends and transport layers through a simple API and data model.
Simulations instrumented with the SENSEI API can process data using any of
these back-ends interchangeably. The back-ends are selected and configured at
run-time via an XML configuration file. This document is targeted at scientists
and developers wishing to run simulations instrumented with SENSEI, instrument
a new simulation, or develop new analysis back-ends.

.. toctree::
   :maxdepth: 3

   installation
   system_components
   mini-apps
   examples
   developer_guide
   glossary
