.. _AmesLdav19:

****************************************************
Low-overhead in situ visualization using halo replay
****************************************************

Jeff Ames, Silvio Rizzi, Joseph Insley, Saumil Patel, Benjamín Hernández, Erik W Draeger, Amanda Randles

============
Full Text
============

Link to the full text `PDF <https://www.osti.gov/servlets/purl/1572627>`_.

========
Abstract
========

In situ visualization and analysis is of increasing importance as the compute and I/O gap further widens with the advance to exascale capable computing. Yet, currently in situ methods impose resource constraints leading to the difficult task of balancing simulation code performance and the quality of analysis. Applications with tightly-coupled in situ visualization often achieve performance through spatial and temporal downsampling, a tradeoff which risks not capturing transient phenomena at sufficient fidelity. Determining a priori visualization parameters such as sampling rate is difficult without time and resource intensive experimentation. We present a method for reducing resource contention between in situ visualization and stencil codes on heterogeneous systems. This method permits full-resolution replay through recording halos and the communication-free reconstruction of interior values uncoupled from the main simulation. We apply this method in the computational fluid dynamics (CFD) code HARVEY on the Summit supercomputer. We demonstrate minimal overhead, in situ visualization relative to simulation alone, and compare the Halo Replay performance to tightly-coupled in situ approaches.

