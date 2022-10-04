.. _Ayachit_ISAV16:

***********************************************
The SENSEI Generic In Situ Interface
***********************************************

Utkarsh Ayachit, Brad Whitlock, Matthew Wolf, Burlen Loring,
Berk Geveci, David Lonie, and E. Wes Bethel

============
Full Text
============

Link to the full text PDF.

Abstract
========
The SENSEI generic in situ interface is an API that promotes code
portability and reusability. From the simulation view, a developer
can instrument their code with the SENSEI API and then make make use
of any number of in situ infrastructures. From the method view, a
developer can write an in situ method using the SENSEI API, then
expect it to run in any number of in situ infrastructures, or be
invoked directly from a simulation code, with little or no
modification. This paper presents the design principles underlying
the SENSEI generic interface, along with some simplified coding
examples.

Interface Design
=================

Data Model: A key part of the design of the common interface was a
decision on a common data description model. Our choice was to extend
a variant on the VTK data model. There were several reasons for this
choice. The VTK data model is already widely used in applications like
VisIt and ParaView, which are important codes for the post-hoc
development of the sorts of analysis and visualization that are
required in situ. The VTK data model has native support for a plethora
of common scientific data structures, including regular grids,
curvilinear grids, unstructured grids, graphs, tables, and AMR. There
is also already a dedicated community looking to carry forward VTK to
exascale computing, so our efforts can cross-leverage those.

Despite its many strengths, there were some key additions we wanted
for the SENSEI model. To minimize effort and memory overhead when
mapping memory layouts for data arrays from applications to VTK, we
extended the VTK data model to support arbitrary layouts for
multicomponent arrays through a new API called generic arrays. Through
this work, this capability has been back-ported to the core VTK data
model. VTK now natively supports the commonly encountered
structure-of-arrays and array-of-structures layouts utilizing
zero-copy memory techniques.

Interface: The SENSEI interface comprises of three components: data
adaptor that helps map sim data to VTK data model, analysis adaptor
that maps VTK data model for analysis methods, and in situ bridge that
links together the data adaptor and the analysis adaptor, and provides
the API that the simulation uses to trigger the in situ analysis.

The data adaptor defines an API to access the simulation data as VTK
data objects. The analysis adaptor uses this API to access the data to
pass to the analysis method. To instrument a simulation code for
SENSEI, one has to provide a concrete implementation for this data
adaptor API. The API treats connectivity and attribute array
information separately, providing specific API calls for requesting
each. This helps to avoid compute cycles needed to map the
connectivity and/or data attributes to the VTK data model unless
needed by active analysis methods.

The analysis adaptor’s role is to take the data adaptor and pass the
data to the analysis method, doing any transformations as necessary.
For a specific analysis method, the analysis adaptor is provided the
data adaptor in its Execute method. Using the sensei::DataAdaptor API,
the analysis adaptor can obtain the mesh (geometry, and connectivity)
and attribute or field arrays necessary for the analysis method.

