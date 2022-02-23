*************
Adaptor API's
*************
SENSEI makes heavy use of the adaptor design pattern. This pattern is used to
abstract away the details of complex and diverse systems exposing them through
a single API. SENSEI has 2 types of adaptor. The DataAdaptor abstracts away the
details of accessing simulation data. This let's analysis back-ends access any
simulation's data through a single API. The AnalysisAdaptor abstarcts away the
details of the analysis back-ends. This let's the simulation invoke all of the
various analysis back-ends through a single API. When a simulation invokes an
analysis back-end it passes it a DataAdaptor that can be used to access
simulation data.

.. include:: data_adaptor_api.rst
.. include:: analysis_adaptor_api.rst
