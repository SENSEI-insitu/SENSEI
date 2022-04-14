AnalysisAdaptor API
===================

Extending SENSEI for customized Analysis capabilities requires implementing a
`sensei::AnalysisAdaptor <https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_analysis_adaptor.html>`_ .

At a minimum one must implement the `sensei::AnalysisAdaptor::Execute` method.
In your implementation you will make use of the passed `sensei::DataAdaptor <https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_data_adaptor.html>`_
instance to fetch the necessary simulation data.

The following template can be used to add a new C++ based analysis capability.

.. code-block:: c++

   class MyAnalysis : public sensei::AnalysisAdaptor
   {
   public:

       virtual bool Execute(DataAdaptor* dataIn, DataAdaptor** dataOut)
       {
           // YOUR ANALYSIS CODE HERE. USE THE PASSED DATA ADAPTOR TO ACCESS
           // SIMULATION DATA

           if (dataOut)
           {
               // IF YOUR CODE CAN RETURN DATA, CREATE AND RETURN A DATA
               // ADAPTOR HERE THAT CAN BE USED TO ACCESS IT
               *dataOut = nullptr;
           }

           return true;
       }

   };

Python API
----------
The `sensei::PythonAnalysis <https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_python_analysis.html>`_
adaptor enables the use of a Python scripts as an analysis
back end in C,C++, and Fortran based simulation codes.
It accomplishes this by embedding a Python interpreter and includes a
minimal set of the sensei python bindings. To author a new python analysis one
must provide a python script that implements three functions in a user provided Python script that is loaded at run time.
The three functions are: `Inititalize`, `Execute` and `Finalize`. These functions implement the `sensei::AnalysisAdaptor`
API.

The `Execute` function is required while `Initialize` and `Finalize` functions
are optional. The `Execute` function is passed a `sensei::DataAdaptor` instance
from which one has access to simulation data structures. If an error occurs
during processing one should raise an exception. If the analysis required MPI
communication, one must make use of the adaptor’s MPI communicator which is
stored in the global variable comm. Additionally one can provide a secondary
script that is executed prior to the API functions. This script can set global
variables that control runtime behavior.

End users will make use of the `sensei::ConfigurableAnalysis` and point to the
python script containing the three functions described above. The script can be
loaded in one of two ways: via python’s import machinery or via a customized
mechanism that reads the file on MPI rank 0 and broadcasts it to the other
ranks. The latter is the recommended approach.


.. code-block:: python

    def Initialize():
        """ Initialization code here """
        return

    def Execute(dataAdaptor):
        """ Use sensei::DataAdaptor API to process data here """
        return

    def Finalize():
        """ Finalization code here """
        return


