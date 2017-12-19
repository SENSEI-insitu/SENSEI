%define SENSEI_PY_DOC
"SENSEI Python module
"
%enddef
%module (docstring=SENSEI_PY_DOC) senseiPython
%feature("autodoc", "3");

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  PyArray_API_SENSEI
#include <numpy/arrayobject.h>
#include "senseiConfig.h"
#include "LibsimImageProperties.h"
%}

%init %{
PyEval_InitThreads();
import_array();
%}

%include <std_string.i>
%include <std_vector.i>
%template(vector_string) std::vector<std::string>;
%include <mpi4py/mpi4py.i>
%include <vtk.i>
%include <senseiTypeMaps.i>

%mpi4py_typemap(Comm, MPI_Comm);

%import "senseiConfig.h"

/****************************************************************************
 * VTK objects used in our API
 ***************************************************************************/
VTK_SWIG_INTEROP(vtkObjectBase)
VTK_SWIG_INTEROP(vtkDataObject)
VTK_SWIG_INTEROP(vtkInformation)

/****************************************************************************
 * DataAdaptor
 ***************************************************************************/
/* SWIG generates bogus code for the following overloads, it looks
 like the fact that these static methods overload non-static
 methods is causing the problem */
%ignore sensei::DataAdaptor::SetDataTime(vtkInformation *,double);
%ignore sensei::DataAdaptor::SetDataTimeStep(vtkInformation *,int);
%ignore sensei::DataAdaptor::GetDataTime(vtkInformation *);
%ignore sensei::DataAdaptor::GetDataTimeStep(vtkInformation *);
VTK_DERIVED(DataAdaptor)

/****************************************************************************
 * AnalysisAdaptor
 ***************************************************************************/
VTK_DERIVED(AnalysisAdaptor)

/****************************************************************************
 * VTKDataAdaptor
 ***************************************************************************/
VTK_DERIVED(VTKDataAdaptor)

/****************************************************************************
 * ConfigurableAnalysis
 ***************************************************************************/
VTK_DERIVED(ConfigurableAnalysis)

/****************************************************************************
 * Histogram
 ***************************************************************************/
%extend sensei::Histogram
{
  PyObject *GetHistogram()
  {
    // invoke the C++ method
    double hmin = 0.0;
    double hmax = 0.0;
    std::vector<unsigned int> hist;
    if (self->GetHistogram(hmin, hmax, hist))
      {
      PyErr_Format(PyExc_RuntimeError,
        "Failed to get the histogram");
      return nullptr;
      }

    // pass the result back
    PyObject *retTup = PyTuple_New(3);
    PyTuple_SetItem(retTup, 0, senseiPyObject::PyTT<double>::NewObject(hmin));
    PyTuple_SetItem(retTup, 1, senseiPyObject::PyTT<double>::NewObject(hmax));
    PyTuple_SetItem(retTup, 2, senseiPySequence::NewList<unsigned int>(hist));

    return retTup;
  }
}
VTK_DERIVED(Histogram)

/****************************************************************************
 * Autocorrelation
 ***************************************************************************/
VTK_DERIVED(Autocorrelation)

/****************************************************************************
 * CatalystAnalysisAdaptor
 ***************************************************************************/
#ifdef ENABLE_CATALYST
VTK_DERIVED(CatalystAnalysisAdaptor)
#endif

/****************************************************************************
 * LibsimAnalysisAdaptor
 ***************************************************************************/
#ifdef ENABLE_LIBSIM
VTK_DERIVED(LibsimAnalysisAdaptor)
%include "LibsimImageProperties.h"
#endif

/****************************************************************************
 * ADIOSAnalysisAdaptor/DataAdaptor
 ***************************************************************************/
#ifdef ENABLE_ADIOS
VTK_DERIVED(ADIOSAnalysisAdaptor)
VTK_DERIVED(ADIOSDataAdaptor)
#endif

/****************************************************************************
 * VTKPosthocIO
 ***************************************************************************/
#ifdef ENABLE_VTK_XMLP
VTK_DERIVED(VTKPosthocIO)
#endif
