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
#include "senseiPyDataAdaptor.h"
#include "LibsimImageProperties.h"
#include "DataRequirements.h"
#include "MeshMetadata.h"
#include "VTKUtils.h"
#include <sstream>
%}

%init %{
PyEval_InitThreads();
import_array();
%}


%include <mpi4py/mpi4py.i>
%include "vtk.i"
%include "senseiTypeMaps.i"
%include "senseiSTL.i"

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
%include "DataAdaptor.i"

/****************************************************************************
 * DataRequirements
 ***************************************************************************/
%ignore sensei::MeshRequirementsIterator::operator++;
%ignore sensei::MeshRequirementsIterator::operator bool() const;
%extend sensei::MeshRequirementsIterator
{
  // ------------------------------------------------------------------------
  int __bool__()
  {
    return static_cast<bool>(*self);
  }

  // ------------------------------------------------------------------------
  sensei::MeshRequirementsIterator &__iadd__(int n)
  {
    for (int i = 0; (i < n) && *self; ++i)
      self->operator++();
    return *self;
  }
}
%ignore sensei::ArrayRequirementsIterator::operator++;
%ignore sensei::ArrayRequirementsIterator::operator bool() const;
%extend sensei::ArrayRequirementsIterator
{
  // ------------------------------------------------------------------------
  int __bool__()
  {
    return static_cast<bool>(*self);
  }

  // ------------------------------------------------------------------------
  sensei::ArrayRequirementsIterator &__iadd__(int n)
  {
    for (int i = 0; i < n; ++i)
      self->operator++();
    return *self;
  }
}
%include "DataRequirements.h"

/****************************************************************************
 * AnalysisAdaptor
 ***************************************************************************/
VTK_DERIVED(AnalysisAdaptor)

/****************************************************************************
 * VTKDataAdaptor
 ***************************************************************************/
SENSEI_DATA_ADAPTOR(VTKDataAdaptor)

/****************************************************************************
 * ProgrammableDataAdaptor
 ***************************************************************************/
%extend sensei::ProgrammableDataAdaptor
{
  /* replace the callback setter's. we'll use objects
     that forward to/from a Python callable as there
     is no direct mapping from Python to C/C++ */
  void SetGetNumberOfMeshesCallback(PyObject *f)
  {
    self->SetGetNumberOfMeshesCallback(
      senseiPyDataAdaptor::PyGetNumberOfMeshesCallback(f));
  }

  void SetGetMeshMetadataCallback(PyObject *f)
  {
    self->SetGetMeshMetadataCallback(
      senseiPyDataAdaptor::PyGetMeshMetadataCallback(f));
  }

  void SetGetMeshCallback(PyObject *f)
  {
    self->SetGetMeshCallback(
      senseiPyDataAdaptor::PyGetMeshCallback(f));
  }

  void SetAddArrayCallback(PyObject *f)
  {
    self->SetAddArrayCallback(
      senseiPyDataAdaptor::PyAddArrayCallback(f));
  }

  void SetReleaseDataCallback(PyObject *f)
  {
    self->SetReleaseDataCallback(
      senseiPyDataAdaptor::PyReleaseDataCallback(f));
  }
}
%ignore sensei::ProgrammableDataAdaptor::SetGetNumberOfMeshesCallback;
%ignore sensei::ProgrammableDataAdaptor::SetGetMeshMetadataCallback;
%ignore sensei::ProgrammableDataAdaptor::SetGetMeshCallback;
%ignore sensei::ProgrammableDataAdaptor::SetAddArrayCallback;
%ignore sensei::ProgrammableDataAdaptor::SetReleaseDataCallback;
SENSEI_DATA_ADAPTOR(ProgrammableDataAdaptor)

/****************************************************************************
 * ConfigurableAnalysis
 ***************************************************************************/
VTK_DERIVED(ConfigurableAnalysis)

/****************************************************************************
 * Histogram
 ***************************************************************************/
%extend sensei::Histogram
{
  /* hide the C++ implementation, as Python doesn't pass by referemce
     and instead return a tuple (min, max, bins) or raise an exception
     if an error occurred */
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

    // pass the result back in a tuple
    PyObject *retTup = PyTuple_New(3);
    PyTuple_SetItem(retTup, 0, senseiPyObject::PyTT<double>::NewObject(hmin));
    PyTuple_SetItem(retTup, 1, senseiPyObject::PyTT<double>::NewObject(hmax));
    PyTuple_SetItem(retTup, 2, senseiPySequence::NewList<unsigned int>(hist));

    return retTup;
  }
}
%ignore sensei::Histogram::GetHistogram;
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
 * ADIOS1AnalysisAdaptor/DataAdaptor
 ***************************************************************************/
#ifdef ENABLE_ADIOS1
VTK_DERIVED(ADIOS1AnalysisAdaptor)
SENSEI_IN_TRANSIT_DATA_ADAPTOR(ADIOS1DataAdaptor)
#endif

#ifdef ENABLE_VTK_IO
/****************************************************************************
 * VTKPosthocIO
 ***************************************************************************/
VTK_DERIVED(VTKPosthocIO)

/****************************************************************************
 * VTKAmrWriter
 ***************************************************************************/
#ifdef ENABLE_VTK_MPI
VTK_DERIVED(VTKAmrWriter)
#endif
#endif
