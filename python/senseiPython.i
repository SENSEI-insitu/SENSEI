%define SENSEI_PY_DOC
"SENSEI Python module
"
%enddef
%module (docstring=SENSEI_PY_DOC) senseiPython
%feature("autodoc", "3");
#pragma SWIG nowarn=302

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
#include "SVTKUtils.h"
#include "Profiler.h"
#include <sstream>

#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
%}

/* SWIG does not understand attributes */
#define __attribute__(x)

%init %{
#if PY_VERSION_HEX < 0x03070000
PyEval_InitThreads();
#endif
import_array();
%}

/* import the SVTK module */
%{
#include "svtk.h"
%}
%import "svtk.i"

%include <mpi4py/mpi4py.i>
%mpi4py_typemap(Comm, MPI_Comm);

%include "senseiTypeMaps.i"
%include "senseiSTL.i"

/* wraps the passed class */
%define SENSEI_WRAP_ANALYSIS_ADAPTOR(CLASS)
%{
#include <CLASS##.h>
%}
SVTK_OBJECT_FACTORY(sensei::##CLASS)
SVTK_OBJECT_IGNORE_CPP_API(sensei::##CLASS)
%ignore sensei::##CLASS::Execute(DataAdaptor*, DataAdaptor**);
%extend sensei::##CLASS
{
    SENSEI_CONSTRUCTOR(CLASS)
    SVTK_OBJECT_STR(sensei::##CLASS)

    PyObject *Execute(sensei::DataAdaptor *daIn)
    {
        sensei::DataAdaptor *daOut = nullptr;

        // invoke the analysis
        int execOk = self->Execute(daIn, &daOut);

        // package the returned optional data adaptor.
        PyObject *pyDaOut = nullptr;
        if (daOut)
        {
            pyDaOut = SWIG_NewPointerObj((void*)daOut,
                SWIGTYPE_p_sensei__DataAdaptor, SWIG_POINTER_OWN);
        }
        else
        {
            pyDaOut = Py_None;
            Py_INCREF(pyDaOut);
        }

        PyObject *res = Py_BuildValue("(IN)", execOk, pyDaOut);
        return res;
    }
};
%include <CLASS##.h>
%enddef

/****************************************************************************
 * comnpile time settings
 ***************************************************************************/
%import "senseiConfig.h"

/****************************************************************************
 * timer
 ***************************************************************************/
%include "Profiler.h"

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
SENSEI_WRAP_ADAPTOR(AnalysisAdaptor)

/****************************************************************************
 * SVTKDataAdaptor
 ***************************************************************************/
SENSEI_WRAP_DATA_ADAPTOR(SVTKDataAdaptor)

/****************************************************************************
 * ProgrammableDataAdaptor
 ***************************************************************************/
%{
#include <ProgrammableDataAdaptor.h>
%}
SVTK_OBJECT_FACTORY(sensei::ProgrammableDataAdaptor)
%extend sensei::ProgrammableDataAdaptor
{
  SENSEI_CONSTRUCTOR(ProgrammableDataAdaptor)
  SVTK_OBJECT_STR(sensei::ProgrammableDataAdaptor)
  SENSEI_DATA_ADAPTOR_PYTHON_API(sensei::ProgrammableDataAdaptor)
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
};
SVTK_OBJECT_IGNORE_CPP_API(sensei::ProgrammableDataAdaptor)
SENSEI_DATA_ADAPTOR_IGNORE_CPP_API(sensei::ProgrammableDataAdaptor)
%ignore sensei::ProgrammableDataAdaptor::SetGetNumberOfMeshesCallback;
%ignore sensei::ProgrammableDataAdaptor::SetGetMeshMetadataCallback;
%ignore sensei::ProgrammableDataAdaptor::SetGetMeshCallback;
%ignore sensei::ProgrammableDataAdaptor::SetAddArrayCallback;
%ignore sensei::ProgrammableDataAdaptor::SetReleaseDataCallback;
%include <ProgrammableDataAdaptor.h>

/****************************************************************************
 * ConfigurableAnalysis
 ***************************************************************************/
SENSEI_WRAP_ANALYSIS_ADAPTOR(ConfigurableAnalysis)

/****************************************************************************
 * Histogram
 ***************************************************************************/
%{
#include <Histogram.h>
%}
SVTK_OBJECT_IGNORE_CPP_API(sensei::Histogram)
SVTK_OBJECT_FACTORY(sensei::Histogram)
%ignore sensei::Histogram::Data;
%ignore sensei::Histogram::GetHistogram;
%extend sensei::Histogram
{
    SENSEI_CONSTRUCTOR(Histogram)
    SVTK_OBJECT_STR(sensei::Histogram)
  /* hide the C++ implementation, as Python doesn't pass by referemce
     and instead return a tuple (min, max, bins) or raise an exception
     if an error occurred */
  PyObject *GetHistogram()
  {
    // invoke the C++ method
    sensei::Histogram::Data result;
    if (self->GetHistogram(result))
      {
      PyErr_Format(PyExc_RuntimeError,
        "Failed to get the histogram");
      return nullptr;
      }

    // pass the result back in a tuple
    PyObject *retTup = PyTuple_New(3);
    PyTuple_SetItem(retTup, 0, senseiPyObject::PyTT<double>::NewObject(result.BinMin));
    PyTuple_SetItem(retTup, 1, senseiPyObject::PyTT<double>::NewObject(result.BinMax));
    PyTuple_SetItem(retTup, 2, senseiPySequence::NewList<unsigned int>(result.Histogram));

    return retTup;
  }
};
%include <Histogram.h>

/****************************************************************************
 * Autocorrelation
 ***************************************************************************/
SENSEI_WRAP_ANALYSIS_ADAPTOR(Autocorrelation)

/****************************************************************************
 * CatalystAnalysisAdaptor
 ***************************************************************************/
#ifdef ENABLE_CATALYST
SENSEI_WRAP_ANALYSIS_ADAPTOR(CatalystAnalysisAdaptor)
#endif

/****************************************************************************
 * LibsimAnalysisAdaptor
 ***************************************************************************/
#ifdef ENABLE_LIBSIM
SENSEI_WRAP_ANALYSIS_ADAPTOR(LibsimAnalysisAdaptor)
%include "LibsimImageProperties.h"
#endif

/****************************************************************************
 * ADIOS1AnalysisAdaptor/DataAdaptor
 ***************************************************************************/
#ifdef ENABLE_ADIOS1
SENSEI_WRAP_ANALYSIS_ADAPTOR(ADIOS1AnalysisAdaptor)
SENSEI_WRAP_IN_TRANSIT_DATA_ADAPTOR(ADIOS1DataAdaptor)
#endif

/****************************************************************************
 * ADIOS2AnalysisAdaptor/DataAdaptor
 ***************************************************************************/
#ifdef ENABLE_ADIOS2
SENSEI_WRAP_ANALYSIS_ADAPTOR(ADIOS2AnalysisAdaptor)
SENSEI_WRAP_IN_TRANSIT_DATA_ADAPTOR(ADIOS2DataAdaptor)
#endif


#ifdef ENABLE_VTK_IO
/****************************************************************************
 * VTKPosthocIO
 ***************************************************************************/
SENSEI_WRAP_ANALYSIS_ADAPTOR(VTKPosthocIO)

/****************************************************************************
 * VTKAmrWriter
 ***************************************************************************/
#ifdef ENABLE_VTK_MPI
SENSEI_WRAP_ANALYSIS_ADAPTOR(VTKAmrWriter)
#endif

/****************************************************************************
 * SliceExtract
 ***************************************************************************/
#ifdef ENABLE_VTK_FILTERS
SENSEI_WRAP_ANALYSIS_ADAPTOR(SliceExtract)
#endif
#endif
