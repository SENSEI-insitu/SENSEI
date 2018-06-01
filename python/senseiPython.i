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
#include "VTKUtils.h"
%}

%init %{
PyEval_InitThreads();
import_array();
%}

%include <std_string.i>
%include <std_vector.i>
%template(vector_string) std::vector<std::string>;
%include <mpi4py/mpi4py.i>
%include "vtk.i"
%include "senseiTypeMaps.i"
%include "senseiDataAdaptor.i"

%mpi4py_typemap(Comm, MPI_Comm);

%import "senseiConfig.h"

/****************************************************************************
 * VTK objects used in our API
 ***************************************************************************/
VTK_SWIG_INTEROP(vtkObjectBase)
VTK_SWIG_INTEROP(vtkDataObject)
VTK_SWIG_INTEROP(vtkInformation)

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
 * DataAdaptor
 ***************************************************************************/
%extend sensei::DataAdaptor
{
  /* Modify the DataAdaptor API for Python. Python doesn't
     support pass by reference. Hence, we need to wrap the
     core API. Rather than return an error code we will ask
     that Python codes raise and exception if there is an
     error and return function results(or void for cases when
     there are none) instead of using pass by reference/output
     parameters */
  // ------------------------------------------------------------------------
  unsigned int GetNumberOfMeshes()
  {
    unsigned int nMeshes = 0;
    if (self->GetNumberOfMeshes(nMeshes))
      {
      SENSEI_ERROR("Failed to get the number of meshes")
      }
    return nMeshes;
  }

  // ------------------------------------------------------------------------
  std::string GetMeshName(unsigned int id)
  {
    std::string meshName;
    if (self->GetMeshName(id, meshName))
      {
      SENSEI_ERROR("Failed to get the mesh name for " << id)
      }
    return meshName;
  }

  // ------------------------------------------------------------------------
  vtkDataObject *GetMesh(const std::string &meshName, bool structureOnly)
  {
    vtkDataObject *mesh = nullptr;
    if (self->GetMesh(meshName, structureOnly, mesh))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      }
    return mesh;
  }

  // ------------------------------------------------------------------------
  void AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
  {
     if (self->AddArray(mesh, meshName, association, arrayName))
       {
       SENSEI_ERROR("Failed to add "
        << sensei::VTKUtils::GetAttributesName(association)
        << " data array \"" << arrayName << "\" to mesh \""
        << meshName << "\"")
       }
  }

  // ------------------------------------------------------------------------
  unsigned int GetNumberOfArrays(const std::string &meshName, int association)
  {
    unsigned int nArrays = 0;
    if (self->GetNumberOfArrays(meshName, association, nArrays))
      {
      SENSEI_ERROR("Failed to get the number of "
        << sensei::VTKUtils::GetAttributesName(association)
        << " arrays on mesh \"" << meshName << "\"")
      }
    return nArrays;
  }

  // ------------------------------------------------------------------------
  std::string GetArrayName(const std::string &meshName, int association,
    unsigned int index)
  {
    std::string arrayName;
    if (self->GetArrayName(meshName, association, index, arrayName))
      {
      SENSEI_ERROR("Failed to get "
        << sensei::VTKUtils::GetAttributesName(association)
        << " data array name " << index << " on mesh \""
        << meshName << "\"")
      }
    return arrayName;
  }

  // ------------------------------------------------------------------------
  void ReleaseData()
  {
    if (self->ReleaseData())
      {
      SENSEI_ERROR("Failed to release data")
      }
  }
}
SENSEI_DATA_ADAPTOR(DataAdaptor)

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

  void SetGetMeshNameCallback(PyObject *f)
  {
    self->SetGetMeshNameCallback(
      senseiPyDataAdaptor::PyGetMeshNameCallback(f));
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

  void SetGetNumberOfArraysCallback(PyObject *f)
  {
    self->SetGetNumberOfArraysCallback(
      senseiPyDataAdaptor::PyGetNumberOfArraysCallback(f));
  }

  void SetGetArrayNameCallback(PyObject *f)
  {
    self->SetGetArrayNameCallback(
      senseiPyDataAdaptor::PyGetArrayNameCallback(f));
  }

  void SetReleaseDataCallback(PyObject *f)
  {
    self->SetReleaseDataCallback(
      senseiPyDataAdaptor::PyReleaseDataCallback(f));
  }
}
%ignore sensei::ProgrammableDataAdaptor::SetGetNumberOfMeshesCallback;
%ignore sensei::ProgrammableDataAdaptor::SetGetMeshNameCallback;
%ignore sensei::ProgrammableDataAdaptor::SetGetMeshCallback;
%ignore sensei::ProgrammableDataAdaptor::SetAddArrayCallback;
%ignore sensei::ProgrammableDataAdaptor::SetGetNumberOfArraysCallback;
%ignore sensei::ProgrammableDataAdaptor::SetGetArrayNameCallback;
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
 * ADIOSAnalysisAdaptor/DataAdaptor
 ***************************************************************************/
#ifdef ENABLE_ADIOS
VTK_DERIVED(ADIOSAnalysisAdaptor)
SENSEI_DATA_ADAPTOR(ADIOSDataAdaptor)
#endif

/****************************************************************************
 * VTKPosthocIO
 ***************************************************************************/
#ifdef ENABLE_VTK_IO
VTK_DERIVED(VTKPosthocIO)
#endif

/****************************************************************************
 * VTKAmrWriter
 ***************************************************************************/
#ifdef ENABLE_VTK_IO
VTK_DERIVED(VTKAmrWriter)
#endif
