%define PYTHON_ANALYSIS_DOC
"SENSEI PythonAnalysis. Provides a minimal sensei wrapping to
code up analysis adaptors in a Python script."
%enddef
%module (docstring=PYTHON_ANALYSIS_DOC) PythonAnalysis
%feature("autodoc", "3");

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  PyArray_API_SENSEI_PYTHON_ANALYSIS
#include <numpy/arrayobject.h>
#include "senseiConfig.h"
#include "VTKUtils.h"
#include "DataRequirements.h"
#include "DataAdaptor.h"
#include "Error.h"
#include <mpi.h>

/* PythonAnalysis sources included directly as they
   need direct access to the SWIG run time and type table */
#include "PythonAnalysis.h"
#include "PythonAnalysis.cxx"
%}

%init %{
PyEval_InitThreads();
import_array();
%}

%include <std_string.i>
%include <std_vector.i>
%include <std_map.i>
%template(vector_string) std::vector<std::string>;
%template(map_string_bool) std::map<std::string, bool>;
%template(map_int_vector_string) std::map<int, std::vector<std::string>>;
%include <mpi4py/mpi4py.i>
%include "vtk.i"
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
