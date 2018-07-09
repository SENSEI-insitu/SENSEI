%{
#include "senseiConfig.h"
#include "VTKUtils.h"
#include "MeshMetadata.h"
#include "DataAdaptor.h"
#include "Error.h"
#include <sstream>
%}

%define SENSEI_DATA_ADAPTOR(DA)
/* Modify the DataAdaptor API for Python. Python doesn't
   support pass by reference. Hence, we need to wrap the
   core API. Rather than return an error code we will ask
   that Python codes raise and exception if there is an
   error and return function results(or void for cases when
   there are none) instead of using pass by reference/output
   parameters */
%ignore sensei::DA::GetNumberOfMeshes;
%ignore sensei::DA::GetMeshMetadata;
%ignore sensei::DA::GetMesh;
%ignore sensei::DA::AddArray;
%ignore sensei::DA::ReleaseData;
/* memory management */
VTK_DERIVED(DA)
%enddef

/****************************************************************************
 * BinaryStream
 ***************************************************************************/
%include "BinaryStream.h"

/****************************************************************************
 * MeshMetadata
 ***************************************************************************/
%include "MeshMetadata.h"
%extend sensei::MeshMetadata
{
  PyObject *__str__()
    {
    std::ostringstream oss;
    self->ToStream(oss);
    return PyString_FromString(oss.str().c_str());
    }
}

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
      PyErr_Format(PyExc_RuntimeError,
        "Failed to get the number of meshes");
      PyErr_Print();
      }
    return nMeshes;
  }

  // ------------------------------------------------------------------------
  sensei::MeshMetadataPtr GetMeshMetadata(unsigned int id)
  {
    sensei::MeshMetadataPtr pmd;
    if (self->GetMeshMetadata(id, pmd))
      {
      PyErr_Format(PyExc_RuntimeError,
        "Failed to get metadata for mesh %d", id);
      PyErr_Print();
      }
    return pmd;
  }

  // ------------------------------------------------------------------------
  vtkDataObject *GetMesh(const std::string &meshName, bool structureOnly)
  {
    vtkDataObject *mesh = nullptr;
    if (self->GetMesh(meshName, structureOnly, mesh))
      {
      PyErr_Format(PyExc_RuntimeError,
        "Failed to get mesh \"%s\"", meshName.c_str());
      PyErr_Print();
      }
    return mesh;
  }

  // ------------------------------------------------------------------------
  void AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
  {
     if (self->AddArray(mesh, meshName, association, arrayName))
       {
       PyErr_Format(PyExc_RuntimeError,
         "Failed to add %s data array \"%s\" to mesh \"%s\"",
         sensei::VTKUtils::GetAttributesName(association),
         arrayName.c_str(), meshName.c_str());
       PyErr_Print();
       }
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
