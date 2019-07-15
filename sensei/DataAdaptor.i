%{
#include "senseiConfig.h"
#include "MeshMetadata.h"
#include "DataAdaptor.h"
#include "InTransitDataAdaptor.h"
#include "ConfigurableInTransitDataAdaptor.h"
#include "Partitioner.h"
#include "BlockPartitioner.h"
#include "PlanarPartitioner.h"
#include "MappedPartitioner.h"
#include "PlanarSlicePartitioner.h"
#include "IsoSurfacePartitioner.h"
#include "ConfigurablePartitioner.h"
#include "VTKUtils.h"
#include "Error.h"
#include "senseiPyString.h"
#include <sstream>
#include <string>
#include <vector>
using namespace std;
%}

%include "std_shared_ptr.i"

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

%define SENSEI_IN_TRANSIT_DATA_ADAPTOR(DA)
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
%ignore sensei::DA::GetSenderMeshMetadata;
%ignore sensei::DA::GetReceiverMeshMetadata;
%ignore sensei::DA::SetReceiverMeshMetadata;
/* memory management */
VTK_DERIVED(DA)
%enddef

/****************************************************************************
 * BinaryStream
 ***************************************************************************/
%ignore sensei::BinaryStream::operator=;
%ignore sensei::BinaryStream::BinaryStream(BinaryStream &&);
%include "BinaryStream.h"

/****************************************************************************
 * MeshMetadata
 ***************************************************************************/
%shared_ptr(sensei::MeshMetadata)

// this lets you assign directly to the member variable
%naturalvar sensei::MeshMetadata::GlobalView;
%naturalvar sensei::MeshMetadata::MeshName;
%naturalvar sensei::MeshMetadata::MeshType;
%naturalvar sensei::MeshMetadata::BlockType;
%naturalvar sensei::MeshMetadata::NumBlocks;
%naturalvar sensei::MeshMetadata::NumBlocksLocal;
%naturalvar sensei::MeshMetadata::Extent;
%naturalvar sensei::MeshMetadata::Bounds;
%naturalvar sensei::MeshMetadata::CoordinateType;
%naturalvar sensei::MeshMetadata::NumPoints;
%naturalvar sensei::MeshMetadata::NumCells;
%naturalvar sensei::MeshMetadata::CellArraySize;
%naturalvar sensei::MeshMetadata::NumArrays;
%naturalvar sensei::MeshMetadata::NumGhostCells;
%naturalvar sensei::MeshMetadata::NumGhostNodes;
%naturalvar sensei::MeshMetadata::NumLevels;
%naturalvar sensei::MeshMetadata::StaticMesh;
%naturalvar sensei::MeshMetadata::ArrayName;
%naturalvar sensei::MeshMetadata::ArrayCentering;
%naturalvar sensei::MeshMetadata::ArrayComponents;
%naturalvar sensei::MeshMetadata::ArrayType;
%naturalvar sensei::MeshMetadata::ArrayRange;
%naturalvar sensei::MeshMetadata::BlockOwner;
%naturalvar sensei::MeshMetadata::BlockIds;
%naturalvar sensei::MeshMetadata::BlockNumPoints;
%naturalvar sensei::MeshMetadata::BlockNumCells;
%naturalvar sensei::MeshMetadata::BlockCellArraySize;
%naturalvar sensei::MeshMetadata::BlockExtents;
%naturalvar sensei::MeshMetadata::BlockBounds;
%naturalvar sensei::MeshMetadata::RefRatio;
%naturalvar sensei::MeshMetadata::BlocksPerLevel;
%naturalvar sensei::MeshMetadata::BlockLevel;
%naturalvar sensei::MeshMetadata::PeriodicBoundary;
%naturalvar sensei::MeshMetadata::BlockNeighbors;
%naturalvar sensei::MeshMetadata::BlockParents;
%naturalvar sensei::MeshMetadata::BlockChildren;
%naturalvar sensei::MeshMetadata::BlockArrayRange;
%naturalvar sensei::MeshMetadata::Flags;

%extend sensei::MeshMetadata
{
  PyObject *__str__()
    {
    std::ostringstream oss;
    self->ToStream(oss);
    return C_STRING_TO_PY_STRING(oss.str().c_str());
    }
}

%extend sensei::MeshMetadataFlags
{
  PyObject *__str__()
    {
    std::ostringstream oss;
    self->ToStream(oss);
    return C_STRING_TO_PY_STRING(oss.str().c_str());
    }
}
%include "MeshMetadata.h"

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
  sensei::MeshMetadataPtr GetMeshMetadata(unsigned int id,
    sensei::MeshMetadataFlags flags = sensei::MeshMetadataFlags())
  {
    sensei::MeshMetadataPtr pmd = sensei::MeshMetadata::New(flags);

    if (self->GetMeshMetadata(id, pmd) || !pmd)
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
%include "DataAdaptor.h"

/****************************************************************************
 * Partitioners
 ***************************************************************************/
%shared_ptr(sensei::Partitioner)
%shared_ptr(sensei::BlockPartitioner)
%shared_ptr(sensei::PlanarPartitioner)
%shared_ptr(sensei::MappedPartitioner)
%shared_ptr(sensei::PlanarSlicePartitioner)
%shared_ptr(sensei::IsoSurfacePartitioner)
%shared_ptr(sensei::ConfigurablePartitioner)

%define PARTITIONER_API(cname)
%extend sensei::##cname
{
  sensei::MeshMetadataPtr GetPartition(MPI_Comm comm, const sensei::MeshMetadataPtr &in)
  {
    sensei::MeshMetadataPtr out = sensei::MeshMetadata::New();
    if (self->GetPartition(comm, in, out))
    {
      PyErr_Format(PyExc_RuntimeError,
        "Failed to get partition");
    }
    return out;
  }
}
%ingnore sensei::##cname##::GetPartition;
%enddef

PARTITIONER_API(Partitioner)
PARTITIONER_API(BlockPartitioner)
PARTITIONER_API(PlanarPartitioner)
PARTITIONER_API(MappedPartitioner)
PARTITIONER_API(PlanarSlicePartitioner)
PARTITIONER_API(IsoSurfacePartitioner)
PARTITIONER_API(ConfigurablePartitioner)

%include "Partitioner.h"
%include "BlockPartitioner.h"
%include "PlanarPartitioner.h"
%include "MappedPartitioner.h"
%include "PlanarSlicePartitioner.h"
%include "IsoSurfacePartitioner.h"
%include "ConfigurablePartitioner.h"

/****************************************************************************
 * InTransitDataAdaptor
 ***************************************************************************/
%extend sensei::InTransitDataAdaptor
{
  // ------------------------------------------------------------------------
  sensei::MeshMetadataPtr GetSenderMeshMetadata(unsigned int id)
  {
    sensei::MeshMetadataPtr pmd = sensei::MeshMetadata::New();

    if (self->GetSenderMeshMetadata(id, pmd) || !pmd)
      {
      PyErr_Format(PyExc_RuntimeError,
        "Failed to get sender metadata for mesh %d", id);
      PyErr_Print();
      }

    return pmd;
  }

  // ------------------------------------------------------------------------
  sensei::MeshMetadataPtr GetReceiverMeshMetadata(unsigned int id)
  {
    sensei::MeshMetadataPtr pmd = sensei::MeshMetadata::New();

    if (self->GetReceiverMeshMetadata(id, pmd) || !pmd)
      {
      PyErr_Format(PyExc_RuntimeError,
        "Failed to get receiver metadata for mesh %d", id);
      PyErr_Print();
      }

    return pmd;
  }

  // ------------------------------------------------------------------------
  void SetReceiverMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &md)
  {
    if (self->SetReceiverMeshMetadata(id, md))
      {
      PyErr_Format(PyExc_RuntimeError,
        "Failed to set receiver metadata for mesh %d", id);
      PyErr_Print();
      }
  }
}

SENSEI_IN_TRANSIT_DATA_ADAPTOR(InTransitDataAdaptor)
%include "InTransitDataAdaptor.h"

%inline
%{
// **************************************************************************
sensei::InTransitDataAdaptor *AsInTransitDataAdaptor(sensei::DataAdaptor *da)
{
  return dynamic_cast<sensei::InTransitDataAdaptor*>(da);
}
%}

SENSEI_IN_TRANSIT_DATA_ADAPTOR(ConfigurableInTransitDataAdaptor)
%include "ConfigurableInTransitDataAdaptor.h"
