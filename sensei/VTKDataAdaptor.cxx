#include "VTKDataAdaptor.h"
#include "VTKUtils.h"
#include "Error.h"
#include "MeshMetadata.h"

#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkOverlappingAMR.h>
#include <vtkDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkFieldData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkObjectFactory.h>
#include <vtkDataObject.h>
#include <vtkObjectBase.h>
#include <vtkObject.h>
#include <vtkDataArray.h>
#include <vtkAbstractArray.h>
#include <vtkSmartPointer.h>

#include <functional>
#include <map>
#include <utility>

using vtkDataObjectPtr = vtkSmartPointer<vtkDataObject>;

using MeshMapType = std::map<std::string, vtkDataObjectPtr>;

using vtkCompositeDataIteratorPtr =
  vtkSmartPointer<vtkCompositeDataIterator>;

namespace sensei
{

struct VTKDataAdaptor::InternalsType
{
  MeshMapType MeshMap;
};

//----------------------------------------------------------------------------
senseiNewMacro(VTKDataAdaptor);

//----------------------------------------------------------------------------
VTKDataAdaptor::VTKDataAdaptor()
{
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
VTKDataAdaptor::~VTKDataAdaptor()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
void VTKDataAdaptor::SetDataObject(const std::string &meshName,
  vtkDataObject* dobj)
{
  this->Internals->MeshMap[meshName] = dobj;
}

//----------------------------------------------------------------------------
int VTKDataAdaptor::GetDataObject(const std::string &meshName,
  vtkDataObject *&mesh)
{
 MeshMapType::iterator it = this->Internals->MeshMap.find(meshName);
  if (it == this->Internals->MeshMap.end())
    {
    SENSEI_ERROR("No mesh named \"" << meshName << "\"")
    return -1;
    }

  mesh = it->second.GetPointer();
  return 0;
}

//----------------------------------------------------------------------------
int VTKDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = this->Internals->MeshMap.size();
  return 0;
}

//----------------------------------------------------------------------------
int VTKDataAdaptor::GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata)
{
  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(this->GetCommunicator(), &rank);
  MPI_Comm_size(this->GetCommunicator(), &nRanks);

  if (id >= this->Internals->MeshMap.size())
    {
    SENSEI_ERROR("Index " << id << " out of bounds")
    return -1;
    }

  // get i'th mesh
  MeshMapType::iterator it = this->Internals->MeshMap.begin();

  for (unsigned int i = 0; i < id; ++i)
    ++it;

  std::string meshName = it->first;
  vtkDataObject *dobj = it->second;

  // fill in metadata
  metadata->MeshName = meshName;

  // multiblock and amr
  if (vtkCompositeDataSet *cd = dynamic_cast<vtkCompositeDataSet*>(dobj))
    {
    if (VTKUtils::GetMetadata(this->GetCommunicator(), cd, metadata))
      {
      SENSEI_ERROR("Failed to get metadata for composite mesh \""
        << meshName << "\"")
      return -1;
      }
    return 0;
    }

  // ParaView's legacy domain decomp
  if (vtkDataSet *ds = dynamic_cast<vtkDataSet*>(dobj))
    {
    if (VTKUtils::GetMetadata(this->GetCommunicator(), ds, metadata))
      {
      SENSEI_ERROR("Failed to get metadata for dataset mesh \""
        << meshName << "\"")
      return -1;
      }
    return 0;
    }

  SENSEI_ERROR("Unsupoorted data object type " << dobj->GetClassName())
  return -1;
}

//----------------------------------------------------------------------------
int VTKDataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
  vtkDataObject *&mesh)
{
  vtkDataObject *dobj = nullptr;
  if (this->GetDataObject(meshName, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return -1;
    }

  if (vtkCompositeDataSet *cd = dynamic_cast<vtkCompositeDataSet*>(dobj))
    {
    vtkCompositeDataSet *cdo = cd->NewInstance();
    cdo->CopyStructure(cd);

    vtkCompositeDataIterator *cdit = cd->NewIterator();
    while (!cdit->IsDoneWithTraversal())
      {
      vtkDataObject *dobj = cd->GetDataSet(cdit);
      vtkDataObject *dobjo = dobj->NewInstance();
      if (!structureOnly)
        {
        if (vtkDataSet *ds = dynamic_cast<vtkDataSet*>(dobj))
          {
          vtkDataSet *dso = static_cast<vtkDataSet*>(dobjo);
          dso->CopyStructure(ds);
          }
        }
      cdo->SetDataSet(cdit, dobjo);
      cdit->GoToNextItem();
      }

    mesh = cdo;
    return 0;
    }

  if (vtkDataSet *ds = dynamic_cast<vtkDataSet*>(dobj))
    {
    vtkDataSet *dsOut = ds->NewInstance();

    if (!structureOnly)
      dsOut->CopyStructure(ds);

    mesh = dsOut;
    return 0;
    }

  SENSEI_ERROR("Unsupoorted data object type " << dobj->GetClassName())
  return -1;
}

//----------------------------------------------------------------------------
int VTKDataAdaptor::AddArray(vtkDataObject* mesh, const std::string &meshName,
  int association, const std::string &arrayName)
{
  // define helper function to add the array to the mesh
  VTKUtils::BinaryDatasetFunction addArray =
    [&](vtkDataSet *ds, vtkDataSet *dsOut) -> int
    {
    vtkFieldData *dsa = VTKUtils::GetAttributes(ds, association);
    vtkFieldData *dsaOut = VTKUtils::GetAttributes(dsOut, association);

    vtkDataArray *da = dsa->GetArray(arrayName.c_str());
    if (!da)
      {
      SENSEI_ERROR("No " << VTKUtils::GetAttributesName(association)
        << " data array \"" << arrayName << "\" in mesh \""
        << meshName)
      return -1;
      }

    dsaOut->AddArray(da);
    return 0;
    };

  // get the cached copy of the mesh
  vtkDataObject *dobj = nullptr;
  if (this->GetDataObject(meshName, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return -1;
    }

  // apply the helper function
  if (VTKUtils::Apply(dobj, mesh, addArray))
    {
    SENSEI_ERROR("Failed to add array \"" << arrayName
      << "\" to mesh \"" << meshName  << "\"")
    return -1;
    }

  return 0;
}

// TODO
/*
//----------------------------------------------------------------------------
int VTKDataAdaptor::GetNumberOfArrays(const std::string &meshName,
  int association, unsigned int &numberOfArrays)
{
  numberOfArrays = 0;

  // define helper function to compute number of arrays
  VTKUtils::DatasetFunction getNumberOfArrays = [&](vtkDataSet *ds) -> int
    {
    vtkFieldData *dsa = VTKUtils::GetAttributes(ds, association);

    if (!dsa)
      return -1;

    numberOfArrays = dsa->GetNumberOfArrays();
    return 1;
    };

  // locate the cached mesh
  vtkDataObject *dobj = nullptr;
  if (this->GetDataObject(meshName, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return -1;
    }

  // apply the helper function
  if (VTKUtils::Apply(dobj, getNumberOfArrays))
    {
    SENSEI_ERROR("Failed to get the number of "
      << VTKUtils::GetAttributesName(association) << " data arrays for mesh \""
      << meshName  << "\"")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int VTKDataAdaptor::GetArrayName(const std::string &meshName, int association,
  unsigned int index, std::string &arrayName)
{
  // define helper function to get the available data arrays
  VTKUtils::DatasetFunction getArrayName = [&](vtkDataSet *ds) -> int
    {
    vtkFieldData *dsa =  VTKUtils::GetAttributes(ds, association);

    if (!dsa)
      return -1;

    unsigned int nArrays = dsa->GetNumberOfArrays();

    if (index > nArrays)
      {
      SENSEI_ERROR("Index " << index << " is out of bounds. "
        << VTKUtils::GetAttributesName(association) << " data on mesh \""
        << meshName << "\" has " << nArrays << " arrays.")
      return -1;
      }

    arrayName = dsa->GetArray(index)->GetName();

    return 1;
    };

  // locate the cached mesh
  vtkDataObject *dobj = nullptr;
  if (this->GetDataObject(meshName, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return -1;
    }

  // apply the helper function
  if (VTKUtils::Apply(dobj, getArrayName))
    {
    SENSEI_ERROR("Failed to get the number of arrays for mesh \""
      << meshName  << "\"")
    return -1;
    }

  return 0;
}
*/
//----------------------------------------------------------------------------
int VTKDataAdaptor::ReleaseData()
{
  this->Internals->MeshMap.clear();
  return 0;
}

}
