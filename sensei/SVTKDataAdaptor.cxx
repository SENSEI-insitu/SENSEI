#include "SVTKDataAdaptor.h"
#include "SVTKUtils.h"
#include "Error.h"
#include "MeshMetadata.h"

#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkOverlappingAMR.h>
#include <svtkDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkFieldData.h>
#include <svtkPointData.h>
#include <svtkCellData.h>
#include <svtkObjectFactory.h>
#include <svtkDataObject.h>
#include <svtkObjectBase.h>
#include <svtkObject.h>
#include <svtkDataArray.h>
#include <svtkAbstractArray.h>
#include <svtkSmartPointer.h>

#include <functional>
#include <map>
#include <utility>

using svtkDataObjectPtr = svtkSmartPointer<svtkDataObject>;

using MeshMapType = std::map<std::string, svtkDataObjectPtr>;

using svtkCompositeDataIteratorPtr =
  svtkSmartPointer<svtkCompositeDataIterator>;

namespace sensei
{

struct SVTKDataAdaptor::InternalsType
{
  MeshMapType MeshMap;
};

//----------------------------------------------------------------------------
senseiNewMacro(SVTKDataAdaptor);

//----------------------------------------------------------------------------
SVTKDataAdaptor::SVTKDataAdaptor()
{
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
SVTKDataAdaptor::~SVTKDataAdaptor()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
void SVTKDataAdaptor::SetDataObject(const std::string &meshName,
  svtkDataObject* dobj)
{
  this->Internals->MeshMap[meshName] =
    SVTKUtils::AsCompositeData(this->GetCommunicator(), dobj, false);
}

//----------------------------------------------------------------------------
int SVTKDataAdaptor::GetDataObject(const std::string &meshName,
  svtkDataObject *&mesh)
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
int SVTKDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = this->Internals->MeshMap.size();
  return 0;
}

//----------------------------------------------------------------------------
int SVTKDataAdaptor::GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata)
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
  svtkDataObject *dobj = it->second;

  // fill in metadata
  metadata->MeshName = meshName;

  // multiblock and amr
  if (svtkCompositeDataSet *cd = dynamic_cast<svtkCompositeDataSet*>(dobj))
    {
    if (SVTKUtils::GetMetadata(this->GetCommunicator(), cd, metadata))
      {
      SENSEI_ERROR("Failed to get metadata for composite mesh \""
        << meshName << "\"")
      return -1;
      }
    return 0;
    }

  // ParaView's legacy domain decomp
  if (svtkDataSet *ds = dynamic_cast<svtkDataSet*>(dobj))
    {
    if (SVTKUtils::GetMetadata(this->GetCommunicator(), ds, metadata))
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
int SVTKDataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
  svtkDataObject *&mesh)
{
  svtkDataObject *dobj = nullptr;
  if (this->GetDataObject(meshName, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return -1;
    }

  if (svtkCompositeDataSet *cd = dynamic_cast<svtkCompositeDataSet*>(dobj))
    {
    svtkCompositeDataSet *cdo = cd->NewInstance();
    cdo->CopyStructure(cd);

    svtkCompositeDataIterator *cdit = cd->NewIterator();
    while (!cdit->IsDoneWithTraversal())
      {
      svtkDataObject *dobj = cd->GetDataSet(cdit);
      svtkDataObject *dobjo = dobj->NewInstance();
      if (!structureOnly)
        {
        if (svtkDataSet *ds = dynamic_cast<svtkDataSet*>(dobj))
          {
          svtkDataSet *dso = static_cast<svtkDataSet*>(dobjo);
          dso->CopyStructure(ds);
          }
        }
      cdo->SetDataSet(cdit, dobjo);
      dobjo->Delete();

      cdit->GoToNextItem();
      }

    cdit->Delete();

    mesh = cdo;
    return 0;
    }

  if (svtkDataSet *ds = dynamic_cast<svtkDataSet*>(dobj))
    {
    svtkDataSet *dsOut = ds->NewInstance();

    if (!structureOnly)
      dsOut->CopyStructure(ds);

    mesh = dsOut;
    // caller takes ownership

    return 0;
    }

  SENSEI_ERROR("Unsupoorted data object type " << dobj->GetClassName())
  return -1;
}

//----------------------------------------------------------------------------
int SVTKDataAdaptor::AddArray(svtkDataObject* mesh, const std::string &meshName,
  int association, const std::string &arrayName)
{
  // define helper function to add the array to the mesh
  SVTKUtils::BinaryDatasetFunction addArray =
    [&](svtkDataSet *ds, svtkDataSet *dsOut) -> int
    {
    svtkFieldData *dsa = SVTKUtils::GetAttributes(ds, association);
    svtkFieldData *dsaOut = SVTKUtils::GetAttributes(dsOut, association);

    svtkDataArray *da = dsa->GetArray(arrayName.c_str());
    if (!da)
      {
      SENSEI_ERROR("No " << SVTKUtils::GetAttributesName(association)
        << " data array \"" << arrayName << "\" in mesh \""
        << meshName)
      return -1;
      }

    dsaOut->AddArray(da);
    return 0;
    };

  // get the cached copy of the mesh
  svtkDataObject *dobj = nullptr;
  if (this->GetDataObject(meshName, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return -1;
    }

  // apply the helper function
  if (SVTKUtils::Apply(dobj, mesh, addArray))
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
int SVTKDataAdaptor::GetNumberOfArrays(const std::string &meshName,
  int association, unsigned int &numberOfArrays)
{
  numberOfArrays = 0;

  // define helper function to compute number of arrays
  SVTKUtils::DatasetFunction getNumberOfArrays = [&](svtkDataSet *ds) -> int
    {
    svtkFieldData *dsa = SVTKUtils::GetAttributes(ds, association);

    if (!dsa)
      return -1;

    numberOfArrays = dsa->GetNumberOfArrays();
    return 1;
    };

  // locate the cached mesh
  svtkDataObject *dobj = nullptr;
  if (this->GetDataObject(meshName, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return -1;
    }

  // apply the helper function
  if (SVTKUtils::Apply(dobj, getNumberOfArrays))
    {
    SENSEI_ERROR("Failed to get the number of "
      << SVTKUtils::GetAttributesName(association) << " data arrays for mesh \""
      << meshName  << "\"")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int SVTKDataAdaptor::GetArrayName(const std::string &meshName, int association,
  unsigned int index, std::string &arrayName)
{
  // define helper function to get the available data arrays
  SVTKUtils::DatasetFunction getArrayName = [&](svtkDataSet *ds) -> int
    {
    svtkFieldData *dsa =  SVTKUtils::GetAttributes(ds, association);

    if (!dsa)
      return -1;

    unsigned int nArrays = dsa->GetNumberOfArrays();

    if (index > nArrays)
      {
      SENSEI_ERROR("Index " << index << " is out of bounds. "
        << SVTKUtils::GetAttributesName(association) << " data on mesh \""
        << meshName << "\" has " << nArrays << " arrays.")
      return -1;
      }

    arrayName = dsa->GetArray(index)->GetName();

    return 1;
    };

  // locate the cached mesh
  svtkDataObject *dobj = nullptr;
  if (this->GetDataObject(meshName, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return -1;
    }

  // apply the helper function
  if (SVTKUtils::Apply(dobj, getArrayName))
    {
    SENSEI_ERROR("Failed to get the number of arrays for mesh \""
      << meshName  << "\"")
    return -1;
    }

  return 0;
}
*/
//----------------------------------------------------------------------------
int SVTKDataAdaptor::ReleaseData()
{
  this->Internals->MeshMap.clear();
  return 0;
}

}
