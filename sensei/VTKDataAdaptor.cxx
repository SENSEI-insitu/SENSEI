#include "VTKDataAdaptor.h"

#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkObjectFactory.h>
#include <vtkDataObject.h>
#include <vtkObjectBase.h>
#include <vtkObject.h>

namespace
{
vtkDataObject* vtkGetRepresentationDataObject(vtkDataObject* dobj)
{
  if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj))
    {
    vtkCompositeDataIterator* iter = cd->NewIterator();
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      {
      dobj = iter->GetCurrentDataObject();
      break;
      }
    iter->Delete();
    }
  return dobj;
}
}

namespace sensei
{
//----------------------------------------------------------------------------
senseiNewMacro(VTKDataAdaptor);

//----------------------------------------------------------------------------
VTKDataAdaptor::VTKDataAdaptor()
{
}

//----------------------------------------------------------------------------
VTKDataAdaptor::~VTKDataAdaptor()
{
}

//----------------------------------------------------------------------------
void VTKDataAdaptor::SetDataObject(vtkDataObject* dobj)
{
  this->DataObject = dobj;
}

//----------------------------------------------------------------------------
vtkDataObject* VTKDataAdaptor::GetMesh(bool)
{
  return this->DataObject;
}

//----------------------------------------------------------------------------
bool VTKDataAdaptor::AddArray(vtkDataObject* mesh, int association,
  const std::string& arrayname)
{
  (void)mesh;
  vtkDataObject* dobj = ::vtkGetRepresentationDataObject(this->DataObject);
  vtkFieldData* dsa = dobj ? dobj->GetAttributesAsFieldData(association) : nullptr;
  return dsa ? dsa->GetAbstractArray(arrayname.c_str()) != nullptr : false;
}

//----------------------------------------------------------------------------
unsigned int VTKDataAdaptor::GetNumberOfArrays(int association)
{
  vtkDataObject* dobj = ::vtkGetRepresentationDataObject(this->DataObject);
  vtkFieldData* dsa = dobj ? dobj->GetAttributesAsFieldData(association) : NULL;
  return dsa ? static_cast<unsigned int>(dsa->GetNumberOfArrays()) : 0;
}

//----------------------------------------------------------------------------
std::string VTKDataAdaptor::GetArrayName(int association, unsigned int index)
{
  vtkDataObject* dobj = ::vtkGetRepresentationDataObject(this->DataObject);
  vtkFieldData* dsa = dobj ? dobj->GetAttributesAsFieldData(association) : NULL;
  const char* aname = dsa ? dsa->GetArrayName(index) : NULL;
  return aname ? std::string(aname) : std::string();
}

//----------------------------------------------------------------------------
void VTKDataAdaptor::ReleaseData()
{
  this->DataObject = NULL;
}

//----------------------------------------------------------------------------
vtkDataObject* VTKDataAdaptor::GetCompleteMesh()
{
  return this->DataObject;
}

}
