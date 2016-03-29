#include "DataAdaptor.h"

#include "vtkCompositeDataIterator.h"
#include "vtkCompositeDataSet.h"
#include "vtkDataSetAttributes.h"
#include "vtkObjectFactory.h"

namespace sensei
{
namespace vtk
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

vtkStandardNewMacro(DataAdaptor);
//----------------------------------------------------------------------------
DataAdaptor::DataAdaptor()
{
}

//----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataObject(vtkDataObject* dobj)
{
  this->DataObject = dobj;
}

//----------------------------------------------------------------------------
vtkDataObject* DataAdaptor::GetMesh(bool)
{
  return this->DataObject;
}

//----------------------------------------------------------------------------
bool DataAdaptor::AddArray(vtkDataObject* mesh, int association, const std::string& arrayname)
{
  vtkDataObject* dobj = vtkGetRepresentationDataObject(this->DataObject);
  vtkFieldData* dsa = dobj? dobj->GetAttributesAsFieldData(association) : NULL;
  return dsa? dsa->GetAbstractArray(arrayname.c_str()) : NULL;
}

//----------------------------------------------------------------------------
unsigned int DataAdaptor::GetNumberOfArrays(int association)
{
  vtkDataObject* dobj = vtkGetRepresentationDataObject(this->DataObject);
  vtkFieldData* dsa = dobj? dobj->GetAttributesAsFieldData(association) : NULL;
  return dsa? static_cast<unsigned int>(dsa->GetNumberOfArrays()) : 0;
}

//----------------------------------------------------------------------------
std::string DataAdaptor::GetArrayName(int association, unsigned int index)
{
  vtkDataObject* dobj = vtkGetRepresentationDataObject(this->DataObject);
  vtkFieldData* dsa = dobj? dobj->GetAttributesAsFieldData(association) : NULL;
  const char* aname = dsa? dsa->GetArrayName(index) : NULL;
  return aname? std::string(aname) : std::string();
}

//----------------------------------------------------------------------------
void DataAdaptor::ReleaseData()
{
  this->DataObject = NULL;
}

//----------------------------------------------------------------------------
vtkDataObject* DataAdaptor::GetCompleteMesh()
{
  return this->DataObject;
}

} // vtk
} // sensei
