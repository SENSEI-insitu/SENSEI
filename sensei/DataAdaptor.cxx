#include "DataAdaptor.h"

#include "vtkDataObject.h"
#include "vtkInformation.h"
#include "vtkInformationIntegerKey.h"
#include "vtkObjectFactory.h"

namespace sensei
{

//----------------------------------------------------------------------------
vtkInformationKeyMacro(DataAdaptor, DATA_TIME_STEP_INDEX, Integer);

//----------------------------------------------------------------------------
DataAdaptor::DataAdaptor()
{
  this->Information = vtkInformation::New();
}

//----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
  this->Information->Delete();
}

//----------------------------------------------------------------------------
double DataAdaptor::GetDataTime(vtkInformation* info)
{
  return info->Has(vtkDataObject::DATA_TIME_STEP())?
    info->Get(vtkDataObject::DATA_TIME_STEP()) : 0.0;
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTime(vtkInformation* info, double time)
{
  info->Set(vtkDataObject::DATA_TIME_STEP(), time);
}

//----------------------------------------------------------------------------
int DataAdaptor::GetDataTimeStep(vtkInformation* info)
{
  return info->Has(DataAdaptor::DATA_TIME_STEP_INDEX())?
    info->Get(DataAdaptor::DATA_TIME_STEP_INDEX()) : 0;
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTimeStep(vtkInformation* info, int index)
{
  info->Set(DataAdaptor::DATA_TIME_STEP_INDEX(), index);
}

//----------------------------------------------------------------------------
vtkDataObject* DataAdaptor::GetCompleteMesh()
{
  vtkDataObject* mesh = this->GetMesh();
  for (int attr=vtkDataObject::FIELD_ASSOCIATION_POINTS;
    attr < vtkDataObject::NUMBER_OF_ASSOCIATIONS; ++attr)
    {
    for (unsigned int cc=0, max=this->GetNumberOfArrays(attr); cc < max; ++cc)
      {
      this->AddArray(mesh, attr, this->GetArrayName(attr, cc));
      }
    }
  return mesh;
}

//----------------------------------------------------------------------------
void DataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
