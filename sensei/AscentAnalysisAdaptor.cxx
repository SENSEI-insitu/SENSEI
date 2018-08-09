#include "AscentAnalysisAdaptor.h"
#include "DataAdaptor.h"
#include "Error.h"


#include <vtkObjectFactory.h>
#include <vtkDataObject.h>
#include <vtkFieldData.h>


namespace sensei
{

//------------------------------------------------------------------------------
senseiNewMacro(AscentAnalysisAdaptor);


//------------------------------------------------------------------------------
AscentAnalysisAdaptor::AscentAnalysisAdaptor()
{
}

//------------------------------------------------------------------------------
AscentAnalysisAdaptor::~AscentAnalysisAdaptor()
{
}

//------------------------------------------------------------------------------
void
AscentAnalysisAdaptor::Initialize()
{

return;
}


//------------------------------------------------------------------------------
bool
AscentAnalysisAdaptor::Execute(DataAdaptor* dataAdaptor)
{
  std::cout << "IN EzXECUTE" << std::endl;
  vtkIndent indent;
  vtkDataObject* obj = nullptr;
  if(dataAdaptor->GetMesh("mesh", false, obj))
  {
    SENSEI_ERROR("Failed to get mesh")
    return -1;
  }
//  obj->Print(std::cout); 
  unsigned int nArrays = 0;
  dataAdaptor->GetNumberOfArrays("mesh", 1, nArrays);
  std::cout << "Number of arrays " << nArrays << std::endl;
  std::string arrayName;
  dataAdaptor->GetArrayName("mesh", 1, 0, arrayName);
  std::cout << "ArrayName " << arrayName <<std::endl;
  dataAdaptor->AddArray(obj, "mesh", 1, "dist");
  obj->Print(std::cout);
  vtkFieldData* fd = obj->GetFieldData();
  fd->Print(std::cout);
  return true;
}


//------------------------------------------------------------------------------
int
AscentAnalysisAdaptor::Finalize()
{

return 0;
}

}
