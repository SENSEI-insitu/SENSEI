#include "Calculator.h"

#include "DataRequirements.h"
#include "senseiConfig.h"
#include "Error.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "Profiler.h"

#include "SVTKDataAdaptor.h"
#include "SVTKUtils.h"

#include <svtkDataObject.h>

#include <vtkSmartPointer.h>
#include <vtkArrayCalculator.h>
#include <vtkObjectFactory.h>

#include <string>

namespace sensei
{

static void replace_all(std::string& data, const std::string& oldtxt, const std::string& newtxt)
{
	size_t pos = data.find(oldtxt);
	while (pos != std::string::npos)
	  {
		data.replace(pos, oldtxt.size(), newtxt);
	  pos = data.find(oldtxt, pos + newtxt.size());
	  }
}


//-----------------------------------------------------------------------------
senseiNewMacro(Calculator);

//-----------------------------------------------------------------------------
Calculator::Calculator()
{
}

//-----------------------------------------------------------------------------
Calculator::~Calculator()
{
}

//-----------------------------------------------------------------------------
void Calculator::Initialize(const std::string& meshName, int association, const std::string& expression,
    const std::string& result)
{
  this->MeshName = meshName;
  this->Association = association;
  this->Expression = expression;
  this->Result = result;
}

//-----------------------------------------------------------------------------
bool Calculator::Execute(DataAdaptor* data, DataAdaptor*& result)
{
  TimeEvent<128> mark("Calculator::Execute");

  // see what the simulation is providing
  MeshMetadataMap mdMap;
  if (mdMap.Initialize(data))
    {
    SENSEI_ERROR("Failed to get metadata")
    return false;
    }

  // get the current time and step
  int step = data->GetDataTimeStep();
  double time = data->GetDataTime();

  // get the mesh metadata object
  MeshMetadataPtr mmd;
  if (mdMap.GetMeshMetadata(this->MeshName, mmd))
    {
    SENSEI_ERROR("Failed to get metadata for mesh \"" << this->MeshName << "\"")
    return false;
    }

  // get the mesh object
  svtkDataObject *meshIn = nullptr;
  if (data->GetMesh(this->MeshName, false, meshIn))
    {
    SENSEI_ERROR("Failed to get mesh \"" << this->MeshName << "\"")
    return false;
    }

  vtkNew<vtkArrayCalculator> calculator;
  calculator->SetAttributeType(this->Association);
  calculator->AddCoordinateScalarVariable("coordsX", 0);
  calculator->AddCoordinateScalarVariable("coordsY", 1);
  calculator->AddCoordinateScalarVariable("coordsZ", 2);
  calculator->AddCoordinateVectorVariable("coords", 0, 1, 2);

  DataRequirements requirements;
  requirements.Initialize(data, /*structureOnly=*/false);

  for (auto ait = requirements.GetArrayRequirementsIterator(this->MeshName); meshIn && ait; ++ait)
  {
    data->AddArray(meshIn, this->MeshName, ait.Association(), ait.Array());
    if (ait.Association() == this->Association)
    {
      // fixme: handle multicomponent arrays.
      calculator->AddScalarArrayName(ait.Array().c_str(), 0);
      calculator->AddScalarVariable(("\"" + ait.Array() + "\"").c_str(), ait.Array().c_str(), 0);
    }
  }

  auto function = this->Expression;
  replace_all(function, "data_time", std::to_string(time));
  replace_all(function, "data_time_step", std::to_string(step));

  // convert input to VTK
  vtkDataObject *vmeshIn = SVTKUtils::VTKObjectFactory::New(meshIn);
  if (!vmeshIn)
  {
    SENSEI_ERROR("Conversion from " << meshIn->GetClassName() << " to VTK failed")
    return false;
  }

  calculator->SetInputDataObject(vmeshIn);
  calculator->SetFunction(function.c_str());
  calculator->SetCoordinateResults(this->Result == "coords"? 1:0);
  calculator->SetResultArrayName(this->Result.c_str());
  calculator->Update();

  vtkDataObject *vmeshOut = calculator->GetOutputDataObject(0);

  // convert the output to SVTK
  svtkDataObject *meshOut = SVTKUtils::SVTKObjectFactory::New(vmeshOut);

  // configure the return adaptor
  SVTKDataAdaptor *ra = SVTKDataAdaptor::New();
  ra->SetDataObject(this->MeshName, meshOut);
  result = ra;

  vmeshIn->Delete();
  meshIn->Delete();

  return true;
}

//-----------------------------------------------------------------------------
int Calculator::Finalize()
{
  return 0;
}


} // end of sensei
