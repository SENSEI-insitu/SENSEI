#include "ADIOSAnalysisAdaptor.h"

#include "ADIOSSchema.h"
#include "DataAdaptor.h"
#include "VTKUtils.h"
#include "Timer.h"
#include "Error.h"

#include <vtkCellTypes.h>
#include <vtkCellData.h>
#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkLongArray.h>
#include <vtkUnsignedLongArray.h>
#include <vtkCharArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkIdTypeArray.h>
#include <vtkCellArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <mpi.h>
#include <adios.h>
#include <vector>

namespace sensei
{

//----------------------------------------------------------------------------
senseiNewMacro(ADIOSAnalysisAdaptor);

//----------------------------------------------------------------------------
ADIOSAnalysisAdaptor::ADIOSAnalysisAdaptor() : Comm(MPI_COMM_WORLD),
   MaxBufferSize(500), Schema(nullptr), Method("MPI"), FileName("sensei.bp")
{
}

//----------------------------------------------------------------------------
ADIOSAnalysisAdaptor::~ADIOSAnalysisAdaptor()
{
}

//-----------------------------------------------------------------------------
int ADIOSAnalysisAdaptor::SetDataRequirements(const DataRequirements &reqs)
{
  this->Requirements = reqs;
  return 0;
}

//-----------------------------------------------------------------------------
int ADIOSAnalysisAdaptor::AddDataRequirement(const std::string &meshName,
  int association, const std::vector<std::string> &arrays)
{
  this->Requirements.AddRequirement(meshName, association, arrays);
  return 0;
}

//----------------------------------------------------------------------------
bool ADIOSAnalysisAdaptor::Execute(DataAdaptor* dataAdaptor)
{
  timer::MarkEvent mark("ADIOSAnalysisAdaptor::Execute");

  // if no dataAdaptor requirements are given, push all the data
  // fill in the requirements with every thing
  if (this->Requirements.Empty())
    {
    if (this->Requirements.Initialize(dataAdaptor))
      {
      SENSEI_ERROR("Failed to initialze dataAdaptor description")
      return -1;
      }
    SENSEI_WARNING("No subset specified. Writing all available data")
    }

  // collect the specified data objects
  std::vector<vtkDataObject*> objects;
  std::vector<std::string> objectNames;

  MeshRequirementsIterator mit =
    this->Requirements.GetMeshRequirementsIterator();

  for (; mit; ++mit)
    {
    // get the mesh
    vtkDataObject* dobj = nullptr;
    if (dataAdaptor->GetMesh(mit.MeshName(), mit.StructureOnly(), dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << mit.MeshName() << "\"")
      return -1;
      }

    // add the required arrays
    ArrayRequirementsIterator ait =
      this->Requirements.GetArrayRequirementsIterator(mit.MeshName());

    for (; ait; ++ait)
      {
      if (dataAdaptor->AddArray(dobj, mit.MeshName(),
         ait.Association(), ait.Array()))
        {
        SENSEI_ERROR("Failed to add "
          << VTKUtils::GetAttributesName(ait.Association())
          << " data array \"" << ait.Array() << "\" to mesh \""
          << mit.MeshName() << "\"")
        return -1;
        }
      }

    // add to the collection
    objects.push_back(dobj);
    objectNames.push_back(mit.MeshName());
    }

  unsigned long timeStep = dataAdaptor->GetDataTimeStep();
  double time = dataAdaptor->GetDataTime();

  if (this->InitializeADIOS(objectNames, objects) ||
    this->WriteTimestep(timeStep, time, objectNames, objects))
    return false;

  return true;
}

//----------------------------------------------------------------------------
int ADIOSAnalysisAdaptor::InitializeADIOS(
  const std::vector<std::string> &objectNames,
  const std::vector<vtkDataObject*> &objects)
{
  if (!this->Schema)
    {
    timer::MarkEvent mark("ADIOSAnalysisAdaptor::IntializeADIOS");

    // initialize adios
    adios_init_noxml(this->Comm);

    int64_t gHandle = 0;

#if ADIOS_VERSION_GE(1,11,0)
    adios_set_max_buffer_size(this->MaxBufferSize);
    adios_declare_group(&gHandle, "SENSEI", "",
      static_cast<ADIOS_STATISTICS_FLAG>(adios_flag_yes));
#else
    adios_allocate_buffer(ADIOS_BUFFER_ALLOC_NOW, this->MaxBufferSize);
    adios_declare_group(&gHandle, "SENSEI", "", adios_flag_yes);
#endif

    adios_select_method(gHandle, this->Method.c_str(), "", "");

    // define ADIOS variables
    this->Schema = new senseiADIOS::DataObjectCollectionSchema;

    if (this->Schema->DefineVariables(this->Comm, gHandle, objectNames, objects))
      {
      SENSEI_ERROR("Failed to define variables")
      return -1;
      }
    }
  return 0;
}

//----------------------------------------------------------------------------
int ADIOSAnalysisAdaptor::FinalizeADIOS()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  adios_finalize(rank);
  return 0;
}

//----------------------------------------------------------------------------
int ADIOSAnalysisAdaptor::Finalize()
{
  timer::MarkEvent mark("ADIOSAnalysisAdaptor::Finalize");

  if (this->Schema)
    this->FinalizeADIOS();

  delete this->Schema;
  this->Schema = nullptr;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSAnalysisAdaptor::WriteTimestep(unsigned long timeStep,
  double time, const std::vector<std::string> &objectNames,
  const std::vector<vtkDataObject*> &objects)
{
  timer::MarkEvent mark("ADIOSAnalysisAdaptor::WriteTimestep");

  int64_t handle = 0;

  adios_open(&handle, "sensei", this->FileName.c_str(),
    timeStep == 0 ? "w" : "a", this->Comm);

  uint64_t group_size = this->Schema->GetSize(this->Comm, objectNames, objects);
  adios_group_size(handle, group_size, &group_size);

  if (this->Schema->Write(this->Comm, handle, timeStep, time,
    objectNames, objects))
    {
    SENSEI_ERROR("Failed to write step " << timeStep
      << " to \"" << this->FileName << "\"")
    return -1;
    }

  adios_close(handle);

  return 0;
}

//----------------------------------------------------------------------------
void ADIOSAnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
