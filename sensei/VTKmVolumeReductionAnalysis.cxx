#include "VTKmVolumeReductionAnalysis.h"
#include "DataAdaptor.h"
#include "CinemaHelper.h"
#include <Timer.h>

#include <vtkMPI.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMultiBlockDataSet.h>

#include <algorithm>
#include <vector>

namespace sensei
{


//-----------------------------------------------------------------------------
senseiNewMacro(VTKmVolumeReductionAnalysis);

//-----------------------------------------------------------------------------
VTKmVolumeReductionAnalysis::VTKmVolumeReductionAnalysis() : Communicator(MPI_COMM_WORLD), Helper(NULL)
{
}

//-----------------------------------------------------------------------------
VTKmVolumeReductionAnalysis::~VTKmVolumeReductionAnalysis()
{
    delete this->Helper;
}

//-----------------------------------------------------------------------------
void VTKmVolumeReductionAnalysis::Initialize(
  MPI_Comm comm, const std::string& workingDirectory, int reductionFactor)
{
  this->Communicator = comm;

  vtkNew<vtkMPICommunicator> vtkComm;
  vtkMPICommunicatorOpaqueComm h(&this->Communicator);
  vtkComm->InitializeExternal(&h);

  vtkNew<vtkMPIController> con;
  con->SetCommunicator(vtkComm.GetPointer());
  vtkMultiProcessController::SetGlobalController(con.GetPointer());
  con->Register(NULL); // Keep ref

  this->Helper = new CinemaHelper();
  this->Helper->SetWorkingDirectory(workingDirectory);
  this->Helper->SetExportType("vtk-volume");
}

//-----------------------------------------------------------------------------
bool VTKmVolumeReductionAnalysis::Execute(DataAdaptor* data)
{
  timer::MarkEvent mark("VTKmVolumeReductionAnalysis::execute");
  vtkMultiProcessController* controller = vtkMultiProcessController::GetGlobalController();

  this->Helper->AddTimeEntry();

  vtkDataObject* mesh = data->GetMesh(/*structure_only*/true);
  bool dataError = !data->AddArray(mesh, vtkDataObject::FIELD_ASSOCIATION_CELLS, "data");

  vtkMultiBlockDataSet* blocks = vtkMultiBlockDataSet::SafeDownCast(mesh);
  std::cout << blocks->GetBlock(0)->GetClassName() << " " << blocks->GetNumberOfBlocks() << std::endl;
  this->Helper->WriteVolume(vtkImageData::SafeDownCast(blocks->GetBlock(0)));
  this->Helper->WriteMetadata();

  cout << "Done " << controller->GetLocalProcessId() << endl;

  return true;
}

}
