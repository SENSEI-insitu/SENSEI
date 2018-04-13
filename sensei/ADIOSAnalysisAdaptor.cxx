#include "ADIOSAnalysisAdaptor.h"

#include "ADIOSSchema.h"
#include "DataAdaptor.h"
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
   Schema(nullptr), Method("MPI"), FileName("sensei.bp")
{
}

//----------------------------------------------------------------------------
ADIOSAnalysisAdaptor::~ADIOSAnalysisAdaptor()
{
}

//----------------------------------------------------------------------------
bool ADIOSAnalysisAdaptor::Execute(DataAdaptor* data)
{
  timer::MarkEvent mark("ADIOSAnalysisAdaptor::Execute");

  vtkDataObject* dobj = data->GetCompleteMesh();
  unsigned long timeStep = data->GetDataTimeStep();
  double time = data->GetDataTime();

  this->InitializeADIOS(dobj);
  this->WriteTimestep(timeStep, time, dobj);

  return true;
}

//----------------------------------------------------------------------------
void ADIOSAnalysisAdaptor::InitializeADIOS(vtkDataObject *dobj)
{
  if (this->Schema)
    return;

  timer::MarkEvent mark("ADIOSAnalysisAdaptor::IntializeADIOS");

  // initialize adios
  adios_init_noxml(this->Comm);

  int64_t gHandle = 0;
  int64_t bufferSizeMB = 500;

#if ADIOS_VERSION_GE(1,11,0)
  adios_set_max_buffer_size(bufferSizeMB);
  adios_declare_group(&gHandle, "sensei", "",
    static_cast<ADIOS_STATISTICS_FLAG>(adios_flag_yes));
#else
  adios_allocate_buffer(ADIOS_BUFFER_ALLOC_NOW, bufferSizeMB);
  adios_declare_group(&gHandle, "sensei", "", adios_flag_yes);
#endif

  adios_select_method(gHandle, this->Method.c_str(), "", "");

  // define ADIOS variables
  this->Schema = new senseiADIOS::DataObjectSchema;
  this->Schema->DefineVariables(this->Comm, gHandle, dobj);
}

//----------------------------------------------------------------------------
void ADIOSAnalysisAdaptor::FinalizeADIOS()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  adios_finalize(rank);
}

//----------------------------------------------------------------------------
int ADIOSAnalysisAdaptor::Finalize()
{
  timer::MarkEvent mark("ADIOSAnalysisAdaptor::Finalize");

  if (this->Schema)
    this->FinalizeADIOS();

  delete Schema;

  return 0;
}

//----------------------------------------------------------------------------
void ADIOSAnalysisAdaptor::WriteTimestep(unsigned long timeStep,
  double time, vtkDataObject *dobj)
{
  timer::MarkEvent mark("ADIOSAnalysisAdaptor::WriteTimestep");

  int64_t handle = 0;

  adios_open(&handle, "sensei", this->FileName.c_str(),
    timeStep == 0 ? "w" : "a", this->Comm);

  uint64_t group_size = this->Schema->GetSize(this->Comm, dobj);
  adios_group_size(handle, group_size, &group_size);

  if (this->Schema->Write(this->Comm, handle, dobj) ||
    this->Schema->WriteTimeStep(this->Comm, handle, timeStep, time))
    {
    SENSEI_ERROR("Failed to write step " << timeStep
      << " to \"" << this->FileName << "\"")
    return;
    }

  adios_close(handle);
}

//----------------------------------------------------------------------------
void ADIOSAnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
