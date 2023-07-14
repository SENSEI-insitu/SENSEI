#include "AnalysisAdaptor.h"
#include "Error.h"

#include <cstdlib>
#if defined(SENSEI_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace sensei
{

//----------------------------------------------------------------------------
AnalysisAdaptor::AnalysisAdaptor() : Verbose(0), DeviceId(-1),
  DevicesPerNode(0), DeviceStart(0), Asynchronous(0)
{
  // give each analysis its own communication space.
  MPI_Comm_dup(MPI_COMM_WORLD, &this->Comm);

  int rank = 0;
  MPI_Comm_rank(this->Comm, &rank);

  // set verbosity level default
  if (const char *aVal = getenv("SENSEI_VERBOSE"))
  {
    this->Verbose = atoi(aVal);
    SENSEI_STATUS("SENSEI_VERBOSE = " << this->Verbose)
  }

  // set asynchronous execution default
  if (const char *aVal = getenv("SENSEI_ASYNCHRONOUS"))
  {
    this->Asynchronous = atoi(aVal);
    SENSEI_STATUS("SENSEI_ASYNCHRONOUS = " << this->Asynchronous)
  }

  // set the device default
  if (const char *aVal = getenv("SENSEI_DEVICE_ID"))
  {
    this->DeviceId = atoi(aVal);
    SENSEI_STATUS("SENSEI_DEVICE_ID = " << this->DeviceId)
  }

  // set the devices per node default
  if (const char *aVal = getenv("SENSEI_DEVICES_PER_NODE"))
  {
    this->DevicesPerNode = atoi(aVal);
    SENSEI_STATUS("SENSEI_DEVICES_PER_NODE = " << this->DevicesPerNode)
  }
#if defined(SENSEI_ENABLE_CUDA)
  else
  {
    cudaError_t ierr = cudaGetDeviceCount(&this->DevicesPerNode);
    if (ierr != cudaSuccess)
    {
      SENSEI_ERROR("Failed to query the number of CUDA devices. "
        << cudaGetErrorString(ierr))
    }
  }
#endif

  // set device start default
  if (const char *aVal = getenv("SENSEI_DEVICE_START"))
  {
    this->DeviceStart = atoi(aVal);
    SENSEI_STATUS("SENSEI_DEVICE_START = " << this->DeviceStart)
  }
}

//----------------------------------------------------------------------------
AnalysisAdaptor::~AnalysisAdaptor()
{
  MPI_Comm_free(&this->Comm);
}

//----------------------------------------------------------------------------
int AnalysisAdaptor::SetCommunicator(MPI_Comm comm)
{
  MPI_Comm_free(&this->Comm);
  MPI_Comm_dup(comm, &this->Comm);
  return 0;
}

//----------------------------------------------------------------------------
int AnalysisAdaptor::GetDeviceId()
{
  if (this->DeviceId == AnalysisAdaptor::DEVICE_AUTO)
  {
    // automatic device selection
    if (this->DevicesPerNode == 0)
    {
      // no devices available
      this->DeviceId = AnalysisAdaptor::DEVICE_HOST;
    }
    else
    {
      // select the device
      int rank = 0;
      MPI_Comm_rank(this->GetCommunicator(), &rank);
      this->DeviceId = rank % this->DevicesPerNode + this->DeviceStart;
    }
  }
  return this->DeviceId;
}

//----------------------------------------------------------------------------
void AnalysisAdaptor::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
