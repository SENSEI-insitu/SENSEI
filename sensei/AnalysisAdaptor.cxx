#include "AnalysisAdaptor.h"
#include "Error.h"

#include <cstdlib>
#if defined(SENSEI_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace sensei
{

//----------------------------------------------------------------------------
AnalysisAdaptor::AnalysisAdaptor() : Verbose(0), DeviceId(-2),
  DevicesPerNode(1), DevicesToUse(1), DeviceStart(0), Asynchronous(0)
{
  // give each analysis its own communication space.
  MPI_Comm_dup(MPI_COMM_WORLD, &this->Comm);

  int rank = 0;
  MPI_Comm_rank(this->Comm, &rank);

  // set verbosity level default
  if (const char *aVal = getenv("SENSEI_VERBOSE"))
  {
    this->Verbose = atoi(aVal);
    if (this->Verbose)
      SENSEI_STATUS("SENSEI_VERBOSE = " << this->Verbose)
  }

  // set asynchronous execution default
  if (const char *aVal = getenv("SENSEI_ASYNCHRONOUS"))
  {
    this->Asynchronous = atoi(aVal);
    if (this->Verbose)
      SENSEI_STATUS("SENSEI_ASYNCHRONOUS = " << this->Asynchronous)
  }

  // set the device default
  if (const char *aVal = getenv("SENSEI_DEVICE_ID"))
  {
    this->DeviceId = atoi(aVal);
    if (this->Verbose)
      SENSEI_STATUS("SENSEI_DEVICE_ID = " << this->DeviceId)
  }

#if defined(SENSEI_ENABLE_CUDA)
  cudaError_t ierr = cudaGetDeviceCount(&this->DevicesPerNode);
  if (ierr != cudaSuccess)
  {
    SENSEI_ERROR("Failed to query the number of CUDA devices. "
      << cudaGetErrorString(ierr))
  }
#endif

  // set the devices to use per node default
  if (const char *aVal = getenv("SENSEI_DEVICES_TO_USE"))
  {
    this->DevicesToUse = atoi(aVal);
    if (this->Verbose)
      SENSEI_STATUS("SENSEI_DEVICES_TO_USE = " << this->DevicesToUse)
  }
#if defined(SENSEI_ENABLE_CUDA)
  else
  {
    this->DevicesToUse = this->DevicesPerNode;
  }
#endif

  // set device start default
  if (const char *aVal = getenv("SENSEI_DEVICE_START"))
  {
    this->DeviceStart = atoi(aVal);
    if (this->Verbose)
      SENSEI_STATUS("SENSEI_DEVICE_START = " << this->DeviceStart)
  }

  // set device stride default
  if (const char *aVal = getenv("SENSEI_DEVICE_STRIDE"))
  {
    this->DeviceStride = atoi(aVal);
    if (this->Verbose)
      SENSEI_STATUS("SENSEI_DEVICE_STRIDE = " << this->DeviceStride)
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
#if defined(SENSEI_ENABLE_CUDA)
  if (this->DeviceId == AnalysisAdaptor::DEVICE_AUTO)
  {
    if ((this->DevicesToUse < 1) || (this->DevicesPerNode < 1))
    {
      this->DeviceId = AnalysisAdaptor::DEVICE_HOST;
    }
    else
    {
      // select the device
      int rank = 0;
      MPI_Comm_rank(this->GetCommunicator(), &rank);
      this->DeviceId = ( rank % this->DevicesToUse * this->DeviceStride
                         + this->DeviceStart ) % this->DevicesPerNode;
    }
  }
#else
  this->DeviceId = AnalysisAdaptor::DEVICE_HOST;
#endif
  return this->DeviceId;
}

//----------------------------------------------------------------------------
void AnalysisAdaptor::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
