#include "svtkHAMRDataArray.h"
#include "svtkImageData.h"
#include "svtkPointData.h"

#include "senseiConfig.h"

#if defined(SENSEI_ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

void example1(size_t nElem, int devId)
{
  // OpenMP allocates this array on device memory
  omp_set_default_device(devId);
  double *devPtr = (double*)malloc(nElem*sizeof(double));
  #pragma omp target enter data map(alloc: devPtr[0:nElem])

  // OpenMP initializes the memory on the device
  #pragma omp target teams distribute \
    parallel for map(alloc: devPtr[0:nElem])
  for (size_t i = 0; i < nElem; ++i)
    devPtr[i] = -3.14;

  // zero-copy construct with the device pointer
  svtkHAMRDoubleArray *simData;
  #pragma omp target data use_device_addr(devPtr)
  {
  simData = svtkHAMRDoubleArray::New("simData", devPtr, nElem, 1,
                              svtkAllocator::openmp, svtkStream(),
                              svtkStreamMode::async, devId, 0);
  }

  // do something with the array
  simData->Print(std::cerr);

  // delete the array
  simData->Delete();

  // now it is safe to deallocate the device memory
  #pragma omp target exit data map(release: devPtr[0:nElem])
}


void example2(size_t nElem, int devId)
{
  omp_set_default_device(devId);

  // allocate device memory
  double *devPtr = (double*)omp_target_alloc(nElem*sizeof(double), devId);

  // wrap it in a shared pointer so it is eventually deallocated
  std::shared_ptr<double> spDev(devPtr,
    [devId](double *ptr){ omp_target_free(ptr, devId); });

  // initialize the array on the device
  #pragma omp target teams distribute parallel for is_device_ptr(devPtr)
  for (size_t i = 0; i < nElem; ++i)
    devPtr[i] = -3.14;

  // zero-copy construct with coordinated life cycle management
  auto simData = svtkHAMRDoubleArray::New("simData", spDev, nElem, 1,
                                          svtkAllocator::openmp, svtkStream(),
                                          svtkStreamMode::async, devId);

  // do something with the data in SENSEI
  simData->Print(std::cerr);

  // free up the container
  simData->Delete();
}

void example3(size_t nElem, int srcDev, int destDev)
{
  // allocate device memory
  omp_set_default_device(srcDev);
  double *devPtr = (double*)omp_target_alloc(nElem*sizeof(double), srcDev);

  // initialize
  #pragma omp target teams distribute parallel for is_device_ptr(devPtr)
  for (size_t i = 0; i < nElem; ++i)
    devPtr[i] = -3.14;

  // zero-copy construct from a device pointer, and take ownership
  auto simData = svtkHAMRDoubleArray::New("simData", devPtr, nElem, 1,
                                          svtkAllocator::openmp, svtkStream(),
                                          svtkStreamMode::async, srcDev, 1);

  // move to destDev in place
  omp_set_default_device(destDev);
  simData->SetOwner();

  // do something with the data in SENSEI
  simData->Print(std::cerr);

  // free up the container
  simData->Delete();
}

void example4(size_t nElem, int srcDev, int destDev)
{
  // allocate and value initialize on one device using OpenMP
  omp_set_default_device(srcDev);
  auto simData = svtkHAMRDoubleArray::New("simData", nElem, 1,
                                  svtkAllocator::openmp, svtkStream(),
                                  svtkStreamMode::async, -3.14);

  // deep-copy to another device using CUDA
  cudaSetDevice(destDev);
  auto dataCpy = svtkHAMRDoubleArray::New(simData, svtkAllocator::cuda_async,
                                  svtkStream(), svtkStreamMode::async);

  // make sure movement is complete before freeing up the source
  dataCpy->Synchronize();
  simData->Delete();

  // do something with the data in SENSEI
  dataCpy->Print(std::cerr);

  // free up the container
  dataCpy->Delete();
}



svtkHAMRDoubleArray *
add_arrays_mp(int dev, svtkHAMRDoubleArray *a1, svtkHAMRDoubleArray *a2)
{
  // get a view of the incoming data on the device we will use
  omp_set_default_device(dev);

  auto spA1 = a1->GetOpenMPAccessible();
  auto pA1 = spA1.get();

  auto spA2 = a2->GetOpenMPAccessible();
  auto pA2 = spA2.get();

  // allocate space for the result
  size_t nElem = a1->GetNumberOfTuples();

  auto a3 = svtkHAMRDoubleArray::New("sum", nElem, 1,
                             svtkAllocator::openmp, svtkStream(),
                             svtkStreamMode::async);

  // direct access to the result since we know it is in place
  auto pA3 = a3->GetData();

  // do the calculation
  #pragma omp target teams distribute parallel for is_device_ptr(pA1, pA2)
  for (size_t i = 0; i < nElem; ++i)
    pA3[i] = pA2[i] + pA1[i];

  return a3;
}

void example5(size_t nElem, int dev1, int dev2)
{
  // this data is located in host main memory
  auto a1 = svtkHAMRDoubleArray::New("a1", nElem, 1,
                             svtkAllocator::malloc, svtkStream(),
                             svtkStreamMode::async, 1.0);

  // this data is located in device 1 main memory
  omp_set_default_device(dev1);
  auto a2 = svtkHAMRDoubleArray::New("a2", nElem, 1,
                             svtkAllocator::openmp, svtkStream(),
                             svtkStreamMode::async, 2.0);

  // do the calculation on device 2
  auto a3 = add_arrays_mp(dev2, a1, a2);

  // do something with the result
  a3->Print(std::cerr);

  // clean up
  a1->Delete();
  a2->Delete();
  a3->Delete();
}


namespace libA {
__global__
void add(double *a3, const double *a1, const double *a2, size_t n)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= n) return;
  a3[i] = a1[i] + a2[i];
  //printf("a3[%d]=%g  a1[%d]=%g  a2[%d]=%g \n", i, a3[i], i, a1[i], i, a2[i]);
}

svtkHAMRDoubleArray *
Add(int dev, svtkHAMRDoubleArray *a1, svtkHAMRDoubleArray *a2)
{
  // use this stream for the calculation
  cudaStream_t strm = svtkStream();

  // get a view of the incoming data on the device we will use
  cudaSetDevice(dev);

  auto spA1 = a1->GetCUDAAccessible();
  auto pA1 = spA1.get();

  auto spA2 = a2->GetCUDAAccessible();
  auto pA2 = spA2.get();

  // allocate space for the result
  size_t nElem = a1->GetNumberOfTuples();

  auto a3 = svtkHAMRDoubleArray::New("sum", nElem, 1,
                             svtkAllocator::cuda_async, strm,
                             svtkStreamMode::async);

  // direct access to the result since we know it is in place
  auto pA3 = a3->GetData();

  // make sure the data in flight, if it was moved, has arrived
  a1->Synchronize();
  a2->Synchronize();

  // do the calculation
  int threads = 128;
  int blocks = nElem / threads + ( nElem % threads ? 1 : 0 );
  add<<<blocks,threads,0,strm>>>(pA3, pA1, pA2, nElem);

  return a3;
}
}

namespace libB {
void Write(std::ofstream &ofs, svtkHAMRDoubleArray *a)
{
  // get a view of the data on the host
  auto spA = a->GetHostAccessible();
  auto pA = spA.get();

  // send name while data may be in flight
  ofs << a->GetName() << " = ";

  // make sure the data if moved has arrived
  a->Synchronize();

  // send the data to the file
  size_t nElem = a->GetNumberOfTuples();
  for (size_t i = 0; i < nElem; ++i)
    ofs << pA[i] << " ";
  ofs << std::endl;
}
}

void example6(size_t nElem, int dev1, int dev2)
{
  // this data is located in host memory, initialized to 1
  auto a1 = svtkHAMRDoubleArray::New("a1", nElem, 1,
                             svtkAllocator::malloc, svtkStream(),
                             svtkStreamMode::async, 1.0);

  // this data is located in device 1 memory, unitialized
  omp_set_default_device(dev1);
  auto a2 = svtkHAMRDoubleArray::New("a2", nElem, 1,
                             svtkAllocator::openmp, svtkStream(),
                             svtkStreamMode::async);

  // initialize with OpenMP target offload
  auto pA2 = a2->GetData();

  #pragma omp target teams distribute parallel for is_device_ptr(pA2)
  for (size_t i = 0; i < nElem; ++i)
    pA2[i] = 2.0;

  // pass data to libA for the calculations
  auto a3 = libA::Add(dev2, a1, a2);

  // pass data to libB for I/O
  auto ofile = std::ofstream("data.txt");
  libB::Write(ofile, a1);
  libB::Write(ofile, a2);
  libB::Write(ofile, a3);
  ofile.close();

  // clean up
  a1->Delete();
  a2->Delete();
  a3->Delete();
}


svtkHAMRDoubleArray *example7(int dev, svtkHAMRDoubleArray *a1, svtkHAMRDoubleArray *a2)
{
  // get a view of the incoming data on the device we will use
  omp_set_default_device(dev);

  auto spA1 = a1->GetOpenMPAccessible();
  auto pA1 = spA1.get();

  auto spA2 = a2->GetOpenMPAccessible();
  auto pA2 = spA2.get();

  // allocate space for the result
  size_t nElem = a1->GetNumberOfTuples();

  auto a3 = svtkHAMRDoubleArray::New("sum", nElem, 1,
                             GetDeviceAllocator(), svtkStream(),
                             svtkStreamMode::async);

  // direct access to the result since we know it is in place
  auto pA3 = a3->GetData();

  // do the calculation
  #pragma omp target teams distribute parallel for is_device_ptr(pA1, pA2)
  for (size_t i = 0; i < nElem; ++i)
    pA3[i] = pA2[i] + pA1[i];

  return a3;
}





int main(int argc, char **argv)
{
  if (argc != 3)
  {
      std::cerr << "usage: testHAMRDataArray [device id] [device id]" << std::endl;
      return -1;
  }

  size_t nElem = 64;
  int dev = atoi(argv[1]);
  int destDev = atoi(argv[2]);

  std::cerr << "zero-copy construct manual life cycle management ... " << std::endl;
  example1(nElem, dev);

  std::cerr << "zero-copy construct automatic life cycle management ... " << std::endl;
  example2(nElem, dev);

  std::cerr << "move in place ..." << std::endl;
  example3(nElem, dev, destDev);

  std::cerr << "deep copy construct on another device ... " << std::endl;
  example4(nElem, dev, destDev);

  std::cerr << "add two arrays OpenMP ... " << std::endl;
  example5(nElem, dev, destDev);

  std::cerr << "add two arrays OpenMP CUDA interop ... " << std::endl;
  example6(nElem, dev, destDev);

  return 0;
}
