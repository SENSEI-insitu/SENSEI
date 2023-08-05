#include "ParticleDensity.h"
#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "Profiler.h"
#include "SVTKUtils.h"
#include "SVTKDataAdaptor.h"
#include "Error.h"

#include <svtkObjectFactory.h>
#include <svtkDataObject.h>
#include <svtkImageData.h>
#include <svtkCompositeDataSet.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkTable.h>
#include <svtkPointData.h>
#include <svtkCompositeDataIterator.h>
#include <svtkSmartPointer.h>
#include <svtkUnsignedCharArray.h>
#include <svtkHAMRDataArray.h>

#if defined(SENSEI_ENABLE_CUDA)
#include "CUDAUtils.h"
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#endif

#include <algorithm>
#include <vector>
#include "sys/time.h"

namespace
{
// helper to print the name of the projection
const char *getProjectionName(int proj)
{
  if (proj == sensei::ParticleDensity::PROJECT_XY)
    return "XY";
  else if (proj == sensei::ParticleDensity::PROJECT_XZ)
    return "XZ";
  else if (proj == sensei::ParticleDensity::PROJECT_YZ)
    return "YZ";
  return "INVALID PROJECTION";
}

/*
// writes a PGM format file
template <typename data_t>
int writePgm(const char *fileName, const data_t *data, long xres, long yres)
{
  FILE *fh = fopen(fileName, "wb");
  if (!fh)
  {
    SENSEI_ERROR("Failed to open \"" << fileName << "\" for writing")
    return -1;
  }

  fprintf(fh, "P5\n%ld\n%ld\n65535\n", xres, yres);

  // find the max
  long xyres = xres*yres;
  float maxv = std::numeric_limits<float>::lowest();
  for (long i = 0; i < xyres; ++i)
    maxv = std::max<float>(maxv, data[i]);

  // allocate the image
  unsigned short *img = (unsigned short*)malloc(xyres*sizeof(data_t));

  // reverse and scale the image.
  long xyresm1 = xyres - 1;
  for (long i = 0; i < xyres; ++i)
    img[i] = float(data[xyresm1 - i]) / maxv * 65535.0f;

  if (fwrite(img, sizeof(unsigned short), xyres, fh) != (size_t)xyres)
  {
    SENSEI_ERROR("Write failed")
    return -1;
  }

  fclose(fh);
  free(img);
  return 0;
}
*/

#if defined(SENSEI_ENABLE_CUDA)
namespace CudaImpl
{
/** Computes 2D density projection on the GPU. The density must be
 * pre-initialized to zero multiple invokations of the kernel accumulate
 * results for new data.
 *
 * @param[in] posx        the x position of particles
 * @param[in] posy        the y position of particles
 * @param[in] mass        the particle mass
 * @param[in] nVals       the number of particles
 * @param[in] minx        the minimum in the x-direction
 * @param[in] miny        the minimum in the y-direction
 * @param[in] dx          the grid spacing in the x-direction
 * @param[in] dy          the grid spacing in the y-direction
 * @param[in,out] numDen  the number of particles in the grid cell
 * @param[in,out] massDen the total mass in the grid cell
 * @param[in] res         the grid resolution + 1 in all directions.
 */
template <typename data_t>
__global__
void density(const data_t *posx, const data_t *posy,
  const data_t *mass, long nVals, data_t minx, data_t miny,
  data_t dx, data_t dy, long *numDen, double *massDen,
  long resx, long resy)
{
  unsigned long q = sensei::CUDAUtils::ThreadIdToArrayIndex();

  if (q >= nVals)
    return;

  // look up the coordinates
  data_t x = posx[q];
  data_t y = posy[q];

  // calculate the cell the point falls into. the logic here handles a point
  // that is out of bounds by adding its contribution to the first or last bin.
  long i = 0;
  if (minx < x)
    i = (x - minx) / dx;
  if (i >= resx)
    i = resx - 1;

  long j = 0;
  if (miny < y)
    j = (y - miny) / dy;
  if (j >= resy)
    j = resy - 1;

  // the array index of the cell
  size_t qq = j * resx + i;

  // update the bin count
  atomicAdd((unsigned long long*)&(numDen[qq]), 1l);
  atomicAdd(&(massDen[qq]), mass[q]);

#if SENSEI_DEBUG > 1
  printf("%g,%g in bin %lu,%lu idx %lu\n", x,y, i,j, qq);
#endif
}

/** launch the density kernel */
template <typename data_t>
int blockDensity(cudaStream_t strm,
  const data_t *posx, const data_t *posy,
  const data_t *mass, long nVals, data_t minx, data_t miny,
  data_t dx, data_t dy, long *numDen, double *massDen,
  long resx, long resy)
{
  // determine kernel launch parameters
  dim3 blockGrid;
  int nBlocks = 0;
  dim3 threadGrid;
  if (sensei::CUDAUtils::PartitionThreadBlocks(0,
   nVals, 8, blockGrid, nBlocks, threadGrid))
  {
    SENSEI_ERROR("Failed to partition thread blocks")
    return -1;
  }

  // compute the histgram for this block's worth of data on the GPU. It is
  // left on the GPU until data for all blocks has been processed.
  density<<<blockGrid, threadGrid, 0, strm>>>(posx, posy, mass,
    nVals, minx, miny, dx, dy, numDen, massDen, resx, resy);

  return 0;
}
}
#endif

namespace CpuImpl
{
/** Computes 2D density projection on the Host. The density must be
 * pre-initialized to zero multiple invokations of the kernel accumulate
 * results for new data.
 *
 * @param[in] data        the array to calculate the density for
 * @param[in] nVals       the length of the array
 * @param[in] minVal      the minimum bin value
 * @param[in] spacing     the spacing of density bins
 * @param[in] resolution  the number of bins + 1.
 * @param[in,out] density the current density
 */
template <typename data_t>
int blockDensity(const data_t *posx, const data_t *posy,
  const data_t *mass, long nVals, data_t minx, data_t miny,
  data_t dx, data_t dy, long *numDen, double *massDen,
  long resx, long resy)
{
  for (long q = 0; q < nVals; ++q)
  {
    // look up the coordinates
    data_t x = posx[q];
    data_t y = posy[q];

    // calculate the cell the point falls into. the logic here handles a point
    // that is out of bounds by adding its contribution to the first or last bin.
    long i = 0;
    if (minx < x)
      i = (x - minx) / dx;
    if (i >= resx)
      i = resx - 1;

    long j = 0;
    if (miny < y)
      j = (y - miny) / dy;
    if (j >= resy)
      j = resy - 1;

    // array index of the cell
    size_t qq = j * resx + i;

    // update the bin count
    numDen[qq] += 1;
    massDen[qq] += mass[q];
  }
  return 0;
}
}
}

namespace sensei
{
//-----------------------------------------------------------------------------
senseiNewMacro(ParticleDensity);

//-----------------------------------------------------------------------------
ParticleDensity::ParticleDensity() : Projection(PROJECT_XY),
  XRes(256), YRes(-1), OutDir("./"), Iteration(0),
  MeshName(), XPosArray(), YPosArray(), ZPosArray(), MassArray(),
  ReturnData(0)
{
}

//-----------------------------------------------------------------------------
ParticleDensity::~ParticleDensity()
{
}

//-----------------------------------------------------------------------------
void ParticleDensity::Initialize(const std::string &meshName,
  const std::string &xPosArray, const std::string &yPosArray,
  const std::string &zPosArray, const std::string &massArray,
  const std::string proj, long xres, long yres, const std::string &outDir,
  int returnData)
{
  this->MeshName = meshName;
  this->XPosArray = xPosArray;
  this->YPosArray = yPosArray;
  this->ZPosArray = zPosArray;
  this->MassArray = massArray;

  if (((proj[0] == 'x') && (proj[1] == 'y')) || ((proj[0] == 'X') && (proj[1] == 'Y')))
    this->Projection = PROJECT_XY;
  else if (((proj[0] == 'x') && (proj[1] == 'z')) || ((proj[0] == 'X') && (proj[1] == 'Z')))
    this->Projection = PROJECT_XZ;
  else if (((proj[0] == 'y') && (proj[1] == 'z')) || ((proj[0] == 'Y') && (proj[1] == 'Z')))
    this->Projection = PROJECT_YZ;

  this->XRes = xres;
  this->YRes = yres;
  this->OutDir = outDir;
  this->ReturnData = returnData;
}

//-----------------------------------------------------------------------------
bool ParticleDensity::Execute(DataAdaptor* daIn, DataAdaptor** dataOut)
{
  TimeEvent<128> mark("ParticleDensity::Execute");

  int rank = 0;
  int n_ranks = 1;
  MPI_Comm comm = this->GetCommunicator();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);

  timeval startTime{};
  if (rank == 0)
    gettimeofday(&startTime, nullptr);

  if (dataOut)
    *dataOut = nullptr;

  // see what the simulation is providing
  MeshMetadataMap mdMap;

  MeshMetadataFlags mdFlags;
  mdFlags.SetBlockBounds();

  if (mdMap.Initialize(daIn, mdFlags))
  {
    SENSEI_ERROR("Failed to get metadata")
    return false;
  }

  // get the mesh metadata object
  MeshMetadataPtr mmd;
  if (mdMap.GetMeshMetadata(this->MeshName, mmd))
  {
    SENSEI_ERROR("Failed to get metadata for mesh \"" << this->MeshName << "\"")
    return false;
  }

  // get the global bounds and compute the resolution and spacing
  if (!mmd->GlobalView)
      mmd->GlobalizeView(comm);

  long xres = this->XRes;
  long yres = this->YRes;

  double minx = 0.;
  double maxx = 0.;

  double miny = 0.;
  double maxy = 0.;

  double dx = 0.;
  double dy = 0.;

  const char *xcoord = nullptr;
  const char *ycoord = nullptr;
  const char *massAr = this->MassArray.c_str();

  if (this->Projection == PROJECT_XY)
  {
    minx = mmd->Bounds[0];
    maxx = mmd->Bounds[1];
    miny = mmd->Bounds[2];
    maxy = mmd->Bounds[3];
    xcoord = this->XPosArray.c_str();
    ycoord = this->YPosArray.c_str();
  }
  else if (this->Projection == PROJECT_XZ)
  {
    minx = mmd->Bounds[0];
    maxx = mmd->Bounds[1];
    miny = mmd->Bounds[4];
    maxy = mmd->Bounds[5];
    xcoord = this->XPosArray.c_str();
    ycoord = this->ZPosArray.c_str();
  }
  else if (this->Projection == PROJECT_YZ)
  {
    minx = mmd->Bounds[2];
    maxx = mmd->Bounds[3];
    miny = mmd->Bounds[4];
    maxy = mmd->Bounds[5];
    xcoord = this->YPosArray.c_str();
    ycoord = this->ZPosArray.c_str();
  }

  dx = maxx - minx;
  dy = maxy - miny;

  if ((xres < 0) && (yres < 0))
      xres = 512;

  if (xres < 0)
      xres = yres * dx / dy;

  if (yres < 0)
      yres = xres * dy / dx;

  dx /= xres;
  dy /= yres;

  long xyres = xres * yres;

  // get the mesh object
  svtkDataObject *dobj = nullptr;
  if (daIn->GetMesh(this->MeshName, true, dobj))
  {
    SENSEI_ERROR("Failed to get mesh \"" << this->MeshName << "\"")
    return false;
  }

  // this lets one load balance across multiple GPU's and Host's
  // set -1 to execute on the Host and 0 to N_CUDA_DEVICES -1 to specify
  // the specific GPU to run on.
#if defined(SENSEI_ENABLE_CUDA)
  int deviceId = 0;
  const char *aDevId = getenv("SENSEI_DEVICE_ID");
  if (aDevId)
    deviceId = atoi(aDevId);
#else
  int deviceId = -1;
#endif

  // get the current time and step
  int step = daIn->GetDataTimeStep();
  double time = daIn->GetDataTime();

  if (!dobj)
  {
    // it is not an necessarilly an error if all ranks do not have
    // a dataset to process.
    return true;
  }

  // fetch the needed arrays from the simulation
  if (daIn->AddArray(dobj, this->MeshName, svtkDataObject::CELL, xcoord) ||
    daIn->AddArray(dobj, this->MeshName, svtkDataObject::CELL, ycoord) ||
    daIn->AddArray(dobj, this->MeshName, svtkDataObject::CELL, massAr))
  {
    SENSEI_ERROR(<< daIn->GetClassName()
      << " failed to fetch the position and mass arrays")

    MPI_Abort(comm, -1);
    return false;
  }

  // allocate arrays for the result
  svtkAllocator alloc = svtkAllocator::malloc;
  auto smode = svtkStreamMode::async;
#if defined(SENSEI_ENABLE_CUDA)
  cudaStream_t ndstr, mdstr, calcstr;
  cudaStreamCreate(&ndstr);
  cudaStreamCreate(&mdstr);
  if (deviceId >= 0)
  {
    alloc = svtkAllocator::cuda;
    sensei::CUDAUtils::SetDevice(deviceId);
    cudaStreamCreate(&calcstr);
  }
#else
  svtkStream ndstr, mdstr;
#endif
  auto numDen = svtkHAMRLongArray::New("number_density", xyres, 1, alloc, ndstr, smode, 0);
  auto massDen = svtkHAMRDoubleArray::New("mass_density", xyres, 1, alloc, mdstr, smode, 0.);

  // process the blocks of data
  long nVals = 0;
  svtkCompositeDataSetPtr mesh = SVTKUtils::AsCompositeData(comm, dobj, true);
  svtkSmartPointer<svtkCompositeDataIterator> iter;
  iter.TakeReference(mesh->NewIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    // get the next local block of data.
    auto curObj = iter->GetCurrentDataObject();

    // downcast to the block type. for simplicity of this example we require
    // tabular data.
    auto tab = dynamic_cast<svtkTable*>(curObj);
    if (!tab)
    {
      SENSEI_ERROR("Unsupported dataset type "
        << (curObj ? curObj->GetClassName() : "nullptr"))

      MPI_Abort(comm, -1);
      return false;
    }

    // get the positions and mass. for simplicity of this example we require
    // double precision arrays here.
    auto ax = dynamic_cast<svtkHAMRDoubleArray*>(tab->GetColumnByName(xcoord));
    auto ay = dynamic_cast<svtkHAMRDoubleArray*>(tab->GetColumnByName(ycoord));
    auto am = dynamic_cast<svtkHAMRDoubleArray*>(tab->GetColumnByName(massAr));

    if (!ax || !ay || !am)
    {
      SENSEI_ERROR("Failed to get arrays : " << xcoord << (ax ? "(N) " : "(Y) ")
        << ycoord << (ay ? "(N) " : "(Y) ") << massAr << (am ? "(N)" : "(Y)"))

      MPI_Abort(comm, -1);
      return false;
    }

    nVals = ax->GetNumberOfTuples();

#if defined(SENSEI_ENABLE_CUDA)
    if (deviceId >= 0)
    {
      // make sure the data is on the active GPU
      auto pax = ax->GetDeviceAccessible();
      auto pay = ay->GetDeviceAccessible();
      auto pam = am->GetDeviceAccessible();

      ax->Synchronize();
      ay->Synchronize();
      am->Synchronize();

      // compute the projections on the GPU
      if (::CudaImpl::blockDensity(calcstr, pax.get(), pay.get(), pam.get(),
        nVals, minx, miny, dx, dy, numDen->GetData(), massDen->GetData(),
        xres, yres))
      {
        SENSEI_ERROR("Failed to compute the density projection using CUDA")
        return false;
      }
    }
    else
    {
#endif
      // process on the Host
      auto pax = ax->GetHostAccessible();
      auto pay = ay->GetHostAccessible();
      auto pam = am->GetHostAccessible();

      ax->Synchronize();
      ay->Synchronize();
      am->Synchronize();

      // compute the projections on the GPU
      if (::CpuImpl::blockDensity(pax.get(), pay.get(), pam.get(),
        nVals, minx, miny, dx, dy, numDen->GetData(), massDen->GetData(),
        xres, yres))
      {
        SENSEI_ERROR("Failed to compute the density projection using the Host")
        return false;
      }
#if defined(SENSEI_ENABLE_CUDA)
    }
#endif
  }

  // move the results to the Host for comm and I/O
#if defined(SENSEI_ENABLE_CUDA)
  if (deviceId >= 0)
  {
    // move data back to the Host for I/O
    cudaStreamSynchronize(calcstr);

    alloc = svtkAllocator::cuda_host;

    numDen->SetAllocator(alloc);
    massDen->SetAllocator(alloc);

    numDen->Synchronize();
    massDen->Synchronize();

    cudaStreamDestroy(calcstr);
  }
  cudaStreamDestroy(ndstr);
  cudaStreamDestroy(mdstr);
#endif

  // accumulate contributions from all ranks
  if (rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, numDen->GetData(), xyres, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(MPI_IN_PLACE, massDen->GetData(), xyres, MPI_DOUBLE, MPI_SUM, 0, comm);
  }
  else
  {
    MPI_Reduce(numDen->GetData(), nullptr, xyres, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(massDen->GetData(), nullptr, xyres, MPI_DOUBLE, MPI_SUM, 0, comm);
  }

  // write the results
  if (rank == 0)
  {
    char fn[512];
    fn[511] = '\0';

    snprintf(fn, 511, "%s/%s_density_%06ld.vtk",
        this->OutDir.c_str(), this->MeshName.c_str(), this->Iteration);

    if (SVTKUtils::WriteVTK(fn, xres + 1, yres + 1, 1, minx, miny,
                            0., dx, dy, 1., {numDen, massDen}, {}))
    {
      SENSEI_ERROR("Failed to write file \"" << fn << "\"")
      return false;
    }

    this->Iteration += 1;
  }

  if (this->ReturnData && dataOut)
  {
    auto mbo = svtkMultiBlockDataSet::New();
    mbo->SetNumberOfBlocks(n_ranks);

    if (rank == 0)
    {
      auto imo = svtkImageData::New();
      imo->SetOrigin(minx, miny, 0.0);
      imo->SetSpacing(dx, dy, 0.0);
      imo->SetDimensions(xres, yres, 1);
      imo->GetPointData()->AddArray(massDen);
      imo->GetPointData()->AddArray(numDen);

      mbo->SetBlock(0, imo);

      imo->Delete();
    }

    auto va = SVTKDataAdaptor::New();
    va->SetDataObject(this->MeshName, mbo);

    mbo->Delete();

    *dataOut = va;
  }

  if (rank == 0)
  {
    timeval endTime{};
    gettimeofday(&endTime, nullptr);

    double runTimeUs = (endTime.tv_sec * 1e6 + endTime.tv_usec) -
      (startTime.tv_sec * 1e6 + startTime.tv_usec);

    SENSEI_STATUS("ParticleDensity: Step = " << step
      << " Time = " << time << " Completed " << xres << "x" << yres
      << " " << getProjectionName(this->Projection)
      << " projection of " << nVals << " particles using "
      << (deviceId < 0 ? "the host" : "CUDA GPU")
      << "(" << deviceId << ") in " << runTimeUs / 1e6 << " s")
  }

  numDen->Delete();
  massDen->Delete();

  return true;
}

//-----------------------------------------------------------------------------
int ParticleDensity::Finalize()
{
  return 0;
}

}
