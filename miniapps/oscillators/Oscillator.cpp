#include "Oscillator.h"

#include <SVTKDataAdaptor.h>
#include <MeshMetadata.h>
#include <MeshMetadataMap.h>

#include <svtkPolyData.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkPoints.h>
#include <svtkPointData.h>
#include <svtkFloatArray.h>
#include <svtkIntArray.h>
#include <svtkSmartPointer.h>

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include <sdiy/point.hpp>

namespace
{
// **************************************************************************
static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(),
        std::find_if(s.begin(), s.end(),
        [](int c) -> bool { return !std::isspace(c); }));
    return s;
}

// **************************************************************************
static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](int c) -> bool { return !std::isspace(c); }).base(), s.end());
    return s;
}

// **************************************************************************
static inline std::string &trim(std::string &s)  { return ltrim(rtrim(s)); }

// **************************************************************************
std::vector<Oscillator> read(const std::string &fn)
{
    std::vector<Oscillator> res;

    std::ifstream in(fn);
    if (!in)
        throw std::runtime_error("Unable to open " + fn);
    std::string line;
    while(std::getline(in, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);

        std::string stype;
        iss >> stype;

        auto type = Oscillator::periodic;
        if (stype == "damped")
            type = Oscillator::damped;
        else if (stype == "decaying")
            type = Oscillator::decaying;

        float x,y,z;
        iss >> x >> y >> z;

        float r, omega0, zeta=0.0f;
        iss >> r >> omega0;

        if (type == Oscillator::damped)
            iss >> zeta;
        res.emplace_back(Oscillator { x,y,z, r, omega0, zeta, type });
    }
    return res;
}

// **************************************************************************
std::vector<Oscillator> fetch(sensei::DataAdaptor *data)
{
  std::vector<Oscillator> oscillators;

  sensei::MeshMetadataMap mdMap;
  if (mdMap.Initialize(data))
    {
    SENSEI_ERROR("Failed to get metadata")
    return oscillators;
    }

  // get the mesh metadata object
  sensei::MeshMetadataPtr mmd;
  if (mdMap.GetMeshMetadata("oscillators", mmd))
    {
    SENSEI_ERROR("Failed to get metadata for mesh \"oscillators\"")
    return oscillators;
    }

  if (mmd->NumBlocksLocal == std::vector<int>{ 1 })
    {
    svtkDataObject* mesh;
    if (data->GetMesh("oscillators", false, mesh) ||
        data->AddArrays(mesh, "oscillators", svtkDataObject::POINT, mmd->ArrayName))
      {
      SENSEI_ERROR(<< data->GetClassName() << " failed to get mesh or add arrays.");
      return oscillators;
      }

    auto pd = svtkPolyData::SafeDownCast(svtkMultiBlockDataSet::SafeDownCast(mesh)->GetBlock(0));
    auto radius = pd->GetPointData()->GetArray("radius");
    auto omega0 = pd->GetPointData()->GetArray("omega0");
    auto zeta =   pd->GetPointData()->GetArray("zeta");
    auto type = pd->GetPointData()->GetArray("type");
    for (size_t cc=0, numPts = pd->GetNumberOfPoints(); cc < numPts; ++cc)
    {
      double pt[3];
      pd->GetPoint(cc, pt);

      float x = static_cast<float>(pt[0]);
      float y = static_cast<float>(pt[1]);
      float z = static_cast<float>(pt[2]);
      float fradius = static_cast<float>(radius->GetTuple1(cc));
      float fomega0 = static_cast<float>(omega0->GetTuple1(cc));
      float fzeta = static_cast<float>(zeta->GetTuple1(cc));
      Oscillator::Type itype = static_cast<Oscillator::Type>(static_cast<int>(type->GetTuple1(cc)));

      oscillators.emplace_back(Oscillator { x,y,z, fradius, fomega0, fzeta, itype });
    }

    mesh->Delete();
    }

  return oscillators;
}

// **************************************************************************
OscillatorArray bcast(const sdiy::mpi::communicator &comm,
  std::vector<Oscillator>& oscillatorsIn)
{
  OscillatorArray oscillatorsOut;

  if (comm.rank() == 0)
  {
    // distribute
    unsigned long n = oscillatorsIn.size();
    unsigned long nBytes = n*sizeof(Oscillator);

    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, comm);
    MPI_Bcast((void*)oscillatorsIn.data(), nBytes, MPI_BYTE, 0, comm);

    // move to buffer memory
    oscillatorsOut.Allocate(n);
    memcpy(oscillatorsOut.Data(), oscillatorsIn.data(), nBytes);
  }
  else
  {
    // recieve the number of oscillatorsOut
    unsigned long n = 0;
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, comm);

    // allocate the buffer for the oscillator array
    oscillatorsOut.Allocate(n);

    // recieve the list of oscillatorsOut
    unsigned long nBytes = n*sizeof(Oscillator);
    MPI_Bcast(oscillatorsOut.Data(), nBytes, MPI_BYTE, 0, comm);
  }

  return oscillatorsOut;
}
}

// --------------------------------------------------------------------------
void OscillatorArray::Clear()
{
  mSize = 0;
  mData = nullptr;
}

// --------------------------------------------------------------------------
void OscillatorArray::Allocate(unsigned long n)
{
  mSize = n;

  size_t nBytes = n*sizeof(Oscillator);
#if defined(OSCILLATOR_CUDA)
  Oscillator *pTmpOsc = nullptr;
  cudaMallocManaged(&pTmpOsc, nBytes);
  mData = std::shared_ptr<Oscillator>(pTmpOsc, cudaFree);
#else
  mData = std::shared_ptr<Oscillator>((Oscillator*)malloc(nBytes), free);
#endif
}

// --------------------------------------------------------------------------
void OscillatorArray::Initialize(const sdiy::mpi::communicator &comm,
  const std::string &fileName)
{
  std::vector<Oscillator> tmp;

  if (comm.rank() == 0)
    tmp = ::read(fileName);

  *this = bcast(comm, tmp);
}

// --------------------------------------------------------------------------
void OscillatorArray::Initialize(const sdiy::mpi::communicator &comm,
  sensei::DataAdaptor *da)
{
  std::vector<Oscillator> tmp;

  if (comm.rank() == 0)
    tmp = ::fetch(da);

  *this = bcast(comm, tmp);
}

// --------------------------------------------------------------------------
void OscillatorArray::Print(std::ostream &os) const
{
  os << mSize << " Oscillators" << std::endl;
  for (unsigned long i = 0; i < mSize; ++i)
  {
    Oscillator &o = (mData.get())[i];
    os << i << " center = " << o.center_x << ", " << o.center_y << ", "
      << o.center_z << " radius = " << o.radius << " omega0 = "
      << o.omega0 << " zeta = " << o.zeta << std::endl;
  }
}





// **************************************************************************
sensei::DataAdaptor* new_adaptor(const sdiy::mpi::communicator& comm,
  const std::vector<Oscillator>& oscillators)
{
  svtkNew<svtkMultiBlockDataSet> mb;
  mb->SetNumberOfBlocks(1);

  if (comm.rank() > 0)
  {
    auto data = sensei::SVTKDataAdaptor::New();
    data->SetDataObject("oscillators", mb);
    return data;
  }

  size_t numPts = oscillators.size();

  svtkNew<svtkPolyData> pd;
  svtkNew<svtkPoints> pts;
  pts->SetDataTypeToFloat();

  svtkNew<svtkFloatArray> radius;
  radius->SetNumberOfComponents(1);
  radius->SetNumberOfTuples(numPts);
  radius->SetName("radius");

  svtkNew<svtkFloatArray> omega0;
  omega0->SetNumberOfComponents(1);
  omega0->SetNumberOfTuples(numPts);
  omega0->SetName("omega0");

  svtkNew<svtkFloatArray> zeta;
  zeta->SetNumberOfComponents(1);
  zeta->SetNumberOfTuples(numPts);
  zeta->SetName("zeta");

  svtkNew<svtkIntArray> type;
  type->SetNumberOfComponents(1);
  type->SetNumberOfTuples(numPts);
  type->SetName("type");

  pts->SetNumberOfPoints(numPts);
  for (size_t cc=0; cc < numPts; ++cc)
  {
    pts->SetPoint(cc, oscillators[cc].center_x,
      oscillators[cc].center_y, oscillators[cc].center_z);

    radius->SetTypedComponent(cc, 0, oscillators[cc].radius);
    omega0->SetTypedComponent(cc, 0, oscillators[cc].omega0);
    zeta->SetTypedComponent(cc, 0, oscillators[cc].zeta);
    type->SetTypedComponent(cc, 0, static_cast<int>(oscillators[cc].type));
  }

  pd->GetPointData()->AddArray(radius);
  pd->GetPointData()->AddArray(omega0);
  pd->GetPointData()->AddArray(zeta);
  pd->GetPointData()->AddArray(type);
  pd->SetPoints(pts);
  mb->SetBlock(0, pd);

  auto data = sensei::SVTKDataAdaptor::New();
  data->SetDataObject("oscillators", mb);
  return data;
}
