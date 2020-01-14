#include "Oscillator.h"

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include <sdiy/point.hpp>

static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(),
        std::find_if(s.begin(), s.end(),
        [](int c) -> bool { return !std::isspace(c); }));
    return s;
}

static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](int c) -> bool { return !std::isspace(c); }).base(), s.end());
    return s;
}

static inline std::string &trim(std::string &s)  { return ltrim(rtrim(s)); }

std::vector<Oscillator> read_oscillators(std::string fn)
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

//-----------------------------------------------------------------------------
#include <VTKDataAdaptor.h>
#include <MeshMetadata.h>
#include <MeshMetadataMap.h>
#include <vtkPolyData.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkSmartPointer.h>

sensei::DataAdaptor* new_adaptor(sdiy::mpi::communicator& world, const std::vector<Oscillator>& oscillators)
{
  vtkNew<vtkMultiBlockDataSet> mb;
  mb->SetNumberOfBlocks(1);

  if (world.rank() > 0)
  {
    auto data = sensei::VTKDataAdaptor::New();
    data->SetDataObject("oscillators", mb);
    return data;
  }

  const vtkIdType numPts = static_cast<vtkIdType>(oscillators.size());

  vtkNew<vtkPolyData> pd;
  vtkNew<vtkPoints> pts;
  pts->SetDataTypeToFloat();

  vtkNew<vtkFloatArray> radius;
  radius->SetNumberOfComponents(1);
  radius->SetNumberOfTuples(numPts);
  radius->SetName("radius");

  vtkNew<vtkFloatArray> omega0;
  omega0->SetNumberOfComponents(1);
  omega0->SetNumberOfTuples(numPts);
  omega0->SetName("omega0");

  vtkNew<vtkFloatArray> zeta;
  zeta->SetNumberOfComponents(1);
  zeta->SetNumberOfTuples(numPts);
  zeta->SetName("zeta");

  vtkNew<vtkIntArray> type;
  type->SetNumberOfComponents(1);
  type->SetNumberOfTuples(numPts);
  type->SetName("type");

  pts->SetNumberOfPoints(numPts);
  for (vtkIdType cc=0; cc < numPts; ++cc)
  {
    pts->SetPoint(cc, oscillators[cc].center[0], oscillators[cc].center[1], oscillators[cc].center[2]);
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

  auto data = sensei::VTKDataAdaptor::New();
  data->SetDataObject("oscillators", mb);
  return data;
}

std::vector<Oscillator> read_oscillators(sensei::DataAdaptor* data)
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
    vtkDataObject* mesh;
    if (data->GetMesh("oscillators", false, mesh) ||
        data->AddArrays(mesh, "oscillators", vtkDataObject::POINT, mmd->ArrayName))
      {
      SENSEI_ERROR(<< data->GetClassName() << " failed to get mesh or add arrays.");
      return oscillators;
      }

    auto pd = vtkPolyData::SafeDownCast(vtkMultiBlockDataSet::SafeDownCast(mesh)->GetBlock(0));
    auto radius = pd->GetPointData()->GetArray("radius");
    auto omega0 = pd->GetPointData()->GetArray("omega0");
    auto zeta =   pd->GetPointData()->GetArray("zeta");
    auto type = pd->GetPointData()->GetArray("type");
    for (vtkIdType cc=0, numPts = pd->GetNumberOfPoints(); cc < numPts; ++cc)
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
      oscillators.emplace_back(Oscillator { {x,y,z}, fradius, fomega0, fzeta, itype });
    }

    mesh->Delete();
    }
  return oscillators;
}

