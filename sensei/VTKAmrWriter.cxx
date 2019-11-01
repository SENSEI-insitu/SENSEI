#include "VTKAmrWriter.h"
#include "senseiConfig.h"
#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "VTKUtils.h"
#include "Error.h"

#include <vtkCompositeDataIterator.h>
#include <vtkDataObject.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkOverlappingAMR.h>
#include <vtkCompositeDataPipeline.h>
#include <vtkXMLPUniformGridAMRWriter.h>
#include <vtkAlgorithm.h>
#include <vtkMultiProcessController.h>
#include <vtkMPIController.h>
#include <vtkMPICommunicator.h>
#include <vtkMPI.h>

#include <algorithm>
#include <sstream>
#include <fstream>
#include <cassert>

#include <mpi.h>

static
std::string getFileName(const std::string &outputDir,
  const std::string &meshName, unsigned long fileId,
  const std::string &blockExt)
{
  std::ostringstream fss;
  fss << outputDir << "/" << meshName << "_"
    << std::setw(6) << std::setfill('0') << fileId << blockExt;
  return fss.str();
}


namespace sensei
{
//-----------------------------------------------------------------------------
senseiNewMacro(VTKAmrWriter);

//-----------------------------------------------------------------------------
VTKAmrWriter::VTKAmrWriter() : OutputDir("./"), Mode(MODE_PARAVIEW)
{}

//-----------------------------------------------------------------------------
VTKAmrWriter::~VTKAmrWriter()
{}

//-----------------------------------------------------------------------------
int VTKAmrWriter::Initialize()
{
  MPI_Comm comm = this->GetCommunicator();
  vtkMPICommunicatorOpaqueComm ocomm(&comm);

  vtkMPICommunicator *vcomm = vtkMPICommunicator::New();
  vcomm->InitializeExternal(&ocomm);

  vtkMPIController *controller = vtkMPIController::New();
  controller->Initialize(0,0,1);
  controller->SetCommunicator(vcomm);
  vcomm->Delete();

  vtkMultiProcessController::SetGlobalController(controller);

  vtkCompositeDataPipeline* cexec=vtkCompositeDataPipeline::New();
  vtkAlgorithm::SetDefaultExecutivePrototype(cexec);
  cexec->Delete();

  return 0;
}

//-----------------------------------------------------------------------------
int VTKAmrWriter::SetOutputDir(const std::string &outputDir)
{
  this->OutputDir = outputDir;
  return 0;
}

//-----------------------------------------------------------------------------
int VTKAmrWriter::SetMode(int mode)
{
  if (!(mode == VTKAmrWriter::MODE_VISIT) ||
    (mode == VTKAmrWriter::MODE_PARAVIEW))
    {
    SENSEI_ERROR("Invalid mode " << mode)
    return -1;
    }

  this->Mode = mode;
  return 0;
}

//-----------------------------------------------------------------------------
int VTKAmrWriter::SetMode(std::string modeStr)
{
  unsigned int n = modeStr.size();
  for (unsigned int i = 0; i < n; ++i)
    modeStr[i] = tolower(modeStr[i]);

  int mode = 0;
  if (modeStr == "visit")
    {
    mode = VTKAmrWriter::MODE_VISIT;
    }
  else if (modeStr == "paraview")
    {
    mode = VTKAmrWriter::MODE_PARAVIEW;
    }
  else
    {
    SENSEI_ERROR("invalid mode \"" << modeStr << "\"")
    return -1;
    }

  this->Mode = mode;
  return 0;
}

//-----------------------------------------------------------------------------
int VTKAmrWriter::SetDataRequirements(const DataRequirements &reqs)
{
  this->Requirements = reqs;
  return 0;
}

//-----------------------------------------------------------------------------
int VTKAmrWriter::AddDataRequirement(const std::string &meshName,
  int association, const std::vector<std::string> &arrays)
{
  this->Requirements.AddRequirement(meshName, association, arrays);
  return 0;
}

//-----------------------------------------------------------------------------
bool VTKAmrWriter::Execute(DataAdaptor* dataAdaptor)
{
  int rank = 0;
  MPI_Comm_rank(this->GetCommunicator(), &rank);

  // see what the simulation is providing
  MeshMetadataMap mdMap;
  if (mdMap.Initialize(dataAdaptor))
    {
    SENSEI_ERROR("Failed to get metadata")
    return false;
    }

  // if no dataAdaptor requirements are given, push all the data
  // fill in the requirements with every thing
  if (this->Requirements.Empty())
    {
    if (this->Requirements.Initialize(dataAdaptor, false))
      {
      SENSEI_ERROR("Failed to initialze dataAdaptor description")
      return false;
      }
    SENSEI_WARNING("No subset specified. Writing all available data")
    }

  MeshRequirementsIterator mit =
    this->Requirements.GetMeshRequirementsIterator();

  while (mit)
    {
    // get the mesh
    vtkDataObject* dobj = nullptr;
    std::string meshName = mit.MeshName();
    if (dataAdaptor->GetMesh(meshName, mit.StructureOnly(), dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      return false;
      }

    // make sure we have amr dataset
    if (!dynamic_cast<vtkOverlappingAMR*>(dobj))
      {
      SENSEI_ERROR("Data \"" << dobj->GetClassName() << "\" is not an AMR data set")
      return false;
      }

    MeshMetadataPtr metadata;
    if (mdMap.GetMeshMetadata(mit.MeshName(), metadata))
      {
      SENSEI_ERROR("Failed to get metadata for mesh \"" << mit.MeshName() << "\"")
      return false;
      }

    // add the ghost cell arrays to the mesh
    if (metadata->NumGhostCells &&
      dataAdaptor->AddGhostCellsArray(dobj, mit.MeshName()))
      {
      SENSEI_ERROR("Failed to get ghost cells for mesh \"" << mit.MeshName() << "\"")
      return false;
      }

    // add the ghost node arrays to the mesh
    if (metadata->NumGhostNodes &&
      dataAdaptor->AddGhostNodesArray(dobj, mit.MeshName()))
      {
      SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << mit.MeshName() << "\"")
      return false;
      }

    // add the required arrays
    ArrayRequirementsIterator ait =
      this->Requirements.GetArrayRequirementsIterator(meshName);

    while (ait)
      {
      if (dataAdaptor->AddArray(dobj, mit.MeshName(),
         ait.Association(), ait.Array()))
        {
        SENSEI_ERROR("Failed to add "
          << VTKUtils::GetAttributesName(ait.Association())
          << " data array \"" << ait.Array() << "\" to mesh \""
          << meshName << "\"")
        return false;
        }

      ++ait;
      }

    // initialize file id and time and step records
    if (this->HaveBlockInfo.count(meshName) == 0)
      {
      this->HaveBlockInfo[meshName] = 1;
      this->FileId[meshName] = 0;
      }

    // write to disk
    std::string fileName =
      getFileName(this->OutputDir, meshName, this->FileId[meshName], ".vth");

    vtkXMLPUniformGridAMRWriter *w = vtkXMLPUniformGridAMRWriter::New();
    w->SetInputData(dobj);
    w->SetFileName(fileName.c_str());
    w->Write();
    w->Delete();

    // update file id
    this->FileId[meshName] += 1;

    // record time and step info for meta files
    if (rank == 0)
      {
      double time = dataAdaptor->GetDataTime();
      this->Time[meshName].push_back(time);

      long step = dataAdaptor->GetDataTimeStep();
      this->TimeStep[meshName].push_back(step);
      }

    dobj->Delete();

    ++mit;
    }

  return true;
}

//-----------------------------------------------------------------------------
int VTKAmrWriter::Finalize()
{
  int rank = 0;
  MPI_Comm_rank(this->GetCommunicator(), &rank);

  // clean up VTK
  vtkMultiProcessController *controller =
    vtkMultiProcessController::GetGlobalController();

  controller->Finalize(1);
  controller->Delete();

  vtkMultiProcessController::SetGlobalController(nullptr);
  vtkAlgorithm::SetDefaultExecutivePrototype(nullptr);

  // rank 0 will write meta files
  if (rank != 0)
    return 0;

  std::vector<std::string> meshNames;
  this->Requirements.GetRequiredMeshes(meshNames);

  unsigned int nMeshes = meshNames.size();
  for (unsigned int i = 0; i < nMeshes; ++i)
    {
    const std::string &meshName = meshNames[i];

    if (this->HaveBlockInfo.find(meshName) == this->HaveBlockInfo.end())
      {
      SENSEI_ERROR("No blocks have been written for a mesh named \""
        << meshName << "\"")
      return -1;
      }

    std::vector<double> &times = this->Time[meshName];
    long nSteps = times.size();

    if (this->Mode == VTKAmrWriter::MODE_PARAVIEW)
      {
      std::string pvdFileName = this->OutputDir + "/" + meshName + ".pvd";
      ofstream pvdFile(pvdFileName);

      if (!pvdFile)
        {
        SENSEI_ERROR("Failed to open " << pvdFileName << " for writing")
        return -1;
        }

      pvdFile << "<?xml version=\"1.0\"?>" << endl
        << "<VTKFile type=\"Collection\" version=\"0.1\"" << endl
        << "  byte_order=\"LittleEndian\" compressor=\"\">" << endl
        << "<Collection>" << endl;

      for (long i = 0; i < nSteps; ++i)
        {
        std::string fileName =
          getFileName(this->OutputDir, meshName, i, ".vth");

        pvdFile << "<DataSet timestep=\"" << times[i]
          << "\" group=\"\" part=\"\" file=\"" << fileName
          << "\"/>" << endl;
        }

      pvdFile << "</Collection>" << endl
        << "</VTKFile>" << endl;
      }
    else if (this->Mode == VTKAmrWriter::MODE_VISIT)
      {
      std::string visitFileName = this->OutputDir + "/" + meshName + ".visit";
      ofstream visitFile(visitFileName);

      if (!visitFile)
        {
        SENSEI_ERROR("Failed to open " << visitFileName << " for writing")
        return -1;
        }

      visitFile << "!NBLOCKS " << 1 << endl;

      for (long i = 0; i < nSteps; ++i)
        {
        visitFile << "!TIME " << times[i] << endl;
        }

      for (long i = 0; i < nSteps; ++i)
        {
        std::string fileName =
          getFileName(this->OutputDir, meshName, i, ".vth");

        visitFile << fileName << endl;
        }
      }
    else
      {
      SENSEI_ERROR("Invalid mode \"" << this->Mode << "\"")
      return -1;
      }
    }

  return 0;
}

}
