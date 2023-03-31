#include "VTKPosthocIO.h"
#include "senseiConfig.h"
#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "SVTKUtils.h"
#include "Error.h"

#include <svtkCellData.h>
#include <svtkCompositeDataIterator.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkUniformGridAMR.h>
#include <svtkDataArray.h>
#include <svtkDataObject.h>
#include <svtkDataSetAttributes.h>
#include <svtkImageData.h>
#include <svtkPolyData.h>
#include <svtkRectilinearGrid.h>
#include <svtkStructuredGrid.h>
#include <svtkUnstructuredGrid.h>
#include <svtkObjectFactory.h>
#include <svtkPointData.h>
#include <svtkSmartPointer.h>

#include <algorithm>
#include <sstream>
#include <fstream>
#include <cassert>

#include <sys/stat.h>
#include <errno.h>
#include <string.h>

#include <vtkAlgorithm.h>
#include <vtkCompositeDataPipeline.h>
#include <vtkXMLDataSetWriter.h>
#include <vtkDataSetWriter.h>
#include <vtkDataSet.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>

#include <mpi.h>

//-----------------------------------------------------------------------------
static
std::string getBlockExtension(svtkDataObject *dob)
{
  if (dynamic_cast<svtkPolyData*>(dob))
  {
    return ".vtp";
  }
  else if (dynamic_cast<svtkUnstructuredGrid*>(dob))
  {
    return ".vtu";
  }
  else if (dynamic_cast<svtkImageData*>(dob))
  {
    return ".vti";
  }
  else if (dynamic_cast<svtkRectilinearGrid*>(dob))
  {
    return ".vtr";
  }
  else if (dynamic_cast<svtkStructuredGrid*>(dob))
  {
    return ".vts";
  }
  else if (dynamic_cast<svtkMultiBlockDataSet*>(dob))
  {
    return ".vtm";
  }

  SENSEI_ERROR("Failed to determine file extension for \""
    << dob->GetClassName() << "\"")
  return "";
}

//-----------------------------------------------------------------------------
static
std::string getBlockFileName(const std::string &outputDir,
  const std::string &meshName, long blockId, long fileId,
  const std::string &blockExt)
{
  std::ostringstream oss;

  oss << outputDir << "/" << meshName << "_"
    << std::setw(6) << std::setfill('0') << blockId << "_"
    << std::setw(6) << std::setfill('0') << fileId << blockExt;

  return oss.str();
}

namespace sensei
{
//-----------------------------------------------------------------------------
senseiNewMacro(VTKPosthocIO);

//-----------------------------------------------------------------------------
VTKPosthocIO::VTKPosthocIO() :
  Frequency(1), OutputDir("./"), Mode(MODE_PARAVIEW), Writer(WRITER_VTK_XML)
{}

//-----------------------------------------------------------------------------
VTKPosthocIO::~VTKPosthocIO()
{}

//-----------------------------------------------------------------------------
int VTKPosthocIO::SetOutputDir(const std::string &outputDir)
{
  int rank = 0;
  MPI_Comm_rank(this->GetCommunicator(), &rank);

  // rank 0 ensures that directory is present
  if (rank == 0)
    {
    int ierr = mkdir(outputDir.c_str(), S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH);
    if (ierr && (errno != EEXIST))
      {
      const char *estr = strerror(errno);
      SENSEI_ERROR("Directory \"" << outputDir
        << "\" does not exist and we could not create it. " << estr)
      return -1;
      }
    }

  this->OutputDir = outputDir;
  return 0;
}

//-----------------------------------------------------------------------------
int VTKPosthocIO::SetMode(int mode)
{
  if ((mode != VTKPosthocIO::MODE_VISIT) && (mode != VTKPosthocIO::MODE_PARAVIEW))
    {
    SENSEI_ERROR("Invalid mode " << mode)
    return -1;
    }

  this->Mode = mode;
  return 0;
}

//-----------------------------------------------------------------------------
int VTKPosthocIO::SetMode(std::string modeStr)
{
  unsigned int n = modeStr.size();
  for (unsigned int i = 0; i < n; ++i)
    modeStr[i] = tolower(modeStr[i]);

  int mode = 0;
  if (modeStr == "visit")
    {
    mode = VTKPosthocIO::MODE_VISIT;
    }
  else if (modeStr == "paraview")
    {
    mode = VTKPosthocIO::MODE_PARAVIEW;
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
int VTKPosthocIO::SetWriter(int writer)
{
  if ((writer != VTKPosthocIO::WRITER_VTK_LEGACY) &&
    (writer != VTKPosthocIO::WRITER_VTK_XML))
    {
    SENSEI_ERROR("Invalid writer " << writer)
    return -1;
    }

  this->Writer = writer;
  return 0;
}

//-----------------------------------------------------------------------------
int VTKPosthocIO::SetWriter(std::string writerStr)
{
  unsigned int n = writerStr.size();
  for (unsigned int i = 0; i < n; ++i)
    writerStr[i] = tolower(writerStr[i]);

  int writer = 0;
  if (writerStr == "legacy")
    {
    writer = VTKPosthocIO::WRITER_VTK_LEGACY;
    }
  else if (writerStr == "xml")
    {
    writer = VTKPosthocIO::WRITER_VTK_XML;
    }
  else
    {
    SENSEI_ERROR("invalid writer \"" << writerStr << "\"")
    return -1;
    }

  this->Writer = writer;
  return 0;
}

//-----------------------------------------------------------------------------
void VTKPosthocIO::SetGhostArrayName(const std::string &name)
{
  this->GhostArrayName = name;
}

//-----------------------------------------------------------------------------
std::string VTKPosthocIO::GetGhostArrayName()
{
  if (this->GhostArrayName.empty())
    {
    if (this->Mode == VTKPosthocIO::MODE_VISIT)
      return "avtGhostZones";

    if (this->Mode == VTKPosthocIO::MODE_PARAVIEW)
      return "vtkGhostType";
    }

  return this->GhostArrayName;
}

//-----------------------------------------------------------------------------
int VTKPosthocIO::SetDataRequirements(const DataRequirements &reqs)
{
  this->Requirements = reqs;
  return 0;
}

//-----------------------------------------------------------------------------
int VTKPosthocIO::SetFrequency(unsigned int frequency)
{
  this->Frequency = frequency;
  return 0;
}


//-----------------------------------------------------------------------------
int VTKPosthocIO::AddDataRequirement(const std::string &meshName,
  int association, const std::vector<std::string> &arrays)
{
  this->Requirements.AddRequirement(meshName, association, arrays);
  return 0;
}


//-----------------------------------------------------------------------------
bool VTKPosthocIO::Execute(DataAdaptor* dataIn, DataAdaptor** dataOut)
{
  // we do not return anything
  if (dataOut)
    {
    *dataOut = nullptr;
    }

  long step = dataIn->GetDataTimeStep();

  if ((this->Frequency > 0) && (step % this->Frequency != 0))
    {
    return true;
    }

  // see what the simulation is providing
  MeshMetadataFlags flags;
  flags.SetBlockDecomp();
  flags.SetBlockSize();

  MeshMetadataMap mdMap;
  if (mdMap.Initialize(dataIn, flags))
    {
    SENSEI_ERROR("Failed to get metadata")
    return false;
    }

  // if no dataIn requirements are given, push all the data
  // fill in the requirements with every thing
  if (this->Requirements.Empty())
    {
    if (this->Requirements.Initialize(dataIn, false))
      {
      SENSEI_ERROR("Failed to initialze dataIn description")
      return false;
      }

    if (this->GetVerbose())
      SENSEI_WARNING("No subset specified. Writing all available data")
    }

  MeshRequirementsIterator mit =
    this->Requirements.GetMeshRequirementsIterator();

  while (mit)
    {
    const std::string &meshName = mit.MeshName();
    // get the metadta
    MeshMetadataPtr mmd;
    if (mdMap.GetMeshMetadata(meshName, mmd))
      {
      SENSEI_ERROR("Failed to get metadata for mesh \"" << meshName << "\"")
      return false;
      }

    // generate a global view of the metadata.
    if (!mmd->GlobalView)
      mmd->GlobalizeView(this->GetCommunicator());

    // get the mesh
    svtkDataObject* dobj = nullptr;
    if (dataIn->GetMesh(meshName, mit.StructureOnly(), dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      return false;
      }

    // add the ghost cell arrays to the mesh
    if ((mmd->NumGhostCells || SVTKUtils::AMR(mmd)) &&
      dataIn->AddGhostCellsArray(dobj, meshName))
      {
      SENSEI_ERROR("Failed to get ghost cells for mesh \"" << meshName << "\"")
      return false;
      }

    // add the ghost node arrays to the mesh
    if (mmd->NumGhostNodes && dataIn->AddGhostNodesArray(dobj, meshName))
      {
      SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << meshName << "\"")
      return false;
      }

    // add the required arrays
    ArrayRequirementsIterator ait =
      this->Requirements.GetArrayRequirementsIterator(meshName);

    while (ait)
      {
      if (dataIn->AddArray(dobj, mit.MeshName(),
         ait.Association(), ait.Array()))
        {
        SENSEI_ERROR("Failed to add "
          << SVTKUtils::GetAttributesName(ait.Association())
          << " data array \"" << ait.Array() << "\" to mesh \""
          << meshName << "\"")
        return false;
        }
      ++ait;
      }

    // This class does not use SVTK's parallel writers because at this
    // time those writers gather some data to rank 0 and this results
    // in OOM crashes when run with 45k cores on Cori.

    // make sure we have composite dataset if not create one
    svtkCompositeDataSetPtr cd =
      SVTKUtils::AsCompositeData(this->GetCommunicator(), dobj, false);

    svtkCompositeDataIterator *it = cd->NewIterator();
    it->SetSkipEmptyNodes(1);
    it->InitTraversal();

    // figure out block distribution, assume that it does not change, and
    // that block types are homgeneous
    if (!it->IsDoneWithTraversal() && !this->HaveBlockInfo[meshName])
      {
      this->BlockExt[meshName] = this->Writer == VTKPosthocIO::WRITER_VTK_LEGACY ?
        ".svtk" : getBlockExtension(it->GetCurrentDataObject());

      this->HaveBlockInfo[meshName] = 1;
      }

    // amr meshes indices start from 0 while multiblock starts at 1
    long bidShift = 1;
    if (dynamic_cast<svtkUniformGridAMR*>(cd.GetPointer()))
      bidShift = 0;

    // write the blocks
    for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
      {
      svtkDataSet *ds = dynamic_cast<svtkDataSet*>(it->GetCurrentDataObject());
      if (!ds)
        {
        // this should never happen
        SENSEI_ERROR("Block at " << it->GetCurrentFlatIndex() << " is null")
        return false;
        }
      // skip writing blocks that have no data
      if (ds->GetNumberOfCells() < 1 && ds->GetNumberOfPoints() < 1)
        continue;

      long blockId = it->GetCurrentFlatIndex() - bidShift;

      if (blockId < 0)
        {
        // this should never happen
        SENSEI_ERROR("Negative index! Dataset is " << cd->GetClassName())
        return false;
        }

      std::string fileName =
        getBlockFileName(this->OutputDir, meshName, blockId,
          this->FileId[meshName], this->BlockExt[meshName]);


      // convert from SVTK to VTK
      vtkDataSet *vds = SVTKUtils::VTKObjectFactory::New(ds);
      vtkDataArray *ga = vds->GetCellData()->GetArray("vtkGhostType");
      if (ga)
        {
        ga->SetName(this->GetGhostArrayName().c_str());
#if VTK_MAJOR_VERSION < 9 ||                                       \
    (VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION < 2) ||           \
    (VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION == 2 && VTK_BUILD_VERSION < 20220823)
        // deprecation happen after VTK 9.2 but before build 20220823
        // which is the VTK version included in ParaView
        vds->UpdateCellGhostArrayCache();
#endif
        }

      if (this->Writer == VTKPosthocIO::WRITER_VTK_LEGACY)
        {
        vtkDataSetWriter *writer = vtkDataSetWriter::New();
        writer->SetInputData(vds);
        writer->SetFileName(fileName.c_str());
        writer->SetFileTypeToBinary();
        writer->Write();
        writer->Delete();
        }
      else
        {
        vtkXMLDataSetWriter *writer = vtkXMLDataSetWriter::New();
        writer->SetInputData(vds);
        writer->SetDataModeToAppended();
        writer->EncodeAppendedDataOff();
        writer->SetCompressorTypeToNone();
        writer->SetFileName(fileName.c_str());
        writer->Write();
        writer->Delete();
        }

      vds->Delete();
      }
    it->Delete();

    // we count empty steps
    NameMap<long>::iterator fidIt = this->FileId.find(meshName);
    if (fidIt == this->FileId.end())
        this->FileId[meshName] = 0;
    else
        fidIt->second += 1;

    // rank 0 keeps track of time info for meta file
    int rank = 0;
    MPI_Comm_rank(this->GetCommunicator(), &rank);

    if (rank == 0)
      {
      double time = dataIn->GetDataTime();
      this->Time[meshName].push_back(time);

      long step = dataIn->GetDataTimeStep();
      this->TimeStep[meshName].push_back(step);

      this->Metadata[meshName].push_back(mmd);
      }

    dobj->Delete();

    ++mit;
    }

  dataIn->ReleaseData();

  return true;
}

//-----------------------------------------------------------------------------
int VTKPosthocIO::Finalize()
{
  int rank = 0;
  MPI_Comm_rank(this->GetCommunicator(), &rank);

  if (rank != 0)
    return 0;

  int nRanks = 1;
  MPI_Comm_size(this->GetCommunicator(), &nRanks);

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

    const std::vector<MeshMetadataPtr> &mmd = this->Metadata[meshName];

    std::vector<double> &times = this->Time[meshName];
    long nSteps = times.size();

    std::string &blockExt = this->BlockExt[meshName];

    if (this->Mode == VTKPosthocIO::MODE_PARAVIEW)
      {
      std::string pvdFileName = this->OutputDir + "/" + meshName + ".pvd";
      std::ofstream pvdFile(pvdFileName);

      if (!pvdFile)
        {
        SENSEI_ERROR("Failed to open " << pvdFileName << " for writing")
        return -1;
        }

      pvdFile << "<?xml version=\"1.0\"?>" << endl
        << "<VTKFile type=\"Collection\" version=\"0.1\""
           " byte_order=\"LittleEndian\" compressor=\"\">" << endl
        << "<Collection>" << endl;

      for (long i = 0; i < nSteps; ++i)
        {
        for (long j = 0, k = 0; j < mmd[i]->NumBlocks; ++j)
          {
          if (mmd[i]->BlockNumCells[j] > 0)
            {
            std::string fileName =
              getBlockFileName("./", meshName, mmd[i]->BlockIds[j], i, blockExt);

            pvdFile << "<DataSet timestep=\"" << times[i]
              << "\" group=\"\" part=\"" << k << "\" file=\"" << fileName
              << "\"/>" << endl;

            ++k;
            }
          }
        }

      pvdFile << "</Collection>" << endl
        << "</VTKFile>" << endl;

      return 0;
      }
    else if (this->Mode == VTKPosthocIO::MODE_VISIT)
      {
      // does the number of blocks change?
      // if so dump one visit file per timestep, otherwise one visit file for
      // the series
      int staticMesh = 1;
      int nBlocks = mmd[0]->NumBlocks;
      for (long i = 0; staticMesh && (i < nSteps); ++i)
        {
        if (nBlocks != mmd[i]->NumBlocks)
          staticMesh = 0;

        for (long j = 0; j < mmd[i]->NumBlocks; ++j)
          {
          if (mmd[i]->BlockNumCells[j] < 1)
            staticMesh = 0;
          }
        }

      if (staticMesh)
        {
        // write a single .visit file for the time series
        std::string visitFileName = this->OutputDir + "/" + meshName + ".visit";
        std::ofstream visitFile(visitFileName);

        if (!visitFile)
          {
          SENSEI_ERROR("Failed to open " << visitFileName << " for writing")
          return -1;
          }

        visitFile << "!NBLOCKS " << mmd[0]->NumBlocks << std::endl;

        for (long i = 0; i < nSteps; ++i)
          visitFile << "!TIME " << times[i] << std::endl;

        for (long i = 0; i < nSteps; ++i)
          {
          for (long j = 0; j < mmd[i]->NumBlocks; ++j)
            {
            std::string fileName =
              getBlockFileName("./", meshName, mmd[i]->BlockIds[j], i, blockExt);

            visitFile << fileName << std::endl;
            }
          }

        visitFile.close();
        }
      else
        {
        // write a .visit file per step
        for (long i = 0; i < nSteps; ++i)
          {
          if (mmd[i]->NumBlocks < 1)
            continue;

          long numActiveBlocks = 0;
          for (long j = 0; j < mmd[i]->NumBlocks; ++j)
            if (mmd[i]->BlockNumCells[j] > 0)
                ++numActiveBlocks;

          if (numActiveBlocks < 1)
            continue;

          std::ostringstream oss;
          oss << this->OutputDir << "/" << meshName << "_"
            <<  std::setw(5) << std::setfill('0') << i << ".visit";

          std::string visitFileName = oss.str();

          std::ofstream visitFile(visitFileName);
          if (!visitFile)
            {
            SENSEI_ERROR("Failed to open \"" << visitFileName << "\" for writing")
            return -1;
            }

          visitFile << "!NBLOCKS " << numActiveBlocks << std::endl;
          visitFile << "!TIME " << times[i] << std::endl;

          for (long j = 0; j < mmd[i]->NumBlocks; ++j)
            {
            if (mmd[i]->BlockNumCells[j] > 0)
              {
              std::string fileName =
                getBlockFileName("./", meshName, mmd[i]->BlockIds[j], i, blockExt);

              visitFile << fileName << std::endl;
              }
            }

          visitFile.close();
          }
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
