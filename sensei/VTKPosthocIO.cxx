#include "VTKPosthocIO.h"
#include "senseiConfig.h"
#include "DataAdaptor.h"
#include "Error.h"

#include <vtkCellData.h>
#include <vtkCompositeDataIterator.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkDataArray.h>
#include <vtkDataArrayTemplate.h>
#include <vtkDataObject.h>
#include <vtkDataSetAttributes.h>
#include <vtkImageData.h>
#include <vtkPolyData.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkInformation.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>

#include <algorithm>
#include <sstream>
#include <fstream>
#include <cassert>

#include <vtkAlgorithm.h>
#include <vtkCompositeDataPipeline.h>
#include <vtkXMLDataSetWriter.h>

#include <mpi.h>

//-----------------------------------------------------------------------------
static
std::string getBlockExtension(vtkDataObject *dob)
{
  if (dynamic_cast<vtkPolyData*>(dob))
  {
    return ".vtp";
  }
  else if (dynamic_cast<vtkUnstructuredGrid*>(dob))
  {
    return ".vtu";
  }
  else if (dynamic_cast<vtkImageData*>(dob))
  {
    return ".vti";
  }
  else if (dynamic_cast<vtkRectilinearGrid*>(dob))
  {
    return ".vtr";
  }
  else if (dynamic_cast<vtkStructuredGrid*>(dob))
  {
    return ".vts";
  }
  else if (dynamic_cast<vtkMultiBlockDataSet*>(dob))
  {
    return ".vtm";
  }

  SENSEI_ERROR("Failed to determine file extension for \""
    << dob->GetClassName() << "\"")
  return "";
}

namespace sensei
{
//-----------------------------------------------------------------------------
senseiNewMacro(VTKPosthocIO);

//-----------------------------------------------------------------------------
VTKPosthocIO::VTKPosthocIO() : Comm(MPI_COMM_WORLD),
  OutputDir("./"), FileId(0), Mode(MODE_PARAVIEW), Period(1), HaveBlockInfo(0)
{}

//-----------------------------------------------------------------------------
VTKPosthocIO::~VTKPosthocIO()
{}

//-----------------------------------------------------------------------------
void VTKPosthocIO::Initialize(MPI_Comm comm, const std::string &outputDir,
  const std::string &fileName, const std::vector<std::string> &cellArrays,
  const std::vector<std::string> &pointArrays, int mode, int period)
{
  this->Comm = comm;
  this->OutputDir = outputDir;
  this->FileName = fileName;
  this->CellArrays = cellArrays;
  this->PointArrays = pointArrays;
  this->Mode = mode;
  this->Period = period;
}

//-----------------------------------------------------------------------------
bool VTKPosthocIO::Execute(DataAdaptor* data)
{
  vtkCompositeDataSet* cd =
    dynamic_cast<vtkCompositeDataSet*>(data->GetMesh(false));

  if (!cd)
    {
    SENSEI_ERROR("unsupported dataset type")
    return false;
    }

  int timeStep = data->GetDataTimeStep();
  if (timeStep%this->Period)
      return true;

  int rank = 0;
  MPI_Comm_rank(this->Comm, &rank);

  vtkCompositeDataIterator *it = cd->NewIterator();
  it->SetSkipEmptyNodes(1);

  // figure out block distribution, assume that it does not change, and
  // that block types are homgeneous
  if (!this->HaveBlockInfo)
    {
    if (!it->IsDoneWithTraversal())
      {
      this->BlockExt = getBlockExtension(it->GetCurrentDataObject());
      }

    long numBlocks = 0;
    for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
      {
      numBlocks += 1;
      }

    int nRanks = 1;
    MPI_Comm_size(this->Comm, &nRanks);
    this->NumBlocks.resize(nRanks);
    this->NumBlocks[rank] = numBlocks;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, this->NumBlocks.data(),
      1, MPI_LONG, this->Comm);

    this->BlockStarts.resize(nRanks,0);
    for (int i = 1; i < nRanks; ++i)
      {
      int ii = i - 1;
      this->BlockStarts[i] = this->BlockStarts[ii] + this->NumBlocks[ii];
      }

    this->HaveBlockInfo = 1;
    }

  // write the blocks
  vtkXMLDataSetWriter *writer = vtkXMLDataSetWriter::New();
  long blockId = this->BlockStarts[rank];
  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
    {
    std::ostringstream oss;
    oss << this->OutputDir << "/" << this->FileName << "_"
      << std::setw(5) << std::setfill('0') << blockId << "_"
      << std::setw(7) << std::setfill('0') << this->FileId
      << this->BlockExt;

    writer->SetInputData(it->GetCurrentDataObject());
    writer->SetDataModeToAppended();
    writer->EncodeAppendedDataOff();
    writer->SetCompressorTypeToNone();
    writer->SetFileName(oss.str().c_str());
    writer->Write();

    blockId += 1;
    }
  writer->Delete();

  this->FileId += 1;

  // rank 0 keeps track of time info for meta file
  if (rank == 0)
    {
    double time = data->GetDataTime();
    this->Time.push_back(time);

    long step = data->GetDataTimeStep();
    this->TimeStep.push_back(step);
    }

  return true;
}

//-----------------------------------------------------------------------------
int VTKPosthocIO::Finalize()
{
  int rank = 0;
  MPI_Comm_rank(this->Comm, &rank);

  if (rank != 0)
    return 0;

  int nRanks = 1;
  MPI_Comm_size(this->Comm, &nRanks);

  long nBlocks = 0;
  for (int i = 0; i < nRanks; ++i)
    nBlocks += this->NumBlocks[i];

  if (this->Mode == VTKPosthocIO::MODE_PARAVIEW)
  {
    std::string pvdFileName = this->OutputDir + "/" + this->FileName + ".pvd";
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

    long nSteps = this->Time.size();
    for (long i = 0; i < nSteps; ++i)
      {
      for (long j = 0; j < nBlocks; ++j)
        {
        std::ostringstream oss;
        oss << this->OutputDir << "/" << this->FileName << "_"
          << std::setw(5) << std::setfill('0') << j << "_"
          << std::setw(7) << std::setfill('0') << i << this->BlockExt;

        pvdFile << "<DataSet timestep=\"" << this->Time[i]
          << "\" group=\"\" part=\"" << j << "\" file=\"" << oss.str() << "\"/>" << endl;
        }
      }

    pvdFile << "</Collection>" << endl
      << "</VTKFile>" << endl;

    return true;
  }
  else if (this->Mode == VTKPosthocIO::MODE_VISIT)
  {
    std::string visitFileName = this->OutputDir + "/" + this->FileName + ".visit";
    ofstream visitFile(visitFileName);

    if (!visitFile)
      {
      SENSEI_ERROR("Failed to open " << visitFileName << " for writing")
      return false;
      }

    visitFile << "!NBLOCKS " << nBlocks << endl;
    long nSteps = this->Time.size();
    for (long i = 0; i < nSteps; ++i)
      {
      visitFile << "!TIME " << this->Time[i] << endl;
      }

    for (long i = 0; i < nSteps; ++i)
      {
      for (long j = 0; j < nBlocks; ++j)
        {
        std::ostringstream oss;
        oss << this->OutputDir << "/" << this->FileName << "_"
          << std::setw(5) << std::setfill('0') << j << "_"
          << std::setw(7) << std::setfill('0') << i << this->BlockExt;

        visitFile << oss.str() << endl;
        }
      }

    return true;
  }

  SENSEI_ERROR("Invalid mode \"" << this->Mode << "\"")
  return -1;
}

}
