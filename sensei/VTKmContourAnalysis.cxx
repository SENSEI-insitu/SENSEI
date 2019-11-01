#include "VTKmContourAnalysis.h"

#include "DataAdaptor.h"
#include "Profiler.h"
#include "Error.h"

#include <vtkObjectFactory.h>
#include <vtkmAverageToPoints.h>
#include <vtkmContour.h>
#include <vtkNew.h>
#include <vtkDataObject.h>
#include <vtkSmartPointer.h>
#include <vtkCompositeDataIterator.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataSet.h>
#include <vtkMPIController.h>
#include <vtkMPICommunicator.h>
#include <vtkMPI.h>
#include <vtkXMLPMultiBlockDataWriter.h>
#include <vtkCompositeDataIterator.h>
#include <vtkImageData.h>
#include <vtkStructuredGridConnectivity.h>
#include <vtkStructuredExtent.h>
#include <vtkExtractVOI.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkStructuredPointsReader.h>
#include <vtkStructuredPoints.h>

#include <diy/master.hpp>
#include <diy/mpi.hpp>

#include <algorithm>
#include <vector>
#include <sstream>
#include <map>


namespace sensei
{

#if VTK_MAJOR_VERSION == 6 && VTK_MINOR_VERSION == 1
VTKmContourAnalysis *VTKmContourAnalysis::New() { return new VTKmContourAnalysis; }
#else
vtkStandardNewMacro(VTKmContourAnalysis);
#endif

//-----------------------------------------------------------------------------
VTKmContourAnalysis::VTKmContourAnalysis()
{
}

//-----------------------------------------------------------------------------
VTKmContourAnalysis::~VTKmContourAnalysis()
{
}

//-----------------------------------------------------------------------------
void VTKmContourAnalysis::Initialize(const std::string& meshName,
  const std::string& arrayName, double value, bool writeOutput)
{
  this->MeshName = meshName;
  this->ArrayName = arrayName;
  this->Value = value;
  this->WriteOutput = writeOutput;
}

struct Block
{
  Block(int* extent, int gl, vtkImageData* data,  vtkImageData* ghosted,
        vtkStructuredGridConnectivity* con)
       : GhostLevels(gl), Data(data), GhostedData(ghosted), Connectivity(con), Neighbors(0)
  {
    memcpy(this->Extent, extent, 6*sizeof(int));
  }

  ~Block()
  {
    delete this->Neighbors;
  }

  int Extent[6];
  int GhostLevels;
  vtkImageData* Data;
  vtkImageData* GhostedData;
  vtkStructuredGridConnectivity* Connectivity;
  std::map<vtkIdType, vtkStructuredNeighbor>* Neighbors;
};

void SendGhostRegion(Block* b, const diy::Master::ProxyWithLink& cp,
  std::vector<int>& ext, diy::BlockID& target)
{
  vtkNew<vtkExtractVOI> evoi;
  evoi->SetInputData(b->GhostedData);
  evoi->SetVOI(&ext[0]);
  evoi->Update();

  vtkNew<vtkStructuredPointsWriter> writer;
  writer->SetWriteToOutputString(1);
  writer->SetFileTypeToBinary();
  writer->SetInputData(evoi->GetOutput());
  writer->SetWriteExtent(true);
  writer->Write();

  unsigned char* buffer = writer->GetBinaryOutputString();
  int len = writer->GetOutputStringLength();

  std::vector<unsigned char> bufferv(buffer, buffer+len);

  cp.enqueue(target, bufferv);
}

void SendGhosts(Block* b,
                const diy::Master::ProxyWithLink& cp,
                void*)
{
  diy::Link*    l = cp.link();
  for (int i = 0; i < l->size(); ++i)
    {
    vtkIdType neighborID = l->target(i).gid;
    vtkStructuredNeighbor& ninfo = (*b->Neighbors)[neighborID];
    const int* sExt = ninfo.OverlapExtent;
    std::vector<int> sExtC(sExt, sExt+6);
    vtkStructuredExtent::Clamp(&sExtC[0], b->Data->GetExtent());
    SendGhostRegion(b, cp, sExtC, l->target(i));
    }

}

void ReceiveGhosts(Block* b,
                   const diy::Master::ProxyWithLink& cp,
                   void*)
{
  vtkNew<vtkStructuredPointsReader> reader;

  // gids of incoming neighbors in the link
  std::vector<int> in;
  cp.incoming(in);

  for (unsigned int i = 0; i < in.size(); ++i)
    {
    std::vector<unsigned char> buffer;
    cp.dequeue(in[i], buffer);
    reader->SetInputString(reinterpret_cast<char*>(&buffer[0]), buffer.size());
    reader->SetReadFromInputString(1);
    reader->Update();

    // Here we copy the ghost cells values to the output
    // dataset. For now, we copy only cell data.
    vtkImageData* ghosted = b->GhostedData;

    vtkImageData* ghosts = reader->GetOutput();

    // Extents are based on points so convert them
    // to cell extents.
    int gExt[6];
    ghosted->GetExtent(gExt);
    int recvExt[6];
    ghosts->GetExtent(recvExt);
    for (unsigned int i=0; i<3; i++)
      {
      gExt[2*i+1]--;
      recvExt[2*i+1]--;
      }

    ghosted->GetCellData()->CopyStructuredData(
      ghosts->GetCellData(), recvExt, gExt, false);
    }

}

// This method exchanges 2 layers of ghost cells between
// blocks. A new set of datasets with ghost levels is
// returned
vtkSmartPointer<vtkMultiBlockDataSet> ExchangeGhosts(
  vtkDataObject* mesh, vtkMPIController* contr)
{
  vtkMultiBlockDataSet* mb = vtkMultiBlockDataSet::SafeDownCast(
    mesh);
  if (!mb)
    {
    return nullptr;
    }

  vtkSmartPointer<vtkCompositeDataIterator> iter;
  iter.TakeReference(mb->NewIterator());

  // Create a flat vector of datasets for simplicity.
  std::vector<vtkSmartPointer<vtkImageData> > datasets;
  iter->InitTraversal();
  while (!iter->IsDoneWithTraversal())
    {
    vtkImageData* cur = vtkImageData::SafeDownCast(
      iter->GetCurrentDataObject());
    if (cur)
      {
      datasets.push_back(cur);
      }
    iter->GoToNextItem();
    }

  vtkIdType nblocks = datasets.size();
  int nranks = contr->GetNumberOfProcesses();
  int rank = contr->GetLocalProcessId();

  // Find out how many blocks all processes have.
  std::vector<vtkIdType> allnblocks(nranks);
  contr->AllGather(&nblocks, &allnblocks[0], 1);

  // We need to gather information about extents to
  // figure out block neighbors as well as which
  // regions to exchange.
  std::vector<vtkIdType> offsets(nranks);
  std::vector<vtkIdType> extoffsets(nranks);
  std::vector<vtkIdType> recvLength(nranks);
  offsets[0] = 0;
  extoffsets[0] = 0;
  recvLength[0] = allnblocks[0] * 6;
  for(vtkIdType i=1; i<nranks; i++)
    {
    offsets[i] = offsets[i-1] + allnblocks[i-1];
    extoffsets[i] = offsets[i]*6;
    recvLength[i] = allnblocks[i] * 6;
    }
  vtkIdType totnblocks = offsets[nranks - 1] + allnblocks[nranks - 1];
  std::vector<int> localExtents(nblocks*6);
  for(vtkIdType i=0; i<nblocks; i++)
    {
    datasets[i]->GetExtent(&localExtents[i*6]);
    }
  std::vector<int> allExtents(totnblocks*6);
  contr->AllGatherV(
    &localExtents[0], &allExtents[0], nblocks*6, &recvLength[0], &extoffsets[0]);

  // Figure out the overall extent of all blocks.
  int wholeExtent[6] = {VTK_INT_MAX, -VTK_INT_MAX,
                        VTK_INT_MAX, -VTK_INT_MAX,
                        VTK_INT_MAX, -VTK_INT_MAX};
  for (vtkIdType i=0; i<totnblocks; i++)
    {
    int* extent = &allExtents[i*6];
    for (vtkIdType j=0; j<3; j++)
      {
      if (extent[2*j] < wholeExtent[2*j])
        {
        wholeExtent[2*j] = extent[2*j];
        }
      if (extent[2*j+1] > wholeExtent[2*j+1])
        {
        wholeExtent[2*j+1] = extent[2*j+1];
        }
      }
    }

  // Create a map from global block id to rank
  std::vector<int> blockToRank(totnblocks);
  for(vtkIdType irank=0; irank<nranks; irank++)
    {
    vtkIdType offset = offsets[irank];
    vtkIdType nblocks = allnblocks[irank];
    for (vtkIdType iblock=offset; iblock<offset+nblocks; iblock++)
      {
      blockToRank[iblock] = irank;
      }
    }

  const int nGhosts = 2;

  // Use vtkStructuredGridConnectivity to compute
  // neighborhood info.
  std::vector<int> ghostedExtents(allExtents);

  vtkNew<vtkStructuredGridConnectivity> sgc;
  sgc->SetNumberOfGrids(totnblocks);
  sgc->SetWholeExtent(wholeExtent);
  for (vtkIdType i=0; i<totnblocks; i++)
    {
    int* gExt = &ghostedExtents[i*6];
    vtkStructuredExtent::Grow(gExt, nGhosts, wholeExtent);
    sgc->RegisterGrid(i, gExt, 0, 0, 0, 0, 0);
    }
  sgc->ComputeNeighbors();

  // Allocate a new set of blocks that will contain the
  // ghost cells. Also copy the information from input
  // datasets.
  std::vector<vtkSmartPointer<vtkImageData> > ghostedDatasets;
  ghostedDatasets.resize(nblocks);
  vtkIdType offset = offsets[rank];
  for (vtkIdType id=0; id<nblocks; id++)
    {
    vtkImageData* dataset = datasets[id].GetPointer();
    vtkImageData* ghosted = vtkImageData::New();
    ghosted->SetExtent(&ghostedExtents[(id+offset)*6]);
    vtkDataSetAttributes* cd = dataset->GetCellData();
    vtkDataSetAttributes* gcd = ghosted->GetCellData();
    gcd->CopyAllocate(cd, ghosted->GetNumberOfCells());

    int numArrays = gcd->GetNumberOfArrays();

    int ext[6], gExt[6];
    dataset->GetExtent(ext);
    ghosted->GetExtent(gExt);
    for (int i=0; i<3; i++)
      {
      ext[2*i+1]--;
      gExt[2*i+1]--;
      }
    gcd->CopyStructuredData(cd, ext, gExt, false);
    for (int i=0; i<numArrays; i++)
      {
      gcd->GetAbstractArray(i)->SetNumberOfTuples(ghosted->GetNumberOfCells());
      }
    ghostedDatasets[id].TakeReference(ghosted);
    }

  vtkMPICommunicator *vtkcomm = vtkMPICommunicator::SafeDownCast(
    contr->GetCommunicator());
  diy::mpi::communicator comm(*vtkcomm->GetMPIComm()->GetHandle());

  diy::Master master(comm);

  std::vector<Block> blocks;
  blocks.reserve(nblocks);

  // Setup the blocks for DIY.
  for (vtkIdType iblock=offset; iblock<offset+nblocks; iblock++)
    {
    vtkIdType myBlock = iblock - offset;
    blocks[myBlock] = Block(&localExtents[myBlock*6], nGhosts, datasets[myBlock], ghostedDatasets[myBlock], sgc.GetPointer());

    diy::Link* link = new diy::Link;

    std::map<vtkIdType, vtkStructuredNeighbor>* ns =
      new std::map<vtkIdType, vtkStructuredNeighbor>;
    blocks[myBlock].Neighbors = ns;

    int nNeighbors = sgc->GetNumberOfNeighbors(iblock);
    for (int j=0; j<nNeighbors; j++)
      {
      vtkStructuredNeighbor neighborInfo =
        sgc->GetGridNeighbor(iblock, j);
      diy::BlockID  neighbor;
      vtkIdType id = neighborInfo.NeighborID;
      (*ns)[id] = neighborInfo;
      neighbor.gid = id;
      neighbor.proc = blockToRank[id];
      link->add_neighbor(neighbor);
      }
    master.add(iblock, &blocks[myBlock], link);
    }

  // Now exchange ghosts.
  master.foreach<Block>(&SendGhosts);
  master.exchange();
  master.foreach<Block>(&ReceiveGhosts);

  // Finalize the return data structure.
  vtkSmartPointer<vtkMultiBlockDataSet> ghosted =
    vtkSmartPointer<vtkMultiBlockDataSet>::New();
  for (vtkIdType id=0; id<nblocks; id++)
    {
    // We need ghost cell information so that ghost cells
    // can be ignored in post processing.
    ghostedDatasets[id]->GenerateGhostArray(datasets[id]->GetExtent(), true);

    ghosted->SetBlock(id, ghostedDatasets[id]);
    }
  return ghosted;
}

//-----------------------------------------------------------------------------
bool VTKmContourAnalysis::Execute(sensei::DataAdaptor* data)
{
  TimeEvent<128> mark("VTKmContourAnalysis::Execute");

  vtkMultiProcessController* prev =
    vtkMultiProcessController::GetGlobalController();

  if (prev)
    {
    prev->Register(0);
    }

  MPI_Comm comm = this->GetCommunicator();
  vtkMPICommunicatorOpaqueComm ocomm(&comm);
  vtkNew<vtkMPICommunicator> vtkComm;
  vtkComm->InitializeExternal(&ocomm);

  vtkNew<vtkMPIController> con;
  con->SetCommunicator(vtkComm.GetPointer());

  vtkMultiProcessController::SetGlobalController(con.GetPointer());

  vtkDataObject* mesh = nullptr;
  if (data->GetMesh(this->MeshName, false, mesh))
    {
    SENSEI_ERROR("Failed to get mesh \"" << this->MeshName << "\"");
    return false;
    }

  if (data->AddArray(mesh, this->MeshName,
    vtkDataObject::FIELD_ASSOCIATION_CELLS, this->ArrayName))
    {
    SENSEI_ERROR("Failed to add cell data array \"" << this->ArrayName
      << "\" to mesh \"" << this->MeshName << "\"")
    return false;
    }

  vtkSmartPointer<vtkMultiBlockDataSet> ghosted =
    ExchangeGhosts(mesh, con.GetPointer());

  vtkNew<vtkmAverageToPoints> cell2Point;
  cell2Point->SetInputDataObject(0, ghosted.GetPointer());
  cell2Point->SetInputArrayToProcess(0, 0, 0,
    vtkDataObject::FIELD_ASSOCIATION_CELLS, this->ArrayName.c_str());

  vtkNew<vtkmContour> contour;
  contour->SetInputConnection(cell2Point->GetOutputPort());
  contour->SetValue(0, /*0.3*/ this->Value);
  contour->SetInputArrayToProcess(0, 0, 0,
    vtkDataObject::FIELD_ASSOCIATION_POINTS, this->ArrayName.c_str());
  contour->Update();

  // vtkDataObject* output = contour->GetOutputDataObject(0);
  vtkDataObject* output = contour->GetOutputDataObject(0);
  if (vtkMultiBlockDataSet* outputCD = vtkMultiBlockDataSet::SafeDownCast(output))
    {
    vtkSmartPointer<vtkCompositeDataIterator> iter;
    iter.TakeReference(outputCD->NewIterator());

    vtkIdType nCells = 0;
    iter->InitTraversal();
    while (!iter->IsDoneWithTraversal())
      {
      vtkPolyData* cur = vtkPolyData::SafeDownCast(
        iter->GetCurrentDataObject());
      if (cur)
        {
        nCells += cur->GetNumberOfCells();
        }
      iter->GoToNextItem();
      }
    vtkIdType numTotCells;
    con->Reduce(&nCells, &numTotCells, 1, vtkCommunicator::SUM_OP, 0);

    SENSEI_STATUS("Number of contour cells: " << numTotCells)

    if (this->WriteOutput)
      {
      std::stringstream fname;
      fname << "contour" << data->GetDataTimeStep() << ".vtm";

      vtkNew<vtkXMLPMultiBlockDataWriter> writer;
      writer->SetInputData(outputCD);
      writer->SetDataModeToAppended();
      writer->EncodeAppendedDataOff();
      writer->SetCompressorTypeToNone();
      writer->SetFileName(fname.str().c_str());
      writer->SetWriteMetaFile(1);
      writer->Write();
      }
    }

  if (prev)
    {
    vtkMultiProcessController::SetGlobalController(prev);
    prev->UnRegister(0);
    }

  return true;
}

//-----------------------------------------------------------------------------
int VTKmContourAnalysis::Finalize()
{
  return 0;
}

}
