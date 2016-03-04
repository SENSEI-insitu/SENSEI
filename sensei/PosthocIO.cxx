#include "PosthocIO.h"

#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataArray.h>
#include <vtkDataObject.h>
#include <vtkDataSetAttributes.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkDataArrayTemplate.txx>
#include <vtkCellData.h>
#include <vtkPointData.h>

#include "DataAdaptor.h"

#include <algorithm>
#include <sstream>
#include <fstream>

#include <ArrayIO.h>

#if defined(ENABLE_VTK_XMLP)
#include <vtkAlgorithm.h>
#include <vtkCompositeDataPipeline.h>
#include <vtkMultiProcessController.h>
#include <vtkMPI.h>
#include <vtkMPIController.h>
#include <vtkXMLPMultiBlockDataWriter.h>
#endif

//#define PosthocIO_DEBUG

#define PosthocIOError(_arg) \
  cerr << "ERROR: " << __FILE__ " : "  << __LINE__ << std::endl \
    << "" _arg << std::endl;

#define posthocIO_status(_cond, _arg) \
  if (_cond) \
    cerr << "" _arg << std::endl;

namespace sensei
{
namespace impl
{
// **************************************************************************
void getWholePointExtents(vtkInformation *info, int *wholePointExtent)
{
  memcpy(wholePointExtent, info->Get(vtkDataObject::DATA_EXTENT()),
    6*sizeof(int));
}

// **************************************************************************
void getWholeCellExtents(vtkInformation *info, int *wholeCellExtent)
{
  int *wholePointExtent = info->Get(vtkDataObject::DATA_EXTENT());
  for (int i = 0; i < 6; ++i)
    wholeCellExtent[i] = wholePointExtent[i] - (i%2 ? 1 : 0);
}

// **************************************************************************
void getLocalCellExtents(vtkImageData *id, int *localCellExtent)
{
  id->GetExtent(localCellExtent);
  localCellExtent[1] -= 1;
  localCellExtent[3] -= 1;
  localCellExtent[5] -= 1;
}

/*/ **************************************************************************
void getValidCellExtents(vtkImageData *id, int *validCellExtent)
{ getLocalCellExtents(id, validCellExtent); } */

// **************************************************************************
void getLocalPointExtents(vtkImageData *id, int *localPointExtent)
{
  id->GetExtent(localPointExtent);
}

// **************************************************************************
void getValidPointExtents(vtkImageData *id,
    int *wholePointExtent, int *validPointExtent)
{
  // deal with coincident block faces
  id->GetExtent(validPointExtent);

  validPointExtent[1] = (validPointExtent[1] == wholePointExtent[1]
        ? validPointExtent[1] : validPointExtent[1] - 1);

  validPointExtent[3] = (validPointExtent[3] == wholePointExtent[3]
        ? validPointExtent[3] : validPointExtent[3] - 1);

  validPointExtent[5] = (validPointExtent[5] == wholePointExtent[5]
        ? validPointExtent[5] : validPointExtent[5] - 1);
}

// ****************************************************************************
int write(MPI_File file, MPI_Info hints,
      int domain[6], int decomp[6], int valid[6], vtkDataArray *da,
      bool useCollectives)
{
  switch (da->GetDataType())
    {
    vtkTemplateMacro(
      vtkDataArrayTemplate<VTK_TT> *ta =
        static_cast<vtkDataArrayTemplate<VTK_TT>*>(da);
      if ((useCollectives && arrayIO::write_all(file, hints,
            domain, decomp, valid, ta->GetPointer(0))) ||
            arrayIO::write(file, hints, domain, decomp,
            valid, ta->GetPointer(0)))
        {
        PosthocIOError("write failed");
        return -1;
        }
        );
    default:
      PosthocIOError("Unhandled data type");
      return -1;
    }
  return 0;
}
} // namespace impl
} // namespace sensei

namespace sensei
{
//-----------------------------------------------------------------------------
vtkStandardNewMacro(PosthocIO);

//-----------------------------------------------------------------------------
PosthocIO::PosthocIO() : Comm(MPI_COMM_WORLD), CommRank(0), CommSize(1),
   OutputDir("./"), HeaderFile("ImageHeader"), BlockExt(".sensei"),
   HaveHeader(true), Mode(mpiIO), Period(1) {}

//-----------------------------------------------------------------------------
PosthocIO::~PosthocIO()
{
#if defined(ENABLE_VTK_XMLP)
  // teardown for parallel vtk I/O
  vtkMultiProcessController *mpc =
    vtkMultiProcessController::GetGlobalController();
  mpc->Finalize(1);
  mpc->Delete();
  vtkMultiProcessController::SetGlobalController(nullptr);
  vtkAlgorithm::SetDefaultExecutivePrototype(nullptr);
#endif
}

//-----------------------------------------------------------------------------
void PosthocIO::Initialize(MPI_Comm comm,
    const std::string &outputDir, const std::string &headerFile,
    const std::string &blockExt, const std::vector<std::string> &cellArrays,
    const std::vector<std::string> &pointArrays, int mode, int period)
{
#ifdef PosthocIO_DEBUG
  posthocIO_status((this->CommRank==0), "PosthocIO::Initialize");
#endif
  this->Comm = comm;
  MPI_Comm_rank(this->Comm, &this->CommRank);
  MPI_Comm_size(this->Comm, &this->CommSize);
  this->OutputDir = outputDir;
  this->HeaderFile = headerFile;
  this->BlockExt = blockExt;
  this->CellArrays = cellArrays;
  this->PointArrays = pointArrays;
  this->HaveHeader = (this->CommRank==0 ? false : true);
  this->Mode = mode;
  this->Period = period;
#if defined(ENABLE_VTK_XMLP)
  // setup for parallel vtk i/o
  vtkMPICommunicator* vtkComm = vtkMPICommunicator::New();
  vtkMPICommunicatorOpaqueComm h(&comm);
  vtkComm->InitializeExternal(&h);

  vtkMPIController *con = vtkMPIController::New();
  con->SetCommunicator(vtkComm);
  vtkComm->Delete();

  vtkMultiProcessController::SetGlobalController(con);

  vtkCompositeDataPipeline* cexec = vtkCompositeDataPipeline::New();
  vtkAlgorithm::SetDefaultExecutivePrototype(cexec);
  cexec->Delete();
#endif
}

//-----------------------------------------------------------------------------
bool PosthocIO::Execute(sensei::DataAdaptor* data)
{
#ifdef PosthocIO_DEBUG
  posthocIO_status((this->CommRank==0), "PosthocIO::Execute");
#endif
  // validate the input dataset.
  // TODO:for now we need composite data, to support non-composite
  // data we will wrap it in a composite dataset.
  vtkCompositeDataSet* cd =
    dynamic_cast<vtkCompositeDataSet*>(data->GetMesh(false));

  if (!cd)
    {
    PosthocIOError("unsupported dataset type")
    return false;
    }

  // we need whole extents
  vtkInformation *info = data->GetInformation();
  if (!info->Has(vtkDataObject::DATA_EXTENT()))
    {
    PosthocIOError("missing vtkDataObject::DATA_EXTENT");
    return false;
    }

  // grab the current time step
  int timeStep = data->GetDataTimeStep();

  // option to reduce the amount of data written
  if (timeStep%this->Period)
      return true;

  // dispatch the write
  switch (this->Mode)
    {
    case mpiIO:
      this->WriteBOVHeader(info);
      this->WriteBOV(cd, info, timeStep);
      break;
    case vtkXmlP:
      this->WriteXMLP(cd, info, timeStep);
      break;
    default:
      PosthocIOError("invalid mode \"" << this->Mode << "\"")
      return false;
    }

  return true;
}

//-----------------------------------------------------------------------------
int PosthocIO::WriteXMLP(vtkCompositeDataSet *cd,
    vtkInformation *info, int timeStep)
{
#if defined(ENABLE_VTK_XMLP)
  (void)info;

  std::ostringstream oss;
  oss << this->OutputDir << "/" << this->HeaderFile
      << "_" << timeStep << ".vtmb";

  vtkXMLPMultiBlockDataWriter *writer = vtkXMLPMultiBlockDataWriter::New();
  writer->SetInputData(cd);
  writer->SetDataModeToAppended();
  writer->EncodeAppendedDataOff();
  writer->SetCompressorTypeToNone();
  writer->SetFileName(oss.str().c_str());
  writer->Write();
  writer->Delete();

#ifdef PosthocIO_DEBUG
  posthocIO_status((this->CommRank==0),
    "PosthocIO::WriteXMLP \"" << oss.str() << "\"");
#endif

#else
  (void)cd;
  (void)info;
  (void)timeStep;
  PosthocIOError("built without vtk xmlp writer")
  return -1;
#endif
  return 0;
}

//-----------------------------------------------------------------------------
int PosthocIO::WriteBOVHeader(vtkInformation *info)
{
  if (this->CommRank || this->HaveHeader)
    return 0;

  // handle both cell and point data
  for (int dType = 0; dType < 2; ++dType)
    {
    // get the arrays
    std::vector<std::string> &arrays =
      dType ? this->CellArrays : this->PointArrays;

    if (arrays.empty())
      continue;

    // get the extents
    int wholeExt[6];
    if (dType)
      impl::getWholeCellExtents(info, wholeExt);
    else
      impl::getWholePointExtents(info, wholeExt);

    const char *dTypeId =
      (dType ? "CellData.bov" : "PointData.bov");

    std::string headerFile =
        this->OutputDir + "/" + this->HeaderFile + dTypeId;

    this->WriteBOVHeader(headerFile, arrays, wholeExt);
    }

  this->HaveHeader = true;
  return 0;
}

//-----------------------------------------------------------------------------
int PosthocIO::WriteBOVHeader(const std::string &fileName,
    const std::vector<std::string> &arrays, const int *wholeExtent)
{
  std::ofstream ff(fileName, std::ofstream::out);
  if (!ff.good())
    {
    PosthocIOError("Failed to write the header file \"" << fileName << "\"")
    return -1;
    }

  int dims[3] = {wholeExtent[1] - wholeExtent[0] + 1,
      wholeExtent[3] - wholeExtent[2] + 1,
      wholeExtent[5] - wholeExtent[4] + 1};

  ff << "# SciberQuest MPI-IO BOV Reader" << std::endl
    << "nx=" << dims[0] << ", ny=" << dims[1] << ", nz=" << dims[2] << std::endl
    << "ext=" << this->BlockExt << std::endl
    << "dtype=f32" << std::endl
    << std::endl;

  size_t n = arrays.size();
  for (size_t i = 0; i < n; ++i)
    ff << "scalar:" << arrays[i] << std::endl;

  ff << std::endl;
  ff.close();

#ifdef posthocIO_DEBUG
  posthocIO_status((this->CommRank==0),
    "wrote BOV header \"" << fileName << "\"");
#endif
  return 0;
}

//-----------------------------------------------------------------------------
int PosthocIO::WriteBOV(vtkCompositeDataSet *cd,
    vtkInformation *info, int timeStep)
{
#ifdef PosthocIO_DEBUG
  posthocIO_status((this->CommRank==0), "PosthocIO::WriteBOV");
#endif

  // handle both cell and point data
  for (int dType = 0; dType < 2; ++dType)
    {
    std::vector<std::string> &arrays =
      dType ? this->CellArrays : this->PointArrays;

    size_t n_arrays = arrays.size();
    for (size_t i = 0; i < n_arrays; ++i)
      {
      const std::string &arrayName = arrays[i];

      // construct the array's file name
      std::ostringstream oss;
      oss << this->OutputDir << "/"  << arrayName
        << "_" << timeStep << "." << this->BlockExt;
      std::string fileName = oss.str();
   
      // open the file 
      MPI_File fh;
      if (arrayIO::open(this->Comm, fileName.c_str(), MPI_INFO_NULL, fh))
        {
        PosthocIOError("Open failed \"" << fileName);
        return -1;
        }

      // get the extents
      int wholeExt[6];
      if (dType)
        impl::getWholeCellExtents(info, wholeExt);
      else
        impl::getWholePointExtents(info, wholeExt);
    
      // count the number of local blocks. if there are more than 1
      // block on any process then collective buffering is problematic.
      // there are a couple of ways to work around.
      //
      // 1) make the write in a number of rounds.
      //    a) if all of the processes have the same number of blocks
      //       there is no problem. If there are varying numbers of blocks
      //       per process each process participates in each round passing
      //       empty arrays.
      //    b) disable collective buffering
      //
      // 2) copy local blocks into an array with a regular rectangular
      //    shape (ie describable by mpi subarray).
      //
      // 3) modify the io code to use an aggregate data type describing the
      //    irregular shaped region (this could be very slow though!)
      //
      // the approach below disables collective buffering when there is
      // more than 1 block per process on any process (1b above). 2 is
      // an more attractive scenario but I'm not sure diy decomposes this
      // way.
      vtkSmartPointer<vtkCompositeDataIterator> iter;
      iter.TakeReference(cd->NewIterator());

      int nLocalBlocks = 0;
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal();
          iter->GoToNextItem()) ++nLocalBlocks;

      MPI_Allreduce(MPI_IN_PLACE, &nLocalBlocks,
          1, MPI_INT, MPI_MAX, this->Comm);

      bool useCollectives = (nLocalBlocks==1);

      if (!useCollectives)
        {
        posthocIO_status((this->CommRank==0),
          "WARNING: COLLECTIVE BUFFERING IS DISABLED "
          "BECAUSE THERE IS AT LEAST ONE PROCESS WITH "
          "MORE THAN ONE BLOCK")
        }

      // write the array
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal();
          iter->GoToNextItem())
        {
        // get the block
        vtkImageData *id =
          dynamic_cast<vtkImageData*>(iter->GetCurrentDataObject());

        if (!id)
          {
          PosthocIOError("input not an image.");
          continue;
          }

        // get the local and valid extents
        int localExt[6];
        int validExt[6];
        if (dType)
          {
          impl::getLocalCellExtents(id, localExt);
          memcpy(validExt, localExt, 6*sizeof(int));
          }
        else
          {
          impl::getLocalPointExtents(id, localExt);
          impl::getValidPointExtents(id, wholeExt, validExt);
          }

        // grab the requested array
        vtkDataSetAttributes *atts = (dType ?
            static_cast<vtkDataSetAttributes*>(id->GetCellData()) :
            static_cast<vtkDataSetAttributes*>(id->GetPointData()));

        vtkDataArray *da = atts->GetArray(arrayName.c_str());
        if (!da)
          {
          PosthocIOError("no array named \"" << arrayName << "\"");
          atts->Print(cerr);
          continue;
          }

        // dispatch the write
        if (impl::write(fh, MPI_INFO_NULL, wholeExt, localExt,
              validExt, da, useCollectives))
          {
          PosthocIOError("write failed \"" << fileName)
          return -1;
          }
        }
      // close file
      arrayIO::close(fh);
      }
    }
  return 0;
}


}
