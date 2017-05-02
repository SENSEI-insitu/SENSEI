#include "PosthocIO.h"
#include "DataAdaptor.h"
#include "senseiConfig.h"
#include "Error.h"

#include <vtkCellData.h>
#include <vtkCompositeDataIterator.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkDataArray.h>
#include <vtkDataArrayTemplate.h>
#include <vtkDataObject.h>
#include <vtkDataSetAttributes.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkNew.h>


#include <algorithm>
#include <sstream>
#include <fstream>
#include <cassert>

#include <ArrayIO.h>

#if defined(ENABLE_VTK_XMLP)
#include <vtkAlgorithm.h>
#include <vtkCompositeDataPipeline.h>
#include <vtkXMLMultiBlockDataWriter.h>
#endif

//#define PosthocIO_DEBUG
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
        SENSEI_ERROR("write failed");
        return -1;
        }
        );
    default:
      SENSEI_ERROR("Unhandled data type");
      return -1;
    }
  return 0;
}
} // namespace impl

//-----------------------------------------------------------------------------
senseiNewMacro(PosthocIO);

//-----------------------------------------------------------------------------
PosthocIO::PosthocIO() : Comm(MPI_COMM_WORLD), CommRank(0), CommSize(1),
   OutputDir("./"), HeaderFile("ImageHeader"), BlockExt(".sensei"),
   HaveHeader(true), Mode(mpiIO), Period(1) {}

//-----------------------------------------------------------------------------
PosthocIO::~PosthocIO()
{
}

//-----------------------------------------------------------------------------
void PosthocIO::Initialize(MPI_Comm comm,
    const std::string &outputDir, const std::string &headerFile,
    const std::string &blockExt, const std::vector<std::string> &cellArrays,
    const std::vector<std::string> &pointArrays, int mode, int period)
{
#ifdef PosthocIO_DEBUG
  if (!this->CommRank)
    SENSEI_STATUS("PosthocIO::Initialize")
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
}

//-----------------------------------------------------------------------------
bool PosthocIO::Execute(DataAdaptor* data)
{
#ifdef PosthocIO_DEBUG
  if (!this->CommRank)
    SENSEI_STATUS("PosthocIO::Execute");
#endif
  // validate the input dataset.
  // TODO:for now we need composite data, to support non-composite
  // data we will wrap it in a composite dataset.
  vtkCompositeDataSet* cd =
    dynamic_cast<vtkCompositeDataSet*>(data->GetMesh(false));

  if (!cd)
    {
    SENSEI_ERROR("unsupported dataset type")
    return false;
    }

  // we need whole extents
  vtkInformation *info = data->GetInformation();
  if (!info->Has(vtkDataObject::DATA_EXTENT()))
    {
    SENSEI_ERROR("missing vtkDataObject::DATA_EXTENT");
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
      SENSEI_ERROR("invalid mode \"" << this->Mode << "\"")
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

  std::ostringstream fprefix;
  fprefix << this->HeaderFile << "_" << timeStep;
  std::ostringstream oss;
  oss << this->OutputDir << "/" << fprefix.str().c_str() << ".vtmb";
  vtkNew<vtkXMLMultiBlockDataWriter> writer;
  writer->SetInputData(cd);
  writer->SetDataModeToAppended();
  writer->EncodeAppendedDataOff();
  writer->SetCompressorTypeToNone();
  writer->SetFileName(oss.str().c_str());
  writer->SetWriteMetaFile(0);
  writer->Write();

  // Write meta-file on root node.
  if (this->CommRank == 0)
    {
    assert(vtkMultiBlockDataSet::SafeDownCast(cd) &&
      vtkMultiBlockDataSet::SafeDownCast(cd)->GetNumberOfBlocks() ==
        static_cast<unsigned int>(this->CommSize));
    ofstream ofp(oss.str().c_str());
    if (ofp)
      {
      // This is hard-coded to only work with a multiblock of image dataset with
      // non-null leaf node only on 1 rank and block index corresponding to the process rank.
      ofp << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
      ofp << "  <vtkMultiBlockDataSet>\n";

      for (int cc=0; cc < this->CommSize; ++cc)
        {
        ofp << "    <DataSet index=\"" << cc << "\" file=\""
            << fprefix.str().c_str() << "/" << fprefix.str().c_str() << "_" << cc << ".vti\">\n";
        ofp << "    </DataSet>\n";
        }
      ofp << "  </vtkMultiBlockDataSet>\n";
      ofp << "</VTKFile>\n";
      }
    ofp.close();
    }

#ifdef PosthocIO_DEBUG
  if (!this->CommRank)
    SENSEI_STATUS("PosthocIO::WriteXMLP \"" << oss.str() << "\"");
#endif

#else
  (void)cd;
  (void)info;
  (void)timeStep;
  SENSEI_ERROR("built without vtk xmlp writer")
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
  std::ofstream ff(fileName.c_str(), std::ofstream::out);
  if (!ff.good())
    {
    SENSEI_ERROR("Failed to write the header file \"" << fileName << "\"")
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

#ifdef PosthocIO_DEBUG
  if (!this->CommRank)
    SENSEI_STATUS("wrote BOV header \"" << fileName << "\"");
#endif
  return 0;
}

//-----------------------------------------------------------------------------
int PosthocIO::WriteBOV(vtkCompositeDataSet *cd,
    vtkInformation *info, int timeStep)
{
#ifdef PosthocIO_DEBUG
  if (!this->CommRank)
    SENSEI_STATUS("PosthocIO::WriteBOV");
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
        SENSEI_ERROR("Open failed \"" << fileName);
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
        if (!this->CommRank)
          SENSEI_WARNING("COLLECTIVE BUFFERING IS DISABLED BECAUSE THERE "
            "IS AT LEAST ONE PROCESS WITH MORE THAN ONE BLOCK")
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
          SENSEI_ERROR("input not an image.");
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
          SENSEI_ERROR("no array named \"" << arrayName << "\"");
          atts->Print(cerr);
          continue;
          }

        // dispatch the write
        if (impl::write(fh, MPI_INFO_NULL, wholeExt, localExt,
              validExt, da, useCollectives))
          {
          SENSEI_ERROR("write failed \"" << fileName)
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
