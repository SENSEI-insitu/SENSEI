#include "DataAdaptor.h"

#include <timer/Timer.h>

#include <vtkCompositeDataIterator.h>
#include <vtkDataSetAttributes.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>


namespace sensei
{
namespace adios
{

namespace internals
{
  vtkDataArray* createVTKArray(ADIOS_DATATYPES type)
    {
    switch (type)
      {
    case adios_real:
      return vtkFloatArray::New();
    case adios_double:
      return vtkDoubleArray::New();
    default:
      abort();
      }
    }

  vtkSmartPointer<vtkDataSet> ReadBlockMetaData(int blockno, ADIOS_FILE* file)
    {
    vtkSmartPointer<vtkImageData> img = vtkSmartPointer<vtkImageData>::New();

    ADIOS_SELECTION* selection = adios_selection_writeblock(blockno);

    double origin[3], spacing[3];
    int extent[6];
    adios_schedule_read(file, selection, "block/origin", 0, 1, origin);
    adios_schedule_read(file, selection, "block/spacing", 0, 1, spacing);
    adios_schedule_read(file, selection, "block/extents", 0, 1, extent);
    adios_perform_reads(file, 1);

    img->SetOrigin(origin);
    img->SetSpacing(spacing);
    img->SetExtent(extent);
    return img;
    }

  void ReleaseData(vtkDataSet* ds)
    {
    if (ds)
      {
      ds->GetAttributes(vtkDataObject::CELL)->Initialize();
      ds->GetAttributes(vtkDataObject::POINT)->Initialize();
      }
    }

  void ReleaseData(vtkDataObject* dobj)
    {
    if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj))
      {
      vtkSmartPointer<vtkCompositeDataIterator> iter;
      iter.TakeReference(cd->NewIterator());
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
        {
        ReleaseData(vtkDataSet::SafeDownCast(iter->GetCurrentDataObject()));
        }
      }
    else
      {
      ReleaseData(vtkDataSet::SafeDownCast(dobj));
      }
    }


  bool AddArray(vtkDataSet* ds, int association, const char* arrayname,
    unsigned int block, ADIOS_FILE* file)
    {
    std::string path = (association == vtkDataObject::CELL)?
      "block/celldata/" : "block/pointdata";
    path += arrayname;

    ADIOS_VARINFO* varinfo = adios_inq_var(file, path.c_str());
    vtkDataArray* vtkarray = createVTKArray(varinfo->type);
    adios_free_varinfo(varinfo);

    vtkarray->SetName(arrayname);
    vtkarray->SetNumberOfTuples(
      association == vtkDataObject::FIELD_ASSOCIATION_POINTS?
      ds->GetNumberOfPoints() : ds->GetNumberOfCells());

    ADIOS_SELECTION* selection = adios_selection_writeblock(block);
    adios_schedule_read(file, selection, path.c_str(), 0, 1, vtkarray->GetVoidPointer(0));
    ds->GetAttributes(association)->AddArray(vtkarray);
    vtkarray->FastDelete();
    vtkarray->Modified();
    return true;
    }
}

vtkStandardNewMacro(DataAdaptor);
//----------------------------------------------------------------------------
DataAdaptor::DataAdaptor()
  : File(NULL),
  Comm(MPI_COMM_WORLD)
{
}

//----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
}

//----------------------------------------------------------------------------
bool DataAdaptor::Open(MPI_Comm comm,
  ADIOS_READ_METHOD method, const std::string& filename)
{
  timer::MarkEvent mark("adios::dataadaptor::open");

  this->Comm = comm;
  adios_read_init_method (method, comm, "verbose=1");
  this->File = adios_read_open(filename.c_str(), method, comm, ADIOS_LOCKMODE_ALL, -1);

  int mpi_size, version, data_type, extents[6];
  int64_t num_blocks, num_points, num_cells, ntimestep;
  double time;

  adios_schedule_read(this->File, NULL, "mpi_size", 0, 1, &mpi_size);
  adios_schedule_read(this->File, NULL, "data_type", 0, 1, &data_type);
  adios_schedule_read(this->File, NULL, "version", 0, 1, &version);
  adios_schedule_read(this->File, NULL, "extents", 0, 1, extents);

  adios_schedule_read(this->File, NULL, "num_blocks", 0, 1, &num_blocks);
  adios_schedule_read(this->File, NULL, "num_points", 0, 1, &num_points);
  adios_schedule_read(this->File, NULL, "num_cells", 0, 1, &num_cells);
  adios_schedule_read(this->File, NULL, "ntimestep", 0, 1, &ntimestep);
  adios_schedule_read(this->File, NULL, "time", 0, 1, &time);
  adios_perform_reads(this->File, 1);

  if (num_blocks == 1)
    {
    num_blocks = num_blocks * mpi_size;
    }

  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(this->Comm, &rank);
  if (size > num_blocks)
    {
    cerr << "MPI group size cannot be smaller than number of blocks in the dataset!" << endl;
    abort();
    return false;
    }

  vtkDataObject* dobj = NULL;
  if (size == num_blocks)
    {
    this->Mesh = internals::ReadBlockMetaData(rank, this->File);
    }
  else
    {
    vtkMultiBlockDataSet* mb = vtkMultiBlockDataSet::New();
    mb->SetNumberOfBlocks(num_blocks);

    int64_t blocks_to_read = num_blocks / size;
    int64_t remainder = num_blocks % size;

    int64_t start_block = (rank >= remainder)?
      ((remainder * ( blocks_to_read + 1)) + (rank - remainder) * blocks_to_read) :
      rank * (blocks_to_read + 1);
    blocks_to_read += (rank < remainder)?  1 : 0;

    for (int64_t cc = start_block; cc < start_block + blocks_to_read; ++cc)
      {
      mb->SetBlock(static_cast<unsigned int>(cc), internals::ReadBlockMetaData(cc, this->File));
      }
    this->Mesh.TakeReference(mb);
    }
  this->GetInformation()->Set(vtkDataObject::DATA_EXTENT(), extents, 6);

  this->AssociatedArrays.clear();
  for (int cc=0; cc < this->File->nvars; ++cc)
    {
    int association = vtkDataObject::FIELD_ASSOCIATION_POINTS;
    const char* name = NULL;
    if (strncmp(this->File->var_namelist[cc], "block/celldata/", strlen("block/celldata/")) == 0)
      {
      association = vtkDataObject::FIELD_ASSOCIATION_CELLS;
      name = this->File->var_namelist[cc] + strlen("block/celldata/");
      }
    else if (strncmp(this->File->var_namelist[cc], "block/point/", strlen("block/point/")) == 0)
      {
      association = vtkDataObject::FIELD_ASSOCIATION_POINTS;
      name = this->File->var_namelist[cc] + strlen("block/point/");
      }
    // TODO: handle all other possible assocations.
    if (!name) { continue; }
    this->AssociatedArrays[association][name] = false;
    }

  return adios_errno != err_end_of_stream;
}

//----------------------------------------------------------------------------
bool DataAdaptor::Advance()
{
  adios_release_step(this->File);
  return adios_advance_step(this->File, 0, /*timeout*/0.5) == 0;
}

//----------------------------------------------------------------------------
bool DataAdaptor::ReadStep()
{
  timer::MarkStartEvent("adios::dataadaptor::read-timestep-metadata");

  int tstep = 0; double time = 0;
  ADIOS_FILE* f = this->File;
  adios_schedule_read(f, NULL, "ntimestep", 0, 1, &tstep);
  adios_schedule_read(f, NULL, "time", 0, 1, &time);
  adios_perform_reads(f, 1);
  this->SetDataTimeStep(tstep);
  this->SetDataTime(time);
  timer::MarkEndEvent("adios::dataadaptor::read-timestep-metadata");
  return true;
}

//----------------------------------------------------------------------------
vtkDataObject* DataAdaptor::GetMesh(bool /*structure_only*/)
{
  return this->Mesh;
}

//----------------------------------------------------------------------------
bool DataAdaptor::AddArray(vtkDataObject* mesh, int association, const char* arrayname)
{
  ArraysType& arrays = this->AssociatedArrays[association];
  ArraysType::iterator iter = arrays.find(arrayname);
  if (iter == arrays.end())
    {
    return false;
    }

  if (iter->second == false)
    {
    timer::MarkEvent mark("adios::dataadaptor::read-arrays");
    if (vtkDataSet* image = vtkDataSet::SafeDownCast(mesh))
      {
      int rank;
      MPI_Comm_rank(this->Comm, &rank);
      if (internals::AddArray(image, association, arrayname, rank, this->File) == false)
        {
        return false;
        }
      }
    else
      {
      vtkMultiBlockDataSet* mb = vtkMultiBlockDataSet::SafeDownCast(mesh);
      for (unsigned int cc=0, max = mb->GetNumberOfBlocks(); cc < max; ++cc)
        {
        if (vtkDataSet* ds = vtkDataSet::SafeDownCast(mb->GetBlock(cc)))
          {
          if (internals::AddArray(ds, association, arrayname, cc, this->File) == false)
            {
            return false;
            }
          }
        }
      }
    adios_perform_reads(this->File, 1);
    iter->second = true;
    }
  return true;
}

//----------------------------------------------------------------------------
unsigned int DataAdaptor::GetNumberOfArrays(int association)
{
  return static_cast<unsigned int>(this->AssociatedArrays[association].size());
}

//----------------------------------------------------------------------------
const char* DataAdaptor::GetArrayName(int association, unsigned int index)
{
  ArraysType& arrays = this->AssociatedArrays[association];
  unsigned int cc=0;
  for (ArraysType::iterator iter = arrays.begin(); iter != arrays.end(); ++iter, ++cc)
    {
    if (cc == index)
      {
      return iter->first.c_str();
      }
    }
  return NULL;
}

//----------------------------------------------------------------------------
void DataAdaptor::ReleaseData()
{
  timer::MarkEvent mark("adios::dataadaptor::release-data");

  for (AssociatedArraysType::iterator iter1 = this->AssociatedArrays.begin();
    iter1 != this->AssociatedArrays.end(); ++iter1)
    {
    for (ArraysType::iterator iter2 = iter1->second.begin();
      iter2 != iter1->second.end(); ++iter2)
      {
      iter2->second = false;
      }
    }
  internals::ReleaseData(this->Mesh);
}

//----------------------------------------------------------------------------
void DataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

} // adios
} // sensei
