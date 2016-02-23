#include "AnalysisAdaptor.h"

#include <sensei/DataAdaptor.h>

#include <vtkCellData.h>
#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <mpi.h>
#include <adios.h>
#include <vector>

namespace sensei
{
namespace adios
{

namespace internals
{
  ADIOS_DATATYPES adiosType(vtkDataArray* da)
    {
    if (vtkFloatArray::SafeDownCast(da))
      {
      return adios_real;
      }
    if (vtkDoubleArray::SafeDownCast(da))
      {
      return adios_double;
      }
    // TODO:
    abort();
    }
  int64_t CountBlocks(vtkCompositeDataSet* cd, bool skip_null)
    {
    vtkSmartPointer<vtkCompositeDataIterator> iter;
    iter.TakeReference(cd->NewIterator());
    iter->SetSkipEmptyNodes(skip_null? 1 : 0);
    int64_t count = 0;
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      {
      count++;
      }
    return count;
    }

  int64_t CountTotalBlocks(vtkDataObject* dobj)
    {
    vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj);
    return cd? CountBlocks(cd, false) : 1;
    }

  int64_t CountLocalBlocks(vtkDataObject* dobj)
    {
    vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj);
    return cd? CountBlocks(cd, true) : 1;
    }

  void define_attribute_data_vars(vtkDataSet* ds, int blockno, int association,
    int64_t g_handle, const char* c_prefix, const char* ldims, const char* gdims, const char* offsets,
    int64_t& databytes)
    {
    vtkDataSetAttributes* dsa = ds->GetAttributes(association);
    std::string prefix(c_prefix);
    for (unsigned int cc=0, max=dsa->GetNumberOfArrays(); cc<max;++cc)
      {
      if (vtkDataArray* da = dsa->GetArray(cc))
        {
        const char* aname = da->GetName();
        adios_define_var(g_handle, (prefix + aname).c_str(), "",
          adiosType(da), ldims, gdims, offsets);
        }
      }
    }

  void write_attribute_data_vars(vtkDataSet* ds, int association, int64_t io_handle, const char* c_prefix)
    {
    std::string prefix(c_prefix);
    vtkDataSetAttributes* dsa = ds->GetAttributes(association);
    for (unsigned int cc=0, max=dsa->GetNumberOfArrays(); cc<max; ++cc)
      {
      if (vtkDataArray* da = dsa->GetArray(cc))
        {
        const char* aname = da->GetName();
        adios_write(io_handle, (prefix + aname).c_str(), da->GetVoidPointer(0));
        }
      }
    }

  int64_t ComputeVariableLengthVarSize(vtkDataSet* data)
    {
    if (data)
      {
      return data->GetNumberOfCells()*data->GetCellData()->GetNumberOfArrays()*sizeof(double) +
        data->GetNumberOfPoints()*data->GetPointData()->GetNumberOfArrays()*sizeof(double);
      }
    return 0;
    }

  int64_t ComputeVariableLengthVarSize(vtkDataObject* data)
    {
    if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(data))
      {
      vtkSmartPointer<vtkCompositeDataIterator> iter;
      iter.TakeReference(cd->NewIterator());
      int64_t dsize = 0;
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
        {
        dsize += ComputeVariableLengthVarSize(vtkDataSet::SafeDownCast(iter->GetCurrentDataObject()));
        }
      return dsize;
      }
    else
      {
      return ComputeVariableLengthVarSize(vtkDataSet::SafeDownCast(data));
      }
    }

  void CountLocalElements(vtkDataSet* ds, int64_t* num_points, int64_t* num_cells)
    {
    *num_points += ds? ds->GetNumberOfPoints() : 0;
    *num_cells += ds? ds->GetNumberOfCells() : 0;
    }

  void CountLocalElements(vtkDataObject* dobj, int64_t* num_points, int64_t* num_cells)
    {
    if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj))
      {
      vtkSmartPointer<vtkCompositeDataIterator> iter;
      iter.TakeReference(cd->NewIterator());
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
        {
        CountLocalElements(vtkDataSet::SafeDownCast(iter->GetCurrentDataObject()), num_points, num_cells);
        }
      }
    else
      {
      CountLocalElements(vtkDataSet::SafeDownCast(dobj), num_points, num_cells);
      }
    }

  vtkDataSet* GetRepresentativeBlock(vtkDataObject* dobj)
    {
    if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj))
      {
      vtkSmartPointer<vtkCompositeDataIterator> iter;
      iter.TakeReference(cd->NewIterator());
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
        {
        return vtkDataSet::SafeDownCast(iter->GetCurrentDataObject());
        }
      }
    else
      {
      return vtkDataSet::SafeDownCast(dobj);
      }
    }

  void GetBlocks(vtkDataObject* dobj, std::vector<vtkDataSet*> &blocks)
    {
    if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj))
      {
      vtkSmartPointer<vtkCompositeDataIterator> iter;
      iter.TakeReference(cd->NewIterator());
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
        {
        if (vtkDataSet* ds = vtkDataSet::SafeDownCast(iter->GetCurrentDataObject()))
          {
          blocks.push_back(ds);
          }
        }
      }
    else if (vtkDataSet* ds = vtkDataSet::SafeDownCast(dobj))
      {
      blocks.push_back(ds);
      }
    }
}

vtkStandardNewMacro(AnalysisAdaptor);
//----------------------------------------------------------------------------
AnalysisAdaptor::AnalysisAdaptor() :
  Initialized(false),
  FixedLengthVarSize(0)
{
  this->SetMethod("MPI");
  this->SetFileName("sensei.bp");
}

//----------------------------------------------------------------------------
AnalysisAdaptor::~AnalysisAdaptor()
{
}

//----------------------------------------------------------------------------
bool AnalysisAdaptor::Execute(DataAdaptor* data)
{
  vtkDataObject* mesh = data->GetCompleteMesh();
  this->InitializeADIOS(data);
  this->WriteTimestep(data);
  return true;
}

//----------------------------------------------------------------------------
void AnalysisAdaptor::InitializeADIOS(DataAdaptor* data)
{
  // Ideally, we'd develop code that can write any VTK data object to ADIOS
  // using a schema that we can call VTK_ADIOS_SCHEMA. VTK and consequently
  // ParaView, Catalyst, VisIt, Libsim all can then develop readers/writers that
  // read/write this schema. Should this simply be the ADIOS Vis Schema? Maybe.
  if (this->Initialized)
    {
    return;
    }

  adios_init_noxml(MPI_COMM_WORLD);
  adios_allocate_buffer(ADIOS_BUFFER_ALLOC_NOW, 500);

  vtkDataObject* structure = data->GetMesh(/*structure_only*/ true);
  int64_t local_blocks = internals::CountLocalBlocks(structure);
  int64_t total_blocks = internals::CountTotalBlocks(structure);
  vtkDataSet* ds = internals::GetRepresentativeBlock(structure);


  int64_t databytes = 0;
  int64_t g_handle;
  adios_declare_group(&g_handle, "sensei", "", adios_flag_yes);
  adios_select_method(g_handle, this->Method.c_str(), "", "");

  // API Help:
  // adios_define_var(group_id, name, path, type, dimensions, global_dimensions, local_offsets)

  // global data.
  adios_define_var(g_handle, "rank", "", adios_integer, "", "", ""); databytes+= sizeof(int);
  adios_define_var(g_handle, "mpi_size", "", adios_integer, "", "", ""); databytes += sizeof(int);
  adios_define_var(g_handle, "data_type", "", adios_integer, "", "", ""); databytes += sizeof(int);
  adios_define_var(g_handle, "version", "", adios_integer, "", "", ""); databytes += sizeof(int);
  adios_define_var(g_handle, "extents", "", adios_integer, "6", "", ""); databytes += 6*sizeof(int);
  adios_define_var(g_handle, "num_blocks", "", adios_unsigned_long, "", "", ""); databytes += sizeof(int64_t);
  adios_define_var(g_handle, "num_points", "", adios_unsigned_long, "", "", ""); databytes += sizeof(int64_t);
  adios_define_var(g_handle, "num_cells", "", adios_unsigned_long, "", "", ""); databytes += sizeof(int64_t);
  adios_define_var (g_handle, "ntimestep", "", adios_unsigned_long, "", "", ""); databytes += sizeof(int64_t);
  adios_define_var (g_handle, "time", "", adios_double, "", "", ""); databytes += sizeof(double);

  // per block data.
  for (int64_t block=0; block < local_blocks; ++block)
    {
    adios_define_var(g_handle, "block/origin", "", adios_double, "3", "", ""); databytes += 3*sizeof(double);
    adios_define_var(g_handle, "block/spacing", "", adios_double, "3", "", ""); databytes += 3*sizeof(double);
    adios_define_var(g_handle, "block/extents", "", adios_integer, "6", "", ""); databytes += 6*sizeof(double);
    adios_define_var(g_handle, "block/num_points", "", adios_unsigned_long, "", "", ""); databytes += 6*sizeof(int64_t);
    adios_define_var(g_handle, "block/num_cells", "", adios_unsigned_long, "", "", ""); databytes += 6*sizeof(int64_t);
    adios_define_var(g_handle, "block/offset_points", "", adios_unsigned_long, "", "", ""); databytes += 6*sizeof(int64_t);
    adios_define_var(g_handle, "block/offset_cells", "", adios_unsigned_long, "", "", ""); databytes += 6*sizeof(int64_t);

    internals::define_attribute_data_vars(ds, block, vtkDataObject::POINT, g_handle,
      "block/pointdata/", "block/num_points", "num_points", "block/offset_points", databytes);
    internals::define_attribute_data_vars(ds, block, vtkDataObject::CELL, g_handle,
      "block/celldata/", "block/num_cells", "num_cells", "block/offset_cells", databytes);
    }

  this->FixedLengthVarSize = databytes;
  this->Initialized=true;
}

//----------------------------------------------------------------------------
void AnalysisAdaptor::WriteTimestep(DataAdaptor* data)
{
  vtkDataObject* mesh = data->GetCompleteMesh();

  int nprocs = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int64_t io_handle;
  adios_open(&io_handle, "sensei", this->FileName.c_str(),
    data->GetDataTimeStep() == 0? "w" : "a", MPI_COMM_WORLD);

  uint64_t total_size;
  adios_group_size(io_handle,
    internals::ComputeVariableLengthVarSize(mesh) + this->FixedLengthVarSize,
    &total_size);

  int64_t local_blocks = internals::CountLocalBlocks(mesh);
  int64_t total_blocks = internals::CountTotalBlocks(mesh);
  vtkDataSet* ds = internals::GetRepresentativeBlock(mesh);

  // write global data.
  adios_write(io_handle, "rank", &rank);
  adios_write(io_handle, "mpi_size", &nprocs);
  adios_write(io_handle, "num_blocks", &total_blocks);
  int data_type = ds->GetDataObjectType();
  adios_write(io_handle, "data_type", &data_type);
  int version = 1;
  adios_write(io_handle, "version", &version);

  int extents[6] = {0, -1, 0, -1, 0, -1};
  data->GetInformation()->Get(vtkDataObject::DATA_EXTENT(), extents);
  adios_write(io_handle, "extents", extents);

  int64_t local_num_points=0, local_num_cells=0;
  internals::CountLocalElements(mesh, &local_num_points, &local_num_cells);

  // compute inclusive scan to determine process offsets.
  int64_t global_offset_points=0, global_offset_cells=0, num_points=0, num_cells=0;
    {
    int64_t local_counts[] = { local_num_points, local_num_cells };
    int64_t global_offsets[2];
    MPI_Scan(&local_counts, &global_offsets, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    int64_t totals[2] = {global_offsets[0], global_offsets[1]};
    MPI_Bcast(totals, 2, MPI_UNSIGNED_LONG_LONG, (nprocs-1), MPI_COMM_WORLD);

    global_offset_points = global_offsets[0] - local_counts[0];
    global_offset_cells = global_offsets[1] - local_counts[1];
    num_points = totals[0];
    num_cells = totals[1];
    }

  adios_write(io_handle, "num_points", &num_points);
  adios_write(io_handle, "num_cells", &num_cells);

  int timestep = data->GetDataTimeStep();
  adios_write(io_handle, "ntimestep", &timestep);
	double time = data->GetDataTime();
	adios_write(io_handle, "time", &time);

  // write blocks.
  std::vector<vtkDataSet*> blocks;
  internals::GetBlocks(mesh, blocks);
  assert(blocks.size() == local_blocks);
  for (int64_t cc=0; cc < local_blocks; ++cc)
    {
    vtkDataSet* block = blocks[cc];
    double origin[3] = {0, 0, 0};
    double spacing[3] = {1, 1, 1};
    int lextents[6] = {0, -1, 0, -1, 0, -1};
    if (vtkImageData* img = vtkImageData::SafeDownCast(blocks[cc]))
      {
      img->GetOrigin(origin);
      img->GetSpacing(spacing);
      img->GetExtent(lextents);
      }
    int64_t bnum_points = block->GetNumberOfPoints();
    int64_t bnum_cells = block->GetNumberOfCells();

    adios_write(io_handle, "block/origin", origin);
    adios_write(io_handle, "block/spacing", spacing);
    adios_write(io_handle, "block/extents", lextents);
    adios_write(io_handle, "block/num_points", &bnum_points);
    adios_write(io_handle, "block/num_cells", &bnum_cells);
    adios_write(io_handle, "block/offset_points", &global_offset_points);
    adios_write(io_handle, "block/offset_cells", &global_offset_cells);

    internals::write_attribute_data_vars(block, vtkDataObject::POINT, io_handle, "block/pointdata/");
    internals::write_attribute_data_vars(block, vtkDataObject::CELL, io_handle, "block/celldata/");

    global_offset_cells += bnum_cells;
    global_offset_points += bnum_points;
    }
  adios_close (io_handle);
}

//----------------------------------------------------------------------------
void AnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

} // adios
} // sensei
