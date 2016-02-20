#include "vtkADIOSAnalysisAdaptor.h"

#include <vtkDataSetAttributes.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkInsituDataAdaptor.h>
#include <vtkObjectFactory.h>

#include <mpi.h>
#include <adios.h>

vtkStandardNewMacro(vtkADIOSAnalysisAdaptor);
//----------------------------------------------------------------------------
vtkADIOSAnalysisAdaptor::vtkADIOSAnalysisAdaptor() : Initialized(false)
{
  this->SetMethod("MPI");
  this->SetFileName("sensei.bp");
}

//----------------------------------------------------------------------------
vtkADIOSAnalysisAdaptor::~vtkADIOSAnalysisAdaptor()
{
  this->SetMethod(NULL);
  this->SetFileName(NULL);
}

//----------------------------------------------------------------------------
bool vtkADIOSAnalysisAdaptor::Execute(vtkInsituDataAdaptor* data)
{
  vtkDataObject* mesh = data->GetCompleteMesh();
  vtkImageData* image = vtkImageData::SafeDownCast(mesh);
  if (!image)
    {
    vtkGenericWarningMacro("We currently only support image data.");
    return false;
    }

  this->InitializeADIOS(data);
  this->WriteTimestep(data);
  return true;
}

//----------------------------------------------------------------------------
void vtkADIOSAnalysisAdaptor::InitializeADIOS(vtkInsituDataAdaptor* data)
{
  // Ideally, we'd develop code that can write any VTK data object to ADIOS
  // using a schema that we can call VTK_ADIOS_SCHEMA. VTK and consequently
  // ParaView, Catalyst, VisIt, Libsim all can then develop readers/writers that
  // read/write this schema. Should this simply be the ADIOS Vis Schema? Maybe.
  // As an example, I modifying Matt's code from the histogram miniapp campaign
  // to write out 3D grids.
  if (this->Initialized)
    {
    return;
    }

  int64_t g_handle;
  adios_init_noxml(MPI_COMM_WORLD);
  adios_allocate_buffer(ADIOS_BUFFER_ALLOC_NOW, 500);
  adios_declare_group(&g_handle, "sensei", "", adios_flag_yes);
  adios_select_method(g_handle, this->Method.c_str(), "", "");
  adios_define_var(g_handle, "rank", "", adios_integer, "", "", "");
  adios_define_var(g_handle, "mpi_size", "", adios_integer, "", "", "");

  // save global extents (same value on all ranks).
  // adios_define_var(group_id, name, path, type, dimensions, global_dimensions, local_offsets)
  adios_define_var(g_handle, "whole_extents", "", adios_integer, "6", "", "");
  // save local extents per rank.
  adios_define_var(g_handle, "local_extents", "", adios_integer, "1,6", "mpi_size,6", "rank,0");

  adios_define_var(g_handle, "origin", "", adios_double, "3", "", "");
  adios_define_var(g_handle, "spacing", "", adios_double, "3", "", "");

  // save array sizes.
  adios_define_var(g_handle, "local_point_array_size", "", adios_unsigned_long, "", "", "");
  adios_define_var(g_handle, "total_point_array_size", "", adios_unsigned_long, "", "", "");
  adios_define_var(g_handle, "local_point_array_offset", "", adios_unsigned_long, "", "", "");

  adios_define_var(g_handle, "local_cell_array_size", "", adios_unsigned_long, "", "", "");
  adios_define_var(g_handle, "total_cell_array_size", "", adios_unsigned_long, "", "", "");
  adios_define_var(g_handle, "local_cell_array_offset", "", adios_unsigned_long, "", "", "");

  for (unsigned int cc=0, max=data->GetNumberOfArrays(vtkDataObject::FIELD_ASSOCIATION_POINTS); cc < max; ++cc)
    {
    adios_define_var(g_handle, data->GetArrayName(vtkDataObject::FIELD_ASSOCIATION_POINTS, cc), "",
      adios_double /*FIXME, assume double for now*/,
      "local_point_array_size", "total_point_array_size", "local_point_array_offset");
    }
  for (unsigned int cc=0, max=data->GetNumberOfArrays(vtkDataObject::FIELD_ASSOCIATION_CELLS); cc < max; ++cc)
    {
    adios_define_var(g_handle, data->GetArrayName(vtkDataObject::FIELD_ASSOCIATION_CELLS, cc), "",
      adios_double /*FIXME, assume double for now*/,
      "local_cell_array_size", "total_cell_array_size", "local_cell_array_offset");
    }

  // define the timestep.
  adios_define_var (g_handle, "ntimestep", "", adios_unsigned_long, "", "", "");

  this->Initialized=true;
}

//----------------------------------------------------------------------------
void vtkADIOSAnalysisAdaptor::WriteTimestep(vtkInsituDataAdaptor* data)
{
  vtkDataObject* mesh = data->GetCompleteMesh();
  vtkImageData* image = vtkImageData::SafeDownCast(mesh);
  assert(image);

  int nprocs = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int64_t metadata_bytes = 2*sizeof(int)  + 6*sizeof(int) + 6*sizeof(int)
    + 2*3*sizeof(double) + 6*sizeof(unsigned long);
  int64_t array_bytes =
    sizeof(double)*image->GetNumberOfPoints() * data->GetNumberOfArrays(vtkDataObject::FIELD_ASSOCIATION_POINTS);
  array_bytes +=
    sizeof(double)*image->GetNumberOfCells() * data->GetNumberOfArrays(vtkDataObject::FIELD_ASSOCIATION_CELLS);
  uint64_t total_size;
  int64_t io_handle;
  adios_open(&io_handle, "sensei", this->FileName.c_str(), "w", MPI_COMM_WORLD);

  adios_group_size (io_handle, metadata_bytes + array_bytes, &total_size);
  adios_write (io_handle, "rank", &rank);
  adios_write (io_handle, "mpi_size", &nprocs);

  adios_write(io_handle, "whole_extents", data->GetInformation()->Get(vtkDataObject::DATA_EXTENT()));
  adios_write(io_handle, "local_extents", image->GetExtent());
  adios_write(io_handle, "origin", image->GetOrigin());
  adios_write(io_handle, "spacing", image->GetSpacing());

  uint64_t local_point_array_size = static_cast<uint64_t>(image->GetNumberOfPoints());
  uint64_t local_point_array_offset = rank * local_point_array_size;
  uint64_t total_point_array_size = nprocs * local_point_array_size;

  adios_write(io_handle, "local_point_array_size", &local_point_array_size);
  adios_write(io_handle, "total_point_array_size", &total_point_array_size);
  adios_write(io_handle, "local_point_array_offset", &local_point_array_offset);

  uint64_t local_cell_array_size = static_cast<uint64_t>(image->GetNumberOfCells());
  uint64_t local_cell_array_offset = rank * local_cell_array_size;
  uint64_t total_cell_array_size = nprocs * local_cell_array_size;

  adios_write(io_handle, "local_cell_array_size", &local_cell_array_size);
  adios_write(io_handle, "total_cell_array_size", &total_cell_array_size);
  adios_write(io_handle, "local_cell_array_offset", &local_cell_array_offset);

  int timestep = data->GetDataTimeStep();
  adios_write (io_handle, "ntimestep", &timestep);
  for (int attr=vtkDataObject::FIELD_ASSOCIATION_POINTS; attr <= vtkDataObject::FIELD_ASSOCIATION_CELLS; ++attr)
    {
    vtkDataSetAttributes* dsa = image->GetAttributes(attr);
    for (int cc=0, max=dsa->GetNumberOfArrays(); cc<max; ++cc)
      {
      if (vtkDoubleArray* da = vtkDoubleArray::SafeDownCast(dsa->GetArray(cc)))
        {
        adios_write(io_handle, da->GetName(), da->GetPointer(0));
        }
      }
    }
  adios_close (io_handle);
}

//----------------------------------------------------------------------------
void vtkADIOSAnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
