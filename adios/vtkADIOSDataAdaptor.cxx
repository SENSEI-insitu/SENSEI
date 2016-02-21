#include "vtkADIOSDataAdaptor.h"

#include <vtkObjectFactory.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkDataSetAttributes.h>

vtkStandardNewMacro(vtkADIOSDataAdaptor);
//----------------------------------------------------------------------------
vtkADIOSDataAdaptor::vtkADIOSDataAdaptor()
  : File(NULL),
  Comm(MPI_COMM_WORLD)
{
}

//----------------------------------------------------------------------------
vtkADIOSDataAdaptor::~vtkADIOSDataAdaptor()
{
}

//----------------------------------------------------------------------------
bool vtkADIOSDataAdaptor::Open(MPI_Comm comm,
  ADIOS_READ_METHOD method, const std::string& filename)
{
  this->Comm = comm;
  adios_read_init_method (method, comm, "verbose=1");
  this->File = adios_read_open(filename.c_str(), method, comm, ADIOS_LOCKMODE_ALL, -1);

  int mpi_size;
  adios_schedule_read(this->File, NULL, "mpi_size", 0, 1, &mpi_size);
  adios_perform_reads(this->File, 1);
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(this->Comm, &rank);
  if (size != mpi_size)
    {
    cerr << "Current implementation of vtkADIOSDataAdaptor only support "
      "matching mpi ranks. This will be fixed soon.";
    return false;
    }

  this->Mesh = vtkSmartPointer<vtkImageData>::New();

  int whole_extents[6], local_extents[6];
  double origin[3], spacing[3];

  ADIOS_SELECTION* selection = adios_selection_writeblock(rank);
  adios_schedule_read(this->File, NULL, "whole_extents", 0, 1, whole_extents);
  adios_schedule_read(this->File, selection, "local_extents", 0, 1, local_extents);
  adios_schedule_read(this->File, NULL, "origin", 0, 1, origin);
  adios_schedule_read(this->File, NULL, "spacing", 0, 1, spacing);
  adios_perform_reads(this->File, 1);

  this->Mesh->SetExtent(local_extents);
  this->Mesh->SetOrigin(origin);
  this->Mesh->SetSpacing(spacing);
  this->GetInformation()->Set(vtkDataObject::DATA_EXTENT(), whole_extents, 6);

  return adios_errno != err_end_of_stream;
}

//----------------------------------------------------------------------------
bool vtkADIOSDataAdaptor::Advance()
{
  adios_release_step(this->File);
  return adios_advance_step(this->File, 0, /*timeout*/0.5) == 0;
}

//----------------------------------------------------------------------------
bool vtkADIOSDataAdaptor::ReadStep()
{
  int tstep = 0; double time = 0;
  ADIOS_FILE* f = this->File;
  adios_schedule_read(f, NULL, "ntimestep", 0, 1, &tstep);
  adios_schedule_read(f, NULL, "time", 0, 1, &time);
  adios_perform_reads(f, 1);
  this->SetDataTimeStep(tstep);
  this->SetDataTime(time);

  this->AssociatedArrays.clear();
  for (int cc=0; cc < this->File->nvars; ++cc)
    {
    int association = vtkDataObject::FIELD_ASSOCIATION_POINTS;
    const char* name = NULL;
    if (strncmp(this->File->var_namelist[cc], "cellcentered/", strlen("cellcentered/")) == 0)
      {
      association = vtkDataObject::FIELD_ASSOCIATION_CELLS;
      name = this->File->var_namelist[cc] + strlen("cellcentered/");
      }
    if (strncmp(this->File->var_namelist[cc], "pointcentered/", strlen("pointcentered/")) == 0)
      {
      association = vtkDataObject::FIELD_ASSOCIATION_POINTS;
      name = this->File->var_namelist[cc] + strlen("pointcentered/");
      }
    // TODO: handle all other possible assocations.
    if (!name) { continue; }
    this->AssociatedArrays[association][name] = NULL;
    }
}

//----------------------------------------------------------------------------
vtkDataObject* vtkADIOSDataAdaptor::GetMesh(bool /*structure_only*/)
{
  return this->Mesh;
}

//----------------------------------------------------------------------------
bool vtkADIOSDataAdaptor::AddArray(vtkDataObject* mesh, int association, const char* arrayname)
{
  ArraysType& arrays = this->AssociatedArrays[association];
  ArraysType::iterator iter = arrays.find(arrayname);
  if (iter == arrays.end())
    {
    return false;
    }

  vtkImageData* image = vtkImageData::SafeDownCast(mesh);
  assert(image != NULL);

  if (iter->second.GetPointer() == NULL)
    {
    int rank;
    MPI_Comm_rank(this->Comm, &rank);

    vtkDoubleArray* vtkarray = vtkDoubleArray::New();
    vtkarray->SetName(arrayname);
    vtkarray->SetNumberOfTuples(
      association == vtkDataObject::FIELD_ASSOCIATION_POINTS?
      image->GetNumberOfPoints() : image->GetNumberOfCells());
    vtkarray->FillComponent(0, 0);

    std::string prefix(association == vtkDataObject::FIELD_ASSOCIATION_POINTS?
      "pointcentered/" : "cellcentered/");

    ADIOS_SELECTION* selection = adios_selection_writeblock(rank);
    adios_schedule_read(this->File, selection,
      (prefix + arrayname).c_str(),
      0,
      1,
      vtkarray->GetPointer(0));
    adios_perform_reads(this->File, 1);

    vtkarray->Modified();
    cout << vtkarray->GetNumberOfTuples() << endl;
    cout << "Range: " << vtkarray->GetRange()[0] << ", " << vtkarray->GetRange()[1] << endl;
    iter->second.TakeReference(vtkarray);
    image->GetAttributes(association)->AddArray(vtkarray);
    return true;
    }
  return true;
}

//----------------------------------------------------------------------------
unsigned int vtkADIOSDataAdaptor::GetNumberOfArrays(int association)
{
  return static_cast<unsigned int>(this->AssociatedArrays[association].size());
}

//----------------------------------------------------------------------------
const char* vtkADIOSDataAdaptor::GetArrayName(int association, unsigned int index)
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
void vtkADIOSDataAdaptor::ReleaseData()
{
  this->AssociatedArrays.clear();
}

//----------------------------------------------------------------------------
void vtkADIOSDataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
