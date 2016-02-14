#include "insitu_bridge.h"

#include "infrastructure/vtkInsituAnalysisAdaptor.h"
#include "infrastructure/vtkInsituDataAdaptor.h"

#include "vtkDataObject.h"
#include "vtkDoubleArray.h"
#include "vtkObjectFactory.h"
#include "vtkSmartPointer.h"
#ifndef USE_GENERIC_ARRAYS_API
#include "histogram.h"
#else
#include "histogram_generic_arrays.h"
#endif

// Each simulation code in its bridge, implements a subclass of
// vtkInsituDataAdaptor to map the simulation datastructures to VTK data model.
namespace BridgeInternals
{

class vtk3DGridInsituDataAdaptor : public vtkInsituDataAdaptor
{
public:
  static vtk3DGridInsituDataAdaptor* New();
  vtkTypeMacro(vtk3DGridInsituDataAdaptor, vtkInsituDataAdaptor);

  bool Initialize(
    int g_x, int g_y, int g_z,
    int l_x, int l_y, int l_z,
    uint64_t start_extents_x, uint64_t start_extents_y, uint64_t start_extents_z,
    int tot_blocks_x, int tot_blocks_y, int tot_blocks_z,
    int block_id_x, int block_id_y, int block_id_z)
    {
    // we only really need to save the local extents for our current example. So
    // we'll just save that.
    this->Extent[0] = start_extents_x;
    this->Extent[1] = start_extents_x + l_x - 1;
    this->Extent[2] = start_extents_y;
    this->Extent[3] = start_extents_y + l_y - 1;
    this->Extent[4] = start_extents_z;
    this->Extent[5] = start_extents_z + l_z - 1;
    }

  void SetArrays(double* pressure, double *temperature, double* density)
    {
    this->PressurePtr = pressure;
    this->TemperaturePtr = temperature;
    this->DensityPtr = density;
    }

  virtual vtkDataObject* GetMesh()
    {
    // our analysis doesn't need a mesh so we punt on it for now.
    // In theory, we'll create a new vtkImageData and return that.
    vtkGenericWarningMacro("TODO: Not implemented currently.");
    return NULL;
    }

  // Description:
  // Subclasses should override this method to provide field arrays.
  virtual vtkAbstractArray* GetArray(int association, const char* name)
    {
    if (association != vtkDataObject::FIELD_ASSOCIATION_POINTS)
      {
      return NULL;
      }
    vtkIdType size = (this->Extent[1] - this->Extent[0] + 1) *
      (this->Extent[3] - this->Extent[2] + 1) *
      (this->Extent[5] - this->Extent[4] + 1);

    if (strcmp(name, "pressure") == 0)
      {
      if (this->Pressure == NULL)
        {
        this->Pressure = vtkSmartPointer<vtkDoubleArray>::New();
        this->Pressure->SetName("pressure");
        this->Pressure->SetArray(this->PressurePtr, size, 1);
        }
      return this->Pressure;
      }
    else if (strcmp(name, "temperature") == 0)
      {
      if (this->Temperature == NULL)
        {
        this->Temperature = vtkSmartPointer<vtkDoubleArray>::New();
        this->Temperature->SetName("temperature");
        this->Temperature->SetArray(this->TemperaturePtr, size, 1);
        }
      return this->Temperature;
      }
    else if (strcmp(name, "density") == 0)
      {
      if (this->Density == NULL)
        {
        this->Density = vtkSmartPointer<vtkDoubleArray>::New();
        this->Density->SetName("density");
        this->Density->SetArray(this->DensityPtr, size, 1);
        }
      return this->Density;
      }
    return NULL;
    }

  virtual unsigned int GetNumberOfArrays(int association)
    {
    if (association != vtkDataObject::FIELD_ASSOCIATION_POINTS)
      {
      return 0;
      }
    return 3;
    }

  virtual const char* GetArrayName(int association, unsigned int index)
    {
    if (association != vtkDataObject::FIELD_ASSOCIATION_POINTS)
      {
      return NULL;
      }
    switch (index)
      {
    case 0: return "pressure";
    case 1: return "temperature";
    case 2: return "density";
    default: return NULL;
      }
    }
  // Description:
  // Method called to release data and end of each execution iteration.
  virtual void ReleaseData()
    {
    this->PressurePtr = NULL;
    this->TemperaturePtr = NULL;
    this->DensityPtr = NULL;
    this->Pressure = NULL;
    this->Temperature = NULL;
    this->Density = NULL;
    }

protected:
  vtk3DGridInsituDataAdaptor()
    {
    this->Extent[0] = this->Extent[2] = this->Extent[4] = -1;
    this->Extent[1] = this->Extent[3] = this->Extent[5] = 0;
    this->PressurePtr = NULL;
    this->TemperaturePtr = NULL;
    this->DensityPtr = NULL;
    }
  ~vtk3DGridInsituDataAdaptor()
    {
    }

  int Extent[6];
  double* PressurePtr;
  double* TemperaturePtr;
  double* DensityPtr;

  vtkSmartPointer<vtkDoubleArray> Pressure;
  vtkSmartPointer<vtkDoubleArray> Temperature;
  vtkSmartPointer<vtkDoubleArray> Density;
private:
  vtk3DGridInsituDataAdaptor(const vtk3DGridInsituDataAdaptor&);
  void operator=(const vtk3DGridInsituDataAdaptor);
};
vtkStandardNewMacro(vtk3DGridInsituDataAdaptor);


class vtkHistogramAnalysisAdaptor : public vtkInsituAnalysisAdaptor
{
public:
  static vtkHistogramAnalysisAdaptor* New();
  vtkTypeMacro(vtkHistogramAnalysisAdaptor, vtkInsituAnalysisAdaptor);

  void Initialize(MPI_Comm comm, int bins)
    {
    this->Communicator = comm;
    this->Bins = bins;
    }

  virtual bool Execute(vtkInsituDataAdaptor* data)
    {
#ifndef USE_GENERIC_ARRAYS_API
    // downcase to vtkDoubleArray and call GetPointer() on it.
    if (vtkDoubleArray* array = vtkDoubleArray::SafeDownCast(
        data->GetArray(vtkDataObject::FIELD_ASSOCIATION_POINTS, "pressure")))
      {
      histogram(this->Communicator,
        array->GetPointer(0), array->GetNumberOfTuples(), this->Bins);
      }
    if (vtkDoubleArray* array = vtkDoubleArray::SafeDownCast(
        data->GetArray(vtkDataObject::FIELD_ASSOCIATION_POINTS, "temperature")))
      {
      histogram(this->Communicator,
        array->GetPointer(0), array->GetNumberOfTuples(), this->Bins);
      }
    if (vtkDoubleArray* array = vtkDoubleArray::SafeDownCast(
        data->GetArray(vtkDataObject::FIELD_ASSOCIATION_POINTS, "density")))
      {
      histogram(this->Communicator,
        array->GetPointer(0), array->GetNumberOfTuples(), this->Bins);
      }
#else
    histogram(this->Communicator, vtkDataArray::SafeDownCast(
        data->GetArray(vtkDataObject::FIELD_ASSOCIATION_POINTS, "pressure")), this->Bins);
    histogram(this->Communicator, vtkDataArray::SafeDownCast(
        data->GetArray(vtkDataObject::FIELD_ASSOCIATION_POINTS, "temperature")), this->Bins);
    histogram(this->Communicator, vtkDataArray::SafeDownCast(
        data->GetArray(vtkDataObject::FIELD_ASSOCIATION_POINTS, "density")), this->Bins);
#endif
    return true;
    }

protected:
  vtkHistogramAnalysisAdaptor()
    : Communicator(MPI_COMM_WORLD), Bins(0)
    {
    }
  ~vtkHistogramAnalysisAdaptor()
    {
    }

  MPI_Comm Communicator;
  int Bins;
private:
  vtkHistogramAnalysisAdaptor(const vtkHistogramAnalysisAdaptor&);
  void operator=(const vtkHistogramAnalysisAdaptor&);
};
vtkStandardNewMacro(vtkHistogramAnalysisAdaptor);

static vtkSmartPointer<vtk3DGridInsituDataAdaptor> DataAdaptor;
static vtkSmartPointer<vtkHistogramAnalysisAdaptor> Analysis;
}

//-----------------------------------------------------------------------------
void insitu_bridge_initialize(MPI_Comm comm,
  int g_x, int g_y, int g_z,
  int l_x, int l_y, int l_z,
  uint64_t start_extents_x, uint64_t start_extents_y, uint64_t start_extents_z,
  int tot_blocks_x, int tot_blocks_y, int tot_blocks_z,
  int block_id_x, int block_id_y, int block_id_z,
  int bins)
{
  if (!BridgeInternals::DataAdaptor)
    {
    BridgeInternals::DataAdaptor =
      vtkSmartPointer<BridgeInternals::vtk3DGridInsituDataAdaptor>::New();
    }
  BridgeInternals::DataAdaptor->Initialize(
    g_x, g_y, g_z,
    l_x, l_y, l_z,
    start_extents_x, start_extents_y, start_extents_z,
    tot_blocks_x, tot_blocks_y, tot_blocks_z,
    block_id_x, block_id_y, block_id_z);

  if (!BridgeInternals::Analysis)
    {
    // TODO: based on which insitu infrastructure we want to use for the
    // analysis, we will instantiate the appropriate vtkHistogramAnalysisAdaptor
    // subclass.
    BridgeInternals::Analysis =
      vtkSmartPointer<BridgeInternals::vtkHistogramAnalysisAdaptor>::New();
    }
  BridgeInternals::Analysis->Initialize(comm, bins);
}

//-----------------------------------------------------------------------------
void insitu_bridge_update(double *pressure, double* temperature, double* density)
{
  BridgeInternals::DataAdaptor->SetArrays(pressure, temperature, density);
  BridgeInternals::Analysis->Execute(BridgeInternals::DataAdaptor);
  BridgeInternals::DataAdaptor->ReleaseData();
}

//-----------------------------------------------------------------------------
void insitu_bridge_finalize()
{
  BridgeInternals::Analysis = NULL;
  BridgeInternals::DataAdaptor = NULL;
}

