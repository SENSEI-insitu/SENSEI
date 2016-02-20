#include "Bridge.h"

#include "DataAdaptor.h"
#include "vtkInsituAnalysisAdaptor.h"
#include "vtkNew.h"
#include "vtkDataObject.h"

#ifdef ENABLE_HISTOGRAM
#include "HistogramAnalysisAdaptor.h"
#endif
#ifdef ENABLE_ADIOS
#include <vtkADIOSAnalysisAdaptor.h>
#endif
#ifdef ENABLE_CATALYST
#include <vtkCatalystAnalysisAdaptor.h>
# ifdef ENABLE_CATALYST_SLICE
#include <vtkCatalystSlicePipeline.h>
# endif
#endif

#include <vector>
namespace BridgeInternals
{
  static vtkSmartPointer<parallel3d::DataAdaptor> GlobalDataAdaptor;
  typedef std::vector<vtkSmartPointer<vtkInsituAnalysisAdaptor> > GlobalAnalysesType;
  static GlobalAnalysesType GlobalAnalyses;

  bool ExecuteGlobalAnalyses()
    {
    for (GlobalAnalysesType::const_iterator iter = GlobalAnalyses.begin();
      iter != GlobalAnalyses.end(); ++iter)
      {
      if (!iter->GetPointer()->Execute(GlobalDataAdaptor.GetPointer()))
        {
        return false;
        }
      }
    return true;
    }
}

//-----------------------------------------------------------------------------
void bridge_initialize(MPI_Comm comm,
  int g_x, int g_y, int g_z,
  int l_x, int l_y, int l_z,
  uint64_t start_extents_x, uint64_t start_extents_y, uint64_t start_extents_z,
  int tot_blocks_x, int tot_blocks_y, int tot_blocks_z,
  int block_id_x, int block_id_y, int block_id_z,
  int bins)
{
  if (!BridgeInternals::GlobalDataAdaptor)
    {
    BridgeInternals::GlobalDataAdaptor = vtkSmartPointer<parallel3d::DataAdaptor>::New();
    }
  BridgeInternals::GlobalDataAdaptor->Initialize(
    g_x, g_y, g_z,
    l_x, l_y, l_z,
    start_extents_x, start_extents_y, start_extents_z,
    tot_blocks_x, tot_blocks_y, tot_blocks_z,
    block_id_x, block_id_y, block_id_z);

#ifdef ENABLE_HISTOGRAM
  vtkNew<HistogramAnalysisAdaptor> histogram[3];
  histogram[0]->Initialize(comm, bins, vtkDataObject::FIELD_ASSOCIATION_CELLS, "pressure");
  histogram[1]->Initialize(comm, bins, vtkDataObject::FIELD_ASSOCIATION_CELLS, "temperature");
  histogram[2]->Initialize(comm, bins, vtkDataObject::FIELD_ASSOCIATION_CELLS, "density");

  BridgeInternals::GlobalAnalyses.push_back(histogram[0].GetPointer());
  BridgeInternals::GlobalAnalyses.push_back(histogram[1].GetPointer());
  BridgeInternals::GlobalAnalyses.push_back(histogram[2].GetPointer());
#endif

#ifdef ENABLE_ADIOS
  vtkNew<vtkADIOSAnalysisAdaptor> adios;
  BridgeInternals::GlobalAnalyses.push_back(adios.GetPointer());
#endif

#ifdef ENABLE_CATALYST
  vtkNew<vtkCatalystAnalysisAdaptor> catalyst;
  BridgeInternals::GlobalAnalyses.push_back(catalyst.GetPointer());
# ifdef ENABLE_CATALYST_SLICE
  vtkNew<vtkCatalystSlicePipeline> slicePipeline;
  slicePipeline->SetSliceOrigin(g_x/2.0, g_y/2.0, g_z/2.0);
  slicePipeline->SetSliceNormal(0, 0, 1);
  slicePipeline->ColorBy(vtkDataObject::FIELD_ASSOCIATION_CELLS, "pressure");
  catalyst->AddPipeline(slicePipeline.GetPointer());
# endif
#endif
}

//-----------------------------------------------------------------------------
void bridge_update(double *pressure, double* temperature, double* density)
{
  BridgeInternals::GlobalDataAdaptor->AddArray("pressure", pressure);
  BridgeInternals::GlobalDataAdaptor->AddArray("temperature", temperature);
  BridgeInternals::GlobalDataAdaptor->AddArray("density", density);
  BridgeInternals::ExecuteGlobalAnalyses();
  BridgeInternals::GlobalDataAdaptor->ReleaseData();
}

//-----------------------------------------------------------------------------
void bridge_finalize()
{
  BridgeInternals::GlobalAnalyses.clear();
  BridgeInternals::GlobalDataAdaptor = NULL;
}
