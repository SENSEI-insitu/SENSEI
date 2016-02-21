#include "bridge.h"

#include "dataadaptor.h"

#ifdef ENABLE_HISTOGRAM
#include "HistogramAnalysisAdaptor.h"
#endif
#ifdef ENABLE_AUTOCORRELATION
#include <AutocorrelationAnalysisAdaptor.h>
#endif
#ifdef ENABLE_CATALYST
#include <vtkCatalystAnalysisAdaptor.h>
# ifdef ENABLE_CATALYST_SLICE
#include <vtkCatalystSlicePipeline.h>
# endif
#endif
#ifdef ENABLE_ADIOS
#include <vtkADIOSAnalysisAdaptor.h>
#endif
#include <vtkDataObject.h>
#include <vtkInsituAnalysisAdaptor.h>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vector>

namespace bridge
{
static vtkNew<oscillators::DataAdaptor> GlobalDataAdaptor;
typedef std::vector<vtkSmartPointer<vtkInsituAnalysisAdaptor> > GlobalAnalysesType;
static GlobalAnalysesType GlobalAnalyses;
bool ExecuteGlobalAnalyses()
{
  for (auto a : GlobalAnalyses)
    {
    if (!a->Execute(GlobalDataAdaptor.GetPointer()))
      {
      return false;
      }
    }
  return true;
}

//-----------------------------------------------------------------------------
void initialize(MPI_Comm world,
                size_t window,
                size_t nblocks,
                size_t n_local_blocks,
                int domain_shape_x, int domain_shape_y, int domain_shape_z,
                int* gid,
                int* from_x, int* from_y, int* from_z,
                int* to_x,   int* to_y,   int* to_z)
{
  GlobalDataAdaptor->Initialize(nblocks);
  GlobalDataAdaptor->SetDataTimeStep(-1);
  for (size_t cc=0; cc < n_local_blocks; ++cc)
    {
    GlobalDataAdaptor->SetBlockExtent(gid[cc],
      from_x[cc], to_x[cc],
      from_y[cc], to_y[cc],
      from_z[cc], to_z[cc]);
    }
#ifdef ENABLE_HISTOGRAM
  vtkNew<HistogramAnalysisAdaptor> histogram;
  histogram->Initialize(world, 10, vtkDataObject::FIELD_ASSOCIATION_CELLS, "data");
  GlobalAnalyses.push_back(histogram.GetPointer());
#endif
#ifdef ENABLE_AUTOCORRELATION
  vtkNew<AutocorrelationAnalysisAdaptor> autocorrelation;
  autocorrelation->Initialize(world,
                              window,
                              vtkDataObject::FIELD_ASSOCIATION_CELLS, "data");
  GlobalAnalyses.push_back(autocorrelation.GetPointer());
#endif
#ifdef ENABLE_CATALYST
  vtkNew<vtkCatalystAnalysisAdaptor> catalyst;
  GlobalAnalyses.push_back(catalyst.GetPointer());
# ifdef ENABLE_CATALYST_SLICE
  vtkNew<vtkCatalystSlicePipeline> slicePipeline;
  slicePipeline->SetSliceOrigin(domain_shape_x/2.0, domain_shape_y/2.0, domain_shape_z/2.0);
  slicePipeline->SetSliceNormal(0, 0, 1);
  slicePipeline->ColorBy(vtkDataObject::FIELD_ASSOCIATION_CELLS, "data");
  catalyst->AddPipeline(slicePipeline.GetPointer());
# endif
#endif

#ifdef ENABLE_ADIOS
  vtkNew<vtkADIOSAnalysisAdaptor> adios;
  adios->SetFileName("oscillators.bp");
  GlobalAnalyses.push_back(adios.GetPointer());
#endif

}

//-----------------------------------------------------------------------------
void set_data(int gid, float* data)
{
  GlobalDataAdaptor->SetBlockData(gid, data);
}

//-----------------------------------------------------------------------------
void analyze(float time)
{
  GlobalDataAdaptor->SetDataTime(time);
  GlobalDataAdaptor->SetDataTimeStep(GlobalDataAdaptor->GetDataTimeStep() + 1);
  ExecuteGlobalAnalyses();
  GlobalDataAdaptor->ReleaseData();
}

//-----------------------------------------------------------------------------
void finalize(size_t k_max, size_t nblocks)
{
#ifdef ENABLE_AUTOCORRELATION
  for (auto a : GlobalAnalyses)
    {
    if (AutocorrelationAnalysisAdaptor* aca = AutocorrelationAnalysisAdaptor::SafeDownCast(a))
      {
      aca->PrintResults(k_max);
      }
    }
#endif
  GlobalAnalyses.clear();
}

}
