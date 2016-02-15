#include "bridge.h"

#include "dataadaptor.h"

#ifdef ENABLE_HISTOGRAM
#include "HistogramAnalysisAdaptor.h"
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
  for (size_t cc=0; cc < n_local_blocks; ++cc)
    {
    GlobalDataAdaptor->SetBlockExtent(gid[cc],
      from_x[cc], to_x[cc],
      from_y[cc], to_y[cc],
      from_z[cc], to_z[cc]);
    }
#ifdef ENABLE_HISTOGRAM
  vtkNew<HistogramAnalysisAdaptor> histogram;
  histogram->Initialize(world, 10, vtkDataObject::FIELD_ASSOCIATION_POINTS, "data");
  GlobalAnalyses.push_back(histogram.GetPointer());
#endif
}

//-----------------------------------------------------------------------------
void set_data(int gid, float* data)
{
  GlobalDataAdaptor->SetBlockData(gid, data);
}

//-----------------------------------------------------------------------------
void analyze()
{
  ExecuteGlobalAnalyses();
  GlobalDataAdaptor->ReleaseData();
}

//-----------------------------------------------------------------------------
void finalize(size_t k_max, size_t nblocks)
{
  GlobalAnalyses.clear();
}

}
