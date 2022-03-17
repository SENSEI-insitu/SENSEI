#include "Autocorrelation.h"

#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "SVTKUtils.h"
#include "Profiler.h"
#include "Error.h"

// SVTK includes
#include <svtkCompositeDataIterator.h>
#include <svtkCellData.h>
#include <svtkFieldData.h>
#include <svtkFloatArray.h>
#include <svtkImageData.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkObjectFactory.h>
#include <svtkSmartPointer.h>
#include <svtkStructuredData.h>
#include <svtkUnsignedCharArray.h>

#include <memory>
#include <vector>

#include <sdiy/master.hpp>
#include <sdiy/reduce.hpp>
#include <sdiy/partners/merge.hpp>
#include <sdiy/io/numpy.hpp>
#include <sdiy/grid.hpp>
#include <sdiy/vertices.hpp>


// http://stackoverflow.com/a/12580468
template<typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args&& ...args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template<class T>
struct add_vectors
{
    using Vector = std::vector<T>;
    Vector  operator()(const Vector& lhs, const Vector& rhs) const
    {
      Vector res(lhs.size(), 0);
      for (size_t i = 0; i < lhs.size(); ++i)
        res[i] = lhs[i] + rhs[i];
      return res;
    }
};

namespace sdiy { namespace mpi { namespace detail {
template<class U> struct mpi_op<add_vectors<U>>
{
  static MPI_Op get()
  { return MPI_SUM; }
};
}}}

namespace sensei
{

using GridRef = sdiy::GridRef<float,3>;
using Vertex  = GridRef::Vertex;
using Vertex4D = Vertex::UPoint;

struct AutocorrelationImpl
{
  using Grid = sdiy::Grid<float,4>;
  AutocorrelationImpl(size_t window_, int gid_, Vertex from_, Vertex to_):
    window(window_),
    gid(gid_),
    from(from_), to(to_),
    shape(to - from + Vertex::one()),
    // init grid with (to - from + 1) in 3D, and window in the 4-th dimension
    values(shape.lift(3, window)),
    corr(shape.lift(3, window))
  { corr = 0; }

  static void* create()            { return new AutocorrelationImpl; }
  static void destroy(void* b)    { delete static_cast<AutocorrelationImpl*>(b); }
  void process(float* data, unsigned char *ghostArray)
    {
    GridRef g(data, shape);

    if (ghostArray)
      {
      sdiy::GridRef<unsigned char, 3> ghost(ghostArray, shape);
      // record the values
      sdiy::for_each(g.shape(), [&](const Vertex& v)
        {
        auto gv = (ghost(v) == 0) ? g(v) : 0;

        for (size_t i = 1; i <= window; ++i)
        {
        if (i > count) continue;    // during the initial fill, we don't get contributions to some shifts

        auto uc = v.lift(3, i-1);
        auto uv = v.lift(3, (offset + window - i) % window);
        corr(uc) += values(uv)*gv;
        }

        auto u = v.lift(3, offset);
        values(u) = gv;
        });
      }
    else
      {
      // record the values
      sdiy::for_each(g.shape(), [&](const Vertex& v)
        {
        auto gv = g(v);

        for (size_t i = 1; i <= window; ++i)
        {
        if (i > count) continue;    // during the initial fill, we don't get contributions to some shifts

        auto uc = v.lift(3, i-1);
        auto uv = v.lift(3, (offset + window - i) % window);
        corr(uc) += values(uv)*gv;
        }

        auto u = v.lift(3, offset);
        values(u) = gv;
        });
      }
    offset += 1;
    offset %= window;

    ++count;
    }

  size_t          window;
  int             gid;
  Vertex          from, to, shape;
  Grid            values;     // circular buffer of last `window` values
  Grid            corr;       // autocorrelations for different time shifts

  size_t          offset = 0;
  size_t          count  = 0;

private:
  AutocorrelationImpl() {}        // here just for create; to let Master manage the blocks (+ if we choose to add OOC later)
};

//-----------------------------------------------------------------------------
class Autocorrelation::AInternals
{
public:
  std::unique_ptr<sdiy::Master> Master;
  size_t KMax;
  std::string MeshName;
  int Association;
  std::string ArrayName;
  size_t Window;
  bool BlocksInitialized;
  size_t NumberOfBlocks;

  AInternals() : KMax(3), Association(svtkDataObject::POINT),
    Window(10), BlocksInitialized(false), NumberOfBlocks(0) {}

  void InitializeBlocks(svtkDataObject* dobj)
    {
    if (this->BlocksInitialized)
      {
      return;
      }
    if (svtkImageData* img = svtkImageData::SafeDownCast(dobj))
      {
      int ext[6];
      img->GetExtent(ext);
      if (this->Association == svtkDataObject::CELL)
        {
#if SVTK_MAJOR_VERSION == 6 && SVTK_MINOR_VERSION == 1
        svtkStructuredData::GetCellExtentFromNodeExtent(ext, ext);
#else
        svtkStructuredData::GetCellExtentFromPointExtent(ext, ext);
#endif
        }
      Vertex from { ext[0], ext[2], ext[4] };
      Vertex to   { ext[1], ext[3], ext[5] };
      int bid = this->Master->communicator().rank();
      AutocorrelationImpl* b = new AutocorrelationImpl(this->Window, bid, from, to);
      this->Master->add(bid, b, new sdiy::Link);
      this->NumberOfBlocks = this->Master->communicator().size();
      }
    else if (svtkCompositeDataSet* cd = svtkCompositeDataSet::SafeDownCast(dobj))
      {
      // add blocks
      svtkSmartPointer<svtkCompositeDataIterator> iter;
      iter.TakeReference(cd->NewIterator());
      iter->SkipEmptyNodesOff();

      int bid = 0;
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem(), ++bid)
        {
        if (svtkImageData* id = svtkImageData::SafeDownCast(iter->GetCurrentDataObject()))
          {
          int ext[6];
          id->GetExtent(ext);
          if (this->Association == svtkDataObject::CELL)
            {
#if SVTK_MAJOR_VERSION == 6 && SVTK_MINOR_VERSION == 1
            svtkStructuredData::GetCellExtentFromNodeExtent(ext, ext);
#else
            svtkStructuredData::GetCellExtentFromPointExtent(ext, ext);
#endif
            }
          Vertex from { ext[0], ext[2], ext[4] };
          Vertex to   { ext[1], ext[3], ext[5] };

          AutocorrelationImpl* b = new AutocorrelationImpl(this->Window, bid, from, to);
          this->Master->add(bid, b, new sdiy::Link);
          }
        }
      this->NumberOfBlocks = bid;
      }
    this->BlocksInitialized = true;
    }
};

//-----------------------------------------------------------------------------
senseiNewMacro(Autocorrelation);

//-----------------------------------------------------------------------------
Autocorrelation::Autocorrelation()
  : Internals(new Autocorrelation::AInternals())
{
}

//-----------------------------------------------------------------------------
Autocorrelation::~Autocorrelation()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void Autocorrelation::Initialize(size_t window, const std::string &meshName,
  int association, const std::string &arrayname, size_t kmax, int numThreads)
{
  TimeEvent<128> mark("Autocorrelation::Initialize");

  AInternals& internals = (*this->Internals);

  internals.Master = make_unique<sdiy::Master>(this->GetCommunicator(),
    numThreads, -1, &AutocorrelationImpl::create, &AutocorrelationImpl::destroy);

  internals.MeshName = meshName;
  internals.Association = association;
  internals.ArrayName = arrayname;
  internals.Window = window;
  internals.KMax = kmax;
}

//-----------------------------------------------------------------------------
bool Autocorrelation::Execute(DataAdaptor* dataIn, DataAdaptor** dataOut)
{
  TimeEvent<128> mark("Autocorrelation::Execute");

  // we do not return anything
  if (dataOut)
    {
    *dataOut = nullptr;
    }

  AInternals& internals = (*this->Internals);

  // see what the simulation is providing
  MeshMetadataMap mdMap;
  if (mdMap.Initialize(dataIn))
    {
    SENSEI_ERROR("Failed to get metadata")
    return false;
    }

  // metadata
  MeshMetadataPtr mmd;
  if (mdMap.GetMeshMetadata(internals.MeshName, mmd))
    {
    SENSEI_ERROR("Failed to get metadata for mesh \"" << internals.MeshName << "\"")
    return false;
    }

  // mesh
  svtkDataObject* mesh = nullptr;
  if (dataIn->GetMesh(internals.MeshName, false, mesh))
    {
    SENSEI_ERROR("Failed to get mesh \"" << internals.MeshName << "\"")
    return false;
    }

  // array
  if (dataIn->AddArray(mesh, internals.MeshName,
    internals.Association, internals.ArrayName))
    {
    SENSEI_ERROR("Failed to add array \"" << internals.ArrayName
      << "\" on mesh \"" << internals.MeshName << "\"")
    return false;
    }

  // ghost cells
  if ((mmd->NumGhostCells || SVTKUtils::AMR(mmd)) &&
    dataIn->AddGhostCellsArray(mesh, internals.MeshName))
    {
    SENSEI_ERROR(<< dataIn->GetClassName() << " failed to add ghost cells.")
    return false;
    }

  if ((mmd->NumGhostNodes > 0) &&
    dataIn->AddGhostNodesArray(mesh, internals.MeshName))
    {
    SENSEI_ERROR(<< dataIn->GetClassName() << " failed to add ghost nodes.")
    return false;
    }

  const int association = internals.Association;
  internals.InitializeBlocks(mesh);

  if (svtkCompositeDataSet* cd = svtkCompositeDataSet::SafeDownCast(mesh))
    {
    svtkSmartPointer<svtkCompositeDataIterator> iter;
    iter.TakeReference(cd->NewIterator());
    iter->SkipEmptyNodesOff();

    int bid = 0;
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem(), ++bid)
      {
      if (svtkDataSet* dataObj = svtkDataSet::SafeDownCast(iter->GetCurrentDataObject()))
        {
        int lid = internals.Master->lid(static_cast<int>(bid));
        AutocorrelationImpl* corr = internals.Master->block<AutocorrelationImpl>(lid);
        svtkFloatArray* fa = svtkFloatArray::SafeDownCast(
          dataObj->GetAttributesAsFieldData(association)->GetArray(internals.ArrayName.c_str()));
        svtkUnsignedCharArray *gc = svtkUnsignedCharArray::SafeDownCast(
          dataObj->GetCellData()->GetArray("svtkGhostType"));
        if (fa)
          {
          corr->process(fa->GetPointer(0), gc ? gc->GetPointer(0) : nullptr);
          }
        else
          {
          SENSEI_ERROR("Current implementation only supports float arrays")
          abort();
          }
        }
      }
    }
  else if (svtkDataSet* ds = svtkDataSet::SafeDownCast(mesh))
    {
    int bid = internals.Master->communicator().rank();
    int lid = internals.Master->lid(static_cast<int>(bid));
    AutocorrelationImpl* corr = internals.Master->block<AutocorrelationImpl>(lid);
    svtkFloatArray* fa = svtkFloatArray::SafeDownCast(
      ds->GetAttributesAsFieldData(association)->GetArray(internals.ArrayName.c_str()));
    svtkUnsignedCharArray *gc = svtkUnsignedCharArray::SafeDownCast(
      ds->GetCellData()->GetArray("svtkGhostType"));
    if (fa)
      {
      corr->process(fa->GetPointer(0), gc ? gc->GetPointer(0) : nullptr);
      }
    else
      {
      SENSEI_ERROR("Current implementation only supports float arrays")
      abort();
      }
    }

  mesh->Delete();

  return true;
}

//-----------------------------------------------------------------------------
void Autocorrelation::PrintResults(size_t k_max)
{
  TimeEvent<128> mark("Autocorrelation::PrintResults");

  AInternals& internals = (*this->Internals);
  size_t nblocks = internals.NumberOfBlocks;

    // add up the autocorrellations
  internals.Master->foreach([](AutocorrelationImpl* b, const sdiy::Master::ProxyWithLink& cp)
                                     {
                                        std::vector<float> sums(b->window, 0);
                                        sdiy::for_each(b->corr.shape(), [&](const Vertex4D& v)
                                        {
                                            size_t w = v[3];
                                            sums[w] += b->corr(v);
                                        });

                                        cp.all_reduce(sums, add_vectors<float>());
                                     });
  internals.Master->exchange();
  if (internals.Master->communicator().rank() == 0)
    {
    // print out the autocorrelations
    auto result = internals.Master->proxy(0).get<std::vector<float>>();
    std::cerr << "Autocorrelations:";
    for (size_t i = 0; i < result.size(); ++i)
      std::cerr << ' ' << result[i];
    std::cerr << std::endl;
    }

  internals.Master->foreach(
    [](AutocorrelationImpl*, const sdiy::Master::ProxyWithLink& cp)
    {
    cp.collectives()->clear();
    });


  // select k strongest autocorrelations for each shift
  sdiy::ContiguousAssigner     assigner(internals.Master->communicator().size(), nblocks);     // NB: this is coupled to main(...) in oscillator.cpp
  sdiy::RegularDecomposer<sdiy::DiscreteBounds> decomposer(1, sdiy::interval(0, nblocks-1), nblocks);
  sdiy::RegularMergePartners   partners(decomposer, 2);
  //sdiy::RegularMergePartners   partners(3, nblocks, 2, true);
  sdiy::reduce(*internals.Master, assigner, partners,
              [k_max](void* b_, const sdiy::ReduceProxy& rp, const sdiy::RegularMergePartners&)
              {
                  AutocorrelationImpl* b = static_cast<AutocorrelationImpl*>(b_);
                  //unsigned round = rp.round(); // current round number

                  using MaxHeapVector = std::vector<std::vector<std::tuple<float,Vertex>>>;
                  using Compare       = std::greater<std::tuple<float,Vertex>>;
                  MaxHeapVector maxs(b->window);
                  if (rp.in_link().size() == 0)
                  {
                      sdiy::for_each(b->corr.shape(), [&](const Vertex4D& v)
                      {
                          size_t offset = v[3];
                          float val = b->corr(v);
                          auto& max = maxs[offset];
                          if (max.size() < k_max)
                          {
                              max.emplace_back(val, v.drop(3) + b->from);
                              std::push_heap(max.begin(), max.end(), Compare());
                          } else if (val > std::get<0>(max[0]))
                          {
                              std::pop_heap(max.begin(), max.end(), Compare());
                              maxs[offset].back() = std::make_tuple(val, v.drop(3) + b->from);
                              std::push_heap(max.begin(), max.end(), Compare());
                          }
                      });
                  } else
                  {
                      for (long i = 0; i < rp.in_link().size(); ++i)
                      {
                          MaxHeapVector in_heaps;
                          rp.dequeue(rp.in_link().target(i).gid, in_heaps);
                          for (size_t j = 0; j < in_heaps.size(); ++j)
                          {
                              for (auto& x : in_heaps[j])
                                  maxs[j].push_back(x);
                          }
                      }
                      for (size_t j = 0; j < maxs.size(); ++j)
                      {
                          auto& max = maxs[j];
                          std::make_heap(max.begin(), max.end(), Compare());
                          while(max.size() > k_max)
                          {
                              std::pop_heap(max.begin(), max.end(), Compare());
                              max.pop_back();
                          }
                      }
                  }

                  if (rp.out_link().size() > 0)
                      rp.enqueue(rp.out_link().target(0), maxs);
                  else
                  {
                      // print out the answer
                      for (size_t i = 0; i < maxs.size(); ++i)
                      {
                          std::sort(maxs[i].begin(), maxs[i].end(), Compare());
                          std::cerr << "Max autocorrelations for " << i << ":";
                          for (auto& x : maxs[i])
                              std::cerr << " (" << std::get<0>(x) << " at " << std::get<1>(x) << ")";
                          std::cerr << std::endl;
                      }
                  }
              });
}

//-----------------------------------------------------------------------------
int Autocorrelation::Finalize()
{
  TimeEvent<128> mark("Autocorrelation::Finalize");

  this->PrintResults(this->Internals->KMax);

  delete this->Internals;
  this->Internals = nullptr;

  return 0;
}

}
