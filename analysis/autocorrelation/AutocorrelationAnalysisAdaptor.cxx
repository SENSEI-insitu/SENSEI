#include "AutocorrelationAnalysisAdaptor.h"

#include <vtkCompositeDataIterator.h>
#include <vtkFieldData.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkInsituDataAdaptor.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredData.h>

#include <memory>
#include <vector>

#include <diy/master.hpp>
#include <diy/reduce.hpp>
#include <diy/partners/merge.hpp>
#include <diy/io/numpy.hpp>
#include <grid/grid.h>
#include <grid/vertices.h>

// http://stackoverflow.com/a/12580468
template<typename T, typename ...Args>
std::unique_ptr<T> make_unique( Args&& ...args )
{
    return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
}

template<class T>
struct add_vectors
{
    using Vector = std::vector<T>;
    Vector  operator()(const Vector& lhs, const Vector& rhs) const      { Vector res(lhs.size(), 0); for (size_t i = 0; i < lhs.size(); ++i) res[i] = lhs[i] + rhs[i]; return res; }
};
namespace diy { namespace mpi { namespace detail {
  template<class U> struct mpi_op< add_vectors<U> >         { static MPI_Op  get() { return MPI_SUM; }  };
} } }

using GridRef = grid::GridRef<float,3>;
using Vertex  = GridRef::Vertex;
using Vertex4D = Vertex::UPoint;

struct Autocorrelation
{
  using Grid = grid::Grid<float,4>;
  Autocorrelation(size_t window_, int gid_, Vertex from_, Vertex to_):
    window(window_),
    gid(gid_),
    from(from_), to(to_),
    shape(to - from + Vertex::one()),
    // init grid with (to - from + 1) in 3D, and window in the 4-th dimension
    values(shape.lift(3, window)),
    corr(shape.lift(3, window))
  {}

  static void* create()            { return new Autocorrelation; }
  static void destroy(void* b)    { delete static_cast<Autocorrelation*>(b); }
  void process(float* data)
    {
    GridRef g(data, shape);

    // record the values
    grid::for_each(g.shape(), [&](const Vertex& v)
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
  Autocorrelation() {}        // here just for create; to let Master manage the blocks (+ if we choose to add OOC later)
};

//-----------------------------------------------------------------------------
class AutocorrelationAnalysisAdaptor::AInternals
{
public:
  std::unique_ptr<diy::Master> Master;
  int Association;
  std::string ArrayName;
  size_t Window;
  bool BlocksInitialized;
  AInternals() : Association(vtkDataObject::POINT), Window(10), BlocksInitialized(false) {}

  void InitializeBlocks(vtkDataObject* dobj)
    {
    if (this->BlocksInitialized)
      {
      return;
      }
    if (vtkImageData* img = vtkImageData::SafeDownCast(dobj))
      {
      int ext[6];
      img->GetExtent(ext);
      if (this->Association == vtkDataObject::CELL)
        {
        vtkStructuredData::GetCellExtentFromPointExtent(ext, ext);
        }
      Vertex from { ext[0], ext[2], ext[4] };
      Vertex to   { ext[1], ext[3], ext[5] };
      int bid = this->Master->communicator().rank();
      Autocorrelation* b = new Autocorrelation(this->Window, bid, from, to);
      this->Master->add(bid, b, new diy::Link);
      }
    else if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj))
      {
      // add blocks
      int rank = this->Master->communicator().rank();
      vtkSmartPointer<vtkCompositeDataIterator> iter;
      iter.TakeReference(cd->NewIterator());
      iter->SkipEmptyNodesOff();

      int bid = 0;
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem(), ++bid)
        {
        if (vtkImageData* id = vtkImageData::SafeDownCast(iter->GetCurrentDataObject()))
          {
          int ext[6];
          id->GetExtent(ext);
          if (this->Association == vtkDataObject::CELL)
            {
            vtkStructuredData::GetCellExtentFromPointExtent(ext, ext);
            }
          Vertex from { ext[0], ext[2], ext[4] };
          Vertex to   { ext[1], ext[3], ext[5] };

          Autocorrelation* b = new Autocorrelation(this->Window, bid, from, to);
          this->Master->add(bid, b, new diy::Link);
          }
        }
      }
    this->BlocksInitialized = true;
    }
};

vtkStandardNewMacro(AutocorrelationAnalysisAdaptor);
//-----------------------------------------------------------------------------
AutocorrelationAnalysisAdaptor::AutocorrelationAnalysisAdaptor()
  : Internals(new AutocorrelationAnalysisAdaptor::AInternals())
{
}

//-----------------------------------------------------------------------------
AutocorrelationAnalysisAdaptor::~AutocorrelationAnalysisAdaptor()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void AutocorrelationAnalysisAdaptor::Initialize(MPI_Comm world,
  size_t window,
  int association, const char* arrayname)
{
  AInternals& internals = (*this->Internals);
  internals.Master = make_unique<diy::Master>(world, -1, -1,
                                              &Autocorrelation::create,
                                              &Autocorrelation::destroy);
  internals.Association = association;
  internals.ArrayName = arrayname;
  internals.Window = window;
}

//-----------------------------------------------------------------------------
bool AutocorrelationAnalysisAdaptor::Execute(vtkInsituDataAdaptor* data)
{
  AInternals& internals = (*this->Internals);
  const int association = internals.Association;
  const char* arrayname = internals.ArrayName.c_str();

  vtkDataObject* mesh = data->GetMesh(/*structure-only*/ true);
  if (!data->AddArray(mesh, association, arrayname))
    {
    return false;
    }

  internals.InitializeBlocks(mesh);

  vtkMultiBlockDataSet* md = vtkMultiBlockDataSet::SafeDownCast(mesh);
  if (md == NULL)
    {
    return false;
    }

  for (unsigned int cc=0, max=md->GetNumberOfBlocks(); cc < max; ++cc)
    {
    if (vtkDataObject* dataObj = md->GetBlock(cc))
      {
      int lid = internals.Master->lid(static_cast<int>(cc));
      Autocorrelation* corr = internals.Master->block<Autocorrelation>(lid);
      vtkFloatArray* fa = vtkFloatArray::SafeDownCast(
        dataObj->GetAttributesAsFieldData(association)->GetArray(arrayname));
      if (fa)
        {
        corr->process(fa->GetPointer(0));
        }
      }
    }
  return true;
}

//-----------------------------------------------------------------------------
void AutocorrelationAnalysisAdaptor::PrintResults(size_t k_max, size_t nblocks)
{
  AInternals& internals = (*this->Internals);
    // add up the autocorrellations
  internals.Master->foreach<Autocorrelation>([](Autocorrelation* b, const diy::Master::ProxyWithLink& cp, void*)
                                     {
                                        std::vector<float> sums(b->window, 0);
                                        grid::for_each(b->corr.shape(), [&](const Vertex4D& v)
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
    std::cout << "Autocorrelations:";
    for (size_t i = 0; i < result.size(); ++i)
      std::cout << ' ' << result[i];
    std::cout << std::endl;
    }

  internals.Master->foreach<Autocorrelation>(
    [](Autocorrelation* b, const diy::Master::ProxyWithLink& cp, void*)
    {
    cp.collectives()->clear();
    });

    // select k strongest autocorrelations for each shift
    diy::ContiguousAssigner     assigner(internals.Master->communicator().size(), nblocks);     // NB: this is coupled to main(...) in oscillator.cpp
    diy::RegularMergePartners   partners(3, nblocks, 2, true);
    diy::reduce(*internals.Master, assigner, partners,
                [k_max](void* b_, const diy::ReduceProxy& rp, const diy::RegularMergePartners& partners)
                {
                    Autocorrelation*  b        = static_cast<Autocorrelation*>(b_);
                    unsigned          round    = rp.round();               // current round number

                    using MaxHeapVector = std::vector<std::vector<std::tuple<float,Vertex>>>;
                    using Compare       = std::greater<std::tuple<float,Vertex>>;
                    MaxHeapVector maxs(b->window);
                    if (rp.in_link().size() == 0)
                    {
                        grid::for_each(b->corr.shape(), [&](const Vertex4D& v)
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
                        for (size_t i = 0; i < rp.in_link().size(); ++i)
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
                            std::cout << "Max autocorrelations for " << i << ":";
                            for (auto& x : maxs[i])
                                std::cout << " (" << std::get<0>(x) << " at " << std::get<1>(x) << ")";
                            std::cout << std::endl;
                        }
                    }
                });
}
