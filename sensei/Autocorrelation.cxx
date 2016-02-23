#include "Autocorrelation.h"

#include "DataAdaptor.h"

// VTK includes
#include <vtkCompositeDataIterator.h>
#include <vtkFieldData.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
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

namespace sensei
{

using GridRef = grid::GridRef<float,3>;
using Vertex  = GridRef::Vertex;
using Vertex4D = Vertex::UPoint;

struct AutocorrelationImpl
{
  using Grid = grid::Grid<float,4>;
  AutocorrelationImpl(size_t window_, int gid_, Vertex from_, Vertex to_):
    window(window_),
    gid(gid_),
    from(from_), to(to_),
    shape(to - from + Vertex::one()),
    // init grid with (to - from + 1) in 3D, and window in the 4-th dimension
    values(shape.lift(3, window)),
    corr(shape.lift(3, window))
  {}

  static void* create()            { return new AutocorrelationImpl; }
  static void destroy(void* b)    { delete static_cast<AutocorrelationImpl*>(b); }
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
  AutocorrelationImpl() {}        // here just for create; to let Master manage the blocks (+ if we choose to add OOC later)
};

//-----------------------------------------------------------------------------
class Autocorrelation::AInternals
{
public:
  std::unique_ptr<diy::Master> Master;
  size_t KMax;
  int Association;
  std::string ArrayName;
  size_t Window;
  bool BlocksInitialized;
  size_t NumberOfBlocks;
  AInternals() :
    KMax(3),
    Association(vtkDataObject::POINT),
    Window(10),
    BlocksInitialized(false),
    NumberOfBlocks(0)
  {}

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
      AutocorrelationImpl* b = new AutocorrelationImpl(this->Window, bid, from, to);
      this->Master->add(bid, b, new diy::Link);
      this->NumberOfBlocks = this->Master->communicator().size();
      }
    else if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj))
      {
      // add blocks
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

          AutocorrelationImpl* b = new AutocorrelationImpl(this->Window, bid, from, to);
          this->Master->add(bid, b, new diy::Link);
          }
        }
      this->NumberOfBlocks = bid;
      }
    this->BlocksInitialized = true;
    }
};

vtkStandardNewMacro(Autocorrelation);
//-----------------------------------------------------------------------------
Autocorrelation::Autocorrelation()
  : Internals(new Autocorrelation::AInternals())
{
}

//-----------------------------------------------------------------------------
Autocorrelation::~Autocorrelation()
{
  this->PrintResults(this->Internals->KMax);
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void Autocorrelation::Initialize(MPI_Comm world,
  size_t window,
  int association, const char* arrayname, size_t kmax)
{
  AInternals& internals = (*this->Internals);
  internals.Master = make_unique<diy::Master>(world, -1, -1,
                                              &AutocorrelationImpl::create,
                                              &AutocorrelationImpl::destroy);
  internals.Association = association;
  internals.ArrayName = arrayname;
  internals.Window = window;
  internals.KMax = kmax;
}

//-----------------------------------------------------------------------------
bool Autocorrelation::Execute(DataAdaptor* data)
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


  if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(mesh))
    {
    vtkSmartPointer<vtkCompositeDataIterator> iter;
    iter.TakeReference(cd->NewIterator());
    iter->SkipEmptyNodesOff();

    int bid = 0;
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem(), ++bid)
      {
      if (vtkDataSet* dataObj = vtkDataSet::SafeDownCast(iter->GetCurrentDataObject()))
        {
        int lid = internals.Master->lid(static_cast<int>(bid));
        AutocorrelationImpl* corr = internals.Master->block<AutocorrelationImpl>(lid);
        vtkFloatArray* fa = vtkFloatArray::SafeDownCast(
          dataObj->GetAttributesAsFieldData(association)->GetArray(arrayname));
        if (fa)
          {
          corr->process(fa->GetPointer(0));
          }
        else
          {
          cerr <<"Current implementation only supports float arrays" << endl;
          abort();
          }
        }
      }
    }
  else if (vtkDataSet* ds = vtkDataSet::SafeDownCast(mesh))
    {
    int bid = internals.Master->communicator().rank();
    int lid = internals.Master->lid(static_cast<int>(bid));
    AutocorrelationImpl* corr = internals.Master->block<AutocorrelationImpl>(lid);
    vtkFloatArray* fa = vtkFloatArray::SafeDownCast(
      ds->GetAttributesAsFieldData(association)->GetArray(arrayname));
    if (fa)
      {
      corr->process(fa->GetPointer(0));
      }
    else
      {
      cerr <<"Current implementation only supports float arrays" << endl;
      abort();
      }
    }
  return true;
}

//-----------------------------------------------------------------------------
void Autocorrelation::PrintResults(size_t k_max)
{
  AInternals& internals = (*this->Internals);
  size_t nblocks = internals.NumberOfBlocks;

    // add up the autocorrellations
  internals.Master->foreach<AutocorrelationImpl>([](AutocorrelationImpl* b, const diy::Master::ProxyWithLink& cp, void*)
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

  internals.Master->foreach<AutocorrelationImpl>(
    [](AutocorrelationImpl* b, const diy::Master::ProxyWithLink& cp, void*)
    {
    cp.collectives()->clear();
    });

    // select k strongest autocorrelations for each shift
    diy::ContiguousAssigner     assigner(internals.Master->communicator().size(), nblocks);     // NB: this is coupled to main(...) in oscillator.cpp
    diy::RegularMergePartners   partners(3, nblocks, 2, true);
    diy::reduce(*internals.Master, assigner, partners,
                [k_max](void* b_, const diy::ReduceProxy& rp, const diy::RegularMergePartners& partners)
                {
                    AutocorrelationImpl*  b        = static_cast<AutocorrelationImpl*>(b_);
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

}
