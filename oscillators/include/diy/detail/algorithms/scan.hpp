#ifndef DIY_DETAIL_ALGORITHMS_SORT_HPP
#define DIY_DETAIL_ALGORITHMS_SORT_HPP

#include <diy/partners/common.hpp>

namespace diy
{

namespace detail
{

struct ScanPartners: public RegularPartners
{
  typedef       RegularPartners                                 Parent;

                ScanPartners(int nblocks, int k):
                    Parent(1, nblocks, k)                       {}


  size_t        rounds() const                                  { return 2*Parent::rounds(); }

  inline bool   active(int round, int gid, const Master&) const { if (round < Parent::rounds()) { ... } else { ... } }

  // incoming is only valid for an active gid; it will only be called with an active gid
  inline void   incoming(int round, int gid, std::vector<int>& partners, const Master&) const    { Parent::fill(round - 1, gid, partners); }
  // this is a lazy implementation of outgoing, but it reuses the existing code
  inline void   outgoing(int round, int gid, std::vector<int>& partners, const Master&) const    { std::vector<int> tmp; Parent::fill(round, gid, tmp); partners.push_back(tmp[0]); }
};

// store left sum in out, send the sum on to the parent

template<class Block, class T, class Op>
struct Scan
{
            Scan(const T Block::*          in_,
                       T Block::*          out_,
                 const Op&                 op_):
                in(in_), out(out_), op(op_)             {}

    void    operator()(void* b_, const ReduceProxy& srp, const RegularSwapPartners& partners) const
    {
        Block* b = static_cast<Block*>(b_);

        if (srp.round() == 0)
            b->*out = b->*in;

        if (srp.in_link().size() > 0)
            for (unsigned i = 0; i < srp.in_link().size(); ++i)
            {
                T x;
                srp.dequeue(srp.in_link().target(i).gid, x);
                b->*out = op(x, b->*out);
            }

        if (srp.out_link().size() > 0)
            for (unsigned i = 0; i < srp.out_link().size(); ++i)
                srp.enqueue(srp.out_link().target(i), b->*out);


    }

    const T Block::*        in;
          T Block::*        out;
    const Op&               op;
};

}

}

#endif
