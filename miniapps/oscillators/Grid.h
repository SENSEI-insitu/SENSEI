#ifndef GRID_H
#define GRID_H

#include "sdiy/point.hpp"

#include "senseiConfig.h" // for OSCILLATOR_CUDA

// NOTES:
//
// I'm introducing this class to enable DIY to use CUDA managed memory.
//
// I'm moving this from the DIY namespace to the oscillator namespace
//
// DIY makes use of C++ operator new. I'm using cudaMallocManaged in its place,
// and my implementation does not initialize array elements hence this class is
// limittted to use with POD data types.
//
// The use of cudaMallocManaged simplifies data transfers in that CPU code can
// access GPU memory and vise versa. However, explicit synchronization is
// needed.

namespace oscillator
{

template<class C, unsigned D>
struct Grid;

template<class C, unsigned D>
struct GridRef
{
    public:
        typedef     C                                           Value;

        typedef     sdiy::Point<int, D>                         Vertex;
        typedef     size_t                                      Index;

    public:
        template<class Int>
                GridRef(C* data, const sdiy::Point<Int,D>& shape, bool c_order = true):
                    data_(data), shape_(shape), c_order_(c_order)   { set_stride(); }

                GridRef(Grid<C,D>& g):
                    data_(g.data()), shape_(g.shape()),
                    c_order_(g.c_order())                       { set_stride(); }

        template<class Int>
        C       operator()(const sdiy::Point<Int, D>& v) const        { return (*this)(index(v)); }

        template<class Int>
        C&      operator()(const sdiy::Point<Int, D>& v)              { return (*this)(index(v)); }

        C       operator()(Index i) const                       { return data_[i]; }
        C&      operator()(Index i)                             { return data_[i]; }

        const Vertex&
                shape() const                                   { return shape_; }

        const C*
                data() const                                    { return data_; }
        C*      data()                                          { return data_; }

        // Set every element to the given value
        GridRef&    operator=(C value)                          { Index s = size(); for (Index i = 0; i < s; ++i) data_[i] = value; return *this; }
        GridRef&    operator/=(C value)                         { Index s = size(); for (Index i = 0; i < s; ++i) data_[i] /= value; return *this; }

        inline
        bool        operator==(const GridRef& other) const;
        bool        operator!=(const GridRef& other) const      { return !(*this == other); }


        inline
        Vertex      vertex(Index idx) const;

        Index       index(const Vertex& v) const                { Index idx = 0; for (unsigned i = 0; i < D; ++i) { idx += ((Index) v[i]) * ((Index) stride_[i]); } return idx; }

        Index       size() const                                { return size(shape()); }
        void        swap(GridRef& other)                        { std::swap(data_, other.data_); std::swap(shape_, other.shape_); std::swap(stride_, other.stride_); std::swap(c_order_, other.c_order_); }

        bool        c_order() const                             { return c_order_; }

        static constexpr
        unsigned    dimension()                                 { return D; }

    protected:
        static Index
                size(const Vertex& v)                           { Index res = 1; for (unsigned i = 0; i < D; ++i) res *= v[i]; return res; }

        void    set_stride()
        {
            Index cur = 1;
            if (c_order_)
                for (unsigned i = D; i > 0; --i) { stride_[i-1] = cur; cur *= shape_[i-1]; }
            else
                for (unsigned i = 0; i < D; ++i) { stride_[i] = cur; cur *= shape_[i]; }

        }
        void    set_shape(const Vertex& v)                      { shape_ = v; set_stride(); }
        void    set_data(C* data)                               { data_ = data; }
        void    set_c_order(bool order)                         { c_order_ = order; }

    protected:
        C*      data_;
        Vertex  shape_;
        Vertex  stride_;
        bool    c_order_;
};


template<class C, unsigned D>
struct Grid: public GridRef<C,D>
{
    public:
        typedef     GridRef<C,D>                                Parent;
        typedef     typename Parent::Value                      Value;
        typedef     typename Parent::Index                      Index;
        typedef     typename Parent::Vertex                     Vertex;
        typedef     Parent                                      Reference;

        template<class U>
        struct rebind { typedef Grid<U,D>                       type; };

    public:
                Grid(): Parent(nullptr, Vertex::zero())
                {}

        template<class Int>
                Grid(const sdiy::Point<Int, D>& shape, bool c_order = true):
                    Parent(nullptr, shape, c_order)
                {
                    size_t nBytes = sizeof(C)*size(shape);
#if defined(OSCILLATOR_CUDA)
                    cudaMallocManaged(&this->data_, nBytes);
#else
                    this->data_ = new C[size(shape)];
#endif
                    memset(this->data_, 0, nBytes);
                }

                Grid(Grid&& g): Grid()                          { Parent::swap(g); }

                Grid(const Parent& g):
                    Parent(nullptr, g.shape(), g.c_order())
                {
#if defined(OSCILLATOR_CUDA)
                    size_t nBytes = sizeof(C)*size(g.shape());
                    cudaMallocManaged(&this->data_, sizeof(C)*size(g.shape()));
#else
                    this->data_ = new C[size(g.shape())];
#endif
                    copy_data(g.data());
                }

        template<class OtherGrid>
                Grid(const OtherGrid& g):
                    Parent(g.shape(), g.c_order())
                {
#if defined(OSCILLATOR_CUDA)
                    size_t nBytes = sizeof(C)*size(g.shape());
                    cudaMallocManaged(&this->data_, sizeof(C)*size(g.shape()));
#else
                    this->data_ = new C[size(g.shape())],
#endif
                     copy_data(g.data());
                }

                ~Grid()
                {
#if defined(OSCILLATOR_CUDA)
                    cudaFree(this->data_);
#else
                    delete[] Parent::data();
#endif
                }

        template<class OC>
        Grid&   operator=(const GridRef<OC, D>& other)
        {
            delete[] Parent::data();
            Parent::set_c_order(other.c_order());       // NB: order needs to be set before the shape, to set the stride correctly
            Parent::set_shape(other.shape());
            Index s = size(shape());
            Parent::set_data(new C[s]);
            copy_data(other.data());
            return *this;
        }

        Grid&   operator=(Grid&& g)                             { Parent::swap(g); return *this; }

        using Parent::data;
        using Parent::shape;
        using Parent::operator();
        using Parent::operator=;
        using Parent::size;

    private:
        template<class OC>
        void    copy_data(const OC* data)
        {
            Index s = size(shape());
            for (Index i = 0; i < s; ++i)
                Parent::data()[i] = data[i];
        }
};


template<class C, unsigned D>
bool GridRef<C, D>::operator==(const GridRef<C, D>& other) const
{
    if (c_order() != other.c_order())
        return false;

    if (shape() != other.shape())
        return false;

    for(unsigned i = 0; i < size(); ++i)
        if (data()[i] != other.data()[i])
            return false;

    return true;
}

template<class C, unsigned D>
typename GridRef<C, D>::Vertex
GridRef<C, D>::
vertex(typename GridRef<C, D>::Index idx) const
{
    Vertex v;
    if (c_order())
        for (unsigned i = 0; i < D; ++i)
        {
            v[i] = idx / stride_[i];
            idx %= stride_[i];
        }
    else
        for (int i = D-1; i >= 0; --i)
        {
            v[i] = idx / stride_[i];
            idx %= stride_[i];
        }
    return v;
}

}

#endif
