#ifndef VERTICES_H
#define VERTICES_H

#include <iterator>

namespace grid
{

template<class Vertex_>
class VerticesIterator:
    public std::iterator<std::forward_iterator_tag, Vertex_>
{
    typedef     std::iterator<std::forward_iterator_tag, Vertex_>   Parent;

    public:
        typedef     typename Parent::value_type                     value_type;
        typedef     typename Parent::difference_type                difference_type;
        typedef     typename Parent::reference                      reference;

        typedef     value_type                                      Vertex;
        typedef     typename Vertex::Coordinate                     Coordinate;

                    // upper bounds are non-inclusive
                    VerticesIterator(const Vertex& bounds):
                        to_(bounds - Vertex::one())                 {}

                    VerticesIterator(const Vertex& pos,
                                     const Vertex& bounds):
                        pos_(pos), to_(bounds - Vertex::one())      {}

                    VerticesIterator(const Vertex& pos,
                                     const Vertex& from,
                                     const Vertex& to):
                        pos_(pos), from_(from),
                        to_(to)                                     {}

        static VerticesIterator
                    begin(const Vertex& bounds)                     { return VerticesIterator(bounds); }
        static VerticesIterator
                    end(const Vertex& bounds)                       { Vertex e; e[0] = bounds[0]; return VerticesIterator(e, bounds); }

        static VerticesIterator
                    begin(const Vertex& from, const Vertex& to)     { return VerticesIterator(from, from, to); }
        static VerticesIterator
                    end(const Vertex& from, const Vertex& to)       { Vertex e = from; e[0] = to[0] + 1; return VerticesIterator(e, from, to); }

        const Vertex&       operator*() const                       { return pos_; }
        const Vertex*       operator->() const                      { return &pos_; }

        VerticesIterator&   operator++()                            { increment(); return *this; }
        VerticesIterator    operator++(int)                         { VerticesIterator it = *this; increment(); return it; }

        friend bool operator==(const VerticesIterator& x, const VerticesIterator& y)    { return x.pos_ == y.pos_; }
        friend bool operator!=(const VerticesIterator& x, const VerticesIterator& y)    { return x.pos_ != y.pos_; }

    private:
        void        increment();

    private:
        Vertex      pos_;
        Vertex      from_;
        Vertex      to_;
};

template<class Vertex_>
class VertexRange
{
    public:
        using       Vertex   = Vertex_;
        using       iterator = VerticesIterator<Vertex>;

                    VertexRange(const Vertex& from, const Vertex& to):
                        from_(from), to_(to)                                    {}

                    VertexRange(const Vertex& shape):
                        VertexRange(Vertex::zero(), shape - Vertex::one())      {}

        iterator    begin() const       { return iterator::begin(from_,to_); }
        iterator    end()   const       { return iterator::end(from_,to_); }

    private:
        Vertex  from_;
        Vertex  to_;
};

namespace detail
{
    template<class Vertex, size_t I>
    struct IsLast
    {
        static constexpr bool value = (Vertex::dimension() - 1 == I);
    };

    template<class Vertex, class Callback, size_t I, bool P>
    struct ForEach
    {
        void operator()(Vertex& pos, const Vertex& from, const Vertex& to, const Callback& callback) const
        {
            for (pos[I] = from[I]; pos[I] <= to[I]; ++pos[I])
                ForEach<Vertex, Callback, I+1, IsLast<Vertex,I+1>::value>()(pos, from, to, callback);
        }
    };

    template<class Vertex, class Callback, size_t I>
    struct ForEach<Vertex,Callback,I,true>
    {
        void operator()(Vertex& pos, const Vertex& from, const Vertex& to, const Callback& callback) const
        {
            for (pos[I] = from[I]; pos[I] <= to[I]; ++pos[I])
                callback(pos);
        }
    };
}

template<class Vertex, class Callback>
void for_each(const Vertex& from, const Vertex& to, const Callback& callback)
{
    Vertex pos;
    grid::detail::ForEach<Vertex, Callback, 0, detail::IsLast<Vertex,0>::value>()(pos, from, to, callback);
}

template<class Vertex, class Callback>
void for_each(const Vertex& shape, const Callback& callback)
{
    // specify grid namespace to disambiguate with std::for_each(...)
    grid::for_each(Vertex::zero(), shape - Vertex::one(), callback);
}

}

template<class V>
void
grid::VerticesIterator<V>::
increment()
{
    int j = Vertex::dimension() - 1;
    while (j > 0 && pos_[j] == to_[j])
    {
        pos_[j] = from_[j];
        --j;
    }
    ++pos_[j];
}

#endif
