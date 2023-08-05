#ifndef svtkIterator_h
#define svtkIterator_h

#include <cstddef>

/// replaces std::iterator which was deprecated in C++17.
template<
    typename Category,
    typename T,
    typename Distance = std::ptrdiff_t,
    typename Pointer = T*,
    typename Reference = T&
> struct svtkIterator
{
    using iterator_category = Category;
    using value_type = T;
    using difference_type = Distance;
    using pointer = Pointer;
    using reference = Reference;
};

#endif
