#ifndef DIY_POINT_HPP
#define DIY_POINT_HPP

#include <vector>

namespace diy
{
  template<class T, unsigned D>
  class Point
  {
    private:
      T         data_[D];

    public:
                Point()                         {}

      template<class U>
                Point(const std::vector<U>& x)  { for (unsigned i = 0; i < D; ++i) data_[i] = x[i]; }

      T&        operator[](unsigned i)          { return data_[i]; }
      const T&  operator[](unsigned i) const    { return data_[i]; }

      Point&    operator+=(const Point& other)                  { for (unsigned i = 0; i < D; ++i) data_[i] += other.data_[i]; return *this; }
      friend Point operator+(const Point& x, const Point& y)    { Point r = x; r += y; return r; }

      Point&    operator-=(const Point& other)                  { for (unsigned i = 0; i < D; ++i) data_[i] -= other.data_[i]; return *this; }
      friend Point operator-(const Point& x, const Point& y)    { Point r = x; r -= y; return r; }

      T         norm2() const                                   { T r = 0; for (unsigned i = 0; i < D; ++i) r += data_[i]*data_[i]; return r; }
  };
}

#endif
