#ifndef sensei_STLUtils_h
#define sensei_STLUtils_h

#include <array>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <ostream>


namespace sensei
{

// collection of functions helpful for dealing with STL data structures
namespace STLUtils
{
// --------------------------------------------------------------------------
template<typename con_t>
con_t Sum(const std::vector<con_t> &vec)
{
  con_t sum = con_t();
  unsigned long n = vec.size();
  for (unsigned long i = 0; i < n; ++i)
    sum += vec[i];
  return sum;
}

// --------------------------------------------------------------------------
template<typename con_t, unsigned long n>
void InitializeRange(std::array<con_t,n> &out)
{
  for (unsigned long j = 0; j < n; ++j)
    {
    out[j] = j % 2 ?
      std::numeric_limits<con_t>::lowest() : std::numeric_limits<con_t>::max();
    }
}

// --------------------------------------------------------------------------
template<typename con_t, unsigned long n>
void InitializeRange(std::vector<std::array<con_t,n>> &out)
{
  unsigned long nElem = out.size();
  for (unsigned long j = 0; j < nElem; ++j)
    InitializeRange(out[j]);
}

// --------------------------------------------------------------------------
template<typename con_t, unsigned long n>
void ReduceRange(const std::array<con_t,n> &in, std::array<con_t,n> &out)
{
  for (unsigned long j = 0; j < n; ++j)
    out[j] = j % 2 ? std::max(out[j], in[j]) : std::min(out[j], in[j]);
}

// --------------------------------------------------------------------------
template<typename con_t, unsigned long n>
void ReduceRange(const std::vector<std::array<con_t,n>> &in, std::array<con_t,n> &out)
{
  InitializeRange(out);
  unsigned long nElem = in.size();
  for (unsigned long i = 0; i < nElem; ++i)
    ReduceRange(in[i], out);
}

// --------------------------------------------------------------------------
template<typename con_t, unsigned long n>
void ReduceRange(const std::vector<std::array<con_t,n>> &in, std::vector<std::array<con_t,n>> &out)
{
  unsigned long nElem = in.size();
  for (unsigned long j = 0; j < nElem; ++j)
    ReduceRange(in[j], out[j]);
}

// --------------------------------------------------------------------------
template<typename con_t, unsigned long n>
void ReduceRange(const std::vector<std::vector<std::array<con_t,n>>> &in,
  std::vector<std::array<con_t,n>> &out)
{
  unsigned long nBlocks = in.size();
  if (nBlocks < 1)
    return;

  unsigned long nArrays = in[0].size();
  out.resize(nArrays);
  InitializeRange(out);

  for (unsigned long j = 0; j < nBlocks; ++j)
    {
    for (unsigned long i = 0; i < nArrays; ++i)
      {
      ReduceRange(in[j], out);
      }
    }
}


// --------------------------------------------------------------------------
template<typename T, std::size_t N>
ostream &operator<<(ostream &os, const std::array<T,N> &vec)
{
  os << "{";
  if (N)
    {
    os << vec[0];
    for (size_t i = 1; i < N; ++i)
      os << ", " << vec[i];
    }
  os << "}";
  return os;
}

// --------------------------------------------------------------------------
template<typename T>
ostream &operator<<(ostream &os, const std::vector<T> &vec)
{
  os << "{";
  size_t n = vec.size();
  if (n)
    {
    os << vec[0];
    for (size_t i = 1; i < n; ++i)
      os << ", " << vec[i];
    }
  os << "}";
  return os;
}

// --------------------------------------------------------------------------
template<typename K, typename V>
ostream &operator<<(ostream &os, const std::map<K,V> &amap)
{
  os << "{";
  size_t n = amap.size();
  if (n)
    {
    typename std::map<K,V>::const_iterator it = amap.begin();
    os << "{" << it->first << " : " << it->second << "}";
    ++it;

    for (size_t i = 1; i < n; ++i, ++it)
      os << ", {" << it->first << " : " << it->second << "}";
    }
  os << "}";
  return os;
}

}
}

#endif
