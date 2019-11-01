#ifndef MPIUtils_h
#define MPIUtils_h

#include <algorithm>

namespace sensei
{
namespace MPIUtils
{

// type traits to help convert C++ type
// to MPI enum
template<typename cpp_t> struct mpi_tt {};

#define define_mpi_tt(CT, ME) \
template<> struct mpi_tt<CT> { static MPI_Datatype datatype(){ return ME; } };

define_mpi_tt(char, MPI_CHAR)
define_mpi_tt(int, MPI_INT)
define_mpi_tt(long, MPI_LONG)
define_mpi_tt(unsigned char, MPI_UNSIGNED_CHAR)
define_mpi_tt(unsigned int, MPI_UNSIGNED)
define_mpi_tt(unsigned long, MPI_UNSIGNED_LONG)
define_mpi_tt(float, MPI_FLOAT)
define_mpi_tt(double, MPI_DOUBLE)


// helper to recuce by summation elements in a vector
// it's assumed that the vector is the same size on all
// ranks.
template<typename cpp_t>
void GlobalCounts(MPI_Comm comm, std::vector<cpp_t> &vec)
{
  MPI_Allreduce(MPI_IN_PLACE, vec.data(), vec.size(),
      mpi_tt<cpp_t>::datatype(), MPI_SUM, comm);
}

// helper function to compute an axis aligned bounding box
// that bounds a collection of distrubted axis aligned bounding
// boxes
//
// these can be integer index space bounds (ie VTK extents)
// or floating point world cooridnate system bounds, but for
// index space bounds a signed integer type is required.
//
// local bounds are expected in the layout:
//
//     bx_0_0, bx_1_0, by_0_0, by_1_0, bz_0_0, bz_1_0,
//     ...
//     bx_0_n, bx_1_n, by_0_n, by_1_n, bz_0_n, bz_1_n
//
// where n is the number of blocks minus 1
//
// global bounds are returned in the same layout:
//
//     bx_0, bx_1, by_0, by_1, bz_0, bz_1
//
template <typename cpp_t>
void GlobalBounds(MPI_Comm comm, const std::vector<std::array<cpp_t,6>> &lbounds,
    std::array<cpp_t,6> &gbounds)
{
  int nLocal = lbounds.size();

  gbounds = {std::numeric_limits<cpp_t>::max(), std::numeric_limits<cpp_t>::lowest(),
    std::numeric_limits<cpp_t>::max(), std::numeric_limits<cpp_t>::lowest(),
    std::numeric_limits<cpp_t>::max(), std::numeric_limits<cpp_t>::lowest()};

  // find the smallest bounding covering all local
  for (int q = 0; q < nLocal; ++q)
    {
    const cpp_t *plbounds = lbounds[q].data();
    for (int i = 0; i < 6; ++i)
      {
      int useMax = i % 2;
      cpp_t lval = plbounds[i];
      cpp_t gval = gbounds[i];
      gbounds[i] = useMax ? std::max(gval, lval) : std::min(gval, lval);
      }
    }

  // so we can use MPI_MAX
  for (size_t i = 0; i < 6; i += 2)
    gbounds[i] = -gbounds[i];

  // find the smallest bounding covering all distributed
  MPI_Allreduce(MPI_IN_PLACE, gbounds.data(), 6,
    mpi_tt<cpp_t>::datatype(), MPI_MAX, comm);

  // because we used MPI_MAX
  for (size_t i = 0; i < 6; i += 2)
    gbounds[i] = -gbounds[i];
}

// helper function to compute glpbal array range
template <typename cpp_t>
void GlobalRange(MPI_Comm comm, const std::vector<std::array<cpp_t,2>> &lrange,
    std::array<cpp_t,2> &grange)
{
  int nLocal = lrange.size();

  grange = {std::numeric_limits<cpp_t>::max(),
    std::numeric_limits<cpp_t>::lowest()};

  // find the range over local blocks
  for (int q = 0; q < nLocal; ++q)
    {
    const cpp_t *plrange = lrange[q].data();
    grange[0] = std::min(grange[0], plrange[0]);
    grange[1] = std::max(grange[1], plrange[1]);
    }

  // so we can use MPI_MAX
  grange[0] = -grange[0];

  // find the smallest bounding covering all distributed
  MPI_Allreduce(MPI_IN_PLACE, grange.data(), 2,
    mpi_tt<cpp_t>::datatype(), MPI_MAX, comm);

  // because we used MPI_MAX
  grange[0] = -grange[0];
}

// helper function to generate a global view from a local view.
// here it is assumed that all ranks have the number of items
// in local data. If that is not the case see GlobalViewV
template <typename cpp_t>
void GlobalView(MPI_Comm comm, const std::vector<cpp_t> &ldata,
  std::vector<cpp_t> &gdata)
{
  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nRanks);

  int nLocal = ldata.size();

  gdata.resize(nRanks*nLocal);
  for (int i = 0; i < nLocal; ++i)
      gdata[nLocal*rank+i] = ldata[i];

  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
    gdata.data(), nLocal, mpi_tt<cpp_t>::datatype(), comm);
}

// helper function to generate a global view from a local view. A vector of
// local data items is passed in, this vector could be a different length on
// each rank. a vector of the global data items is returned along with an array
// of counts, and offsets that are used to index into the global data. counts
// is indexed by rank and contains the number of items contributed by each
// rank. offsets contains an offset of each ranks data.
template <typename cpp_t>
void GlobalViewV(MPI_Comm comm, const std::vector<cpp_t> &ldata,
  std::vector<int> &gcounts, std::vector<int> &goffset,
  std::vector<cpp_t> &gdata)
{
  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nRanks);

  gcounts.clear();
  gcounts.resize(nRanks);

  int nLocal = ldata.size();
  gcounts[rank] = nLocal;

  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
    gcounts.data(), 1, MPI_INT, comm);

  goffset.clear();
  goffset.resize(nRanks);

  int q = 0;
  int nTotal = 0;
  std::for_each(gcounts.begin(), gcounts.end(),
    [&nTotal,&goffset,&q](const int &n)
      {
      goffset[q] = nTotal;
      nTotal += n;
      q += 1;
      });

  gdata.resize(nTotal);

  const cpp_t *ld = ldata.data();
  cpp_t *gd = gdata.data() + goffset[rank];
  for (int i = 0; i < nLocal; ++i)
    gd[i] = ld[i];

  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gdata.data(),
    gcounts.data(), goffset.data(), mpi_tt<cpp_t>::datatype(), comm);
}

// use this if you don't need counts & offsets
template <typename cpp_t>
void GlobalViewV(MPI_Comm comm, const std::vector<cpp_t> &ldata,
  std::vector<cpp_t> &gdata)
{
  std::vector<int> counts, offsets;
  GlobalViewV(comm, ldata, counts, offsets, gdata);
}

// use this if you don't need counts & offsets and want the result
// to replace the input.
template <typename cpp_t>
void GlobalViewV(MPI_Comm comm, std::vector<cpp_t> &ldata)
{
  std::vector<int> counts, offsets;
  std::vector<cpp_t> gdata;
  GlobalViewV(comm, ldata, counts, offsets, gdata);
  ldata = std::move(gdata);
}

// helper function to generate a global view from a local view. A vector of
// local data items is passed in, this vector could be a different length on
// each rank. a vector of the global data items is returned along with an array
// of counts, and offsets that are used to index into the global data. counts
// is indexed by rank and contains the number of items contributed by each
// rank. offsets contains an offset of each ranks data.
template <typename cpp_t, std::size_t N>
void GlobalViewV(MPI_Comm comm, const std::vector<std::array<cpp_t,N>> &ldata,
  std::vector<std::array<cpp_t,N>> &gdata)
{
  // serialize the data
  size_t n = ldata.size();
  std::vector<cpp_t> ld(n*N);
  for (size_t i = 0; i < n; ++i)
    {
    const std::array<cpp_t,N> &lda = ldata[i];
    for (size_t j = 0; j < N; ++j)
      {
      ld[i*N + j] = lda[j];
      }
    }

  // send it
  std::vector<cpp_t> gd;
  std::vector<int> counts, offsets;
  GlobalViewV(comm, ld, counts, offsets, gd);

  // deserialize
  n = gd.size()/N;
  gdata.resize(n);
  for (size_t i = 0; i < n; ++i)
    {
    std::array<cpp_t,N> &gda = gdata[i];
    for (size_t j = 0; j < N; ++j)
      {
      gda[j] = gd[i*N + j];
      }
    }
}

// use this if you don't need counts & offsets and want the result
// to replace the input.
template <typename cpp_t, std::size_t N>
void GlobalViewV(MPI_Comm comm, std::vector<std::array<cpp_t,N>> &ldata)
{
  std::vector<std::array<cpp_t,N>> gdata;
  GlobalViewV(comm, ldata, gdata);
  ldata = std::move(gdata);
}

// use this if you don't need counts & offsets and want the result
// to replace the input.
template <typename cpp_t>
void GlobalViewV(MPI_Comm comm, std::vector<std::vector<cpp_t>> &ldata)
{
  // gather local sizes
  std::vector<long> lsizes;
  int nLocal = ldata.size();
  for (int i = 0; i < nLocal; ++i)
    lsizes.push_back(ldata[i].size());

  std::vector<int> scounts, soffsets;
  std::vector<long> gsizes;

  GlobalViewV(comm, lsizes, scounts, soffsets, gsizes);

  // flatten local data
  std::vector<cpp_t> lfdata;
  for (int i = 0; i < nLocal; ++i)
    {
    const std::vector<cpp_t> &elem = ldata[i];
    long nElem = elem.size();
    for (int j = 0; j < nElem; ++j)
      lfdata.push_back(elem[j]);
    }

  // gather flattened data
  std::vector<cpp_t> gfdata;
  GlobalViewV(comm, lfdata, gfdata);

  // un-flatten
  std::vector<std::vector<cpp_t>> gdata;
  unsigned int nranks = scounts.size();
  for (unsigned int i = 0, q  = 0; i < nranks; ++i)
    {
    int nVec = scounts[i];
    int vecOffs = soffsets[i];
    for (int j = 0; j < nVec; ++j)
      {
      long nElem = gsizes[vecOffs + j];
      std::vector<cpp_t> vec;
      for (long k = 0; k < nElem; ++k,++q)
        {
        vec.push_back(gfdata[q]);
        }
      gdata.push_back(vec);
      }
    }

  // return global view
  ldata.swap(gdata);
}

}
}

#endif
