#ifndef ArrayIO_h
#define ArrayIO_h

#include <iostream>
#include <mpi.h>

#define arrayIO_error(_arg) \
  std::cerr << "ERROR: " << __FILE__ " : "  << __LINE__ << std::endl \
    << "" _arg << std::endl;

namespace {
// *****************************************************************************
template <typename T>
size_t size(const T &ext)
{
  size_t n = 1;
  for (int i = 0; i < 3; ++i)
    n *= ext[2*i+1] - ext[2*i] + 1;
  return n;
}

// *****************************************************************************
template <typename T>
void size(const T &ext, int *n)
{
  for (int i = 0; i < 3; ++i)
    n[i] = ext[2*i+1] - ext[2*i] + 1;
}

// *****************************************************************************
template <typename T>
void start(const T &ext, int *id)
{
  for (int i = 0; i < 3; ++i)
    id[i] = ext[2*i];
}

// *****************************************************************************
template <typename T>
void offset(const T &dom, const T &subdom, int *id)
{
  for (int i = 0; i < 3; ++i)
    id[i] = dom[2*i] - subdom[2*i];
}

// *****************************************************************************
template <typename T>
bool equal(const T &l, const T &r)
{
  for (int i = 0; i < 6; ++i)
  {
    if (l[i] != r[i])
      return false;
  }
  return true;
}

// *****************************************************************************
template < typename T> struct mpi_tt;
template <> struct mpi_tt <float> { static MPI_Datatype Type(){ return MPI_FLOAT; } };
template <> struct mpi_tt <double> { static MPI_Datatype Type(){ return MPI_DOUBLE; } };
template <> struct mpi_tt <short int> { static MPI_Datatype Type(){ return MPI_SHORT; } };
template <> struct mpi_tt <unsigned short int> { static MPI_Datatype Type(){ return MPI_UNSIGNED_SHORT; } };
template <> struct mpi_tt <int> { static MPI_Datatype Type(){ return MPI_INT; } };
template <> struct mpi_tt <unsigned int> { static MPI_Datatype Type(){ return MPI_UNSIGNED; } };
template <> struct mpi_tt <long> { static MPI_Datatype Type(){ return MPI_LONG; } };
template <> struct mpi_tt <unsigned long> { static MPI_Datatype Type(){ return MPI_UNSIGNED_LONG; } };
template <> struct mpi_tt <long long> { static MPI_Datatype Type(){ return MPI_LONG_LONG; } };
template <> struct mpi_tt <unsigned long long> { static MPI_Datatype Type(){ return MPI_UNSIGNED_LONG_LONG; } };
template <> struct mpi_tt <signed char> { static MPI_Datatype Type(){ return MPI_CHAR; } };
template <> struct mpi_tt <char> { static MPI_Datatype Type(){ return MPI_CHAR; } };
template <> struct mpi_tt <unsigned char> { static MPI_Datatype Type(){ return MPI_UNSIGNED_CHAR; } };
}

namespace arrayIO
{
// ****************************************************************************
int open(
        MPI_Comm comm,                 // MPI communicator handle
        const char *fileName,          // file name to open
        MPI_Info hints,                // MPI file hints
        MPI_File &file);               // file handle

// ****************************************************************************
inline
int close(MPI_File file)
{
  MPI_File_close(&file);
  return 0;
}

// ****************************************************************************
template < typename T>
int write(
        MPI_File file,                 // File to write to.
        MPI_Info hints,                // MPI file hints
        int domain[6],                 // entire region, dataset extents
        int decomp[6],                 // local memory region, block extents with ghost zones
        int valid[6],                  // region to write to disk
        T *data)                       // pointer to a buffer to write to disk.
{
  int iErr = 0;

  // calculate block offsets and lengths
  int domainDims[3];
  ::size(domain, domainDims);

  int decompDims[3];
  ::size(decomp, decompDims);

  int validDims[3];
  ::size(valid, validDims);

  int validStart[3];
  ::start(valid, validStart);

  int validOffset[3];
  ::offset(decomp, valid, validOffset);

  // file view
  MPI_Datatype nativeType = ::mpi_tt<T>::Type();
  MPI_Datatype fileView;
  if (MPI_Type_create_subarray(3, domainDims,
      validDims, validStart, MPI_ORDER_FORTRAN,
      nativeType, &fileView))
  {
    arrayIO_error(<< "MPI_Type_create_subarray failed.")
    return -1;
  }

  if (MPI_Type_commit(&fileView))
  {
    arrayIO_error(<< "MPI_Type_commit failed.")
    return -1;
  }

  if (MPI_File_set_view(file, 0,
      nativeType, fileView, "native", hints))
  {
    arrayIO_error(<< "MPI_File_set_view failed.")
    return -1;
  }

  // memory view
  MPI_Datatype memView;
  if (MPI_Type_create_subarray(3, decompDims,
      validDims, validOffset, MPI_ORDER_FORTRAN,
      nativeType, &memView))
  {
      arrayIO_error(<< "MPI_Type_create_subarray failed.")
      return -1;
  }

  if (MPI_Type_commit(&memView))
  {
    arrayIO_error(<< "MPI_Type_commit failed.")
    return -1;
  }

  // write
  const int eStrLen = 2048;
  char eStr[eStrLen] = {'\0'};
  MPI_Status status;
  if ((iErr = MPI_File_write(file, data, 1, memView, &status)))
  {
    MPI_Error_string(iErr, eStr, const_cast<int *>(&eStrLen));
    arrayIO_error(<< "write error: " << eStr)
    return -1;
  }

  MPI_Type_free(&fileView);
  MPI_Type_free(&memView);

  return 0;
}

// ****************************************************************************
template < typename T>
int write_all(
        MPI_File file,                 // File to write to.
        MPI_Info hints,                // MPI file hints
        int domain[6],                 // entire region, dataset extents
        int decomp[6],                 // local memory region, block extents with ghost zones
        int valid[6],                  // region to write to disk
        T *data)                       // pointer to a buffer to write to disk.
{
  int iErr = 0;
  // calculate block offsets and lengths
  int domainDims[3];
  size(domain, domainDims);

  int decompDims[3];
  size(decomp, decompDims);

  int validDims[3];
  size(valid, validDims);

  int validStart[3];
  start(valid, validStart);

  int validOffset[3];
  offset(decomp, valid, validOffset);

  // file view
  MPI_Datatype nativeType = ::mpi_tt<T>::Type();
  MPI_Datatype fileView;
  if (MPI_Type_create_subarray(3, domainDims,
      validDims, validStart, MPI_ORDER_FORTRAN,
      nativeType, &fileView))
  {
    arrayIO_error(<< "MPI_Type_create_subarray failed.")
    return -1;
  }

  if (MPI_Type_commit(&fileView))
  {
    arrayIO_error(<< "MPI_Type_commit failed.")
    return -1;
  }

  if (MPI_File_set_view(file, 0,
      nativeType, fileView, "native", hints))
  {
    arrayIO_error(<< "MPI_File_set_view failed.")
    return -1;
  }

  // memory view
  MPI_Datatype memView;
  if (MPI_Type_create_subarray(3, decompDims,
      validDims, validOffset, MPI_ORDER_FORTRAN,
      nativeType, &memView))
  {
      arrayIO_error(<< "MPI_Type_create_subarray failed.")
      return -1;
  }

  if (MPI_Type_commit(&memView))
  {
    arrayIO_error(<< "MPI_Type_commit failed.")
    return -1;
  }

  // write
  const int eStrLen = 2048;
  char eStr[eStrLen] = {'\0'};
  MPI_Status status;
  if ((iErr = MPI_File_write_all(file, data, 1, memView, &status)))
  {
    MPI_Error_string(iErr, eStr, const_cast<int *>(&eStrLen));
    arrayIO_error(<< "write error : " << eStr)
    return -1;
  }

  MPI_Type_free(&fileView);
  MPI_Type_free(&memView);

  return iErr ? -1 : 0;
}

// ****************************************************************************
enum
{
  HINT_DEFAULT = 0,
  HINT_AUTOMATIC = 0,
  HINT_DISABLED = 1,
  HINT_ENABLED = 2
};

// ****************************************************************************
MPI_Info createHints(
    int useCollectiveIO = HINT_ENABLED,
    int numberOfIONodes = -1,
    int collectiveBufferSize = -1,
    int useDirectIO = HINT_AUTOMATIC,
    int useDefferedOpen = HINT_AUTOMATIC,
    int useDataSeive = HINT_AUTOMATIC,
    int sieveBufferSize = -1,
    int stripeCount = -1,
    int stripeSize = -1);

};
#endif
