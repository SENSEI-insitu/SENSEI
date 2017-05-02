#ifndef senseiPyArray_h
#define senseiPyArray_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <cstdlib>

namespace senseiPyArray
{
/// cpp_tt -- traits class for working with PyArrayObject's
/**
cpp_tt::type -- get the C++ type given a numpy enum.

CODE -- numpy type enumeration
CPP_T -- corresponding C++ type
*/
template <int numpy_code> struct cpp_tt
{};

#define senseiPyArray_cpp_tt_declare(CODE, CPP_T)   \
template <> struct cpp_tt<CODE>                     \
{                                                   \
  typedef CPP_T type;                               \
};
senseiPyArray_cpp_tt_declare(NPY_BYTE, char)
senseiPyArray_cpp_tt_declare(NPY_INT32, int)
senseiPyArray_cpp_tt_declare(NPY_INT64, long long)
senseiPyArray_cpp_tt_declare(NPY_UBYTE, unsigned char)
senseiPyArray_cpp_tt_declare(NPY_UINT32, unsigned int)
senseiPyArray_cpp_tt_declare(NPY_UINT64, unsigned long long)
senseiPyArray_cpp_tt_declare(NPY_FLOAT, float)
senseiPyArray_cpp_tt_declare(NPY_DOUBLE, double)

/// numpy_tt - traits class for working with PyArrayObject's
/**
::code - get the numpy type enum given a C++ type.
::is_type - return true if the PyArrayObject has the given type

CODE -- numpy type enumeration
CPP_T -- corresponding C++ type
*/
template <typename cpp_t> struct numpy_tt
{};

#define senseiPyArray_numpy_tt_declare(CODE, CPP_T) \
template <> struct numpy_tt<CPP_T>                  \
{                                                   \
  enum { code = CODE };                             \
  static bool is_type(PyArrayObject *arr)           \
  { return PyArray_TYPE(arr) == CODE; }             \
};
senseiPyArray_numpy_tt_declare(NPY_BYTE, char)
senseiPyArray_numpy_tt_declare(NPY_INT16, short)
senseiPyArray_numpy_tt_declare(NPY_INT32, int)
senseiPyArray_numpy_tt_declare(NPY_LONG, long)
senseiPyArray_numpy_tt_declare(NPY_INT64, long long)
senseiPyArray_numpy_tt_declare(NPY_UBYTE, unsigned char)
senseiPyArray_numpy_tt_declare(NPY_UINT16, unsigned short)
senseiPyArray_numpy_tt_declare(NPY_UINT32, unsigned int)
senseiPyArray_numpy_tt_declare(NPY_ULONG, unsigned long)
senseiPyArray_numpy_tt_declare(NPY_UINT64, unsigned long long)
senseiPyArray_numpy_tt_declare(NPY_FLOAT, float)
senseiPyArray_numpy_tt_declare(NPY_DOUBLE, double)

// CPP_T - array type to match
// OBJ - PyArrayObject* instance
// CODE - code to execute on match
#define SENSEI_PY_ARRAY_DISPATCH_CASE(CPP_T, OBJ, CODE) \
  if (senseiPyArray::numpy_tt<CPP_T>::is_type(OBJ))     \
    {                                                   \
    using AT = CPP_T;                                   \
    CODE                                                \
    }

#define SENSEI_PY_ARRAY_DISPATCH(OBJ, CODE)                     \
  SENSEI_PY_ARRAY_DISPATCH_CASE(float, OBJ, CODE)               \
  SENSEI_PY_ARRAY_DISPATCH_CASE(double, OBJ, CODE)              \
  SENSEI_PY_ARRAY_DISPATCH_CASE(int, OBJ, CODE)                 \
  SENSEI_PY_ARRAY_DISPATCH_CASE(unsigned int, OBJ, CODE)        \
  SENSEI_PY_ARRAY_DISPATCH_CASE(long, OBJ, CODE)                \
  SENSEI_PY_ARRAY_DISPATCH_CASE(unsigned long, OBJ, CODE)       \
  SENSEI_PY_ARRAY_DISPATCH_CASE(long long, OBJ, CODE)           \
  SENSEI_PY_ARRAY_DISPATCH_CASE(unsigned long long, OBJ, CODE)  \
  SENSEI_PY_ARRAY_DISPATCH_CASE(char, OBJ, CODE)        \
  SENSEI_PY_ARRAY_DISPATCH_CASE(unsigned char, OBJ, CODE)

// ****************************************************************************
template <typename cpp_t>
bool copy(cpp_t *va, unsigned long n, PyObject *obj)
{
  // not an array
  if (!PyArray_Check(obj))
    return false;

  PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);

  // nothing to do.
  unsigned long n_elem = PyArray_SIZE(arr);
  if (!n_elem)
    return true;

  // check length of destination buffer
  if (n_elem != n)
    {
    SENSEI_ERROR("destination buffer size(" << n
      << ") does not match source buffer size("
      << n_elem << ")")
    return false;
    }

  // copy
  SENSEI_PY_ARRAY_DISPATCH(arr,
    unsigned long i = 0;
    NpyIter *it = NpyIter_New(arr, NPY_ITER_READONLY,
        NPY_KEEPORDER, NPY_NO_CASTING, nullptr);
    NpyIter_IterNextFunc *next = NpyIter_GetIterNext(it, nullptr);
    AT **ptrptr = reinterpret_cast<AT**>(NpyIter_GetDataPtrArray(it));
    do
      {
      va[i] = **ptrptr;
      ++i;
      }
    while (next(it));
    NpyIter_Deallocate(it);
    return true;
    )

  // unknown type
  return false;
}

}

#endif
