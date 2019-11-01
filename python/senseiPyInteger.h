#ifndef senseiPyInteger_h
#define senseiPyInteger_h

#include "senseiConfig.h"

#if SENSEI_PYTHON_VERSION == 2
#define PY_INTEGER_CHECK PyInt_Check
// conversions from C++
#define C_INT_TO_PY_INTEGER PyInt_FromLong
#define C_UINT_TO_PY_INTEGER PyInt_FromSize_t
#define C_LLINT_TO_PY_INTEGER PyLong_FromLongLong
#define C_ULLINT_TO_PY_INTEGER PyLong_FromUnsignedLongLong
// conversions to C++
#define PY_INTEGER_TO_C_INT PyInt_AsLong
#define PY_INTEGER_TO_C_UINT PyInt_AsSize_t
#define PY_INTEGER_TO_C_LLINT PyInt_AsLongLong
#define PY_INTEGER_TO_C_ULLINT PyInt_AsUnsignedLongLong
#elif SENSEI_PYTHON_VERSION == 3
#define PY_INTEGER_CHECK PyLong_Check
// conversions from C++
#define PY_INTEGER_TO_C_INT PyLong_AsLong
#define PY_INTEGER_TO_C_INTU PyLong_AsUnsignedLong
#define PY_INTEGER_TO_C_INTLL PyLong_AsLongLong
#define PY_INTEGER_TO_C_INTULL PyLong_AsUnsignedLongLong
// conversions to C++
#define C_INT_TO_PY_INTEGER PyLong_FromLong
#define C_UINT_TO_PY_INTEGER PyLong_FromUnsignedLong
#define C_LLINT_TO_PY_INTEGER PyLong_FromLongLong
#define C_ULLINT_TO_PY_INTEGER PyLong_FromUnsignedLongLong
#else
#error #SENSEI_PYTHON_VERSION " must be 2 or 3"
#endif

#endif
