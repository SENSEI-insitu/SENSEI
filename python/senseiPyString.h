#ifndef senseiPyString_h
#define senseiPyString_h

#include "senseiConfig.h"

#if SENSEI_PYTHON_VERSION == 2
#define PY_STRING_CHECK PyString_Check
#define PY_STRING_TO_C_STRING PyString_AsString
#define C_STRING_TO_PY_STRING PyString_FromString
#define C_STRING_LITERAL(arg) const_cast<char*>(arg)
#elif SENSEI_PYTHON_VERSION == 3
#define PY_STRING_CHECK PyUnicode_Check
#define PY_STRING_TO_C_STRING PyUnicode_AsUTF8
#define C_STRING_TO_PY_STRING PyUnicode_FromString
#define C_STRING_LITERAL(arg) L##arg
#else
#error #SENSEI_PYTHON_VERSION " must be 2 or 3"
#endif

#endif
