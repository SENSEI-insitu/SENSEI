%{
#include "senseiConfig.h"
#include "senseiPySequence.h"
#include "senseiPyArray.h"
#include "Error.h"
#include <Python.h>
%}

%define FIXED_LENGTH_ARRAY_TYPEMAP(cpp_t)
%typemap(in) cpp_t[ANY] (cpp_t tmpFla[$1_dim0])
{
  if (!senseiPyArray::Copy(tmpFla, $1_dim0, $input)
    && !senseiPySequence::Copy(tmpFla, $1_dim0, $input))
    {
    const char *estr = "FIXED_LENGTH_ARRAY_TYPEMAP " #cpp_t " $1_dim0 failed.";
	SENSEI_ERROR(<< estr)
    PyErr_SetString(PyExc_RuntimeError, estr);
    }
  $1 = tmpFla;
}
%enddef

FIXED_LENGTH_ARRAY_TYPEMAP(float)
FIXED_LENGTH_ARRAY_TYPEMAP(double)
FIXED_LENGTH_ARRAY_TYPEMAP(char)
FIXED_LENGTH_ARRAY_TYPEMAP(int)
FIXED_LENGTH_ARRAY_TYPEMAP(long)
FIXED_LENGTH_ARRAY_TYPEMAP(long long)
FIXED_LENGTH_ARRAY_TYPEMAP(unsigned char)
FIXED_LENGTH_ARRAY_TYPEMAP(unsigned int)
FIXED_LENGTH_ARRAY_TYPEMAP(unsigned long)
FIXED_LENGTH_ARRAY_TYPEMAP(unsigned long long)
