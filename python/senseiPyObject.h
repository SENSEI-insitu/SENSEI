#ifndef senseiPyObject_h
#define senseiPyObject_h

#include <Python.h>

namespace senseiPyObject
{
/// senseiPyObject::cpp_tt, A traits class for working with PyObject's
/**
if know the Python type tag then this class gives you:

::type -- C++ type that can hold the value of the PyObject.
::is_type -- returns true if the given PyObject has this type
::value -- convert given PyObject to its C++ type

Python type tags and their coresponding PyObject's are:
int --> PyInt, long --> PyLong, bool --> PyBool,
float --> PyFloat, char* --> PyString
*/
template <typename py_t> struct cpp_tt
{};

/*
PY_T -- C-name of python type
CPP_T -- underlying type needed to store it on the C++ side
PY_CHECK -- function that verifies the PyObject is this type
PY_AS_CPP -- function that converts to the C++ type */
#define senseiPyObject_cpp_tt_declare(PY_T, CPP_T, PY_CHECK, PY_AS_CPP) \
template <> struct cpp_tt<PY_T>                                         \
{                                                                       \
  typedef CPP_T type;                                                   \
  static bool is_type(PyObject *obj) { return PY_CHECK(obj); }          \
  static type value(PyObject *obj) { return PY_AS_CPP(obj); }           \
};
senseiPyObject_cpp_tt_declare(int, long, PyInt_Check, PyInt_AsLong)
senseiPyObject_cpp_tt_declare(long, long, PyLong_Check, PyLong_AsLong)
senseiPyObject_cpp_tt_declare(float, double, PyFloat_Check, PyFloat_AsDouble)
senseiPyObject_cpp_tt_declare(char*, std::string, PyString_Check, PyString_AsString)
senseiPyObject_cpp_tt_declare(bool, int, PyBool_Check, PyInt_AsLong)

/// py_tt, traits class for working with PyObject's
/**
if you know the C++ type then this class gives you:

::tag -- Use this in senseiPyObject::cpp_t to find
     the PyObject indentification and conversion
     methods. see example below.

::new_object -- copy construct a new PyObject

here is an example of looking up the PyObject conversion
function(value) from a known C++ type (float).

float val = cpp_tt<py_tt<float>::tag>::value(obj);

py_tt is used to take a C++ type and lookup the Python type
tag. Then the type tag is used to lookup the function.
*/
template <typename type> struct py_tt
{};

/**
CPP_T -- underlying type needed to store it on the C++ side
CPP_AS_PY -- function that converts from the C++ type */
#define senseiPyObject_py_tt_declare(CPP_T, PY_T, CPP_AS_PY)\
template <> struct py_tt<CPP_T>                             \
{                                                           \
  typedef PY_T tag;                                         \
  static PyObject *new_object(CPP_T val)                    \
  { return CPP_AS_PY(val); }                                \
};
senseiPyObject_py_tt_declare(char, int, PyInt_FromLong)
senseiPyObject_py_tt_declare(short, int, PyInt_FromLong)
senseiPyObject_py_tt_declare(int, int, PyInt_FromLong)
senseiPyObject_py_tt_declare(long, int, PyInt_FromLong)
senseiPyObject_py_tt_declare(long long, int, PyInt_FromSsize_t)
senseiPyObject_py_tt_declare(unsigned char, int, PyInt_FromSize_t)
senseiPyObject_py_tt_declare(unsigned short, int, PyInt_FromSize_t)
senseiPyObject_py_tt_declare(unsigned int, int, PyInt_FromSize_t)
senseiPyObject_py_tt_declare(unsigned long, int, PyInt_FromSize_t)
senseiPyObject_py_tt_declare(unsigned long long, int, PyInt_FromSize_t)
senseiPyObject_py_tt_declare(float, float, PyFloat_FromDouble)
senseiPyObject_py_tt_declare(double, float, PyFloat_FromDouble)
// strings are a special case
template <> struct py_tt<std::string>
{
  typedef char* tag;
  static PyObject *new_object(const std::string &s)
  { return PyString_FromString(s.c_str()); }
};

// dispatch macro.
// OBJ -- PyObject* instance
// CODE -- code block to execute on match
// OT -- a typedef to the match type available in
//     the code block
#define SENSEI_PY_OBJECT_DISPATCH_CASE(CPP_T, PY_OBJ, CODE) \
  if (senseiPyObject::cpp_tt<CPP_T>::is_type(PY_OBJ))       \
    {                                                       \
    using OT = CPP_T;                                       \
    CODE                                                    \
    }

#define SENSEI_PY_OBJECT_DISPATCH(PY_OBJ, CODE)             \
  SENSEI_PY_OBJECT_DISPATCH_CASE(int, PY_OBJ, CODE)         \
  else SENSEI_PY_OBJECT_DISPATCH_CASE(float, PY_OBJ, CODE)  \
  else SENSEI_PY_OBJECT_DISPATCH_CASE(char*, PY_OBJ, CODE)  \
  else SENSEI_PY_OBJECT_DISPATCH_CASE(long, PY_OBJ, CODE)

// without string
#define SENSEI_PY_OBJECT_DISPATCH_NUM(PY_OBJ, CODE)         \
  SENSEI_PY_OBJECT_DISPATCH_CASE(int, PY_OBJ, CODE)         \
  else SENSEI_PY_OBJECT_DISPATCH_CASE(float, PY_OBJ, CODE)  \
  else SENSEI_PY_OBJECT_DISPATCH_CASE(long, PY_OBJ, CODE)

// just string
#define SENSEI_PY_OBJECT_DISPATCH_STR(PY_OBJ, CODE)       \
  SENSEI_PY_OBJECT_DISPATCH_CASE(char*, PY_OBJ, CODE)

}

#endif
