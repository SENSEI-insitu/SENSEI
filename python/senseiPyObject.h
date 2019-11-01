#ifndef senseiPyObject_h
#define senseiPyObject_h

#include <Python.h>
#include <string>
#include "senseiPyString.h"
#include "senseiPyInteger.h"

// we are going to be overly verbose in an effort to help
// the user debug their code. package this up for use in all
// the callbacks.
#define SENSEI_PY_CALLBACK_ERROR(_method, _cb_obj)        \
  {                                                       \
  PyObject *cb_str = PyObject_Str(_cb_obj);               \
  const char *cb_c_str = PyString_AsString(cb_str);       \
                                                          \
  SENSEI_ERROR("An exception ocurred when invoking the "  \
  "user supplied Python callback \"" << cb_c_str << "\""  \
  "for " #_method ". The exception that "                 \
  " occurred is:")                                        \
                                                          \
  PyErr_Print();                                          \
                                                          \
  Py_XDECREF(cb_str);                                     \
  }

namespace senseiPyObject
{
/// senseiPyObject::CppTT, A traits class for working with PyObject's
/**
if know the Python type Tag then this class gives you:

::type -- C++ type that can hold the Value of the PyObject.
::IsType -- returns true if the given PyObject has this type
::Value -- convert given PyObject to its C++ type

Python type tags and their coresponding PyObject's are:
int --> PyInt, long --> PyLong, bool --> PyBool,
float --> PyFloat, char* --> PyString
*/
template <typename py_t> struct CppTT
{};

/*
PY_T -- C-name of python type
CPP_T -- underlying type needed to store it on the C++ side
PY_CHECK -- function that verifies the PyObject is this type
PY_AS_CPP -- function that converts to the C++ type */
#define senseiPyObject_CppTT_declare(PY_T, CPP_T, PY_CHECK, PY_AS_CPP) \
template <> struct CppTT<PY_T>                                         \
{                                                                      \
  typedef CPP_T type;                                                  \
  static bool IsType(PyObject *obj) { return PY_CHECK(obj); }          \
  static type Value(PyObject *obj) { return PY_AS_CPP(obj); }          \
};
senseiPyObject_CppTT_declare(int, long, PY_INTEGER_CHECK, PY_INTEGER_TO_C_INT)
senseiPyObject_CppTT_declare(long, long, PY_INTEGER_CHECK, PyLong_AsLong)
senseiPyObject_CppTT_declare(float, double, PyFloat_Check, PyFloat_AsDouble)
senseiPyObject_CppTT_declare(char*, std::string, PY_STRING_CHECK, PY_STRING_TO_C_STRING)
senseiPyObject_CppTT_declare(bool, int, PyBool_Check, PY_INTEGER_TO_C_INT)

/// PyTT, traits class for working with PyObject's
/**
if you know the C++ type then this class gives you:

::Tag -- Use this in senseiPyObject::cpp_t to find
     the PyObject indentification and conversion
     methods. see example below.

::NewObject -- copy construct a new PyObject

here is an example of looking up the PyObject conversion
function(Value) from a known C++ type (float).

float val = CppTT<PyTT<float>::Tag>::Value(obj);

PyTT is used to take a C++ type and lookup the Python type
Tag. Then the type Tag is used to lookup the function.
*/
template <typename type> struct PyTT
{};

/**
CPP_T -- underlying type needed to store it on the C++ side
CPP_AS_PY -- function that converts from the C++ type */
#define senseiPyObject_PyTT_declare(CPP_T, PY_T, CPP_AS_PY)\
template <> struct PyTT<CPP_T>                             \
{                                                          \
  typedef PY_T Tag;                                        \
  static PyObject *NewObject(CPP_T val)                    \
  { return CPP_AS_PY(val); }                               \
};
senseiPyObject_PyTT_declare(char, int, C_INT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(short, int, C_INT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(int, int, C_INT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(long, int, C_INT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(long long, int, C_LLINT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(unsigned char, int, C_UINT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(unsigned short, int, C_UINT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(unsigned int, int, C_UINT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(unsigned long, int, C_UINT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(unsigned long long, int, C_ULLINT_TO_PY_INTEGER)
senseiPyObject_PyTT_declare(float, float, PyFloat_FromDouble)
senseiPyObject_PyTT_declare(double, float, PyFloat_FromDouble)
// strings are a special case
template <> struct PyTT<std::string>
{
  typedef char* Tag;
  static PyObject *NewObject(const std::string &s)
  { return C_STRING_TO_PY_STRING(s.c_str()); }
};

// dispatch macro.
// OBJ -- PyObject* instance
// CODE -- code block to execute on match
// OT -- a typedef to the match type available in
//     the code block
#define SENSEI_PY_OBJECT_DISPATCH_CASE(CPP_T, PY_OBJ, CODE) \
  if (senseiPyObject::CppTT<CPP_T>::IsType(PY_OBJ))         \
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
#define SENSEI_PY_OBJECT_DISPATCH_STR(PY_OBJ, CODE)         \
  SENSEI_PY_OBJECT_DISPATCH_CASE(char*, PY_OBJ, CODE)


// container that keeps a reference to a PyObject
class PyObjectPointer
{
public:
  PyObjectPointer() : Object(nullptr) {}

  PyObjectPointer(PyObject *obj)
    : Object(obj) { Py_XINCREF(this->Object); }

  virtual ~PyObjectPointer() { Py_XDECREF(this->Object); }

  PyObjectPointer(PyObjectPointer &&o)
     : Object(o.Object) { o.Object = nullptr; }

  PyObjectPointer &operator=(PyObjectPointer &&o)
    {
    PyObject *tmp = this->Object;
    this->Object = o.Object;
    o.Object = tmp;
    return *this;
    }

  PyObjectPointer(const PyObjectPointer &o)
     : Object(o.Object) { Py_XINCREF(this->Object); }

  PyObjectPointer &operator=(const PyObjectPointer &o)
    {
    Py_XINCREF(o.Object);
    Py_XDECREF(this->Object);
    this->Object = o.Object;
    return *this;
    }

  explicit operator bool () const
  { return this->Object != nullptr; }

  PyObject *GetObject(){ return this->Object; }

  virtual void SetObject(PyObject *obj)
    {
    Py_XINCREF(obj);
    Py_XDECREF(this->Object);
    this->Object = obj;
    }

private:
  PyObject *Object;
};


// container that keeps a Python callable object
class PyCallablePointer : public PyObjectPointer
{
public:
  PyCallablePointer() : PyObjectPointer() {}

  virtual ~PyCallablePointer()
  { this->PyObjectPointer::SetObject(nullptr); }

  PyCallablePointer(PyObject *f) : PyObjectPointer()
  { this->PyCallablePointer::SetObject(f); }

  PyCallablePointer(const PyCallablePointer &&o)
    : PyObjectPointer(std::move(o)) {}

  PyCallablePointer &operator=(const PyCallablePointer &&o)
    {
    this->PyObjectPointer::operator=(std::move(o));
    return *this;
    }

  PyCallablePointer(const PyCallablePointer &o)
    : PyObjectPointer(o) {}

  PyCallablePointer &operator=(const PyCallablePointer &o)
    {
    this->PyObjectPointer::operator=(o);
    return *this;
    }

  void SetObject(PyObject *f) override
    {
    if (PyCallable_Check(f))
      this->PyObjectPointer::SetObject(f);
    else
      PyErr_Format(PyExc_TypeError,
        "object is not callable");
    }
};

}

#endif
