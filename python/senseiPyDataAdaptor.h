#ifndef senseiPyDataAdaptor_h
#define senseiPyDataAdaptor_h

#include "senseiPyObject.h"
#include "senseiPyGILState.h"
#include "Error.h"

#include <Python.h>

#include <vtkDataObject.h>
#include <vtkPythonUtil.h>
#include <string>

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
  "for DataAdaptor::" #_method ". The exception that "    \
  " occurred is:")                                        \
                                                          \
  PyErr_Print();                                          \
                                                          \
  Py_XDECREF(cb_str);                                     \
  }
#include <iostream>
using std::cerr;
using std::endl;

namespace senseiPyDataAdaptor
{

// a container for the DataAdaptor::GetMesh callable
class PyGetMeshCallback
{
public:
  PyGetMeshCallback(PyObject *f) : Callback(f) {}

  void SetObject(PyObject *f)
  { this->Callback.SetObject(f); }

  explicit operator bool() const
  { return static_cast<bool>(this->Callback); }

  vtkDataObject *operator()(bool structure_only)
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A GetMeshCallback was not provided");
      return nullptr;
      }

    // build arguments list and call the callback
    PyObject *args = Py_BuildValue("(i)", static_cast<int>(structure_only));

    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(GetMeshCallback, f)
      return nullptr;
      }

    Py_DECREF(args);

    // convert the return
    vtkDataObject *mesh = static_cast<vtkDataObject*>(
        vtkPythonUtil::GetPointerFromObject(ret, "vtkDataObject"));

    /*if (!mesh)
      {
      PyErr_Format(PyExc_TypeError,
        "The GetMeshCallback failed to produce data");
      return nullptr;
      }*/

    return mesh;
    }

private:
  senseiPyObject::PyCallablePointer Callback;
};

// a container for the DataAdaptor::AddArray callable
class PyAddArrayCallback
{
public:
  PyAddArrayCallback(PyObject *f) : Callback(f) {}

  void SetObject(PyObject *f)
  { this->Callback.SetObject(f); }

  explicit operator bool() const
  { return static_cast<bool>(this->Callback); }

  bool operator()(vtkDataObject *mesh, int association, const std::string &name)
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A AddArrayCallback was not provided");
      return false;
      }

    // build arguments list and call the callback
    PyObject *pyMesh = vtkPythonUtil::GetObjectFromPointer(
      static_cast<vtkObjectBase*>(mesh));

    PyObject *args = Py_BuildValue("Nis", pyMesh, association, name.c_str());

    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(AddArrayCallback, f)
      return false;
      }

    Py_DECREF(args);

    // convert the return
    if (!senseiPyObject::CppTT<bool>::IsType(ret))
      {
      PyErr_Format(PyExc_TypeError,
        "Bad return type from AddArrayCallback (not a bool).");
      return false;
      }

    return senseiPyObject::CppTT<bool>::Value(ret);
    }

private:
  senseiPyObject::PyCallablePointer Callback;
};

// a container for the DataAdaptor::GetNumberOfArrays callable
class PyGetNumberOfArraysCallback
{
public:
  PyGetNumberOfArraysCallback(PyObject *f) : Callback(f) {}

  void SetObject(PyObject *f)
  { this->Callback.SetObject(f); }

  explicit operator bool() const
  { return static_cast<bool>(this->Callback); }

  unsigned int operator()(int association)
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A GetNumberOfArraysCallback was not provided");
      return 0;
      }

    // build arguments list and call the callback
    PyObject *args = Py_BuildValue("(i)", association);

    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(GetNumberOfArraysCallback, f)
      return 0;
      }

    Py_DECREF(args);

    // convert the return
    if (!senseiPyObject::CppTT<int>::IsType(ret))
      {
      PyErr_Format(PyExc_TypeError,
        "Bad return type from GetNumberOfArraysCallback (not an int)");
      return 0;
      }

    return senseiPyObject::CppTT<int>::Value(ret);
    }

private:
  senseiPyObject::PyCallablePointer Callback;
};

// a container for the DataAdaptor::GetArrayName callable
class PyGetArrayNameCallback
{
public:
  PyGetArrayNameCallback(PyObject *f) : Callback(f) {}

  void SetObject(PyObject *f)
  { this->Callback.SetObject(f); }

  explicit operator bool() const
  { return static_cast<bool>(this->Callback); }

  std::string operator()(int association, unsigned int id)
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A GetArrayNameCallback was not provided");
      return std::string();
      }

    // build arguments list and call the callback
    PyObject *args = Py_BuildValue("iI", association, id);

    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(GetArrayNameCallback, f)
      return std::string();
      }

    Py_DECREF(args);

    // convert the return
    if (!senseiPyObject::CppTT<char*>::IsType(ret))
      {
      PyErr_Format(PyExc_TypeError,
        "Bad return type from GetArrayNameCallback");
      return std::string();
      }

    return senseiPyObject::CppTT<char*>::Value(ret);
    }

private:
  senseiPyObject::PyCallablePointer Callback;
};

// a container for the DataAdaptor::ReleaseData callable
class PyReleaseDataCallback
{
public:
  PyReleaseDataCallback(PyObject *f) : Callback(f) {}

  void SetObject(PyObject *f)
  { this->Callback.SetObject(f); }

  explicit operator bool() const
  { return static_cast<bool>(this->Callback); }

  void operator()()
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A ReleaseDataCallback was not provided");
      return;
      }

    // build arguments list and call the callback
    PyObject_CallObject(f, nullptr);
    if (PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(ReleaseDataCallback, f)
      }
    }

private:
  senseiPyObject::PyCallablePointer Callback;
};

}

#endif
