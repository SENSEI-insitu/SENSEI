#ifndef senseiPyDataAdaptor_h
#define senseiPyDataAdaptor_h

#include "senseiPyObject.h"
#include "senseiPyGILState.h"
#include "MeshMetadata.h"
#include "Error.h"

#include <Python.h>

#include <vtkDataObject.h>
#include <vtkPythonUtil.h>
#include <string>
#include <iostream>
using std::cerr;
using std::endl;

namespace senseiPyDataAdaptor
{

// a container for the DataAdaptor::GetNumberOfMeshes callable
class PyGetNumberOfMeshesCallback
{
public:
  PyGetNumberOfMeshesCallback(PyObject *f) : Callback(f) {}

  void SetObject(PyObject *f)
  { this->Callback.SetObject(f); }

  explicit operator bool() const
  { return static_cast<bool>(this->Callback); }

  int operator()(unsigned int &numberOfMeshes)
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A GetNumberOfMeshesCallback was not provided");
      return -1;
      }

    // build arguments list and call the callback
    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, nullptr)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(DataAdaptor::GetNumberOfMeshesCallback, f)
      return -1;
      }

    // convert the return
    if (!senseiPyObject::CppTT<int>::IsType(ret))
      {
      PyErr_Format(PyExc_TypeError,
        "Bad return type from GetNumberOfMeshesCallback (not an int)");
      return -1;
      }

    numberOfMeshes = senseiPyObject::CppTT<int>::Value(ret);
    return 0;
    }

private:
  senseiPyObject::PyCallablePointer Callback;
};

// a container for the DataAdaptor::GetMeshMetadata callable
class PyGetMeshMetadataCallback
{
public:
  PyGetMeshMetadataCallback(PyObject *f) : Callback(f) {}

  void SetObject(PyObject *f)
  { this->Callback.SetObject(f); }

  explicit operator bool() const
  { return static_cast<bool>(this->Callback); }

  int operator()(unsigned int index, sensei::MeshMetadataPtr &mdp)
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A GetMeshMetadataCallback was not provided");
      return -1;
      }

    // wrap the flags
    PyObject *pyFlags = SWIG_NewPointerObj(
      (void*)&mdp->Flags, SWIGTYPE_p_sensei__MeshMetadataFlags, 0);

    // build arguments list and call the callback
    // the tuple takes owner ship with N
    PyObject *args = Py_BuildValue("(IN)", index, pyFlags);

    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(DataAdaptor::GetMeshMetadataCallback, f)
      return -1;
      }

    // delete the argument tuple
    Py_DECREF(args);

    // convert the return
    int newmem = 0;
    void *tmpvp = nullptr;
    int ierr = SWIG_ConvertPtrAndOwn(ret, &tmpvp,
      SWIGTYPE_p_std__shared_ptrT_sensei__MeshMetadata_t, 0, &newmem);

    if (ierr == SWIG_ERROR)
      {
      SENSEI_ERROR("GetMeshMetadata callback returned an invalid object")
      return -1;
      }

    /* newmem = 1 SWIG_CAST_NEW_MEMORY = 2 tmpvp = 0x557e358ea080
    std::cerr << "newmem = " << newmem << " SWIG_CAST_NEW_MEMORY = "
       << SWIG_CAST_NEW_MEMORY  << " tmpvp = " << tmpvp << std::endl;*/

    if (tmpvp)
      mdp = *(reinterpret_cast< sensei::MeshMetadataPtr * >(tmpvp));

    if (newmem & SWIG_CAST_NEW_MEMORY)
      delete reinterpret_cast<sensei::MeshMetadataPtr*>(tmpvp);

    // delete the python wrapping and release its reference
    // now that we hold one
    Py_DECREF(ret);

    return 0;
    }

private:
  senseiPyObject::PyCallablePointer Callback;
};

// a container for the DataAdaptor::GetMesh callable
class PyGetMeshCallback
{
public:
  PyGetMeshCallback(PyObject *f) : Callback(f) {}

  void SetObject(PyObject *f)
  { this->Callback.SetObject(f); }

  explicit operator bool() const
  { return static_cast<bool>(this->Callback); }

  int operator()(const std::string &meshName,
    bool structureOnly, vtkDataObject *&mesh)
    {
    mesh = nullptr;

    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A GetMeshCallback was not provided");
      return -1;
      }

    // build arguments list and call the callback
    PyObject *args = Py_BuildValue("(si)",
      meshName.c_str(), static_cast<int>(structureOnly));

    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(DataAdaptor::GetMeshCallback, f)
      return -1;
      }

    Py_DECREF(args);

    // convert the return
    mesh = static_cast<vtkDataObject*>(
        vtkPythonUtil::GetPointerFromObject(ret, "vtkDataObject"));

    // because VTK's python runtime akes ownership
    mesh->Register(nullptr);

    return 0;
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

  int operator()(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A AddArrayCallback was not provided");
      return -1;
      }

    // build arguments list and call the callback
    PyObject *pyMesh = vtkPythonUtil::GetObjectFromPointer(
      static_cast<vtkObjectBase*>(mesh));

    PyObject *args = Py_BuildValue("Nsis", pyMesh, meshName.c_str(),
      association, arrayName.c_str());

    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(DataAdaptor::AddArrayCallback, f)
      return -1;
      }

    Py_DECREF(args);

    return 0;
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

  int operator()(const std::string &meshName, int association,
    unsigned int &numberOfArrays)
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A GetNumberOfArraysCallback was not provided");
      return -1;
      }

    // build arguments list and call the callback
    PyObject *args = Py_BuildValue("si", meshName.c_str(), association);

    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(DataAdaptor::GetNumberOfArraysCallback, f)
      return -1;
      }

    Py_DECREF(args);

    // convert the return
    if (!senseiPyObject::CppTT<int>::IsType(ret))
      {
      PyErr_Format(PyExc_TypeError,
        "Bad return type from GetNumberOfArraysCallback (not an int)");
      return -1;
      }

    numberOfArrays = senseiPyObject::CppTT<int>::Value(ret);
    return 0;
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

  int operator()(const std::string &meshName, int association,
    unsigned int index, std::string &arrayName)
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A GetArrayNameCallback was not provided");
      return -1;
      }

    // build arguments list and call the callback
    PyObject *args =
      Py_BuildValue("siI", meshName.c_str(), association, index);

    PyObject *ret = nullptr;
    if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(DataAdaptor::GetArrayNameCallback, f)
      return -1;
      }

    Py_DECREF(args);

    // convert the return
    if (!senseiPyObject::CppTT<char*>::IsType(ret))
      {
      PyErr_Format(PyExc_TypeError,
        "Bad return type from GetArrayNameCallback");
      return -1;
      }

    arrayName = senseiPyObject::CppTT<char*>::Value(ret);
    return 0;
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

  int operator()()
    {
    // lock the GIL
    senseiPyGILState gil;

    // get the callback
    PyObject *f = this->Callback.GetObject();
    if (!f)
      {
      PyErr_Format(PyExc_TypeError,
        "A ReleaseDataCallback was not provided");
      return -1;
      }

    // build arguments list and call the callback
    PyObject_CallObject(f, nullptr);
    if (PyErr_Occurred())
      {
      SENSEI_PY_CALLBACK_ERROR(DataAdaptor::ReleaseDataCallback, f)
      }

    return 0;
    }

private:
  senseiPyObject::PyCallablePointer Callback;
};

}

#endif
