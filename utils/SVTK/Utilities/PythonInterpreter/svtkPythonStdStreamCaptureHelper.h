/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPythonStdStreamCaptureHelper.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPythonStdStreamCaptureHelper
 *
 */

#ifndef svtkPythonStdStreamCaptureHelper_h
#define svtkPythonStdStreamCaptureHelper_h

#include "structmember.h"
#include "svtkPythonInterpreter.h"

struct svtkPythonStdStreamCaptureHelper
{
  PyObject_HEAD int softspace; // Used by print to keep track of its state.
  bool DumpToError;

  void Write(const char* string)
  {
    if (this->DumpToError)
    {
      svtkPythonInterpreter::WriteStdErr(string);
    }
    else
    {
      svtkPythonInterpreter::WriteStdOut(string);
    }
  }

  void Flush()
  {
    if (this->DumpToError)
    {
      svtkPythonInterpreter::FlushStdErr();
    }
    else
    {
      svtkPythonInterpreter::FlushStdOut();
    }
  }

  svtkStdString Read() { return svtkPythonInterpreter::ReadStdin(); }

  void Close() { this->Flush(); }
};

static PyObject* svtkWrite(PyObject* self, PyObject* args);
static PyObject* svtkRead(PyObject* self, PyObject* args);
static PyObject* svtkFlush(PyObject* self, PyObject* args);
static PyObject* svtkClose(PyObject* self, PyObject* args);

// const_cast since older versions of python are not const correct.
static PyMethodDef svtkPythonStdStreamCaptureHelperMethods[] = {
  { const_cast<char*>("write"), svtkWrite, METH_VARARGS, const_cast<char*>("Dump message") },
  { const_cast<char*>("readline"), svtkRead, METH_VARARGS, const_cast<char*>("Read input line") },
  { const_cast<char*>("flush"), svtkFlush, METH_VARARGS, const_cast<char*>("Flush") },
  { const_cast<char*>("close"), svtkClose, METH_VARARGS, const_cast<char*>("Close") }, { 0, 0, 0, 0 }
};

static PyObject* svtkPythonStdStreamCaptureHelperNew(
  PyTypeObject* type, PyObject* /*args*/, PyObject* /*kwds*/)
{
  return type->tp_alloc(type, 0);
}

static PyMemberDef svtkPythonStdStreamCaptureHelperMembers[] = {
  { const_cast<char*>("softspace"), T_INT, offsetof(svtkPythonStdStreamCaptureHelper, softspace), 0,
    const_cast<char*>("Placeholder so print can keep state.") },
  { 0, 0, 0, 0, 0 }
};

static PyTypeObject svtkPythonStdStreamCaptureHelperType = {
#if PY_VERSION_HEX >= 0x02060000
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
#else
  PyObject_HEAD_INIT(&PyType_Type) 0,
#endif
    "svtkPythonStdStreamCaptureHelper",      // tp_name
  sizeof(svtkPythonStdStreamCaptureHelper),  // tp_basicsize
  0,                                        // tp_itemsize
  0,                                        // tp_dealloc
  0,                                        // tp_print
  0,                                        // tp_getattr
  0,                                        // tp_setattr
  0,                                        // tp_compare
  0,                                        // tp_repr
  0,                                        // tp_as_number
  0,                                        // tp_as_sequence
  0,                                        // tp_as_mapping
  0,                                        // tp_hash
  0,                                        // tp_call
  0,                                        // tp_str
  PyObject_GenericGetAttr,                  // tp_getattro
  PyObject_GenericSetAttr,                  // tp_setattro
  0,                                        // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
  "svtkPythonStdStreamCaptureHelper",        //  tp_doc
  0,                                        //  tp_traverse
  0,                                        //  tp_clear
  0,                                        //  tp_richcompare
  0,                                        //  tp_weaklistoffset
  0,                                        //  tp_iter
  0,                                        //  tp_iternext
  svtkPythonStdStreamCaptureHelperMethods,   //  tp_methods
  svtkPythonStdStreamCaptureHelperMembers,   //  tp_members
  0,                                        //  tp_getset
  0,                                        //  tp_base
  0,                                        //  tp_dict
  0,                                        //  tp_descr_get
  0,                                        //  tp_descr_set
  0,                                        //  tp_dictoffset
  0,                                        //  tp_init
  0,                                        //  tp_alloc
  svtkPythonStdStreamCaptureHelperNew,       //  tp_new
  0,                                        // freefunc tp_free; /* Low-level free-memory routine */
  0,                                        // inquiry tp_is_gc; /* For PyObject_IS_GC */
  0,                                        // PyObject *tp_bases;
  0,                                        // PyObject *tp_mro; /* method resolution order */
  0,                                        // PyObject *tp_cache;
  0,                                        // PyObject *tp_subclasses;
  0,                                        // PyObject *tp_weaklist;
  0,                                        // tp_del
#if PY_VERSION_HEX >= 0x02060000
  0, // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
  0, // tp_finalize
#endif
#if PY_VERSION_HEX >= 0x03080000
  0, // tp_vectorcall
#if PY_VERSION_HEX < 0x03090000
  0, // tp_print
#endif
#endif
};

static PyObject* svtkWrite(PyObject* self, PyObject* args)
{
  if (!self || !PyObject_TypeCheck(self, &svtkPythonStdStreamCaptureHelperType))
  {
    return 0;
  }

  svtkPythonStdStreamCaptureHelper* wrapper =
    reinterpret_cast<svtkPythonStdStreamCaptureHelper*>(self);

  char* string;
  // const_cast since older versions of python are not const correct.
  if (wrapper && PyArg_ParseTuple(args, "s", &string))
  {
    wrapper->Write(string);
  }
  return Py_BuildValue("");
}

static PyObject* svtkRead(PyObject* self, PyObject* args)
{
  (void)args;
  if (!self || !PyObject_TypeCheck(self, &svtkPythonStdStreamCaptureHelperType))
  {
    return 0;
  }

  svtkPythonStdStreamCaptureHelper* wrapper =
    reinterpret_cast<svtkPythonStdStreamCaptureHelper*>(self);

  svtkStdString ret;
  if (wrapper)
  {
    ret = wrapper->Read();
  }
  return Py_BuildValue("s", ret.c_str());
}

static PyObject* svtkFlush(PyObject* self, PyObject* args)
{
  (void)args;
  if (!self || !PyObject_TypeCheck(self, &svtkPythonStdStreamCaptureHelperType))
  {
    return 0;
  }

  svtkPythonStdStreamCaptureHelper* wrapper =
    reinterpret_cast<svtkPythonStdStreamCaptureHelper*>(self);
  if (wrapper)
  {
    wrapper->Flush();
  }
  return Py_BuildValue("");
}

static PyObject* svtkClose(PyObject* self, PyObject* args)
{
  (void)args;
  if (!self || !PyObject_TypeCheck(self, &svtkPythonStdStreamCaptureHelperType))
  {
    return 0;
  }

  svtkPythonStdStreamCaptureHelper* wrapper =
    reinterpret_cast<svtkPythonStdStreamCaptureHelper*>(self);
  if (wrapper)
  {
    wrapper->Close();
  }
  return Py_BuildValue("");
}

static svtkPythonStdStreamCaptureHelper* NewPythonStdStreamCaptureHelper(bool for_stderr = false)
{
  svtkPythonScopeGilEnsurer gilEnsurer;
  if (PyType_Ready(&svtkPythonStdStreamCaptureHelperType) < 0)
  {
    return 0;
  }

  svtkPythonStdStreamCaptureHelper* wrapper =
    PyObject_New(svtkPythonStdStreamCaptureHelper, &svtkPythonStdStreamCaptureHelperType);
  if (wrapper)
  {
    wrapper->DumpToError = for_stderr;
  }

  return wrapper;
}

#endif
// SVTK-HeaderTest-Exclude: svtkPythonStdStreamCaptureHelper.h
