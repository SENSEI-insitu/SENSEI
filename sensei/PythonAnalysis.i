%define PYTHON_ANALYSIS_DOC
"SENSEI PythonAnalysis. Provides a minimal sensei wrapping to
code up analysis adaptors in a Python script."
%enddef
%module (docstring=PYTHON_ANALYSIS_DOC) PythonAnalysis
%feature("autodoc", "3");

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  PyArray_API_SENSEI_PYTHON_ANALYSIS
#include <numpy/arrayobject.h>
#include "senseiConfig.h"
#include "SVTKUtils.h"
#include "DataRequirements.h"
#include "Error.h"
#include <mpi.h>

/* PythonAnalysis sources included directly as they
   need direct access to the SWIG run time and type table */
#include "PythonAnalysis.h"
#include "PythonAnalysis.cxx"

#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
%}

%init %{
#if PY_VERSION_HEX < 0x03070000
PyEval_InitThreads();
#endif
import_array();
%}

/* import the SVTK module */
%{
#include "svtk.h"
%}
%import "svtk.i"

%include <mpi4py/mpi4py.i>
%mpi4py_typemap(Comm, MPI_Comm);

%include "senseiSTL.i"

/****************************************************************************
 * comnpile time settings
 ***************************************************************************/
%import "senseiConfig.h"

/****************************************************************************
 * DataAdaptor
 ***************************************************************************/
%include "DataAdaptor.i"
