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
#include "VTKUtils.h"
#include "DataRequirements.h"
#include "Error.h"
#include <mpi.h>

/* PythonAnalysis sources included directly as they
   need direct access to the SWIG run time and type table */
#include "PythonAnalysis.h"
#include "PythonAnalysis.cxx"
%}

%init %{
PyEval_InitThreads();
import_array();
%}

%include <std_string.i>
%include <std_vector.i>
%include <std_map.i>
%template(vector_string) std::vector<std::string>;
%template(map_string_bool) std::map<std::string, bool>;
%template(map_int_vector_string) std::map<int, std::vector<std::string>>;
%include <mpi4py/mpi4py.i>
%include "vtk.i"

%mpi4py_typemap(Comm, MPI_Comm);

%import "senseiConfig.h"

/****************************************************************************
 * VTK objects used in our API
 ***************************************************************************/
VTK_SWIG_INTEROP(vtkObjectBase)
VTK_SWIG_INTEROP(vtkDataObject)
VTK_SWIG_INTEROP(vtkInformation)

/****************************************************************************
 * DataAdaptor
 ***************************************************************************/
%include "DataAdaptor.i"
