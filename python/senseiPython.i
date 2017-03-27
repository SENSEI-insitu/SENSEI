%define SENSEI_PY_DOC
"SENSEI Python module
"
%enddef
%module (docstring=SENSEI_PY_DOC) senseiPython
%feature("autodoc", "3");

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  PyArray_API_SENSEI
#include <numpy/arrayobject.h>
#include "senseiConfig.h"
#include "LibsimImageProperties.h"
%}

%init %{
import_array();
%}

%include <std_string.i>
%include <std_vector.i>
%template(vector_string) std::vector<std::string>;
%include <mpi4py/mpi4py.i>
%include <vtk.i>
%include <senseiTypeMaps.i>

%mpi4py_typemap(Comm, MPI_Comm);

%import "senseiConfig.h"

VTK_SWIG_INTEROP(vtkObjectBase)
VTK_SWIG_INTEROP(vtkDataObject)
VTK_SWIG_INTEROP(vtkInformation)

/* SWIG generates bogus code for the following overloads, it looks
 like the fact that these static methods overload non-static
 methods is causing the problem */
%ignore sensei::DataAdaptor::SetDataTime(vtkInformation *,double);
%ignore sensei::DataAdaptor::SetDataTimeStep(vtkInformation *,int);

VTK_DERIVED(DataAdaptor)
VTK_DERIVED(AnalysisAdaptor)
VTK_DERIVED(VTKDataAdaptor)
VTK_DERIVED(ConfigurableAnalysis)

#ifdef ENABLE_CATALYST
VTK_DERIVED(CatalystAnalysisAdaptor)
#endif
#ifdef ENABLE_LIBSIM
VTK_DERIVED(LibsimAnalysisAdaptor)
%include "LibsimImageProperties.h"
#endif
#ifdef ENABLE_ADIOS
VTK_DERIVED(ADIOSAnalysisAdaptor)
VTK_DERIVED(ADIOSDataAdaptor)
#endif
#ifdef ENABLE_VTK_XMLP
VTK_DERIVED(VTKPosthocIO)
#endif
