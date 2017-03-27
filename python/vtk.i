%{
#include <vtkPythonUtil.h>
%}

%include "exception.i"

/*---------------------------------------------------------------------------
macro: VTK_SWIG_INTEROP(vtk_t)

arguments:
  vtk_t - a VTK class name that is used in the SWIG generated API.

The macro defines the typemaps needed for SWIG to convert to and
from VTK's Python bindings. Use this when your API containes pointers
to classes defined in VTK.
---------------------------------------------------------------------------*/
%define VTK_SWIG_INTEROP(vtk_t)
%{
#include <vtk_t##.h>
%}
%typemap(out) vtk_t*
{
  $result = vtkPythonUtil::GetObjectFromPointer(
    static_cast<vtkObjectBase*>($1));
}
%typemap(in) vtk_t*
{
  $1 = static_cast<vtk_t*>(
    vtkPythonUtil::GetPointerFromObject($input,#vtk_t));
  if (!$1)
  {
    SWIG_exception(SWIG_TypeError,
      "an object of type " #vtk_t " is required");
  }
}
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) vtk_t*
{
  $1 = vtkPythonUtil::GetPointerFromObject($input,#vtk_t) ? 1 : 0;
}
%enddef

/*---------------------------------------------------------------------------
macro: VTK_DERIVED(derived_t)

arguments:
  derived_t - name of a class that derives from vtkObjectBase.

The macro causes SWIG to wrap the class and defines memory management hooks
that prevent memory leaks when SWIG creates the objects. Use this to wrap
VTK classes defined in your project.
---------------------------------------------------------------------------*/
%define VTK_DERIVED(derived_t)
%{
#include <derived_t##.h>
%}
%feature("ref") derived_t "$this->Register(nullptr);"
%feature("unref") derived_t "$this->UnRegister(nullptr);"
%newobject derived_t##::New();
%include <derived_t##.h>
%enddef
