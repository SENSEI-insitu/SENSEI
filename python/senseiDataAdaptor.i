%define SENSEI_DATA_ADAPTOR(DA)
/* Modify the DataAdaptor API for Python. Python doesn't
   support pass by reference. Hence, we need to wrap the
   core API. Rather than return an error code we will ask
   that Python codes raise and exception if there is an
   error and return function results(or void for cases when
   there are none) instead of using pass by reference/output
   parameters */
%ignore sensei::DA::GetNumberOfMeshes;
%ignore sensei::DA::GetMeshName;
%ignore sensei::DA::GetMesh;
%ignore sensei::DA::AddArray;
%ignore sensei::DA::GetNumberOfArrays;
%ignore sensei::DA::GetArrayName;
%ignore sensei::DA::ReleaseData;
/* memory management */
VTK_DERIVED(DA)
%enddef
