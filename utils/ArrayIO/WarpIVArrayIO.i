%define MDOC
"Mesh and particle I/O using MPI-I/O with collective buffering

int writeArray(
        const char *fileName,
        int domain[6],
        int decomp[6],
        PyArrayObject *adata)
"
%enddef

%module (docstring=MDOC) WarpIVArrayIO

%{
#define SWIG_FILE_WITH_INIT
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  PyArray_API_arrayIO
#include <numpy/arrayobject.h>
#include "WarpIVArrayIO.h"
#include "WarpIVGlue.h"

static
int asIntArray(PyObject *obj, int len, int *array)
{
    if (!PySequence_Check(obj) || !(PySequence_Size(obj) == len))
        return -1;

    for (int i = 0; i < len; ++i)
        array[i] = PyInt_AsLong(PySequence_GetItem(obj, i));

    return 0;
}

%}


%init %{
import_array();
%}

%include "WarpIVTypemaps.i"

%inline %{
// **************************************************************************
int writeArray(
        const char *fileName,
        PyObject *aDomain,
        PyObject *aDecomp,
        PyObject *aValid,
        PyArrayObject *adata)
{
    // create the hints using the defaults
    MPI_Info hints = createWriterHints();

    int domain[6] = {0};
    if (asIntArray(aDomain, 6, domain))
    {
        PyErr_Format(PyExc_TypeError, "aDomain is not a 6 element sequence");
        return -1;
    }

    int decomp[6] = {0};
    if (asIntArray(aDecomp, 6, decomp))
    {
        PyErr_Format(PyExc_TypeError, "aDecomp is not a 6 element sequence");
        return -1;
    }

    int valid[6] = {0};
    if (asIntArray(aValid, 6, valid))
    {
        PyErr_Format(PyExc_TypeError, "aValid is not a 6 element sequence");
        return -1;
    }

    // dispatch based on data type
    switch (getType(adata))
    {
        numpyDispatch(

            // get a pointer to the numpy array
            CPP_TT::Type *data = NULL;
            if (getCPointer(adata, data))
            {
                PyErr_Format(PyExc_ValueError,
                    "failed to get pointer to array");
                return -1;
            }

            return writeArray<CPP_TT::Type>(fileName,
               MPI_COMM_WORLD, hints, domain, decomp, valid, data);
        )
        default:
            PyErr_Format(PyExc_ValueError, "failed dispatch unsupported type");
            return -1;
    }
    return -1;
}
%}
