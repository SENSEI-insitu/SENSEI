%module svtk
%{
#include "svtkType.h"
#include "svtkCommonCoreModule.h"
#include "svtkCommonDataModelModule.h"
#include "svtkConfigure.h"
#include "svtk_kwiml.h"
#include "kwiml/int.h"
#include "kwiml/abi.h"
#include "svtkSetGet.h"

#include <sstream>
%}

%include "std_string.i"

#define __GNUC__
#define __x86_64
#define KWIML_INT_NO_VERIFY
#define KWIML_ABI_NO_VERIFY

/* automatically convert to the derived type */
%typemap(out) svtkDataArray*
{
  if (dynamic_cast<svtkDoubleArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkDoubleArray*>($1),
      SWIGTYPE_p_svtkDoubleArray, 0);
  }
  else if (dynamic_cast<svtkFloatArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkFloatArray*>($1),
      SWIGTYPE_p_svtkFloatArray, 0);
  }
  else if (dynamic_cast<svtkCharArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkCharArray*>($1),
      SWIGTYPE_p_svtkCharArray, 0);
  }
  else if (dynamic_cast<svtkShortArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkShortArray*>($1),
      SWIGTYPE_p_svtkShortArray, 0);
  }
  else if (dynamic_cast<svtkIntArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkIntArray*>($1),
      SWIGTYPE_p_svtkIntArray, 0);
  }
  else if (dynamic_cast<svtkLongArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkLongArray*>($1),
      SWIGTYPE_p_svtkLongArray, 0);
  }
  else if (dynamic_cast<svtkLongLongArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkLongLongArray*>($1),
      SWIGTYPE_p_svtkLongLongArray, 0);
  }
  else if (dynamic_cast<svtkIdTypeArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkIdTypeArray*>($1),
      SWIGTYPE_p_svtkIdTypeArray, 0);
  }
  else if (dynamic_cast<svtkUnsignedCharArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkUnsignedCharArray*>($1),
      SWIGTYPE_p_svtkUnsignedCharArray, 0);
  }
  else if (dynamic_cast<svtkUnsignedShortArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkUnsignedShortArray*>($1),
      SWIGTYPE_p_svtkUnsignedShortArray, 0);
  }
  else if (dynamic_cast<svtkUnsignedIntArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkUnsignedIntArray*>($1),
      SWIGTYPE_p_svtkUnsignedIntArray, 0);
  }
  else if (dynamic_cast<svtkUnsignedLongArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkUnsignedLongArray*>($1),
      SWIGTYPE_p_svtkUnsignedLongArray, 0);
  }
  else if (dynamic_cast<svtkUnsignedLongLongArray*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkUnsignedLongLongArray*>($1),
      SWIGTYPE_p_svtkUnsignedLongLongArray, 0);
  }
  else
  {
    std::cerr << "NOTE: Automatic conversions for "
      << $1->GetClassName() << " are not implemented."
      << std::endl;
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkDataArray*>($1),
      SWIGTYPE_p_svtkDataArray, 0);
  }
}

/* automatically convert to the derived type */
%typemap(out) svtkDataObject*
{
  if (dynamic_cast<svtkOverlappingAMR*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkOverlappingAMR*>($1),
      SWIGTYPE_p_svtkOverlappingAMR, $owner);
  }
  else if (dynamic_cast<svtkMultiBlockDataSet*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkMultiBlockDataSet*>($1),
      SWIGTYPE_p_svtkMultiBlockDataSet, $owner);
  }
  else if (dynamic_cast<svtkCompositeDataSet*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkCompositeDataSet*>($1),
      SWIGTYPE_p_svtkCompositeDataSet, $owner);
  }
  else if (dynamic_cast<svtkImageData*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkImageData*>($1),
      SWIGTYPE_p_svtkImageData, $owner);
  }
  else if (dynamic_cast<svtkRectilinearGrid*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkRectilinearGrid*>($1),
      SWIGTYPE_p_svtkRectilinearGrid, $owner);
  }
  else if (dynamic_cast<svtkPolyData*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkPolyData*>($1),
      SWIGTYPE_p_svtkPolyData, $owner);
  }
  else if (dynamic_cast<svtkUnstructuredGrid*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkUnstructuredGrid*>($1),
      SWIGTYPE_p_svtkUnstructuredGrid, $owner);
  }
  else if (dynamic_cast<svtkStructuredGrid*>($1))
  {
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkStructuredGrid*>($1),
      SWIGTYPE_p_svtkStructuredGrid, $owner);
  }
  else
  {
    std::cerr << "NOTE: Automatic conversions for "
      << $1->GetClassName() << " are not implemented."
      << std::endl;
    $result = SWIG_NewPointerObj(
      (void*)static_cast<svtkDataObject*>($1),
      SWIGTYPE_p_svtkDataObject, $owner);
  }
}

/* SWIG/SVTK memory management */
%define SVTK_OBJECT_FACTORY(OBJ)
%newobject OBJ##::NewInstance;
%feature("ref") OBJ "$this->Register(nullptr);"
%feature("unref") OBJ "$this->UnRegister(nullptr);"
%newobject OBJ##::New();
%delobject OBJ##::Delete();
%typemap(newfree) OBJ* "$1->UnRegister(nullptr);"
%enddef

%define SVTK_OBJECT_IGNORE_CPP_API(OBJ)
%ignore OBJ##::OBJ##();
%ignore OBJ##::operator=;
%enddef

/* Python style constructor */
%define SVTK_OBJECT_CONSTRUCTOR(OBJ)
OBJ##()
{
  OBJ *inst = dynamic_cast<OBJ*>(OBJ##::New());
  return inst;
}
%enddef

/* Python __str__ */
%define SVTK_OBJECT_STR(OBJ)
std::string __str__()
{
    std::ostringstream oss;
    oss << #OBJ << std::endl;
    self->Print(oss);
    return oss.str();
}
%enddef

/* wraps the passed class */
%define SVTK_WRAP_CLASS(CLASS)
%{
#include <CLASS##.h>
%}
%extend CLASS
{
    SVTK_OBJECT_CONSTRUCTOR(CLASS)
    SVTK_OBJECT_STR(CLASS)
};
SVTK_OBJECT_IGNORE_CPP_API(CLASS)
SVTK_OBJECT_FACTORY(CLASS)
%include <CLASS##.h>
%enddef

/* Python __str__ */
%define SVTK_DATA_ARRAY_STR(OBJ)
std::string __str__()
{
    std::ostringstream oss;
    oss << #OBJ << std::endl;
    self->Print(oss);

    size_t nTups = self->GetNumberOfTuples();
    size_t nComps = self->GetNumberOfComponents();
    size_t nElem = nTups*nComps;
    switch (self->GetDataType())
    {
    svtkTemplateMacro(
        SVTK_TT *pData = (SVTK_TT*)self->GetVoidPointer(0);
        for (size_t i = 0; i < nElem; ++i)
        {
            oss << pData[i] << ", ";
            if (((i + 1) % 20) == 0)
                oss << std::endl;
        }
        );
    default:
        std::cerr << "ERROR!!" << std::endl;
    }

    return oss.str();
}
%enddef

/* wraps the passed class */
%define SVTK_WRAP_DATA_ARRAY(CLASS)
%{
#include <CLASS##.h>
%}
%extend CLASS
{
    SVTK_OBJECT_CONSTRUCTOR(CLASS)
    SVTK_DATA_ARRAY_STR(CLASS)
};
SVTK_OBJECT_IGNORE_CPP_API(CLASS)
SVTK_OBJECT_FACTORY(CLASS)
%include <CLASS##.h>
%enddef


%include "svtkCommonCoreModule.h"
%include "svtkCommonDataModelModule.h"

%include "svtkConfigure.h"
%include "svtk_kwiml.h"
%include "kwiml/int.h"
%include "kwiml/abi.h"
%include "svtkType.h"
%include "svtkSetGet.h"
%include "svtkCellType.h"
%include "svtkWrappingHints.h"

%ignore svtkObjectBase::operator<<;
SVTK_WRAP_CLASS(svtkObjectBase)

%ignore svtkObject::svtkClassMemberHandlerPointer::operator=;
SVTK_WRAP_CLASS(svtkObject)

/****************************************************************************
 * Data Arrays
 ***************************************************************************/
SVTK_WRAP_DATA_ARRAY(svtkAbstractArray)
SVTK_WRAP_DATA_ARRAY(svtkDataArray)

%ignore svtkGenericDataArray::DoComputeScalarRange;
SVTK_WRAP_DATA_ARRAY(svtkGenericDataArray)
%template(svtkGenericDataArray_char) svtkGenericDataArray< svtkAOSDataArrayTemplate< char >, char >;
%template(svtkGenericDataArray_double) svtkGenericDataArray< svtkAOSDataArrayTemplate< double >, double  >;
%template(svtkGenericDataArray_float) svtkGenericDataArray< svtkAOSDataArrayTemplate< float >, float >;
/*%template(svtkGenericDataArray_svtkIdType) svtkGenericDataArray< svtkAOSDataArrayTemplate< svtkIdType > >;*/
%template(svtkGenericDataArray_int) svtkGenericDataArray< svtkAOSDataArrayTemplate< int >, int >;
%template(svtkGenericDataArray_long) svtkGenericDataArray< svtkAOSDataArrayTemplate< long >, long >;
%template(svtkGenericDataArray_long_long) svtkGenericDataArray< svtkAOSDataArrayTemplate< long long >, long long >;
%template(svtkGenericDataArray_short) svtkGenericDataArray< svtkAOSDataArrayTemplate< short >, short >;
%template(svtkGenericDataArray_signed_char) svtkGenericDataArray< svtkAOSDataArrayTemplate< signed char >, signed char >;
%template(svtkGenericDataArray_unsigned_char) svtkGenericDataArray< svtkAOSDataArrayTemplate< unsigned char >, unsigned char >;
%template(svtkGenericDataArray_unsigned_int) svtkGenericDataArray< svtkAOSDataArrayTemplate< unsigned int >, unsigned int >;
%template(svtkGenericDataArray_unsigned_long) svtkGenericDataArray< svtkAOSDataArrayTemplate< unsigned long >, unsigned long >;
%template(svtkGenericDataArray_unsigned_long_long) svtkGenericDataArray< svtkAOSDataArrayTemplate< unsigned long long >, unsigned long long >;
%template(svtkGenericDataArray_unsigned_short) svtkGenericDataArray< svtkAOSDataArrayTemplate< unsigned short >, unsigned short >;

SVTK_WRAP_DATA_ARRAY(svtkAOSDataArrayTemplate)
%template(svtkAOSDataArrayTemplate_char) svtkAOSDataArrayTemplate< char >;
%template(svtkAOSDataArrayTemplate_double) svtkAOSDataArrayTemplate< double >;
%template(svtkAOSDataArrayTemplate_float) svtkAOSDataArrayTemplate< float >;
/*%template(svtkAOSDataArrayTemplate_svtkIdType) svtkAOSDataArrayTemplate< svtkIdType >;*/
%template(svtkAOSDataArrayTemplate_int) svtkAOSDataArrayTemplate< int >;
%template(svtkAOSDataArrayTemplate_long) svtkAOSDataArrayTemplate< long >;
%template(svtkAOSDataArrayTemplate_long_long) svtkAOSDataArrayTemplate< long long >;
%template(svtkAOSDataArrayTemplate_short) svtkAOSDataArrayTemplate< short >;
%template(svtkAOSDataArrayTemplate_signed_char) svtkAOSDataArrayTemplate< signed char >;
%template(svtkAOSDataArrayTemplate_unsigned_char) svtkAOSDataArrayTemplate< unsigned char >;
%template(svtkAOSDataArrayTemplate_unsigned_int) svtkAOSDataArrayTemplate< unsigned int >;
%template(svtkAOSDataArrayTemplate_unsigned_long) svtkAOSDataArrayTemplate< unsigned long >;
%template(svtkAOSDataArrayTemplate_unsigned_long_long) svtkAOSDataArrayTemplate< unsigned long long >;
%template(svtkAOSDataArrayTemplate_unsigned_short) svtkAOSDataArrayTemplate< unsigned short >;

SVTK_WRAP_DATA_ARRAY(svtkSOADataArrayTemplate)
SVTK_WRAP_DATA_ARRAY(svtkTypedDataArray)
SVTK_WRAP_DATA_ARRAY(svtkDataArrayTemplate)
SVTK_WRAP_DATA_ARRAY(svtkBitArray)
SVTK_WRAP_DATA_ARRAY(svtkCharArray)
SVTK_WRAP_DATA_ARRAY(svtkShortArray)
SVTK_WRAP_DATA_ARRAY(svtkIntArray)
SVTK_WRAP_DATA_ARRAY(svtkLongArray)
SVTK_WRAP_DATA_ARRAY(svtkLongLongArray)
SVTK_WRAP_DATA_ARRAY(svtkSignedCharArray)
SVTK_WRAP_DATA_ARRAY(svtkUnsignedCharArray)
SVTK_WRAP_DATA_ARRAY(svtkUnsignedShortArray)
SVTK_WRAP_DATA_ARRAY(svtkUnsignedIntArray)
SVTK_WRAP_DATA_ARRAY(svtkUnsignedLongArray)
SVTK_WRAP_DATA_ARRAY(svtkUnsignedLongLongArray)
SVTK_WRAP_DATA_ARRAY(svtkIdTypeArray)
SVTK_WRAP_DATA_ARRAY(svtkDoubleArray)
SVTK_WRAP_DATA_ARRAY(svtkFloatArray)

/*SVTK_WRAP_CLASS(svtkArrayCoordinates)*/
/*SVTK_WRAP_CLASS(svtkArrayDispatch)*/
/*SVTK_WRAP_CLASS(svtkArrayExtents)
SVTK_WRAP_CLASS(svtkArrayExtentsList)*/
/*SVTK_WRAP_CLASS(svtkArrayInterpolate)*/
SVTK_WRAP_CLASS(svtkArrayIterator)
/*SVTK_WRAP_CLASS(svtkArrayIteratorIncludes)*/
SVTK_WRAP_CLASS(svtkArrayIteratorTemplate)
/*SVTK_WRAP_CLASS(svtkArrayPrint)
SVTK_WRAP_CLASS(svtkArrayRange)
SVTK_WRAP_CLASS(svtkArraySort)
SVTK_WRAP_CLASS(svtkArrayWeights)*/
SVTK_WRAP_CLASS(svtkBitArrayIterator)
SVTK_WRAP_CLASS(svtkDataArrayAccessor)
SVTK_WRAP_CLASS(svtkCollection)
SVTK_WRAP_CLASS(svtkCollectionIterator)
SVTK_WRAP_CLASS(svtkDataArrayCollection)
SVTK_WRAP_CLASS(svtkDataArrayCollectionIterator)
/*SVTK_WRAP_CLASS(svtkDataArrayIteratorMacro)
*SVTK_WRAP_CLASS(svtkDataArrayMeta)
SVTK_WRAP_CLASS(svtkDataArrayRange)
SVTK_WRAP_CLASS(svtkDataArraySelection)*/
/*SVTK_WRAP_CLASS(svtkDataArrayTupleRange_AOS)
SVTK_WRAP_CLASS(svtkDataArrayTupleRange_Generic)
SVTK_WRAP_CLASS(svtkDataArrayValueRange_AOS)
SVTK_WRAP_CLASS(svtkDataArrayValueRange_Generic) */
/* SVTK_WRAP_CLASS(svtkDenseArray) */
/*SVTK_WRAP_CLASS(svtkGenericDataArrayLookupHelper) */
/*SVTK_WRAP_CLASS(svtkMappedDataArray)
SVTK_WRAP_CLASS(svtkScaledSOADataArrayTemplate) */
/*SVTK_WRAP_CLASS(svtkSortDataArray)
SVTK_WRAP_CLASS(svtkSparseArray)
SVTK_WRAP_CLASS(svtkStringArray)
SVTK_WRAP_CLASS(svtkTestDataArray) */
/*SVTK_WRAP_CLASS(svtkArray)
SVTK_WRAP_CLASS(svtkTypedArray)
SVTK_WRAP_CLASS(svtkUnicodeStringArray)
SVTK_WRAP_CLASS(svtkVariantArray)
SVTK_WRAP_CLASS(svtkVoidArray)*/

%ignore svtkTypedDataArrayIterator::operator=;
%ignore svtkTypedDataArrayIterator::operator++;
%ignore svtkTypedDataArrayIterator::operator--;
%ignore svtkTypedDataArrayIterator::operator[];
%extend svtkTypedDataArrayIterator
{
    T __getitem__(size_t n)
    {
        return self->Data->GetValueReference(this->Index + n);
    }

    void __setitem__(size_t n, const T &val)
    {
        self->Data->GetValueReference(this->Index + n) = val;
    }

    bool __eq__(self, const svtkTypedDataArrayIterator<T> &other)
    {
        return *self == other;
    }

    svtkTypedDataArrayIterator<T> & __add__(svtkTypedDataArrayIterator<T> &other)
    {
        return *self + other;
    }

    void __iadd__(self, size_t n)
    {
        *self += n;
    }

    svtkTypedDataArrayIterator<T> & __sub__(svtkTypedDataArrayIterator<T> &other)
    {
        return *self - other;
    }

    void __isub__(self, size_t n)
    {
        *self -= n;
    }
}
SVTK_WRAP_CLASS(svtkTypedDataArrayIterator)

/****************************************************************************
 * Data Sets
 ***************************************************************************/
SVTK_WRAP_CLASS(svtkPoints)

%ignore svtkCellArray::Storage;
SVTK_WRAP_CLASS(svtkCellArray)

%ignore svtkFieldData::BasicIterator;
%ignore svtkFieldData::BasicIterator::operator=;
%ignore svtkFieldData::Iterator;
%ignore svtkFieldData::Iterator::operator=;
%ignore svtkFieldData::CopyFieldFlag;
SVTK_WRAP_CLASS(svtkFieldData)

SVTK_WRAP_CLASS(svtkCellTypes)
SVTK_WRAP_CLASS(svtkDataSetAttributes)
SVTK_WRAP_CLASS(svtkCellData)
SVTK_WRAP_CLASS(svtkPointData)

SVTK_WRAP_CLASS(svtkDataObject)
SVTK_WRAP_CLASS(svtkDataSet)
SVTK_WRAP_CLASS(svtkTable)
SVTK_WRAP_CLASS(svtkPointSet)
SVTK_WRAP_CLASS(svtkImageData)
SVTK_WRAP_CLASS(svtkUniformGrid)
SVTK_WRAP_CLASS(svtkRectilinearGrid)
SVTK_WRAP_CLASS(svtkStructuredGrid)
SVTK_WRAP_CLASS(svtkPolyData)
SVTK_WRAP_CLASS(svtkUnstructuredGridBase)
SVTK_WRAP_CLASS(svtkUnstructuredGrid)

SVTK_WRAP_CLASS(svtkCompositeDataIterator)
SVTK_WRAP_CLASS(svtkDataObjectTreeIterator)
SVTK_WRAP_CLASS(svtkUniformGridAMRDataIterator)

%newobject svtkCompositeDataSet::NewIterator;
SVTK_WRAP_CLASS(svtkCompositeDataSet)

%newobject svtkDataObjectTree::NewIterator;
%newobject svtkDataObjectTree::NewTreeIterator;
SVTK_WRAP_CLASS(svtkDataObjectTree)

SVTK_WRAP_CLASS(svtkMultiBlockDataSet)

%newobject svtkUniformGridAMR::NewIterator;
SVTK_WRAP_CLASS(svtkUniformGridAMR)

%newobject svtkOverlappingAMR::NewIterator;
SVTK_WRAP_CLASS(svtkOverlappingAMR)

%newobject svtkNonOverlappingAMR::NewIterator;
SVTK_WRAP_CLASS(svtkNonOverlappingAMR)


/** EXTRAS */
/* SVTK_WRAP_CLASS(svtkGraph)
SVTK_WRAP_CLASS(svtkMolecule)
SVTK_WRAP_CLASS(svtkStructuredData)
SVTK_WRAP_CLASS(svtkGenericDataSet)
SVTK_WRAP_CLASS(svtkGenericAdaptorCell)
SVTK_WRAP_CLASS(svtkGenericAttribute)
SVTK_WRAP_CLASS(svtkBezierWedge)
SVTK_WRAP_CLASS(svtkReebGraphSimplificationMetric)
SVTK_WRAP_CLASS(svtkHyperTreeGrid)
SVTK_WRAP_CLASS(svtkSortFieldData)
SVTK_WRAP_CLASS(svtkCellIterator)
SVTK_WRAP_CLASS(svtkCellLinks)
SVTK_WRAP_CLASS(svtkAbstractCellLinks)
SVTK_WRAP_CLASS(svtkPartitionedDataSet)
SVTK_WRAP_CLASS(svtkDataSetCellIterator)
SVTK_WRAP_CLASS(svtkLocator)
SVTK_WRAP_CLASS(svtkDataSetAttributesFieldList)
SVTK_WRAP_CLASS(svtkStaticCellLinks)
SVTK_WRAP_CLASS(svtkSelectionNode)
SVTK_WRAP_CLASS(svtkPolyDataCollection)
SVTK_WRAP_CLASS(svtkIncrementalOctreePointLocator)
SVTK_WRAP_CLASS(svtkDataSetCollection)
SVTK_WRAP_CLASS(svtkMappedUnstructuredGrid)
SVTK_WRAP_CLASS(svtkGenericInterpolatedVelocityField)
SVTK_WRAP_CLASS(svtkImplicitDataSet)
SVTK_WRAP_CLASS(svtkBezierQuadrilateral)
SVTK_WRAP_CLASS(svtkStaticCellLinksTemplate)
SVTK_WRAP_CLASS(svtkArrayListTemplate)
SVTK_WRAP_CLASS(svtkBezierTriangle)
SVTK_WRAP_CLASS(svtkBezierHexahedron)
SVTK_WRAP_CLASS(svtkGenericCell)
SVTK_WRAP_CLASS(svtkIterativeClosestPointTransform)
SVTK_WRAP_CLASS(svtkFindCellStrategy)
SVTK_WRAP_CLASS(svtkCellType)
SVTK_WRAP_CLASS(svtkPath)
SVTK_WRAP_CLASS(svtkKdTree)
SVTK_WRAP_CLASS(svtkReebGraph)
SVTK_WRAP_CLASS(svtkExplicitStructuredGrid)
SVTK_WRAP_CLASS(svtkBezierTetra)
SVTK_WRAP_CLASS(svtkBezierCurve)
SVTK_WRAP_CLASS(svtkMultiPieceDataSet)
SVTK_WRAP_CLASS(svtkOctreePointLocator)
SVTK_WRAP_CLASS(svtkHierarchicalBoxDataIterator) */

/* TODO -- hack for sharing data with Numpy */
%inline
%{
void *as_void_ptr(size_t pval)
{
  return (void*) pval;
}

size_t as_integer(void *ptr)
{
  return size_t(ptr);
}
%}
/* end of hack */
