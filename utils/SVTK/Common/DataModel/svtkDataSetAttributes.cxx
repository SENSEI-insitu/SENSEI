/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataSetAttributes.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDataSetAttributes.h"

#include "svtkArrayDispatch.h"
#include "svtkArrayIteratorIncludes.h"
#include "svtkAssume.h"
#include "svtkCell.h"
#include "svtkCharArray.h"
#include "svtkDataArrayRange.h"
#include "svtkDoubleArray.h"
#include "svtkFloatArray.h"
#include "svtkIdTypeArray.h"
#include "svtkInformation.h"
#include "svtkIntArray.h"
#include "svtkLongArray.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkShortArray.h"
#include "svtkStructuredExtent.h"
#include "svtkUnsignedCharArray.h"
#include "svtkUnsignedIntArray.h"
#include "svtkUnsignedLongArray.h"
#include "svtkUnsignedShortArray.h"
#include <vector>

svtkStandardNewMacro(svtkDataSetAttributes);
//--------------------------------------------------------------------------
const char svtkDataSetAttributes::AttributeNames[svtkDataSetAttributes::NUM_ATTRIBUTES][19] = {
  "Scalars",
  "Vectors",
  "Normals",
  "TCoords",
  "Tensors",
  "GlobalIds",
  "PedigreeIds",
  "EdgeFlag",
  "Tangents",
  "RationalWeights",
  "HigherOrderDegrees",
};

const char svtkDataSetAttributes::LongAttributeNames[svtkDataSetAttributes::NUM_ATTRIBUTES][42] = {
  "svtkDataSetAttributes::SCALARS",
  "svtkDataSetAttributes::VECTORS",
  "svtkDataSetAttributes::NORMALS",
  "svtkDataSetAttributes::TCOORDS",
  "svtkDataSetAttributes::TENSORS",
  "svtkDataSetAttributes::GLOBALIDS",
  "svtkDataSetAttributes::PEDIGREEIDS",
  "svtkDataSetAttributes::EDGEFLAG",
  "svtkDataSetAttributes::TANGENTS",
  "svtkDataSetAttributes::RATIONALWEIGHTS",
  "svtkDataSetAttributes::HIGHERORDERDEGREES",
};

//--------------------------------------------------------------------------
// Construct object with copying turned on for all data.
svtkDataSetAttributes::svtkDataSetAttributes()
{
  int attributeType;
  for (attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
  {
    this->AttributeIndices[attributeType] = -1;
    this->CopyAttributeFlags[COPYTUPLE][attributeType] = 1;
    this->CopyAttributeFlags[INTERPOLATE][attributeType] = 1;
    this->CopyAttributeFlags[PASSDATA][attributeType] = 1;
  }

  // Global IDs should not be interpolated because they are labels, not "numbers"
  // Global IDs should not be copied either, unless doing so preserves meaning.
  // Passing through is usually OK because it is 1:1.
  this->CopyAttributeFlags[COPYTUPLE][GLOBALIDS] = 0;
  this->CopyAttributeFlags[INTERPOLATE][GLOBALIDS] = 0;

  // Pedigree IDs should not be interpolated because they are labels, not "numbers"
  // Pedigree IDs may be copied since they do not require 1:1 mapping.
  this->CopyAttributeFlags[INTERPOLATE][PEDIGREEIDS] = 0;

  this->TargetIndices = nullptr;
}

//--------------------------------------------------------------------------
// Destructor for the svtkDataSetAttributes objects.
svtkDataSetAttributes::~svtkDataSetAttributes()
{
  this->Initialize();
  delete[] this->TargetIndices;
  this->TargetIndices = nullptr;
}

//--------------------------------------------------------------------------
// Turn on copying of all data.
void svtkDataSetAttributes::CopyAllOn(int ctype)
{
  this->svtkFieldData::CopyAllOn();
  this->SetCopyScalars(1, ctype);
  this->SetCopyVectors(1, ctype);
  this->SetCopyNormals(1, ctype);
  this->SetCopyTCoords(1, ctype);
  this->SetCopyTensors(1, ctype);
  this->SetCopyGlobalIds(1, ctype);
  this->SetCopyPedigreeIds(1, ctype);
  this->SetCopyTangents(1, ctype);
  this->SetCopyRationalWeights(1, ctype);
  this->SetCopyHigherOrderDegrees(1, ctype);
}

//--------------------------------------------------------------------------
// Turn off copying of all data.
void svtkDataSetAttributes::CopyAllOff(int ctype)
{
  this->svtkFieldData::CopyAllOff();
  this->SetCopyScalars(0, ctype);
  this->SetCopyVectors(0, ctype);
  this->SetCopyNormals(0, ctype);
  this->SetCopyTCoords(0, ctype);
  this->SetCopyTensors(0, ctype);
  this->SetCopyGlobalIds(0, ctype);
  this->SetCopyPedigreeIds(0, ctype);
  this->SetCopyTangents(0, ctype);
  this->SetCopyRationalWeights(0, ctype);
  this->SetCopyHigherOrderDegrees(0, ctype);
}

//--------------------------------------------------------------------------
// Deep copy of data (i.e., create new data arrays and
// copy from input data). Note that attribute data is
// not copied.
void svtkDataSetAttributes::DeepCopy(svtkFieldData* fd)
{
  this->Initialize(); // free up memory

  svtkDataSetAttributes* dsa = svtkDataSetAttributes::SafeDownCast(fd);
  // If the source is a svtkDataSetAttributes
  if (dsa)
  {
    int numArrays = fd->GetNumberOfArrays();
    int attributeType, i;
    svtkAbstractArray *data, *newData;

    // Allocate space for numArrays
    this->AllocateArrays(numArrays);
    for (i = 0; i < numArrays; i++)
    {
      data = fd->GetAbstractArray(i);
      newData = data->NewInstance(); // instantiate same type of object
      newData->DeepCopy(data);
      newData->SetName(data->GetName());
      this->AddArray(newData);
      newData->Delete();
    }
    // Copy the copy flags
    for (attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
    {
      // If an array is an attribute in the source, then mark it as a attribute
      // in the clone as well.
      this->AttributeIndices[attributeType] = dsa->AttributeIndices[attributeType];

      this->CopyAttributeFlags[COPYTUPLE][attributeType] =
        dsa->CopyAttributeFlags[COPYTUPLE][attributeType];
      this->CopyAttributeFlags[INTERPOLATE][attributeType] =
        dsa->CopyAttributeFlags[INTERPOLATE][attributeType];
      this->CopyAttributeFlags[PASSDATA][attributeType] =
        dsa->CopyAttributeFlags[PASSDATA][attributeType];
    }
    this->CopyFlags(dsa);
  }
  // If the source is field data, do a field data copy
  else
  {
    this->svtkFieldData::DeepCopy(fd);
  }
}

//--------------------------------------------------------------------------
// Shallow copy of data (i.e., use reference counting).
void svtkDataSetAttributes::ShallowCopy(svtkFieldData* fd)
{
  this->Initialize(); // free up memory

  svtkDataSetAttributes* dsa = svtkDataSetAttributes::SafeDownCast(fd);
  // If the source is a svtkDataSetAttributes
  if (dsa)
  {
    int numArrays = fd->GetNumberOfArrays();
    int attributeType, i;

    // Allocate space for numArrays
    this->AllocateArrays(numArrays);
    this->NumberOfActiveArrays = 0;
    for (i = 0; i < numArrays; i++)
    {
      this->NumberOfActiveArrays++;
      this->SetArray(i, fd->GetAbstractArray(i));
    }

    // Copy the copy flags
    for (attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
    {
      // If an array is an attribute in the source, then mark it as a attribute
      // in the clone as well.
      this->AttributeIndices[attributeType] = dsa->AttributeIndices[attributeType];

      this->CopyAttributeFlags[COPYTUPLE][attributeType] =
        dsa->CopyAttributeFlags[COPYTUPLE][attributeType];
      this->CopyAttributeFlags[INTERPOLATE][attributeType] =
        dsa->CopyAttributeFlags[INTERPOLATE][attributeType];
      this->CopyAttributeFlags[PASSDATA][attributeType] =
        dsa->CopyAttributeFlags[PASSDATA][attributeType];
    }
    this->CopyFlags(dsa);
  }
  // If the source is field data, do a field data copy
  else
  {
    this->svtkFieldData::ShallowCopy(fd);
  }
}

//--------------------------------------------------------------------------
// Initialize all of the object's data to nullptr
void svtkDataSetAttributes::InitializeFields()
{
  this->svtkFieldData::InitializeFields();

  int attributeType;
  for (attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
  {
    this->AttributeIndices[attributeType] = -1;
    this->CopyAttributeFlags[COPYTUPLE][attributeType] = 1;
    this->CopyAttributeFlags[INTERPOLATE][attributeType] = 1;
    this->CopyAttributeFlags[PASSDATA][attributeType] = 1;
  }
  this->CopyAttributeFlags[COPYTUPLE][GLOBALIDS] = 0;
  this->CopyAttributeFlags[INTERPOLATE][GLOBALIDS] = 0;

  this->CopyAttributeFlags[INTERPOLATE][PEDIGREEIDS] = 0;
}

//--------------------------------------------------------------------------
// Initialize all of the object's data to nullptr
void svtkDataSetAttributes::Initialize()
{
  //
  // We don't modify ourselves because the "ReleaseData" methods depend upon
  // no modification when initialized.
  //

  // Call superclass' Initialize()
  this->svtkFieldData::Initialize();
  //
  // Free up any memory
  // And don't forget to reset the attribute copy flags.
  int attributeType;
  for (attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
  {
    this->AttributeIndices[attributeType] = -1;
    this->CopyAttributeFlags[COPYTUPLE][attributeType] = 1;
    this->CopyAttributeFlags[INTERPOLATE][attributeType] = 1;
    this->CopyAttributeFlags[PASSDATA][attributeType] = 1;
  }
  this->CopyAttributeFlags[COPYTUPLE][GLOBALIDS] = 0;
  this->CopyAttributeFlags[INTERPOLATE][GLOBALIDS] = 0;

  this->CopyAttributeFlags[INTERPOLATE][PEDIGREEIDS] = 0;
}

//--------------------------------------------------------------------------
// This method is used to determine which arrays
// will be copied to this object
svtkFieldData::BasicIterator svtkDataSetAttributes::ComputeRequiredArrays(
  svtkDataSetAttributes* pd, int ctype)
{
  if ((ctype < COPYTUPLE) || (ctype > PASSDATA))
  {
    svtkErrorMacro("Must call compute required with COPYTUPLE, INTERPOLATE or PASSDATA");
    ctype = COPYTUPLE;
  }

  // We need to do some juggling to find the number of arrays
  // which will be passed.

  // First, find the number of arrays to be copied because they
  // are in the list of _fields_ to be copied (and the actual data
  // pointer is non-nullptr). Also, we keep those indices in a list.
  int* copyFlags = new int[pd->GetNumberOfArrays()];
  int index, i, numArrays = 0;
  for (i = 0; i < pd->GetNumberOfArrays(); i++)
  {
    const char* arrayName = pd->GetArrayName(i);
    // If there is no blocker for the given array
    // and both CopyAllOff and CopyOn for that array are not true
    if ((this->GetFlag(arrayName) != 0) &&
      !(this->DoCopyAllOff && (this->GetFlag(arrayName) != 1)) && pd->GetAbstractArray(i))
    {
      // Cannot interpolate idtype arrays
      if (ctype != INTERPOLATE || pd->GetAbstractArray(i)->GetDataType() != SVTK_ID_TYPE)
      {
        copyFlags[numArrays] = i;
        numArrays++;
      }
    }
  }

  // Next, we check the arrays to be copied because they are one of
  // the _attributes_ to be copied (and the data array in non-nullptr).
  // We make sure that we don't count anything twice.
  int alreadyCopied;
  int attributeType, j;
  for (attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
  {
    index = pd->AttributeIndices[attributeType];
    int flag = this->GetFlag(pd->GetArrayName(index));
    // If this attribute is to be copied
    if (this->CopyAttributeFlags[ctype][attributeType] && flag)
    {
      // Find out if it is also in the list of fields to be copied
      // Since attributes can only be svtkDataArray, we use GetArray() call.
      if (pd->GetArray(index))
      {
        alreadyCopied = 0;
        for (i = 0; i < numArrays; i++)
        {
          if (index == copyFlags[i])
          {
            alreadyCopied = 1;
          }
        }
        // If not, increment the number of arrays to be copied.
        if (!alreadyCopied)
        {
          // Cannot interpolate idtype arrays
          if (ctype != INTERPOLATE || pd->GetArray(index)->GetDataType() != SVTK_ID_TYPE)
          {
            copyFlags[numArrays] = index;
            numArrays++;
          }
        }
      }
    }
    // If it is not to be copied and it is in the list (from the
    // previous pass), remove it
    else
    {
      for (i = 0; i < numArrays; i++)
      {
        if (index == copyFlags[i])
        {
          for (j = i; j < numArrays - 1; j++)
          {
            copyFlags[j] = copyFlags[j + 1];
          }
          numArrays--;
          i--;
        }
      }
    }
  }

  svtkFieldData::BasicIterator it(copyFlags, numArrays);
  delete[] copyFlags;
  return it;
}

//--------------------------------------------------------------------------
// Pass entire arrays of input data through to output. Obey the "copy"
// flags.
void svtkDataSetAttributes::PassData(svtkFieldData* fd)
{
  if (!fd)
  {
    return;
  }

  svtkDataSetAttributes* dsa = svtkDataSetAttributes::SafeDownCast(fd);

  if (dsa)
  {
    // Create an iterator to iterate over the fields which will
    // be passed, i.e. fields which are either:
    // 1> in the list of _fields_ to be copied or
    // 2> in the list of _attributes_ to be copied.
    // Note that nullptr data arrays are not copied

    svtkFieldData::BasicIterator it = this->ComputeRequiredArrays(dsa, PASSDATA);

    if (it.GetListSize() > this->NumberOfArrays)
    {
      this->AllocateArrays(it.GetListSize());
    }
    if (it.GetListSize() == 0)
    {
      return;
    }

    // Since we are replacing, remove old attributes
    int attributeType; // will change//
    for (attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
    {
      if (this->CopyAttributeFlags[PASSDATA][attributeType])
      {
        this->RemoveArray(this->AttributeIndices[attributeType]);
        this->AttributeIndices[attributeType] = -1;
      }
    }

    int i, arrayIndex;
    for (i = it.BeginIndex(); !it.End(); i = it.NextIndex())
    {
      arrayIndex = this->AddArray(dsa->GetAbstractArray(i));
      // If necessary, make the array an attribute
      if (((attributeType = dsa->IsArrayAnAttribute(i)) != -1) &&
        this->CopyAttributeFlags[PASSDATA][attributeType])
      {
        this->SetActiveAttribute(arrayIndex, attributeType);
      }
    }
  }
  else
  {
    this->svtkFieldData::PassData(fd);
  }
}

//----------------------------------------------------------------------------
namespace
{
struct CopyStructuredDataWorker
{
  const int* OutExt;
  const int* InExt;

  CopyStructuredDataWorker(const int* outExt, const int* inExt)
    : OutExt(outExt)
    , InExt(inExt)
  {
  }

  template <typename Array1T, typename Array2T>
  void operator()(Array1T* dstArray, Array2T* srcArray)
  {
    // Give the compiler a hand -- allow optimizations that require both arrays
    // to have the same stride.
    SVTK_ASSUME(srcArray->GetNumberOfComponents() == dstArray->GetNumberOfComponents());

    // Create some tuple ranges to simplify optimized copying:
    const auto srcTuples = svtk::DataArrayTupleRange(srcArray);
    auto dstTuples = svtk::DataArrayTupleRange(dstArray);

    if (svtkStructuredExtent::Smaller(this->OutExt, this->InExt))
    {
      // get outExt relative to the inExt to keep the logic simple. This assumes
      // that outExt is a subset of the inExt.
      const int relOutExt[6] = {
        this->OutExt[0] - this->InExt[0],
        this->OutExt[1] - this->InExt[0],
        this->OutExt[2] - this->InExt[2],
        this->OutExt[3] - this->InExt[2],
        this->OutExt[4] - this->InExt[4],
        this->OutExt[5] - this->InExt[4],
      };

      const int dims[3] = {
        this->InExt[1] - this->InExt[0] + 1,
        this->InExt[3] - this->InExt[2] + 1,
        this->InExt[5] - this->InExt[4] + 1,
      };

      auto dstTupleIter = dstTuples.begin();
      for (int outz = relOutExt[4]; outz <= relOutExt[5]; ++outz)
      {
        const svtkIdType zfactor = static_cast<svtkIdType>(outz) * dims[1];
        for (int outy = relOutExt[2]; outy <= relOutExt[3]; ++outy)
        {
          const svtkIdType yfactor = (zfactor + outy) * dims[0];
          for (int outx = relOutExt[0]; outx <= relOutExt[1]; ++outx)
          {
            const svtkIdType inTupleIdx = yfactor + outx;
            *dstTupleIter++ = srcTuples[inTupleIdx];
          }
        }
      }
    }
    else
    {
      int writeExt[6];
      memcpy(writeExt, this->OutExt, 6 * sizeof(int));
      svtkStructuredExtent::Clamp(writeExt, this->InExt);

      const svtkIdType inDims[3] = { this->InExt[1] - this->InExt[0] + 1,
        this->InExt[3] - this->InExt[2] + 1, this->InExt[5] - this->InExt[4] + 1 };
      const svtkIdType outDims[3] = { this->OutExt[1] - this->OutExt[0] + 1,
        this->OutExt[3] - this->OutExt[2] + 1, this->OutExt[5] - this->OutExt[4] + 1 };

      for (int idz = writeExt[4]; idz <= writeExt[5]; ++idz)
      {
        const svtkIdType inTupleId1 = (idz - this->InExt[4]) * inDims[0] * inDims[1];
        const svtkIdType outTupleId1 = (idz - this->OutExt[4]) * outDims[0] * outDims[1];
        for (int idy = writeExt[2]; idy <= writeExt[3]; ++idy)
        {
          const svtkIdType inTupleId2 = inTupleId1 + (idy - this->InExt[2]) * inDims[0];
          const svtkIdType outTupleId2 = outTupleId1 + (idy - this->OutExt[2]) * outDims[0];
          for (int idx = writeExt[0]; idx <= writeExt[1]; ++idx)
          {
            const svtkIdType inTupleIdx = inTupleId2 + idx - this->InExt[0];
            const svtkIdType outTupleIdx = outTupleId2 + idx - this->OutExt[0];

            dstTuples[outTupleIdx] = srcTuples[inTupleIdx];
          }
        }
      }
    }

    dstArray->DataChanged();
  }
};

//----------------------------------------------------------------------------
// Handle svtkAbstractArrays that aren't svtkDataArrays.
template <class iterT>
void svtkDataSetAttributesCopyValues(iterT* destIter, const int* outExt, svtkIdType outIncs[3],
  iterT* srcIter, const int* inExt, svtkIdType inIncs[3])
{
  int data_type_size = srcIter->GetArray()->GetDataTypeSize();
  svtkIdType rowLength = outIncs[1];
  unsigned char* inPtr;
  unsigned char* outPtr;
  unsigned char* inZPtr;
  unsigned char* outZPtr;

  // Get the starting input pointer.
  inZPtr = static_cast<unsigned char*>(srcIter->GetArray()->GetVoidPointer(0));
  // Shift to the start of the subextent.
  inZPtr += (outExt[0] - inExt[0]) * inIncs[0] * data_type_size +
    (outExt[2] - inExt[2]) * inIncs[1] * data_type_size +
    (outExt[4] - inExt[4]) * inIncs[2] * data_type_size;

  // Get output pointer.
  outZPtr = static_cast<unsigned char*>(destIter->GetArray()->GetVoidPointer(0));

  // Loop over z axis.
  for (int zIdx = outExt[4]; zIdx <= outExt[5]; ++zIdx)
  {
    inPtr = inZPtr;
    outPtr = outZPtr;
    for (int yIdx = outExt[2]; yIdx <= outExt[3]; ++yIdx)
    {
      memcpy(outPtr, inPtr, rowLength * data_type_size);
      inPtr += inIncs[1] * data_type_size;
      outPtr += outIncs[1] * data_type_size;
    }
    inZPtr += inIncs[2] * data_type_size;
    outZPtr += outIncs[2] * data_type_size;
  }
}

//----------------------------------------------------------------------------
// Specialize for svtkStringArray.
template <>
void svtkDataSetAttributesCopyValues(svtkArrayIteratorTemplate<svtkStdString>* destIter,
  const int* outExt, svtkIdType outIncs[3], svtkArrayIteratorTemplate<svtkStdString>* srcIter,
  const int* inExt, svtkIdType inIncs[3])
{
  svtkIdType inZIndex = (outExt[0] - inExt[0]) * inIncs[0] + (outExt[2] - inExt[2]) * inIncs[1] +
    (outExt[4] - inExt[4]) * inIncs[2];

  svtkIdType outZIndex = 0;
  svtkIdType rowLength = outIncs[1];

  for (int zIdx = outExt[4]; zIdx <= outExt[5]; ++zIdx)
  {
    svtkIdType inIndex = inZIndex;
    svtkIdType outIndex = outZIndex;
    for (int yIdx = outExt[2]; yIdx <= outExt[3]; ++yIdx)
    {
      for (int xIdx = 0; xIdx < rowLength; ++xIdx)
      {
        destIter->GetValue(outIndex + xIdx) = srcIter->GetValue(inIndex + xIdx);
      }
      inIndex += inIncs[1];
      outIndex += outIncs[1];
    }
    inZIndex += inIncs[2];
    outZIndex += outIncs[2];
  }
}

} // end anon namespace

//----------------------------------------------------------------------------
// This is used in the imaging pipeline for copying arrays.
// CopyAllocate needs to be called before this method.
void svtkDataSetAttributes::CopyStructuredData(
  svtkDataSetAttributes* fromPd, const int* inExt, const int* outExt, bool setSize)
{
  int i;

  for (i = this->RequiredArrays.BeginIndex(); !this->RequiredArrays.End();
       i = this->RequiredArrays.NextIndex())
  {
    svtkAbstractArray* inArray = fromPd->Data[i];
    svtkAbstractArray* outArray = this->Data[this->TargetIndices[i]];
    svtkIdType inIncs[3];
    svtkIdType outIncs[3];
    svtkIdType zIdx;

    // Compute increments
    inIncs[0] = inArray->GetNumberOfComponents();
    inIncs[1] = inIncs[0] * (inExt[1] - inExt[0] + 1);
    inIncs[2] = inIncs[1] * (inExt[3] - inExt[2] + 1);
    outIncs[0] = inIncs[0];
    outIncs[1] = outIncs[0] * (outExt[1] - outExt[0] + 1);
    outIncs[2] = outIncs[1] * (outExt[3] - outExt[2] + 1);

    // Make sure the input extents match the actual array lengths.
    zIdx = inIncs[2] / inIncs[0] * (inExt[5] - inExt[4] + 1);
    if (inArray->GetNumberOfTuples() != zIdx)
    {
      svtkErrorMacro("Input extent (" << inExt[0] << ", " << inExt[1] << ", " << inExt[2] << ", "
                                     << inExt[3] << ", " << inExt[4] << ", " << inExt[5]
                                     << ") does not match array length: " << zIdx);
      // Skip copying this array.
      continue;
    }
    // Make sure the output extents match the actual array lengths.
    zIdx = outIncs[2] / outIncs[0] * (outExt[5] - outExt[4] + 1);
    if (outArray->GetNumberOfTuples() != zIdx && setSize)
    {
      // The "CopyAllocate" method only sets the size, not the number of tuples.
      outArray->SetNumberOfTuples(zIdx);
    }

    // We get very little performance improvement from this, but we'll leave the
    // legacy code around until we've done through benchmarking.
    svtkDataArray* inDA = svtkArrayDownCast<svtkDataArray>(inArray);
    svtkDataArray* outDA = svtkArrayDownCast<svtkDataArray>(outArray);
    if (!inDA || !outDA) // String array, etc
    {
      svtkArrayIterator* srcIter = inArray->NewIterator();
      svtkArrayIterator* destIter = outArray->NewIterator();
      switch (inArray->GetDataType())
      {
        svtkArrayIteratorTemplateMacro(svtkDataSetAttributesCopyValues(static_cast<SVTK_TT*>(destIter),
          outExt, outIncs, static_cast<SVTK_TT*>(srcIter), inExt, inIncs));
      }
      srcIter->Delete();
      destIter->Delete();
    }
    else
    {
      CopyStructuredDataWorker worker(outExt, inExt);
      if (!svtkArrayDispatch::Dispatch2SameValueType::Execute(outDA, inDA, worker))
      {
        // Fallback to svtkDataArray API (e.g. svtkBitArray):
        worker(outDA, inDA);
      }
    }
  }
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetupForCopy(svtkDataSetAttributes* pd)
{
  this->InternalCopyAllocate(pd, COPYTUPLE, 0, 0, false, false);
}

//--------------------------------------------------------------------------
// Allocates point data for point-by-point (or cell-by-cell) copy operation.
// If sze=0, then use the input DataSetAttributes to create (i.e., find
// initial size of) new objects; otherwise use the sze variable.
void svtkDataSetAttributes::InternalCopyAllocate(svtkDataSetAttributes* pd, int ctype, svtkIdType sze,
  svtkIdType ext, int shallowCopyArrays, bool createNewArrays)
{
  svtkAbstractArray* newAA;
  int i;

  // Create various point data depending upon input
  //
  if (!pd)
  {
    return;
  }

  if ((ctype < COPYTUPLE) || (ctype > PASSDATA))
  {
    return;
  }

  this->RequiredArrays = this->ComputeRequiredArrays(pd, ctype);
  if (this->RequiredArrays.GetListSize() == 0)
  {
    return;
  }
  delete[] this->TargetIndices;
  this->TargetIndices = new int[pd->GetNumberOfArrays()];
  for (i = 0; i < pd->GetNumberOfArrays(); i++)
  {
    this->TargetIndices[i] = -1;
  }

  svtkAbstractArray* aa = nullptr;
  // If we are not copying on self
  if ((pd != this) && createNewArrays)
  {
    int attributeType;

    for (i = this->RequiredArrays.BeginIndex(); !this->RequiredArrays.End();
         i = this->RequiredArrays.NextIndex())
    {
      // Create all required arrays
      aa = pd->GetAbstractArray(i);
      if (shallowCopyArrays)
      {
        newAA = aa;
      }
      else
      {
        newAA = aa->NewInstance();
        newAA->SetNumberOfComponents(aa->GetNumberOfComponents());
        newAA->CopyComponentNames(aa);
        newAA->SetName(aa->GetName());
        if (aa->HasInformation())
        {
          newAA->CopyInformation(aa->GetInformation(), /*deep=*/1);
        }
        if (sze > 0)
        {
          newAA->Allocate(sze * aa->GetNumberOfComponents(), ext);
        }
        else
        {
          newAA->Allocate(aa->GetNumberOfTuples());
        }
        svtkDataArray* newDA = svtkArrayDownCast<svtkDataArray>(newAA);
        if (newDA)
        {
          svtkDataArray* da = svtkArrayDownCast<svtkDataArray>(aa);
          newDA->SetLookupTable(da->GetLookupTable());
        }
      }
      this->TargetIndices[i] = this->AddArray(newAA);
      // If necessary, make the array an attribute
      if (((attributeType = pd->IsArrayAnAttribute(i)) != -1) &&
        this->CopyAttributeFlags[ctype][attributeType])
      {
        this->CopyAttributeFlags[ctype][attributeType] =
          pd->CopyAttributeFlags[ctype][attributeType];
        this->SetActiveAttribute(this->TargetIndices[i], attributeType);
      }
      if (!shallowCopyArrays)
      {
        newAA->Delete();
      }
    }
  }
  else if (pd == this)
  {
    // If copying on self, resize the arrays and initialize
    // TargetIndices
    for (i = this->RequiredArrays.BeginIndex(); !this->RequiredArrays.End();
         i = this->RequiredArrays.NextIndex())
    {
      aa = pd->GetAbstractArray(i);
      aa->Resize(sze);
      this->TargetIndices[i] = i;
    }
  }
  else
  {
    // All we are asked to do is create a mapping.
    // Here we assume that arrays are the same and ordered
    // the same way.
    for (i = this->RequiredArrays.BeginIndex(); !this->RequiredArrays.End();
         i = this->RequiredArrays.NextIndex())
    {
      this->TargetIndices[i] = i;
    }
  }
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::RemoveArray(int index)
{
  if ((index < 0) || (index >= this->NumberOfActiveArrays))
  {
    return;
  }
  this->Superclass::RemoveArray(index);
  int attributeType;
  for (attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
  {
    if (this->AttributeIndices[attributeType] == index)
    {
      this->AttributeIndices[attributeType] = -1;
    }
    else if (this->AttributeIndices[attributeType] > index)
    {
      this->AttributeIndices[attributeType]--;
    }
  }
}

//--------------------------------------------------------------------------
// Copy the attribute data from one id to another. Make sure CopyAllocate() has
// been invoked before using this method.
void svtkDataSetAttributes::CopyData(svtkDataSetAttributes* fromPd, svtkIdType fromId, svtkIdType toId)
{
  int i;
  for (i = this->RequiredArrays.BeginIndex(); !this->RequiredArrays.End();
       i = this->RequiredArrays.NextIndex())
  {
    this->CopyTuple(fromPd->Data[i], this->Data[this->TargetIndices[i]], fromId, toId);
  }
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::CopyData(
  svtkDataSetAttributes* fromPd, svtkIdList* fromIds, svtkIdList* toIds)
{
  int i;
  for (i = this->RequiredArrays.BeginIndex(); !this->RequiredArrays.End();
       i = this->RequiredArrays.NextIndex())
  {
    this->CopyTuples(fromPd->Data[i], this->Data[this->TargetIndices[i]], fromIds, toIds);
  }
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::CopyData(
  svtkDataSetAttributes* fromPd, svtkIdType dstStart, svtkIdType n, svtkIdType srcStart)
{
  for (int i = this->RequiredArrays.BeginIndex(); !this->RequiredArrays.End();
       i = this->RequiredArrays.NextIndex())
  {
    this->CopyTuples(fromPd->Data[i], this->Data[this->TargetIndices[i]], dstStart, n, srcStart);
  }
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::CopyAllocate(
  svtkDataSetAttributes* pd, svtkIdType sze, svtkIdType ext, int shallowCopyArrays)
{
  this->InternalCopyAllocate(pd, COPYTUPLE, sze, ext, shallowCopyArrays);
}

// Initialize point interpolation method.
void svtkDataSetAttributes::InterpolateAllocate(
  svtkDataSetAttributes* pd, svtkIdType sze, svtkIdType ext, int shallowCopyArrays)
{
  this->InternalCopyAllocate(pd, INTERPOLATE, sze, ext, shallowCopyArrays);
}

//--------------------------------------------------------------------------
// Interpolate data from points and interpolation weights. Make sure that the
// method InterpolateAllocate() has been invoked before using this method.
void svtkDataSetAttributes::InterpolatePoint(
  svtkDataSetAttributes* fromPd, svtkIdType toId, svtkIdList* ptIds, double* weights)
{
  for (int i = this->RequiredArrays.BeginIndex(); !this->RequiredArrays.End();
       i = this->RequiredArrays.NextIndex())
  {
    svtkAbstractArray* fromArray = fromPd->Data[i];
    svtkAbstractArray* toArray = this->Data[this->TargetIndices[i]];

    // check if the destination array needs nearest neighbor interpolation
    int attributeIndex = this->IsArrayAnAttribute(this->TargetIndices[i]);
    if (attributeIndex != -1 && this->CopyAttributeFlags[INTERPOLATE][attributeIndex] == 2)
    {
      svtkIdType numIds = ptIds->GetNumberOfIds();
      svtkIdType maxId = ptIds->GetId(0);
      svtkIdType maxWeight = 0.;
      for (int j = 0; j < numIds; j++)
      {
        if (weights[j] > maxWeight)
        {
          maxWeight = weights[j];
          maxId = ptIds->GetId(j);
        }
      }
      toArray->InsertTuple(toId, maxId, fromArray);
    }
    else
    {
      toArray->InterpolateTuple(toId, ptIds, fromArray, weights);
    }
  }
}

//--------------------------------------------------------------------------
// Interpolate data from the two points p1,p2 (forming an edge) and an
// interpolation factor, t, along the edge. The weight ranges from (0,1),
// with t=0 located at p1. Make sure that the method InterpolateAllocate()
// has been invoked before using this method.
void svtkDataSetAttributes::InterpolateEdge(
  svtkDataSetAttributes* fromPd, svtkIdType toId, svtkIdType p1, svtkIdType p2, double t)
{
  for (int i = this->RequiredArrays.BeginIndex(); !this->RequiredArrays.End();
       i = this->RequiredArrays.NextIndex())
  {
    svtkAbstractArray* fromArray = fromPd->Data[i];
    svtkAbstractArray* toArray = this->Data[this->TargetIndices[i]];

    // check if the destination array needs nearest neighbor interpolation
    int attributeIndex = this->IsArrayAnAttribute(this->TargetIndices[i]);
    if (attributeIndex != -1 && this->CopyAttributeFlags[INTERPOLATE][attributeIndex] == 2)
    {
      if (t < .5)
      {
        toArray->InsertTuple(toId, p1, fromArray);
      }
      else
      {
        toArray->InsertTuple(toId, p2, fromArray);
      }
    }
    else
    {
      toArray->InterpolateTuple(toId, p1, fromArray, p2, fromArray, t);
    }
  }
}

//--------------------------------------------------------------------------
// Interpolate data from the two points p1,p2 (forming an edge) and an
// interpolation factor, t, along the edge. The weight ranges from (0,1),
// with t=0 located at p1. Make sure that the method InterpolateAllocate()
// has been invoked before using this method.
void svtkDataSetAttributes::InterpolateTime(
  svtkDataSetAttributes* from1, svtkDataSetAttributes* from2, svtkIdType id, double t)
{
  for (int attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
  {
    // If this attribute is to be copied
    if (this->CopyAttributeFlags[INTERPOLATE][attributeType])
    {
      if (from1->GetAttribute(attributeType) && from2->GetAttribute(attributeType))
      {
        svtkAbstractArray* toArray = this->GetAttribute(attributeType);
        // check if the destination array needs nearest neighbor interpolation
        if (this->CopyAttributeFlags[INTERPOLATE][attributeType] == 2)
        {
          if (t < .5)
          {
            toArray->InsertTuple(id, id, from1->GetAttribute(attributeType));
          }
          else
          {
            toArray->InsertTuple(id, id, from2->GetAttribute(attributeType));
          }
        }
        else
        {
          toArray->InterpolateTuple(
            id, id, from1->GetAttribute(attributeType), id, from2->GetAttribute(attributeType), t);
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
// Copy a tuple of data from one data array to another. This method (and
// following ones) assume that the fromData and toData objects are of the
// same type, and have the same number of components. This is true if you
// invoke CopyAllocate() or InterpolateAllocate().
void svtkDataSetAttributes::CopyTuple(
  svtkAbstractArray* fromData, svtkAbstractArray* toData, svtkIdType fromId, svtkIdType toId)
{
  toData->InsertTuple(toId, fromId, fromData);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::CopyTuples(
  svtkAbstractArray* fromData, svtkAbstractArray* toData, svtkIdList* fromIds, svtkIdList* toIds)
{
  toData->InsertTuples(toIds, fromIds, fromData);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::CopyTuples(svtkAbstractArray* fromData, svtkAbstractArray* toData,
  svtkIdType dstStart, svtkIdType n, svtkIdType srcStart)
{
  toData->InsertTuples(dstStart, n, srcStart, fromData);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetScalars(svtkDataArray* da)
{
  return this->SetAttribute(da, SCALARS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveScalars(const char* name)
{
  return this->SetActiveAttribute(name, SCALARS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveAttribute(const char* name, int attributeType)
{
  int index;
  this->GetAbstractArray(name, index);
  return this->SetActiveAttribute(index, attributeType);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetScalars()
{
  return this->GetAttribute(SCALARS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetVectors(svtkDataArray* da)
{
  return this->SetAttribute(da, VECTORS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveVectors(const char* name)
{
  return this->SetActiveAttribute(name, VECTORS);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetVectors()
{
  return this->GetAttribute(VECTORS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetNormals(svtkDataArray* da)
{
  return this->SetAttribute(da, NORMALS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveNormals(const char* name)
{
  return this->SetActiveAttribute(name, NORMALS);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetNormals()
{
  return this->GetAttribute(NORMALS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetTangents(svtkDataArray* da)
{
  return this->SetAttribute(da, TANGENTS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveTangents(const char* name)
{
  return this->SetActiveAttribute(name, TANGENTS);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetTangents()
{
  return this->GetAttribute(TANGENTS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetTCoords(svtkDataArray* da)
{
  return this->SetAttribute(da, TCOORDS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveTCoords(const char* name)
{
  return this->SetActiveAttribute(name, TCOORDS);
}
//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetTCoords()
{
  return this->GetAttribute(TCOORDS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetTensors(svtkDataArray* da)
{
  return this->SetAttribute(da, TENSORS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveTensors(const char* name)
{
  return this->SetActiveAttribute(name, TENSORS);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetTensors()
{
  return this->GetAttribute(TENSORS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetGlobalIds(svtkDataArray* da)
{
  return this->SetAttribute(da, GLOBALIDS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveGlobalIds(const char* name)
{
  return this->SetActiveAttribute(name, GLOBALIDS);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetGlobalIds()
{
  return this->GetAttribute(GLOBALIDS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetPedigreeIds(svtkAbstractArray* aa)
{
  return this->SetAttribute(aa, PEDIGREEIDS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActivePedigreeIds(const char* name)
{
  return this->SetActiveAttribute(name, PEDIGREEIDS);
}

//--------------------------------------------------------------------------
svtkAbstractArray* svtkDataSetAttributes::GetPedigreeIds()
{
  return this->GetAbstractAttribute(PEDIGREEIDS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetRationalWeights(svtkDataArray* da)
{
  return this->SetAttribute(da, RATIONALWEIGHTS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveRationalWeights(const char* name)
{
  return this->SetActiveAttribute(name, RATIONALWEIGHTS);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetRationalWeights()
{
  return this->GetAttribute(RATIONALWEIGHTS);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetHigherOrderDegrees(svtkDataArray* da)
{
  return this->SetAttribute(da, HIGHERORDERDEGREES);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveHigherOrderDegrees(const char* name)
{
  return this->SetActiveAttribute(name, HIGHERORDERDEGREES);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetHigherOrderDegrees()
{
  return this->GetAttribute(HIGHERORDERDEGREES);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetScalars(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetScalars();
  }
  return this->GetArray(name);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetVectors(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetVectors();
  }
  return this->GetArray(name);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetNormals(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetNormals();
  }
  return this->GetArray(name);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetTangents(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetTangents();
  }
  return this->GetArray(name);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetTCoords(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetTCoords();
  }
  return this->GetArray(name);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetTensors(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetTensors();
  }
  return this->GetArray(name);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetGlobalIds(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetGlobalIds();
  }
  return this->GetArray(name);
}

//--------------------------------------------------------------------------
svtkAbstractArray* svtkDataSetAttributes::GetPedigreeIds(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetPedigreeIds();
  }
  return this->GetAbstractArray(name);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetRationalWeights(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetRationalWeights();
  }
  return this->GetArray(name);
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetHigherOrderDegrees(const char* name)
{
  if (name == nullptr || name[0] == '\0')
  {
    return this->GetHigherOrderDegrees();
  }
  return this->GetArray(name);
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::SetActiveAttribute(int index, int attributeType)
{
  if ((index >= 0) && (index < this->GetNumberOfArrays()))
  {
    if (attributeType != PEDIGREEIDS)
    {
      svtkDataArray* darray = svtkArrayDownCast<svtkDataArray>(this->Data[index]);
      if (!darray)
      {
        svtkWarningMacro("Can not set attribute "
          << svtkDataSetAttributes::AttributeNames[attributeType]
          << ". Only svtkDataArray subclasses can be set as active attributes.");
        return -1;
      }
      if (!this->CheckNumberOfComponents(darray, attributeType))
      {
        svtkWarningMacro("Can not set attribute "
          << svtkDataSetAttributes::AttributeNames[attributeType]
          << ". Incorrect number of components.");
        return -1;
      }
    }

    this->AttributeIndices[attributeType] = index;
    this->Modified();
    return index;
  }
  else if (index == -1)
  {
    this->AttributeIndices[attributeType] = index;
    this->Modified();
  }

  return -1;
}

//--------------------------------------------------------------------------
const int
  svtkDataSetAttributes ::NumberOfAttributeComponents[svtkDataSetAttributes::NUM_ATTRIBUTES] = { 0, 3,
    3, 3, 9, 1, 1, 1, 3, 1, 3 };

//--------------------------------------------------------------------------
// Scalars set to NOLIMIT
const int svtkDataSetAttributes ::AttributeLimits[svtkDataSetAttributes::NUM_ATTRIBUTES] = { NOLIMIT,
  EXACT, EXACT, MAX, EXACT, EXACT, EXACT, EXACT, EXACT, EXACT, EXACT };

//--------------------------------------------------------------------------
int svtkDataSetAttributes::CheckNumberOfComponents(svtkAbstractArray* aa, int attributeType)
{
  int numComp = aa->GetNumberOfComponents();

  if (svtkDataSetAttributes::AttributeLimits[attributeType] == MAX)
  {
    if (numComp > svtkDataSetAttributes::NumberOfAttributeComponents[attributeType])
    {
      return 0;
    }
    else
    {
      return 1;
    }
  }
  else if (svtkDataSetAttributes::AttributeLimits[attributeType] == EXACT)
  {
    if (numComp == svtkDataSetAttributes::NumberOfAttributeComponents[attributeType] ||
      (numComp == 6 && attributeType == TENSORS)) // TENSORS6 support
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }
  else if (svtkDataSetAttributes::AttributeLimits[attributeType] == NOLIMIT)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

//--------------------------------------------------------------------------
svtkDataArray* svtkDataSetAttributes::GetAttribute(int attributeType)
{
  int index = this->AttributeIndices[attributeType];
  if (index == -1)
  {
    return nullptr;
  }
  else
  {
    return svtkArrayDownCast<svtkDataArray>(this->Data[index]);
  }
}

//--------------------------------------------------------------------------
svtkAbstractArray* svtkDataSetAttributes::GetAbstractAttribute(int attributeType)
{
  int index = this->AttributeIndices[attributeType];
  if (index == -1)
  {
    return nullptr;
  }
  else
  {
    return this->Data[index];
  }
}

//--------------------------------------------------------------------------
// This method lets the user add an array and make it the current
// scalars, vectors etc... (this is determined by the attribute type
// which is an enum defined svtkDataSetAttributes)
int svtkDataSetAttributes::SetAttribute(svtkAbstractArray* aa, int attributeType)
{
  if (aa && attributeType != PEDIGREEIDS && !svtkArrayDownCast<svtkDataArray>(aa))
  {
    svtkWarningMacro("Can not set attribute "
      << svtkDataSetAttributes::AttributeNames[attributeType]
      << ". This attribute must be a subclass of svtkDataArray.");
    return -1;
  }
  if (aa && !this->CheckNumberOfComponents(aa, attributeType))
  {
    svtkWarningMacro("Can not set attribute " << svtkDataSetAttributes::AttributeNames[attributeType]
                                             << ". Incorrect number of components.");
    return -1;
  }

  int currentAttribute = this->AttributeIndices[attributeType];

  // If there is an existing attribute, replace it
  if ((currentAttribute >= 0) && (currentAttribute < this->GetNumberOfArrays()))
  {
    if (this->GetAbstractArray(currentAttribute) == aa)
    {
      return currentAttribute;
    }
    this->RemoveArray(currentAttribute);
  }

  if (aa)
  {
    // Add the array
    currentAttribute = this->AddArray(aa);
    this->AttributeIndices[attributeType] = currentAttribute;
  }
  else
  {
    this->AttributeIndices[attributeType] = -1; // attribute of this type doesn't exist
  }
  this->Modified();
  return this->AttributeIndices[attributeType];
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  // Print the copy flags
  int i;
  os << indent << "Copy Tuple Flags: ( ";
  for (i = 0; i < NUM_ATTRIBUTES; i++)
  {
    os << this->CopyAttributeFlags[COPYTUPLE][i] << " ";
  }
  os << ")" << endl;
  os << indent << "Interpolate Flags: ( ";
  for (i = 0; i < NUM_ATTRIBUTES; i++)
  {
    os << this->CopyAttributeFlags[INTERPOLATE][i] << " ";
  }
  os << ")" << endl;
  os << indent << "Pass Through Flags: ( ";
  for (i = 0; i < NUM_ATTRIBUTES; i++)
  {
    os << this->CopyAttributeFlags[PASSDATA][i] << " ";
  }
  os << ")" << endl;

  // Now print the various attributes
  svtkAbstractArray* aa;
  int attributeType;
  for (attributeType = 0; attributeType < NUM_ATTRIBUTES; attributeType++)
  {
    os << indent << svtkDataSetAttributes::AttributeNames[attributeType] << ": ";
    if ((aa = this->GetAbstractAttribute(attributeType)))
    {
      os << endl;
      aa->PrintSelf(os, indent.GetNextIndent());
    }
    else
    {
      os << "(none)" << endl;
    }
  }
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::GetAttributeIndices(int* indexArray)
{
  int i;
  for (i = 0; i < NUM_ATTRIBUTES; i++)
  {
    indexArray[i] = this->AttributeIndices[i];
  }
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::IsArrayAnAttribute(int idx)
{
  int i;
  for (i = 0; i < NUM_ATTRIBUTES; i++)
  {
    if (idx == this->AttributeIndices[i])
    {
      return i;
    }
  }
  return -1;
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyAttribute(int index, int value, int ctype)
{
  if (index < 0 || ctype < 0 || index >= svtkDataSetAttributes::NUM_ATTRIBUTES ||
    ctype > svtkDataSetAttributes::ALLCOPY)
  {
    svtkErrorMacro("Cannot set copy attribute for attribute type "
      << index << " and copy operation " << ctype << ". These values are out of range.");
    return;
  }

  if (ctype == svtkDataSetAttributes::ALLCOPY)
  {
    int t;
    for (t = COPYTUPLE; t < svtkDataSetAttributes::ALLCOPY; t++)
    {
      if (this->CopyAttributeFlags[t][index] != value)
      {
        this->CopyAttributeFlags[t][index] = value;
        this->Modified();
      }
    }
  }
  else
  {
    if (this->CopyAttributeFlags[ctype][index] != value)
    {
      this->CopyAttributeFlags[ctype][index] = value;
      this->Modified();
    }
  }
}

//--------------------------------------------------------------------------
int svtkDataSetAttributes::GetCopyAttribute(int index, int ctype)
{
  if (index < 0 || ctype < 0 || index >= svtkDataSetAttributes::NUM_ATTRIBUTES ||
    ctype > svtkDataSetAttributes::ALLCOPY)
  {
    svtkWarningMacro("Cannot get copy attribute for attribute type "
      << index << " and copy operation " << ctype << ". These values are out of range.");
    return -1;
  }
  else if (ctype == svtkDataSetAttributes::ALLCOPY)
  {
    return (this->CopyAttributeFlags[COPYTUPLE][index] &&
      this->CopyAttributeFlags[INTERPOLATE][index] && this->CopyAttributeFlags[PASSDATA][index]);
  }
  else
  {
    return this->CopyAttributeFlags[ctype][index];
  }
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyScalars(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(SCALARS, i, ctype);
}

//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyScalars(int ctype)
{
  return this->GetCopyAttribute(SCALARS, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyVectors(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(VECTORS, i, ctype);
}

//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyVectors(int ctype)
{
  return this->GetCopyAttribute(VECTORS, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyNormals(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(NORMALS, i, ctype);
}

//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyNormals(int ctype)
{
  return this->GetCopyAttribute(NORMALS, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyTangents(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(TANGENTS, i, ctype);
}

//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyTangents(int ctype)
{
  return this->GetCopyAttribute(TANGENTS, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyTCoords(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(TCOORDS, i, ctype);
}

//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyTCoords(int ctype)
{
  return this->GetCopyAttribute(TCOORDS, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyTensors(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(TENSORS, i, ctype);
}
//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyTensors(int ctype)
{
  return this->GetCopyAttribute(TENSORS, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyGlobalIds(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(GLOBALIDS, i, ctype);
}

//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyGlobalIds(int ctype)
{
  return this->GetCopyAttribute(GLOBALIDS, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyPedigreeIds(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(PEDIGREEIDS, i, ctype);
}

//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyPedigreeIds(int ctype)
{
  return this->GetCopyAttribute(PEDIGREEIDS, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyRationalWeights(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(RATIONALWEIGHTS, i, ctype);
}

//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyRationalWeights(int ctype)
{
  return this->GetCopyAttribute(RATIONALWEIGHTS, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::SetCopyHigherOrderDegrees(svtkTypeBool i, int ctype)
{
  this->SetCopyAttribute(HIGHERORDERDEGREES, i, ctype);
}

//--------------------------------------------------------------------------
svtkTypeBool svtkDataSetAttributes::GetCopyHigherOrderDegrees(int ctype)
{
  return this->GetCopyAttribute(HIGHERORDERDEGREES, ctype);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::CopyAllocate(
  svtkDataSetAttributes::FieldList& list, svtkIdType sze, svtkIdType ext)
{
  list.CopyAllocate(this, COPYTUPLE, sze, ext);
}

//--------------------------------------------------------------------------
void svtkDataSetAttributes::InterpolateAllocate(
  svtkDataSetAttributes::FieldList& list, svtkIdType sze, svtkIdType ext)
{
  list.CopyAllocate(this, INTERPOLATE, sze, ext);
}

//--------------------------------------------------------------------------
// Description:
// A special form of CopyData() to be used with FieldLists. Use it when you are
// copying data from a set of svtkDataSetAttributes. Make sure that you have
// called the special form of CopyAllocate that accepts FieldLists.
void svtkDataSetAttributes::CopyData(svtkDataSetAttributes::FieldList& list,
  svtkDataSetAttributes* fromDSA, int idx, svtkIdType fromId, svtkIdType toId)
{
  list.CopyData(idx, fromDSA, fromId, this, toId);
}

//--------------------------------------------------------------------------
// Description:
// A special form of CopyData() to be used with FieldLists. Use it when you are
// copying data from a set of svtkDataSetAttributes. Make sure that you have
// called the special form of CopyAllocate that accepts FieldLists.
void svtkDataSetAttributes::CopyData(svtkDataSetAttributes::FieldList& list,
  svtkDataSetAttributes* fromDSA, int idx, svtkIdType dstStart, svtkIdType n, svtkIdType srcStart)
{
  list.CopyData(idx, fromDSA, srcStart, n, this, dstStart);
}

//--------------------------------------------------------------------------
// Interpolate data from points and interpolation weights. Make sure that the
// method InterpolateAllocate() has been invoked before using this method.
void svtkDataSetAttributes::InterpolatePoint(svtkDataSetAttributes::FieldList& list,
  svtkDataSetAttributes* fromPd, int idx, svtkIdType toId, svtkIdList* ptIds, double* weights)
{
  list.InterpolatePoint(idx, fromPd, ptIds, weights, this, toId);
}

//--------------------------------------------------------------------------
const char* svtkDataSetAttributes::GetAttributeTypeAsString(int attributeType)
{
  if (attributeType < 0 || attributeType >= NUM_ATTRIBUTES)
  {
    svtkGenericWarningMacro("Bad attribute type: " << attributeType << ".");
    return nullptr;
  }
  return svtkDataSetAttributes::AttributeNames[attributeType];
}

//--------------------------------------------------------------------------
const char* svtkDataSetAttributes::GetLongAttributeTypeAsString(int attributeType)
{
  if (attributeType < 0 || attributeType >= NUM_ATTRIBUTES)
  {
    svtkGenericWarningMacro("Bad attribute type: " << attributeType << ".");
    return nullptr;
  }
  return svtkDataSetAttributes::LongAttributeNames[attributeType];
}
