/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBitArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkBitArray.h"

#include "svtkBitArrayIterator.h"
#include "svtkIdList.h"
#include "svtkObjectFactory.h"

//----------------------------------------------------------------------------
class svtkBitArrayLookup
{
public:
  svtkBitArrayLookup()
    : Rebuild(true)
  {
    this->ZeroArray = nullptr;
    this->OneArray = nullptr;
  }
  ~svtkBitArrayLookup()
  {
    if (this->ZeroArray)
    {
      this->ZeroArray->Delete();
      this->ZeroArray = nullptr;
    }
    if (this->OneArray)
    {
      this->OneArray->Delete();
      this->OneArray = nullptr;
    }
  }
  svtkIdList* ZeroArray;
  svtkIdList* OneArray;
  bool Rebuild;
};

svtkStandardNewMacro(svtkBitArray);

//----------------------------------------------------------------------------
// Instantiate object.
svtkBitArray::svtkBitArray()
{
  this->Array = nullptr;
  this->TupleSize = 3;
  this->Tuple = new double[this->TupleSize]; // used for conversion
  this->DeleteFunction = ::operator delete[];
  this->Lookup = nullptr;
}

//----------------------------------------------------------------------------
svtkBitArray::~svtkBitArray()
{
  if (this->DeleteFunction)
  {
    this->DeleteFunction(this->Array);
  }
  delete[] this->Tuple;
  delete this->Lookup;
}

//----------------------------------------------------------------------------
unsigned char* svtkBitArray::WritePointer(svtkIdType id, svtkIdType number)
{
  svtkIdType newSize = id + number;
  if (newSize > this->Size)
  {
    this->ResizeAndExtend(newSize);
  }
  if ((--newSize) > this->MaxId)
  {
    this->MaxId = newSize;
  }
  this->DataChanged();
  return this->Array + id / 8;
}

//----------------------------------------------------------------------------
// This method lets the user specify data to be held by the array.  The
// array argument is a pointer to the data.  size is the size of
// the array supplied by the user.  Set save to 1 to keep the class
// from deleting the array when it cleans up or reallocates memory.
// The class uses the actual array provided; it does not copy the data
// from the supplied array.
void svtkBitArray::SetArray(unsigned char* array, svtkIdType size, int save, int deleteMethod)
{

  if ((this->Array) && (this->DeleteFunction))
  {
    svtkDebugMacro(<< "Deleting the array...");
    this->DeleteFunction(this->Array);
  }
  else
  {
    svtkDebugMacro(<< "Warning, array not deleted, but will point to new array.");
  }

  svtkDebugMacro(<< "Setting array to: " << array);

  this->Array = array;
  this->Size = size;
  this->MaxId = size - 1;

  if (save != 0)
  {
    this->DeleteFunction = nullptr;
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_DELETE || deleteMethod == SVTK_DATA_ARRAY_USER_DEFINED)
  {
    this->DeleteFunction = ::operator delete[];
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_ALIGNED_FREE)
  {
#ifdef _WIN32
    this->DeleteFunction = _aligned_free;
#else
    this->DeleteFunction = free;
#endif
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_FREE)
  {
    this->DeleteFunction = free;
  }

  this->DataChanged();
}

//-----------------------------------------------------------------------------
void svtkBitArray::SetArrayFreeFunction(void (*callback)(void*))
{
  this->DeleteFunction = callback;
}

//----------------------------------------------------------------------------
// Get the data at a particular index.
int svtkBitArray::GetValue(svtkIdType id) const
{
  if (this->Array[id / 8] & (0x80 >> (id % 8)))
  {
    return 1;
  }
  return 0;
}

//----------------------------------------------------------------------------
// Allocate memory for this array. Delete old storage only if necessary.
svtkTypeBool svtkBitArray::Allocate(svtkIdType sz, svtkIdType svtkNotUsed(ext))
{
  if (sz > this->Size)
  {
    if (this->DeleteFunction)
    {
      this->DeleteFunction(this->Array);
    }
    this->Size = (sz > 0 ? sz : 1);
    if ((this->Array = new unsigned char[(this->Size + 7) / 8]) == nullptr)
    {
      return 0;
    }
    this->DeleteFunction = ::operator delete[];
  }

  this->MaxId = -1;
  this->DataChanged();

  return 1;
}

//----------------------------------------------------------------------------
// Release storage and reset array to initial state.
void svtkBitArray::Initialize()
{
  if (this->DeleteFunction)
  {
    this->DeleteFunction(this->Array);
  }
  this->Array = nullptr;
  this->Size = 0;
  this->MaxId = -1;
  this->DeleteFunction = ::operator delete[];
  this->DataChanged();
}

//----------------------------------------------------------------------------
// Deep copy of another bit array.
void svtkBitArray::DeepCopy(svtkDataArray* ia)
{
  // Do nothing on a nullptr input.
  if (ia == nullptr)
  {
    return;
  }

  this->DataChanged();

  if (ia->GetDataType() != SVTK_BIT)
  {
    svtkIdType numTuples = ia->GetNumberOfTuples();
    this->NumberOfComponents = ia->GetNumberOfComponents();
    this->SetNumberOfTuples(numTuples);

    for (svtkIdType i = 0; i < numTuples; i++)
    {
      this->SetTuple(i, ia->GetTuple(i));
    }
    return;
  }

  if (this != ia)
  {
    if (this->DeleteFunction)
    {
      this->DeleteFunction(this->Array);
    }

    this->NumberOfComponents = ia->GetNumberOfComponents();
    this->MaxId = ia->GetMaxId();
    this->Size = ia->GetSize();
    this->DeleteFunction = ::operator delete[];

    this->Array = new unsigned char[(this->Size + 7) / 8];
    memcpy(this->Array, static_cast<unsigned char*>(ia->GetVoidPointer(0)),
      static_cast<size_t>((this->Size + 7) / 8) * sizeof(unsigned char));
  }
}

//----------------------------------------------------------------------------
void svtkBitArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  if (this->Array)
  {
    os << indent << "Array: " << this->Array << "\n";
  }
  else
  {
    os << indent << "Array: (null)\n";
  }
}

//----------------------------------------------------------------------------
// Private function does "reallocate". Sz is the number of "bits", and we
// can allocate only 8-bit bytes.
unsigned char* svtkBitArray::ResizeAndExtend(svtkIdType sz)
{
  unsigned char* newArray;
  svtkIdType newSize;

  if (sz > this->Size)
  {
    newSize = this->Size + sz;
  }
  else if (sz == this->Size)
  {
    return this->Array;
  }
  else
  {
    newSize = sz;
  }

  if (newSize <= 0)
  {
    this->Initialize();
    return nullptr;
  }

  if ((newArray = new unsigned char[(newSize + 7) / 8]) == nullptr)
  {
    svtkErrorMacro(<< "Cannot allocate memory\n");
    return nullptr;
  }

  if (this->Array)
  {
    svtkIdType usedSize = (sz < this->Size) ? sz : this->Size;

    memcpy(newArray, this->Array, static_cast<size_t>((usedSize + 7) / 8) * sizeof(unsigned char));
    if (this->DeleteFunction)
    {
      this->DeleteFunction(this->Array);
    }
  }

  if (newSize < this->Size)
  {
    this->MaxId = newSize - 1;
  }
  this->Size = newSize;
  this->Array = newArray;
  this->DeleteFunction = ::operator delete[];
  this->DataChanged();

  return this->Array;
}

//----------------------------------------------------------------------------
svtkTypeBool svtkBitArray::Resize(svtkIdType sz)
{
  unsigned char* newArray;
  svtkIdType newSize = sz * this->NumberOfComponents;

  if (newSize == this->Size)
  {
    return 1;
  }

  if (newSize <= 0)
  {
    this->Initialize();
    return 1;
  }

  if ((newArray = new unsigned char[(newSize + 7) / 8]) == nullptr)
  {
    svtkErrorMacro(<< "Cannot allocate memory\n");
    return 0;
  }

  if (this->Array)
  {
    svtkIdType usedSize = (newSize < this->Size) ? newSize : this->Size;

    memcpy(newArray, this->Array, static_cast<size_t>((usedSize + 7) / 8) * sizeof(unsigned char));
    if (this->DeleteFunction)
    {
      this->DeleteFunction(this->Array);
    }
  }

  if (newSize < this->Size)
  {
    this->MaxId = newSize - 1;
  }
  this->Size = newSize;
  this->Array = newArray;
  this->DeleteFunction = ::operator delete[];
  this->DataChanged();

  return 1;
}

//----------------------------------------------------------------------------
// Set the number of n-tuples in the array.
void svtkBitArray::SetNumberOfTuples(svtkIdType number)
{
  this->SetNumberOfValues(number * this->NumberOfComponents);
}

//----------------------------------------------------------------------------
// Description:
// Set the tuple at the ith location using the jth tuple in the source array.
// This method assumes that the two arrays have the same type
// and structure. Note that range checking and memory allocation is not
// performed; use in conjunction with SetNumberOfTuples() to allocate space.
void svtkBitArray::SetTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source)
{
  svtkBitArray* ba = svtkArrayDownCast<svtkBitArray>(source);
  if (!ba)
  {
    svtkWarningMacro("Input and output arrays types do not match.");
    return;
  }

  svtkIdType loci = i * this->NumberOfComponents;
  svtkIdType locj = j * ba->GetNumberOfComponents();
  for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
  {
    this->SetValue(loci + cur, ba->GetValue(locj + cur));
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
// Description:
// Insert the jth tuple in the source array, at ith location in this array.
// Note that memory allocation is performed as necessary to hold the data.
void svtkBitArray::InsertTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source)
{
  svtkBitArray* ba = svtkArrayDownCast<svtkBitArray>(source);
  if (!ba)
  {
    svtkWarningMacro("Input and output arrays types do not match.");
    return;
  }

  svtkIdType loci = i * this->NumberOfComponents;
  svtkIdType locj = j * ba->GetNumberOfComponents();
  for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
  {
    this->InsertValue(loci + cur, ba->GetValue(locj + cur));
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkBitArray::InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source)
{
  svtkBitArray* ba = svtkArrayDownCast<svtkBitArray>(source);
  if (!ba)
  {
    svtkWarningMacro("Input and output arrays types do not match.");
    return;
  }

  if (ba->NumberOfComponents != this->NumberOfComponents)
  {
    svtkWarningMacro("Number of components do not match.");
    return;
  }

  svtkIdType numIds = dstIds->GetNumberOfIds();
  if (srcIds->GetNumberOfIds() != numIds)
  {
    svtkWarningMacro("Input and output id array sizes do not match.");
    return;
  }

  for (svtkIdType idIndex = 0; idIndex < numIds; ++idIndex)
  {
    svtkIdType numComp = this->NumberOfComponents;
    svtkIdType srcLoc = srcIds->GetId(idIndex) * this->NumberOfComponents;
    svtkIdType dstLoc = dstIds->GetId(idIndex) * this->NumberOfComponents;
    while (numComp-- > 0)
    {
      this->InsertValue(dstLoc++, ba->GetValue(srcLoc++));
    }
  }
  this->DataChanged();
}

//------------------------------------------------------------------------------
void svtkBitArray::InsertTuples(
  svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source)
{
  svtkBitArray* sa = svtkArrayDownCast<svtkBitArray>(source);
  if (!sa)
  {
    svtkWarningMacro("Input and outputs array data types do not match.");
    return;
  }

  if (this->NumberOfComponents != source->GetNumberOfComponents())
  {
    svtkWarningMacro("Input and output component sizes do not match.");
    return;
  }

  svtkIdType srcEnd = srcStart + n;
  if (srcEnd > source->GetNumberOfTuples())
  {
    svtkWarningMacro("Source range exceeds array size (srcStart="
      << srcStart << ", n=" << n << ", numTuples=" << source->GetNumberOfTuples() << ").");
    return;
  }

  for (svtkIdType i = 0; i < n; ++i)
  {
    svtkIdType numComp = this->NumberOfComponents;
    svtkIdType srcLoc = (srcStart + i) * this->NumberOfComponents;
    svtkIdType dstLoc = (dstStart + i) * this->NumberOfComponents;
    while (numComp-- > 0)
    {
      this->InsertValue(dstLoc++, sa->GetValue(srcLoc++));
    }
  }

  this->DataChanged();
}

//----------------------------------------------------------------------------
// Description:
// Insert the jth tuple in the source array, at the end in this array.
// Note that memory allocation is performed as necessary to hold the data.
// Returns the location at which the data was inserted.
svtkIdType svtkBitArray::InsertNextTuple(svtkIdType j, svtkAbstractArray* source)
{
  svtkBitArray* ba = svtkArrayDownCast<svtkBitArray>(source);
  if (!ba)
  {
    svtkWarningMacro("Input and output arrays types do not match.");
    return -1;
  }

  svtkIdType locj = j * ba->GetNumberOfComponents();
  for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
  {
    this->InsertNextValue(ba->GetValue(locj + cur));
  }
  this->DataChanged();
  return (this->GetNumberOfTuples() - 1);
}

//----------------------------------------------------------------------------
// Get a pointer to a tuple at the ith location. This is a dangerous method
// (it is not thread safe since a pointer is returned).
double* svtkBitArray::GetTuple(svtkIdType i)
{
  if (this->TupleSize < this->NumberOfComponents)
  {
    this->TupleSize = this->NumberOfComponents;
    delete[] this->Tuple;
    this->Tuple = new double[this->TupleSize];
  }

  svtkIdType loc = this->NumberOfComponents * i;
  for (int j = 0; j < this->NumberOfComponents; j++)
  {
    this->Tuple[j] = static_cast<double>(this->GetValue(loc + j));
  }

  return this->Tuple;
}

//----------------------------------------------------------------------------
// Copy the tuple value into a user-provided array.
void svtkBitArray::GetTuple(svtkIdType i, double* tuple)
{
  svtkIdType loc = this->NumberOfComponents * i;

  for (int j = 0; j < this->NumberOfComponents; j++)
  {
    tuple[j] = static_cast<double>(this->GetValue(loc + j));
  }
}

//----------------------------------------------------------------------------
// Set the tuple value at the ith location in the array.
void svtkBitArray::SetTuple(svtkIdType i, const float* tuple)
{
  svtkIdType loc = i * this->NumberOfComponents;

  for (int j = 0; j < this->NumberOfComponents; j++)
  {
    this->SetValue(loc + j, static_cast<int>(tuple[j]));
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkBitArray::SetTuple(svtkIdType i, const double* tuple)
{
  svtkIdType loc = i * this->NumberOfComponents;

  for (int j = 0; j < this->NumberOfComponents; j++)
  {
    this->SetValue(loc + j, static_cast<int>(tuple[j]));
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
// Insert (memory allocation performed) the tuple into the ith location
// in the array.
void svtkBitArray::InsertTuple(svtkIdType i, const float* tuple)
{
  svtkIdType loc = this->NumberOfComponents * i;

  for (int j = 0; j < this->NumberOfComponents; j++)
  {
    this->InsertValue(loc + j, static_cast<int>(tuple[j]));
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkBitArray::InsertTuple(svtkIdType i, const double* tuple)
{
  svtkIdType loc = this->NumberOfComponents * i;

  for (int j = 0; j < this->NumberOfComponents; j++)
  {
    this->InsertValue(loc + j, static_cast<int>(tuple[j]));
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
// Insert (memory allocation performed) the tuple onto the end of the array.
svtkIdType svtkBitArray::InsertNextTuple(const float* tuple)
{
  for (int i = 0; i < this->NumberOfComponents; i++)
  {
    this->InsertNextValue(static_cast<int>(tuple[i]));
  }

  this->DataChanged();
  return this->MaxId / this->NumberOfComponents;
}

//----------------------------------------------------------------------------
svtkIdType svtkBitArray::InsertNextTuple(const double* tuple)
{
  for (int i = 0; i < this->NumberOfComponents; i++)
  {
    this->InsertNextValue(static_cast<int>(tuple[i]));
  }

  this->DataChanged();
  return this->MaxId / this->NumberOfComponents;
}

//----------------------------------------------------------------------------
void svtkBitArray::InsertComponent(svtkIdType i, int j, double c)
{
  this->InsertValue(i * this->NumberOfComponents + j, static_cast<int>(c));
  this->DataChanged();
}

//----------------------------------------------------------------------------
// Set the data component at the ith tuple and jth component location.
// Note that i<NumberOfTuples and j<NumberOfComponents. Make sure enough
// memory has been allocated (use SetNumberOfTuples() and
// SetNumberOfComponents()).
void svtkBitArray::SetComponent(svtkIdType i, int j, double c)
{
  this->SetValue(i * this->NumberOfComponents + j, static_cast<int>(c));
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkBitArray::RemoveTuple(svtkIdType id)
{
  if (id < 0 || id >= this->GetNumberOfTuples())
  {
    // Nothing to be done
    return;
  }
  if (id == this->GetNumberOfTuples() - 1)
  {
    // To remove last item, just decrease the size by one
    this->RemoveLastTuple();
    return;
  }
  this->DataChanged();
  svtkErrorMacro("Not yet implemented...");
}

//----------------------------------------------------------------------------
void svtkBitArray::RemoveFirstTuple()
{
  svtkErrorMacro("Not yet implemented...");
  this->RemoveTuple(0);
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkBitArray::RemoveLastTuple()
{
  this->Resize(this->GetNumberOfTuples() - 1);
  this->DataChanged();
}

//----------------------------------------------------------------------------
svtkArrayIterator* svtkBitArray::NewIterator()
{
  svtkArrayIterator* iter = svtkBitArrayIterator::New();
  iter->Initialize(this);
  return iter;
}

//----------------------------------------------------------------------------
void svtkBitArray::UpdateLookup()
{
  if (!this->Lookup)
  {
    this->Lookup = new svtkBitArrayLookup();
    this->Lookup->ZeroArray = svtkIdList::New();
    this->Lookup->OneArray = svtkIdList::New();
  }
  if (this->Lookup->Rebuild)
  {
    int numComps = this->GetNumberOfComponents();
    svtkIdType numTuples = this->GetNumberOfTuples();
    this->Lookup->ZeroArray->Allocate(numComps * numTuples);
    this->Lookup->OneArray->Allocate(numComps * numTuples);
    for (svtkIdType i = 0; i < numComps * numTuples; i++)
    {
      if (this->GetValue(i))
      {
        this->Lookup->OneArray->InsertNextId(i);
      }
      else
      {
        this->Lookup->ZeroArray->InsertNextId(i);
      }
    }
    this->Lookup->Rebuild = false;
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkBitArray::LookupValue(svtkVariant var)
{
  return this->LookupValue(var.ToInt());
}

//----------------------------------------------------------------------------
void svtkBitArray::LookupValue(svtkVariant var, svtkIdList* ids)
{
  this->LookupValue(var.ToInt(), ids);
}

//----------------------------------------------------------------------------
svtkIdType svtkBitArray::LookupValue(int value)
{
  this->UpdateLookup();

  if (value == 1 && this->Lookup->OneArray->GetNumberOfIds() > 0)
  {
    return this->Lookup->OneArray->GetId(0);
  }
  else if (value == 0 && this->Lookup->ZeroArray->GetNumberOfIds() > 0)
  {
    return this->Lookup->ZeroArray->GetId(0);
  }
  return -1;
}

//----------------------------------------------------------------------------
void svtkBitArray::LookupValue(int value, svtkIdList* ids)
{
  this->UpdateLookup();

  if (value == 1)
  {
    ids->DeepCopy(this->Lookup->OneArray);
  }
  else if (value == 0)
  {
    ids->DeepCopy(this->Lookup->ZeroArray);
  }
  else
  {
    ids->Reset();
  }
}

//----------------------------------------------------------------------------
void svtkBitArray::DataChanged()
{
  if (this->Lookup)
  {
    this->Lookup->Rebuild = true;
  }
}

//----------------------------------------------------------------------------
void svtkBitArray::ClearLookup()
{
  delete this->Lookup;
  this->Lookup = nullptr;
}
