/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkFieldData.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkFieldData.h"

#include "svtkDataArray.h"
#include "svtkIdList.h"
#include "svtkInformation.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkFieldData);

//----------------------------------------------------------------------------
svtkFieldData::BasicIterator::BasicIterator(const int* list, unsigned int listSize)
{
  if (list)
  {
    if (listSize > 0)
    {
      this->List = new int[listSize];
      memcpy(this->List, list, listSize * sizeof(int));
    }
    else
    {
      this->List = nullptr;
    }
    this->ListSize = listSize;
  }
  else
  {
    this->List = nullptr;
    this->ListSize = 0;
  }
  this->Position = 0;
}

//----------------------------------------------------------------------------
svtkFieldData::Iterator::Iterator(svtkFieldData* dsa, const int* list, unsigned int listSize)
  : svtkFieldData::BasicIterator(list, listSize)
{
  this->Fields = dsa;
  dsa->Register(nullptr);
  if (!list)
  {
    this->ListSize = dsa->GetNumberOfArrays();
    this->List = new int[this->ListSize];
    for (int i = 0; i < this->ListSize; i++)
    {
      this->List[i] = i;
    }
  }
  this->Detached = 0;
}

//----------------------------------------------------------------------------
svtkFieldData::BasicIterator::BasicIterator()
{
  this->List = nullptr;
  this->ListSize = 0;
}

//----------------------------------------------------------------------------
svtkFieldData::BasicIterator::BasicIterator(const svtkFieldData::BasicIterator& source)
{
  this->ListSize = source.ListSize;

  if (this->ListSize > 0)
  {
    this->List = new int[this->ListSize];
    memcpy(this->List, source.List, this->ListSize * sizeof(int));
  }
  else
  {
    this->List = nullptr;
  }
}

//----------------------------------------------------------------------------
svtkFieldData::Iterator::Iterator(const svtkFieldData::Iterator& source)
  : svtkFieldData::BasicIterator(source)
{
  this->Detached = source.Detached;
  this->Fields = source.Fields;
  if (this->Fields && !this->Detached)
  {
    this->Fields->Register(nullptr);
  }
}

//----------------------------------------------------------------------------
void svtkFieldData::BasicIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "BasicIterator:{";
  if (this->ListSize > 0)
  {
    os << this->List[0];
    for (int i = 1; i < this->ListSize; ++i)
    {
      os << ", " << this->List[i];
    }
  }
  os << "}" << endl;
}

//----------------------------------------------------------------------------
svtkFieldData::BasicIterator& svtkFieldData::BasicIterator::operator=(
  const svtkFieldData::BasicIterator& source)
{
  if (this == &source)
  {
    return *this;
  }
  delete[] this->List;
  this->ListSize = source.ListSize;
  if (this->ListSize > 0)
  {
    this->List = new int[this->ListSize];
    memcpy(this->List, source.List, this->ListSize * sizeof(int));
  }
  else
  {
    this->List = nullptr;
  }
  return *this;
}

//----------------------------------------------------------------------------
svtkFieldData::Iterator& svtkFieldData::Iterator::operator=(const svtkFieldData::Iterator& source)
{
  if (this == &source)
  {
    return *this;
  }
  this->BasicIterator::operator=(source);
  if (this->Fields && !this->Detached)
  {
    this->Fields->UnRegister(nullptr);
  }
  this->Fields = source.Fields;
  this->Detached = source.Detached;
  if (this->Fields && !this->Detached)
  {
    this->Fields->Register(nullptr);
  }
  return *this;
}

//----------------------------------------------------------------------------
svtkFieldData::BasicIterator::~BasicIterator()
{
  delete[] this->List;
}

//----------------------------------------------------------------------------
svtkFieldData::Iterator::~Iterator()
{
  if (this->Fields && !this->Detached)
  {
    this->Fields->UnRegister(nullptr);
  }
}

//----------------------------------------------------------------------------
void svtkFieldData::Iterator::DetachFieldData()
{
  if (this->Fields && !this->Detached)
  {
    this->Fields->UnRegister(nullptr);
    this->Detached = 1;
  }
}

//----------------------------------------------------------------------------
// Construct object with no data initially.
svtkFieldData::svtkFieldData()
{
  this->NumberOfArrays = 0;
  this->Data = nullptr;
  this->NumberOfActiveArrays = 0;

  this->CopyFieldFlags = nullptr;
  this->NumberOfFieldFlags = 0;

  this->DoCopyAllOn = 1;
  this->DoCopyAllOff = 0;

  this->CopyAllOn();
}

//----------------------------------------------------------------------------
svtkFieldData::~svtkFieldData()
{
  this->Initialize();
  this->ClearFieldFlags();
}

//----------------------------------------------------------------------------
// Release all data but do not delete object.
void svtkFieldData::InitializeFields()
{
  int i;

  if (this->Data)
  {
    for (i = 0; i < this->GetNumberOfArrays(); i++)
    {
      this->Data[i]->UnRegister(this);
    }

    delete[] this->Data;
    this->Data = nullptr;
  }

  this->NumberOfArrays = 0;
  this->NumberOfActiveArrays = 0;
  this->Modified();
}

//----------------------------------------------------------------------------
// Release all data but do not delete object.
// Also initialize copy flags.
void svtkFieldData::Initialize()
{
  this->InitializeFields();
  this->CopyAllOn();
  this->ClearFieldFlags();
}

//----------------------------------------------------------------------------
// Allocate data for each array.
svtkTypeBool svtkFieldData::Allocate(svtkIdType sz, svtkIdType ext)
{
  int i;
  int status = 0;

  for (i = 0; i < this->GetNumberOfArrays(); i++)
  {
    if ((status = this->Data[i]->Allocate(sz, ext)) == 0)
    {
      break;
    }
  }

  return status;
}

//----------------------------------------------------------------------------
void svtkFieldData::CopyStructure(svtkFieldData* r)
{
  // Free old fields.
  this->InitializeFields();

  // Allocate new fields.
  this->AllocateArrays(r->GetNumberOfArrays());
  this->NumberOfActiveArrays = r->GetNumberOfArrays();

  // Copy the data array's structure (ie nTups,nComps,name, and info)
  // don't copy their data.
  int i;
  svtkAbstractArray* data;
  for (i = 0; i < r->GetNumberOfArrays(); ++i)
  {
    data = r->Data[i]->NewInstance();
    int numComponents = r->Data[i]->GetNumberOfComponents();
    data->SetNumberOfComponents(numComponents);
    data->SetName(r->Data[i]->GetName());
    for (svtkIdType j = 0; j < numComponents; j++)
    {
      data->SetComponentName(j, r->Data[i]->GetComponentName(j));
    }
    if (r->Data[i]->HasInformation())
    {
      data->CopyInformation(r->Data[i]->GetInformation(), /*deep=*/1);
    }
    this->SetArray(i, data);
    data->Delete();
  }
}

//----------------------------------------------------------------------------
// Set the number of arrays used to define the field.
void svtkFieldData::AllocateArrays(int num)
{
  int i;

  if (num < 0)
  {
    num = 0;
  }

  if (num == this->NumberOfArrays)
  {
    return;
  }

  if (num == 0)
  {
    this->Initialize();
  }
  else if (num < this->NumberOfArrays)
  {
    for (i = num; i < this->NumberOfArrays; i++)
    {
      if (this->Data[i])
      {
        this->Data[i]->UnRegister(this);
      }
    }
    this->NumberOfArrays = num;
  }
  else // num > this->NumberOfArrays
  {
    svtkAbstractArray** data = new svtkAbstractArray*[num];
    // copy the original data
    for (i = 0; i < this->NumberOfArrays; i++)
    {
      data[i] = this->Data[i];
    }

    // initialize the new arrays
    for (i = this->NumberOfArrays; i < num; i++)
    {
      data[i] = nullptr;
    }

    // get rid of the old data
    delete[] this->Data;

    // update object
    this->Data = data;
    this->NumberOfArrays = num;
  }
  this->Modified();
}

//----------------------------------------------------------------------------
// Set an array to define the field.
void svtkFieldData::SetArray(int i, svtkAbstractArray* data)
{
  if (!data || (i > this->NumberOfActiveArrays))
  {
    svtkWarningMacro("Can not set array " << i << " to " << data << endl);
    return;
  }

  if (i < 0)
  {
    svtkWarningMacro("Array index should be >= 0");
    return;
  }
  else if (i >= this->NumberOfArrays)
  {
    this->AllocateArrays(i + 1);
    this->NumberOfActiveArrays = i + 1;
  }

  if (this->Data[i] != data)
  {
    if (this->Data[i] != nullptr)
    {
      this->Data[i]->UnRegister(this);
    }
    this->Data[i] = data;
    if (this->Data[i] != nullptr)
    {
      this->Data[i]->Register(this);
    }
    this->Modified();
  }
}

//----------------------------------------------------------------------------
// Return the ith array in the field. A nullptr is returned if the index i is out
// if range.
svtkDataArray* svtkFieldData::GetArray(int i)
{
  return svtkArrayDownCast<svtkDataArray>(this->GetAbstractArray(i));
}

//----------------------------------------------------------------------------
// Return the ith array in the field. A nullptr is returned if the index i is out
// if range.
svtkAbstractArray* svtkFieldData::GetAbstractArray(int i)
{
  if (i < 0 || i >= this->GetNumberOfArrays() || this->Data == nullptr)
  {
    return nullptr;
  }
  return this->Data[i];
}

//----------------------------------------------------------------------------
// Copy a field by creating new data arrays
void svtkFieldData::DeepCopy(svtkFieldData* f)
{
  svtkAbstractArray *data, *newData;

  this->AllocateArrays(f->GetNumberOfArrays());
  for (int i = 0; i < f->GetNumberOfArrays(); i++)
  {
    data = f->GetAbstractArray(i);
    newData = data->NewInstance(); // instantiate same type of object
    newData->DeepCopy(data);
    newData->SetName(data->GetName());
    if (data->HasInformation())
    {
      newData->CopyInformation(data->GetInformation(), /*deep=*/1);
    }
    this->AddArray(newData);
    newData->Delete();
  }
}

//----------------------------------------------------------------------------
// Copy a field by reference counting the data arrays.
void svtkFieldData::ShallowCopy(svtkFieldData* f)
{
  this->AllocateArrays(f->GetNumberOfArrays());
  this->NumberOfActiveArrays = 0;

  for (int i = 0; i < f->GetNumberOfArrays(); i++)
  {
    this->NumberOfActiveArrays++;
    this->SetArray(i, f->GetAbstractArray(i));
  }
  this->CopyFlags(f);
}

//----------------------------------------------------------------------------
// Squeezes each data array in the field (Squeeze() reclaims unused memory.)
void svtkFieldData::Squeeze()
{
  for (int i = 0; i < this->GetNumberOfArrays(); i++)
  {
    this->Data[i]->Squeeze();
  }
}

//----------------------------------------------------------------------------
// Resets each data array in the field (Reset() does not release memory but
// it makes the arrays look like they are empty.)
void svtkFieldData::Reset()
{
  int i;

  for (i = 0; i < this->GetNumberOfArrays(); i++)
  {
    this->Data[i]->Reset();
  }
}

//----------------------------------------------------------------------------
// Get a field from a list of ids. Supplied field f should have same
// types and number of data arrays as this one (i.e., like
// CopyStructure() creates).
void svtkFieldData::GetField(svtkIdList* ptIds, svtkFieldData* f)
{
  int i, numIds = ptIds->GetNumberOfIds();

  for (i = 0; i < numIds; i++)
  {
    f->InsertTuple(i, ptIds->GetId(i), this);
  }
}

//----------------------------------------------------------------------------
// Return the array containing the ith component of the field. The return value
// is an integer number n 0<=n<this->NumberOfArrays. Also, an integer value is
// returned indicating the component in the array is returned. Method returns
// -1 if specified component is not in field.
int svtkFieldData::GetArrayContainingComponent(int i, int& arrayComp)
{
  int numComp, count = 0;

  for (int j = 0; j < this->GetNumberOfArrays(); j++)
  {
    if (this->Data[j] != nullptr)
    {
      numComp = this->Data[j]->GetNumberOfComponents();
      if (i < (numComp + count))
      {
        arrayComp = i - count;
        return j;
      }
      count += numComp;
    }
  }
  return -1;
}

//----------------------------------------------------------------------------
svtkDataArray* svtkFieldData::GetArray(const char* arrayName, int& index)
{
  int i;
  svtkDataArray* da = svtkArrayDownCast<svtkDataArray>(this->GetAbstractArray(arrayName, i));
  index = (da) ? i : -1;
  return da;
}

//----------------------------------------------------------------------------
svtkAbstractArray* svtkFieldData::GetAbstractArray(const char* arrayName, int& index)
{
  int i;
  const char* name;
  index = -1;
  if (!arrayName)
  {
    return nullptr;
  }
  for (i = 0; i < this->GetNumberOfArrays(); i++)
  {
    name = this->GetArrayName(i);
    if (name && !strcmp(name, arrayName))
    {
      index = i;
      return this->GetAbstractArray(i);
    }
  }
  return nullptr;
}

//----------------------------------------------------------------------------
int svtkFieldData::AddArray(svtkAbstractArray* array)
{
  if (!array)
  {
    return -1;
  }

  int index;
  this->GetAbstractArray(array->GetName(), index);

  if (index == -1)
  {
    index = this->NumberOfActiveArrays;
    this->NumberOfActiveArrays++;
  }
  this->SetArray(index, array);
  return index;
}

//--------------------------------------------------------------------------
void svtkFieldData::RemoveArray(const char* name)
{
  int i;
  this->GetAbstractArray(name, i);
  this->RemoveArray(i);
}

//----------------------------------------------------------------------------
void svtkFieldData::RemoveArray(int index)
{
  if ((index < 0) || (index >= this->NumberOfActiveArrays))
  {
    return;
  }
  this->Data[index]->UnRegister(this);
  this->Data[index] = nullptr;
  this->NumberOfActiveArrays--;
  for (int i = index; i < this->NumberOfActiveArrays; i++)
  {
    this->Data[i] = this->Data[i + 1];
  }
  this->Data[this->NumberOfActiveArrays] = nullptr;
  this->Modified();
}

//----------------------------------------------------------------------------
unsigned long svtkFieldData::GetActualMemorySize()
{
  unsigned long size = 0;

  for (int i = 0; i < this->GetNumberOfArrays(); i++)
  {
    if (this->Data[i] != nullptr)
    {
      size += this->Data[i]->GetActualMemorySize();
    }
  }

  return size;
}

//----------------------------------------------------------------------------
svtkMTimeType svtkFieldData::GetMTime()
{
  svtkMTimeType mTime = this->MTime;
  svtkMTimeType otherMTime;
  svtkAbstractArray* aa;

  for (int i = 0; i < this->NumberOfActiveArrays; i++)
  {
    if ((aa = this->Data[i]))
    {
      otherMTime = aa->GetMTime();
      if (otherMTime > mTime)
      {
        mTime = otherMTime;
      }
    }
  }

  return mTime;
}

//----------------------------------------------------------------------------
void svtkFieldData::CopyFieldOnOff(const char* field, int onOff)
{
  if (!field)
  {
    return;
  }

  int index;
  // If the array is in the list, simply set IsCopied to onOff
  if ((index = this->FindFlag(field)) != -1)
  {
    if (this->CopyFieldFlags[index].IsCopied != onOff)
    {
      this->CopyFieldFlags[index].IsCopied = onOff;
      this->Modified();
    }
  }
  else
  {
    // We need to reallocate the list of fields
    svtkFieldData::CopyFieldFlag* newFlags =
      new svtkFieldData::CopyFieldFlag[this->NumberOfFieldFlags + 1];
    // Copy old flags (pointer copy for name)
    for (int i = 0; i < this->NumberOfFieldFlags; i++)
    {
      newFlags[i].ArrayName = this->CopyFieldFlags[i].ArrayName;
      newFlags[i].IsCopied = this->CopyFieldFlags[i].IsCopied;
    }
    // Copy new flag (strcpy)
    char* newName = new char[strlen(field) + 1];
    strcpy(newName, field);
    newFlags[this->NumberOfFieldFlags].ArrayName = newName;
    newFlags[this->NumberOfFieldFlags].IsCopied = onOff;
    this->NumberOfFieldFlags++;
    delete[] this->CopyFieldFlags;
    this->CopyFieldFlags = newFlags;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
// Turn on copying of all data.
void svtkFieldData::CopyAllOn(int svtkNotUsed(ctype))
{
  if (!DoCopyAllOn || DoCopyAllOff)
  {
    this->DoCopyAllOn = 1;
    this->DoCopyAllOff = 0;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
// Turn off copying of all data.
void svtkFieldData::CopyAllOff(int svtkNotUsed(ctype))
{
  if (DoCopyAllOn || !DoCopyAllOff)
  {
    this->DoCopyAllOn = 0;
    this->DoCopyAllOff = 1;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
// Deallocate and clear the list of fields.
void svtkFieldData::ClearFieldFlags()
{
  if (this->NumberOfFieldFlags > 0)
  {
    for (int i = 0; i < this->NumberOfFieldFlags; i++)
    {
      delete[] this->CopyFieldFlags[i].ArrayName;
    }
  }
  delete[] this->CopyFieldFlags;
  this->CopyFieldFlags = nullptr;
  this->NumberOfFieldFlags = 0;
}

//----------------------------------------------------------------------------
// Find if field is in CopyFieldFlags.
// If it is, it returns the index otherwise it returns -1
int svtkFieldData::FindFlag(const char* field)
{
  if (!field)
    return -1;
  for (int i = 0; i < this->NumberOfFieldFlags; i++)
  {
    if (this->CopyFieldFlags[i].ArrayName && !strcmp(field, this->CopyFieldFlags[i].ArrayName))
    {
      return i;
    }
  }
  return -1;
}

//----------------------------------------------------------------------------
// If there is no flag for this array, return -1.
// If there is one: return 0 if off, 1 if on
int svtkFieldData::GetFlag(const char* field)
{
  int index = this->FindFlag(field);
  if (index == -1)
  {
    return -1;
  }
  else
  {
    return this->CopyFieldFlags[index].IsCopied;
  }
}

//----------------------------------------------------------------------------
// Copy the fields list (with strcpy)
void svtkFieldData::CopyFlags(const svtkFieldData* source)
{
  this->ClearFieldFlags();
  this->NumberOfFieldFlags = source->NumberOfFieldFlags;
  if (this->NumberOfFieldFlags > 0)
  {
    this->CopyFieldFlags = new svtkFieldData::CopyFieldFlag[this->NumberOfFieldFlags];
    for (int i = 0; i < this->NumberOfFieldFlags; i++)
    {
      this->CopyFieldFlags[i].ArrayName = new char[strlen(source->CopyFieldFlags[i].ArrayName) + 1];
      strcpy(this->CopyFieldFlags[i].ArrayName, source->CopyFieldFlags[i].ArrayName);
    }
  }
  else
  {
    this->CopyFieldFlags = nullptr;
  }
}

//----------------------------------------------------------------------------
void svtkFieldData::PassData(svtkFieldData* fd)
{
  for (int i = 0; i < fd->GetNumberOfArrays(); i++)
  {
    const char* arrayName = fd->GetArrayName(i);
    // If there is no blocker for the given array
    // and both CopyAllOff and CopyOn for that array are not true
    if ((this->GetFlag(arrayName) != 0) &&
      !(this->DoCopyAllOff && (this->GetFlag(arrayName) != 1)) && fd->GetAbstractArray(i))
    {
      this->AddArray(fd->GetAbstractArray(i));
    }
  }
}

//----------------------------------------------------------------------------
void svtkFieldData::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Number Of Arrays: " << this->GetNumberOfArrays() << "\n";
  for (int i = 0; i < this->GetNumberOfArrays(); i++)
  {
    if (this->GetArrayName(i))
    {
      os << indent << "Array " << i << " name = " << this->GetArrayName(i) << "\n";
    }
    else
    {
      os << indent << "Array " << i << " name = nullptr\n";
    }
  }
  os << indent << "Number Of Components: " << this->GetNumberOfComponents() << "\n";
  os << indent << "Number Of Tuples: " << this->GetNumberOfTuples() << "\n";
}

//----------------------------------------------------------------------------
// Get the number of components in the field. This is determined by adding
// up the components in each non-nullptr array.
int svtkFieldData::GetNumberOfComponents()
{
  int i, numComp;

  for (i = numComp = 0; i < this->GetNumberOfArrays(); i++)
  {
    if (this->Data[i])
    {
      numComp += this->Data[i]->GetNumberOfComponents();
    }
  }

  return numComp;
}

//----------------------------------------------------------------------------
// Get the number of tuples in the field.
svtkIdType svtkFieldData::GetNumberOfTuples()
{
  svtkAbstractArray* da;
  if ((da = this->GetAbstractArray(0)))
  {
    return da->GetNumberOfTuples();
  }
  else
  {
    return 0;
  }
}

//----------------------------------------------------------------------------
// Set the number of tuples for each data array in the field.
void svtkFieldData::SetNumberOfTuples(const svtkIdType number)
{
  for (int i = 0; i < this->GetNumberOfArrays(); i++)
  {
    this->Data[i]->SetNumberOfTuples(number);
  }
}

//----------------------------------------------------------------------------
// Set the jth tuple in source field data at the ith location.
// Set operations
// means that no range checking is performed, so they're faster.
void svtkFieldData::SetTuple(const svtkIdType i, const svtkIdType j, svtkFieldData* source)
{
  for (int k = 0; k < this->GetNumberOfArrays(); k++)
  {
    this->Data[k]->SetTuple(i, j, source->Data[k]);
  }
}

//----------------------------------------------------------------------------
// Insert the tuple value at the ith location. Range checking is
// performed and memory allocates as necessary.
void svtkFieldData::InsertTuple(const svtkIdType i, const svtkIdType j, svtkFieldData* source)
{
  for (int k = 0; k < this->GetNumberOfArrays(); k++)
  {
    this->Data[k]->InsertTuple(i, j, source->GetAbstractArray(k));
  }
}

//----------------------------------------------------------------------------
// Insert the tuple value at the end of the tuple matrix. Range
// checking is performed and memory is allocated as necessary.
svtkIdType svtkFieldData::InsertNextTuple(const svtkIdType j, svtkFieldData* source)
{
  svtkIdType id = this->GetNumberOfTuples();
  this->InsertTuple(id, j, source);
  return id;
}
