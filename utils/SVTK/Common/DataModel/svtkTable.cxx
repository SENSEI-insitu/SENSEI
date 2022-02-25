/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTable.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------*/

#include "svtkTable.h"
#include "svtkArrayIteratorIncludes.h"

#include "svtkAbstractArray.h"
#include "svtkDataArray.h"
#include "svtkDataSetAttributes.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkStringArray.h"
#include "svtkUnicodeStringArray.h"
#include "svtkVariantArray.h"

#include <vector>

//
// Standard functions
//

svtkStandardNewMacro(svtkTable);
svtkCxxSetObjectMacro(svtkTable, RowData, svtkDataSetAttributes);

//----------------------------------------------------------------------------
svtkTable::svtkTable()
{
  this->RowArray = svtkVariantArray::New();
  this->RowData = svtkDataSetAttributes::New();

  this->Information->Set(svtkDataObject::DATA_EXTENT_TYPE(), SVTK_PIECES_EXTENT);
  this->Information->Set(svtkDataObject::DATA_PIECE_NUMBER(), -1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_PIECES(), 1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS(), 0);
}

//----------------------------------------------------------------------------
svtkTable::~svtkTable()
{
  if (this->RowArray)
  {
    this->RowArray->Delete();
  }
  if (this->RowData)
  {
    this->RowData->Delete();
  }
}

//----------------------------------------------------------------------------
void svtkTable::PrintSelf(ostream& os, svtkIndent indent)
{
  svtkDataObject::PrintSelf(os, indent);
  os << indent << "RowData: " << (this->RowData ? "" : "(none)") << endl;
  if (this->RowData)
  {
    this->RowData->PrintSelf(os, indent.GetNextIndent());
  }
}

//----------------------------------------------------------------------------
void svtkTable::Dump(unsigned int colWidth, int rowLimit)

{
  if (!this->GetNumberOfColumns())
  {
    cout << "++\n++\n";
    return;
  }

  svtkStdString lineStr;
  for (int c = 0; c < this->GetNumberOfColumns(); ++c)
  {
    lineStr += "+-";

    for (unsigned int i = 0; i < colWidth; ++i)
    {
      lineStr += "-";
    }
  }
  lineStr += "-+\n";

  cout << lineStr;

  for (int c = 0; c < this->GetNumberOfColumns(); ++c)
  {
    cout << "| ";
    const char* name = this->GetColumnName(c);
    svtkStdString str = name ? name : "";

    if (colWidth < str.length())
    {
      cout << str.substr(0, colWidth);
    }
    else
    {
      cout << str;
      for (unsigned int i = static_cast<unsigned int>(str.length()); i < colWidth; ++i)
      {
        cout << " ";
      }
    }
  }

  cout << " |\n" << lineStr;

  if (rowLimit != 0)
  {
    for (svtkIdType r = 0; r < this->GetNumberOfRows(); ++r)
    {
      for (int c = 0; c < this->GetNumberOfColumns(); ++c)
      {
        cout << "| ";
        svtkStdString str = this->GetValue(r, c).ToString();

        if (colWidth < str.length())
        {
          cout << str.substr(0, colWidth);
        }
        else
        {
          cout << str;
          for (unsigned int i = static_cast<unsigned int>(str.length()); i < colWidth; ++i)
          {
            cout << " ";
          }
        }
      }
      cout << " |\n";
      if (rowLimit != -1 && r >= rowLimit)
        break;
    }
    cout << lineStr;
    cout.flush();
  }
}

//----------------------------------------------------------------------------
void svtkTable::Initialize()
{
  this->Superclass::Initialize();
  if (this->RowData)
  {
    this->RowData->Initialize();
  }
}

//----------------------------------------------------------------------------
unsigned long svtkTable::GetActualMemorySize()
{
  return this->RowData->GetActualMemorySize() + this->Superclass::GetActualMemorySize();
}

//
// Row functions
//

//----------------------------------------------------------------------------
svtkIdType svtkTable::GetNumberOfRows()
{
  if (this->GetNumberOfColumns() > 0)
  {
    return this->GetColumn(0)->GetNumberOfTuples();
  }
  return 0;
}

//----------------------------------------------------------------------------
void svtkTable::SetNumberOfRows(svtkIdType n)
{
  if (this->RowData)
  {
    this->RowData->SetNumberOfTuples(n);
  }
}

//----------------------------------------------------------------------------
svtkVariantArray* svtkTable::GetRow(svtkIdType row)
{
  svtkIdType ncol = this->GetNumberOfColumns();
  this->RowArray->SetNumberOfTuples(ncol);
  for (svtkIdType i = 0; i < ncol; i++)
  {
    this->RowArray->SetValue(i, this->GetValue(row, i));
  }
  return this->RowArray;
}

//----------------------------------------------------------------------------
void svtkTable::GetRow(svtkIdType row, svtkVariantArray* values)
{
  svtkIdType ncol = this->GetNumberOfColumns();
  values->SetNumberOfTuples(ncol);
  for (svtkIdType i = 0; i < ncol; i++)
  {
    values->SetValue(i, this->GetValue(row, i));
  }
}

//----------------------------------------------------------------------------
void svtkTable::SetRow(svtkIdType row, svtkVariantArray* values)
{
  svtkIdType ncol = this->GetNumberOfColumns();
  if (values->GetNumberOfTuples() != ncol)
  {
    svtkErrorMacro(<< "Incorrect number of tuples in SetRow");
  }
  for (svtkIdType i = 0; i < ncol; i++)
  {
    this->SetValue(row, i, values->GetValue(i));
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkTable::InsertNextBlankRow(double default_num_val)
{
  svtkIdType ncol = this->GetNumberOfColumns();
  std::vector<double> tuple(32, default_num_val);
  for (svtkIdType i = 0; i < ncol; i++)
  {
    svtkAbstractArray* arr = this->GetColumn(i);
    const size_t comps = static_cast<size_t>(arr->GetNumberOfComponents());
    if (svtkArrayDownCast<svtkDataArray>(arr))
    {
      if (comps > tuple.size())
      { // We initialize this to 32 components, but just in case...
        tuple.resize(comps, default_num_val);
      }

      svtkDataArray* data = svtkArrayDownCast<svtkDataArray>(arr);
      data->InsertNextTuple(tuple.data());
    }
    else if (svtkArrayDownCast<svtkStringArray>(arr))
    {
      svtkStringArray* data = svtkArrayDownCast<svtkStringArray>(arr);
      for (size_t j = 0; j < comps; j++)
      {
        data->InsertNextValue(svtkStdString(""));
      }
    }
    else if (svtkArrayDownCast<svtkVariantArray>(arr))
    {
      svtkVariantArray* data = svtkArrayDownCast<svtkVariantArray>(arr);
      for (size_t j = 0; j < comps; j++)
      {
        data->InsertNextValue(svtkVariant());
      }
    }
    else if (svtkArrayDownCast<svtkUnicodeStringArray>(arr))
    {
      svtkUnicodeStringArray* data = svtkArrayDownCast<svtkUnicodeStringArray>(arr);
      for (size_t j = 0; j < comps; j++)
      {
        data->InsertNextValue(svtkUnicodeString::from_utf8(""));
      }
    }
    else
    {
      svtkErrorMacro(<< "Unsupported array type for InsertNextBlankRow");
    }
  }
  return this->GetNumberOfRows() - 1;
}

//----------------------------------------------------------------------------
svtkIdType svtkTable::InsertNextRow(svtkVariantArray* values)
{
  svtkIdType ncol = this->GetNumberOfColumns();
  if (values->GetNumberOfTuples() != ncol)
  {
    svtkErrorMacro(<< "Incorrect number of tuples in SetRow."
                  << " Expected " << ncol << ", but got " << values->GetNumberOfTuples());
  }
  svtkIdType row = this->InsertNextBlankRow();
  for (svtkIdType i = 0; i < ncol; i++)
  {
    this->SetValue(row, i, values->GetValue(i));
  }
  return row;
}

//----------------------------------------------------------------------------
void svtkTable::RemoveRow(svtkIdType row)
{
  svtkIdType ncol = this->GetNumberOfColumns();
  for (svtkIdType i = 0; i < ncol; i++)
  {
    svtkAbstractArray* arr = this->GetColumn(i);
    int comps = arr->GetNumberOfComponents();
    if (svtkArrayDownCast<svtkDataArray>(arr))
    {
      svtkDataArray* data = svtkArrayDownCast<svtkDataArray>(arr);
      data->RemoveTuple(row);
    }
    else if (svtkArrayDownCast<svtkStringArray>(arr))
    {
      // Manually move all elements past the index back one place.
      svtkStringArray* data = svtkArrayDownCast<svtkStringArray>(arr);
      for (int j = comps * row; j < comps * data->GetNumberOfTuples() - 1; j++)
      {
        data->SetValue(j, data->GetValue(j + 1));
      }
      data->Resize(data->GetNumberOfTuples() - 1);
    }
    else if (svtkArrayDownCast<svtkVariantArray>(arr))
    {
      // Manually move all elements past the index back one place.
      svtkVariantArray* data = svtkArrayDownCast<svtkVariantArray>(arr);
      for (int j = comps * row; j < comps * data->GetNumberOfTuples() - 1; j++)
      {
        data->SetValue(j, data->GetValue(j + 1));
      }
      data->Resize(data->GetNumberOfTuples() - 1);
    }
  }
}

//
// Column functions
//

//----------------------------------------------------------------------------
svtkIdType svtkTable::GetNumberOfColumns()
{
  return this->RowData->GetNumberOfArrays();
}

//----------------------------------------------------------------------------
void svtkTable::AddColumn(svtkAbstractArray* arr)
{
  if (this->GetNumberOfColumns() > 0 && arr->GetNumberOfTuples() != this->GetNumberOfRows())
  {
    svtkErrorMacro(<< "Column \"" << arr->GetName() << "\" must have " << this->GetNumberOfRows()
                  << " rows, but has " << arr->GetNumberOfTuples() << ".");
    return;
  }
  this->RowData->AddArray(arr);
}

//----------------------------------------------------------------------------
void svtkTable::RemoveColumnByName(const char* name)
{
  this->RowData->RemoveArray(name);
}

//----------------------------------------------------------------------------
void svtkTable::RemoveColumn(svtkIdType col)
{
  int column = static_cast<int>(col);
  this->RowData->RemoveArray(this->RowData->GetArrayName(column));
}

//----------------------------------------------------------------------------
const char* svtkTable::GetColumnName(svtkIdType col)
{
  int column = static_cast<int>(col);
  return this->RowData->GetArrayName(column);
}

//----------------------------------------------------------------------------
svtkAbstractArray* svtkTable::GetColumnByName(const char* name)
{
  return this->RowData->GetAbstractArray(name);
}

//----------------------------------------------------------------------------
svtkAbstractArray* svtkTable::GetColumn(svtkIdType col)
{
  int column = static_cast<int>(col);
  return this->RowData->GetAbstractArray(column);
}

//
// Table single entry functions
//

//----------------------------------------------------------------------------
void svtkTable::SetValue(svtkIdType row, svtkIdType col, svtkVariant value)
{
  svtkAbstractArray* arr = this->GetColumn(col);
  if (!arr)
  {
    return;
  }
  int comps = arr->GetNumberOfComponents();
  if (svtkArrayDownCast<svtkDataArray>(arr))
  {
    svtkDataArray* data = svtkArrayDownCast<svtkDataArray>(arr);
    if (comps == 1)
    {
      data->SetVariantValue(row, value);
    }
    else
    {
      if (value.IsArray() && svtkArrayDownCast<svtkDataArray>(value.ToArray()) &&
        value.ToArray()->GetNumberOfComponents() == comps)
      {
        data->SetTuple(row, svtkArrayDownCast<svtkDataArray>(value.ToArray())->GetTuple(0));
      }
      else
      {
        svtkWarningMacro("Cannot assign this variant type to multi-component data array.");
        return;
      }
    }
  }
  else if (svtkArrayDownCast<svtkStringArray>(arr))
  {
    svtkStringArray* data = svtkArrayDownCast<svtkStringArray>(arr);
    if (comps == 1)
    {
      data->SetValue(row, value.ToString());
    }
    else
    {
      if (value.IsArray() && svtkArrayDownCast<svtkStringArray>(value.ToArray()) &&
        value.ToArray()->GetNumberOfComponents() == comps)
      {
        data->SetTuple(row, 0, svtkArrayDownCast<svtkStringArray>(value.ToArray()));
      }
      else
      {
        svtkWarningMacro("Cannot assign this variant type to multi-component string array.");
        return;
      }
    }
  }
  else if (svtkArrayDownCast<svtkVariantArray>(arr))
  {
    svtkVariantArray* data = svtkArrayDownCast<svtkVariantArray>(arr);
    if (comps == 1)
    {
      data->SetValue(row, value);
    }
    else
    {
      if (value.IsArray() && value.ToArray()->GetNumberOfComponents() == comps)
      {
        data->SetTuple(row, 0, value.ToArray());
      }
      else
      {
        svtkWarningMacro("Cannot assign this variant type to multi-component string array.");
        return;
      }
    }
  }
  else if (svtkArrayDownCast<svtkUnicodeStringArray>(arr))
  {
    svtkUnicodeStringArray* data = svtkArrayDownCast<svtkUnicodeStringArray>(arr);
    if (comps == 1)
    {
      data->SetValue(row, value.ToUnicodeString());
    }
    else
    {
      if (value.IsArray() && svtkArrayDownCast<svtkUnicodeStringArray>(value.ToArray()) &&
        value.ToArray()->GetNumberOfComponents() == comps)
      {
        data->SetTuple(row, 0, svtkArrayDownCast<svtkUnicodeStringArray>(value.ToArray()));
      }
      else
      {
        svtkWarningMacro("Cannot assign this variant type to multi-component unicode string array.");
        return;
      }
    }
  }
  else
  {
    svtkWarningMacro("Unable to process array named " << col);
  }
}

//----------------------------------------------------------------------------
void svtkTable::SetValueByName(svtkIdType row, const char* col, svtkVariant value)
{
  int colIndex = -1;
  this->RowData->GetAbstractArray(col, colIndex);
  if (colIndex < 0)
  {
    svtkErrorMacro(<< "Could not find column named " << col);
    return;
  }
  this->SetValue(row, colIndex, value);
}

//----------------------------------------------------------------------------
template <typename iterT>
svtkVariant svtkTableGetVariantValue(iterT* it, svtkIdType row)
{
  return svtkVariant(it->GetValue(row));
}

//----------------------------------------------------------------------------
svtkVariant svtkTable::GetValue(svtkIdType row, svtkIdType col)
{
  svtkAbstractArray* arr = this->GetColumn(col);
  if (!arr)
  {
    return svtkVariant();
  }

  int comps = arr->GetNumberOfComponents();
  if (row >= arr->GetNumberOfTuples())
  {
    return svtkVariant();
  }
  if (svtkArrayDownCast<svtkDataArray>(arr))
  {
    if (comps == 1)
    {
      svtkArrayIterator* iter = arr->NewIterator();
      svtkVariant v;
      switch (arr->GetDataType())
      {
        svtkArrayIteratorTemplateMacro(v = svtkTableGetVariantValue(static_cast<SVTK_TT*>(iter), row));
      }
      iter->Delete();
      return v;
    }
    else
    {
      // Create a variant holding an array of the appropriate type
      // with one tuple.
      svtkDataArray* da = svtkDataArray::CreateDataArray(arr->GetDataType());
      da->SetNumberOfComponents(comps);
      da->InsertNextTuple(row, arr);
      svtkVariant v(da);
      da->Delete();
      return v;
    }
  }
  else if (svtkArrayDownCast<svtkStringArray>(arr))
  {
    svtkStringArray* data = svtkArrayDownCast<svtkStringArray>(arr);
    if (comps == 1)
    {
      return svtkVariant(data->GetValue(row));
    }
    else
    {
      // Create a variant holding a svtkStringArray with one tuple.
      svtkStringArray* sa = svtkStringArray::New();
      sa->SetNumberOfComponents(comps);
      sa->InsertNextTuple(row, data);
      svtkVariant v(sa);
      sa->Delete();
      return v;
    }
  }
  else if (svtkArrayDownCast<svtkUnicodeStringArray>(arr))
  {
    svtkUnicodeStringArray* data = svtkArrayDownCast<svtkUnicodeStringArray>(arr);
    if (comps == 1)
    {
      return svtkVariant(data->GetValue(row));
    }
    else
    {
      // Create a variant holding a svtkStringArray with one tuple.
      svtkUnicodeStringArray* sa = svtkUnicodeStringArray::New();
      sa->SetNumberOfComponents(comps);
      sa->InsertNextTuple(row, data);
      svtkVariant v(sa);
      sa->Delete();
      return v;
    }
  }
  else if (svtkArrayDownCast<svtkVariantArray>(arr))
  {
    svtkVariantArray* data = svtkArrayDownCast<svtkVariantArray>(arr);
    if (comps == 1)
    {
      return data->GetValue(row);
    }
    else
    {
      // Create a variant holding a svtkVariantArray with one tuple.
      svtkVariantArray* va = svtkVariantArray::New();
      va->SetNumberOfComponents(comps);
      va->InsertNextTuple(row, data);
      svtkVariant v(va);
      va->Delete();
      return v;
    }
  }
  return svtkVariant();
}

//----------------------------------------------------------------------------
svtkVariant svtkTable::GetValueByName(svtkIdType row, const char* col)
{
  int colIndex = -1;
  this->RowData->GetAbstractArray(col, colIndex);
  if (colIndex < 0)
  {
    return svtkVariant();
  }
  return this->GetValue(row, colIndex);
}

//----------------------------------------------------------------------------
svtkTable* svtkTable::GetData(svtkInformation* info)
{
  return info ? svtkTable::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkTable* svtkTable::GetData(svtkInformationVector* v, int i)
{
  return svtkTable::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkTable::ShallowCopy(svtkDataObject* src)
{
  if (svtkTable* const table = svtkTable::SafeDownCast(src))
  {
    this->RowData->ShallowCopy(table->RowData);
    this->Modified();
  }

  this->Superclass::ShallowCopy(src);
}

//----------------------------------------------------------------------------
void svtkTable::DeepCopy(svtkDataObject* src)
{
  if (svtkTable* const table = svtkTable::SafeDownCast(src))
  {
    this->RowData->DeepCopy(table->RowData);
    this->Modified();
  }

  Superclass::DeepCopy(src);
}

//----------------------------------------------------------------------------
svtkFieldData* svtkTable::GetAttributesAsFieldData(int type)
{
  switch (type)
  {
    case ROW:
      return this->GetRowData();
  }
  return this->Superclass::GetAttributesAsFieldData(type);
}

//----------------------------------------------------------------------------
svtkIdType svtkTable::GetNumberOfElements(int type)
{
  switch (type)
  {
    case ROW:
      return this->GetNumberOfRows();
  }
  return this->Superclass::GetNumberOfElements(type);
}
