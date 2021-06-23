/*=========================================================================

  Program:   ParaView
  Module:    svtkSelectionNode.cxx

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.paraview.org/HTML/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkSelectionNode.h"

#include "svtkDataArrayRange.h"
#include "svtkDataObject.h"
#include "svtkDataSetAttributes.h"
#include "svtkIdTypeArray.h"
#include "svtkInformation.h"
#include "svtkInformationDoubleKey.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationIterator.h"
#include "svtkInformationObjectBaseKey.h"
#include "svtkInformationStringKey.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkSmartPointer.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

svtkStandardNewMacro(svtkSelectionNode);
svtkCxxSetObjectMacro(svtkSelectionNode, SelectionData, svtkDataSetAttributes);

const char svtkSelectionNode ::ContentTypeNames[svtkSelectionNode::NUM_CONTENT_TYPES][14] = {
  "SELECTIONS", // deprecated
  "GLOBALIDS",
  "PEDIGREEIDS",
  "VALUES",
  "INDICES",
  "FRUSTUM",
  "LOCATIONS",
  "THRESHOLDS",
  "BLOCKS",
  "QUERY",
  "USER",
};

const char svtkSelectionNode ::FieldTypeNames[svtkSelectionNode::NUM_FIELD_TYPES][8] = {
  "CELL",
  "POINT",
  "FIELD",
  "VERTEX",
  "EDGE",
  "ROW",
};

svtkInformationKeyMacro(svtkSelectionNode, CONTENT_TYPE, Integer);
svtkInformationKeyMacro(svtkSelectionNode, SOURCE, ObjectBase);
svtkInformationKeyMacro(svtkSelectionNode, SOURCE_ID, Integer);
svtkInformationKeyMacro(svtkSelectionNode, PROP, ObjectBase);
svtkInformationKeyMacro(svtkSelectionNode, PROP_ID, Integer);
svtkInformationKeyMacro(svtkSelectionNode, PROCESS_ID, Integer);
svtkInformationKeyMacro(svtkSelectionNode, COMPOSITE_INDEX, Integer);
svtkInformationKeyMacro(svtkSelectionNode, HIERARCHICAL_LEVEL, Integer);
svtkInformationKeyMacro(svtkSelectionNode, HIERARCHICAL_INDEX, Integer);
svtkInformationKeyMacro(svtkSelectionNode, FIELD_TYPE, Integer);
svtkInformationKeyMacro(svtkSelectionNode, EPSILON, Double);
svtkInformationKeyMacro(svtkSelectionNode, ZBUFFER_VALUE, Double);
svtkInformationKeyMacro(svtkSelectionNode, CONTAINING_CELLS, Integer);
svtkInformationKeyMacro(svtkSelectionNode, CONNECTED_LAYERS, Integer);
svtkInformationKeyMacro(svtkSelectionNode, PIXEL_COUNT, Integer);
svtkInformationKeyMacro(svtkSelectionNode, INVERSE, Integer);
svtkInformationKeyMacro(svtkSelectionNode, INDEXED_VERTICES, Integer);
svtkInformationKeyMacro(svtkSelectionNode, COMPONENT_NUMBER, Integer);

//----------------------------------------------------------------------------
svtkSelectionNode::svtkSelectionNode()
{
  this->SelectionData = svtkDataSetAttributes::New();
  this->Properties = svtkInformation::New();
  this->QueryString = nullptr;
}

//----------------------------------------------------------------------------
svtkSelectionNode::~svtkSelectionNode()
{
  this->Properties->Delete();
  if (this->SelectionData)
  {
    this->SelectionData->Delete();
  }
  this->SetQueryString(nullptr);
}

//----------------------------------------------------------------------------
void svtkSelectionNode::Initialize()
{
  this->Properties->Clear();
  if (this->SelectionData)
  {
    this->SelectionData->Initialize();
  }
  this->Modified();
}

//----------------------------------------------------------------------------
svtkAbstractArray* svtkSelectionNode::GetSelectionList()
{
  if (this->SelectionData && this->SelectionData->GetNumberOfArrays() > 0)
  {
    return this->SelectionData->GetAbstractArray(0);
  }
  return nullptr;
}

//----------------------------------------------------------------------------
void svtkSelectionNode::SetSelectionList(svtkAbstractArray* arr)
{
  if (!this->SelectionData)
  {
    this->SelectionData = svtkDataSetAttributes::New();
  }
  this->SelectionData->Initialize();
  this->SelectionData->AddArray(arr);
}

//----------------------------------------------------------------------------
void svtkSelectionNode::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "ContentType: ";
  if (this->GetContentType() < SelectionContent::NUM_CONTENT_TYPES)
  {
    os << svtkSelectionNode::GetContentTypeAsString(this->GetContentType());
  }
  else
  {
    os << "UNKNOWN";
  }
  os << endl;

  os << indent << "FieldType: ";
  if (this->GetFieldType() < SelectionField::NUM_FIELD_TYPES)
  {
    os << svtkSelectionNode::GetFieldTypeAsString(this->GetFieldType());
  }
  else
  {
    os << "UNKNOWN";
  }
  os << endl;

  os << indent << "Properties: " << (this->Properties ? "" : "(none)") << endl;
  if (this->Properties)
  {
    this->Properties->PrintSelf(os, indent.GetNextIndent());
  }
  os << indent << "SelectionData: " << (this->SelectionData ? "" : "(none)") << endl;
  if (this->SelectionData)
  {
    this->SelectionData->PrintSelf(os, indent.GetNextIndent());
  }
  os << indent << "QueryString: " << (this->QueryString ? this->QueryString : "nullptr") << endl;
}

//----------------------------------------------------------------------------
void svtkSelectionNode::ShallowCopy(svtkSelectionNode* input)
{
  if (!input)
  {
    return;
  }
  this->Initialize();
  this->Properties->Copy(input->Properties, 0);
  this->SelectionData->ShallowCopy(input->SelectionData);
  this->SetQueryString(input->GetQueryString());
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkSelectionNode::DeepCopy(svtkSelectionNode* input)
{
  if (!input)
  {
    return;
  }
  this->Initialize();
  this->Properties->Copy(input->Properties, 1);
  this->SelectionData->DeepCopy(input->SelectionData);
  this->SetQueryString(input->GetQueryString());
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkSelectionNode::SetContentType(int type)
{
  this->GetProperties()->Set(svtkSelectionNode::CONTENT_TYPE(), type);
}

//----------------------------------------------------------------------------
int svtkSelectionNode::GetContentType()
{
  if (this->GetProperties()->Has(svtkSelectionNode::CONTENT_TYPE()))
  {
    return this->GetProperties()->Get(svtkSelectionNode::CONTENT_TYPE());
  }
  return -1;
}

//----------------------------------------------------------------------------
const char* svtkSelectionNode::GetContentTypeAsString(int type)
{
  return svtkSelectionNode::ContentTypeNames[type];
}

//----------------------------------------------------------------------------
void svtkSelectionNode::SetFieldType(int type)
{
  this->GetProperties()->Set(svtkSelectionNode::FIELD_TYPE(), type);
}

//----------------------------------------------------------------------------
int svtkSelectionNode::GetFieldType()
{
  if (this->GetProperties()->Has(svtkSelectionNode::FIELD_TYPE()))
  {
    return this->GetProperties()->Get(svtkSelectionNode::FIELD_TYPE());
  }
  return -1;
}

//----------------------------------------------------------------------------
const char* svtkSelectionNode::GetFieldTypeAsString(int type)
{
  return svtkSelectionNode::FieldTypeNames[type];
}

//----------------------------------------------------------------------------
bool svtkSelectionNode::EqualProperties(svtkSelectionNode* other, bool fullcompare /*=true*/)
{
  if (!other)
  {
    return false;
  }

  svtkSmartPointer<svtkInformationIterator> iterSelf = svtkSmartPointer<svtkInformationIterator>::New();

  iterSelf->SetInformation(this->Properties);

  svtkInformation* otherProperties = other->GetProperties();
  for (iterSelf->InitTraversal(); !iterSelf->IsDoneWithTraversal(); iterSelf->GoToNextItem())
  {
    svtkInformationKey* key = iterSelf->GetCurrentKey();
    svtkInformationIntegerKey* ikey = svtkInformationIntegerKey::SafeDownCast(key);
    svtkInformationObjectBaseKey* okey = svtkInformationObjectBaseKey::SafeDownCast(key);
    if (ikey)
    {
      if (!otherProperties->Has(ikey) || this->Properties->Get(ikey) != otherProperties->Get(ikey))
      {
        return false;
      }
    }
    if (okey)
    {
      if (!otherProperties->Has(okey) || this->Properties->Get(okey) != otherProperties->Get(okey))
      {
        return false;
      }
    }
  }

  // Also check that array names match for certain content types
  // For VALUES and THRESHOLDS, names represent array where values come from.
  // For PEDIGREEIDS, names represent the domain of the selection.
  if (this->GetContentType() == svtkSelectionNode::VALUES ||
    this->GetContentType() == svtkSelectionNode::PEDIGREEIDS ||
    this->GetContentType() == svtkSelectionNode::THRESHOLDS)
  {
    int numArrays = other->SelectionData->GetNumberOfArrays();
    if (this->SelectionData->GetNumberOfArrays() != numArrays)
    {
      return false;
    }
    for (int a = 0; a < numArrays; ++a)
    {
      svtkAbstractArray* arr = this->SelectionData->GetAbstractArray(a);
      svtkAbstractArray* otherArr = other->SelectionData->GetAbstractArray(a);
      if (!arr->GetName() && otherArr->GetName())
      {
        return false;
      }
      if (arr->GetName() && !otherArr->GetName())
      {
        return false;
      }
      if (arr->GetName() && otherArr->GetName() && strcmp(arr->GetName(), otherArr->GetName()))
      {
        return false;
      }
    }
  }

  if (fullcompare)
  {
    return other->EqualProperties(this, false);
  }

  return true;
}

//----------------------------------------------------------------------------
void svtkSelectionNode::UnionSelectionList(svtkSelectionNode* other)
{
  int type = this->Properties->Get(CONTENT_TYPE());
  switch (type)
  {
    case GLOBALIDS:
    case PEDIGREEIDS:
    case VALUES:
    case INDICES:
    case LOCATIONS:
    case THRESHOLDS:
    case BLOCKS:
    {
      svtkDataSetAttributes* fd1 = this->GetSelectionData();
      svtkDataSetAttributes* fd2 = other->GetSelectionData();
      if (fd1->GetNumberOfArrays() != fd2->GetNumberOfArrays())
      {
        svtkErrorMacro(<< "Cannot take the union where the number of arrays do not match.");
      }
      for (int i = 0; i < fd1->GetNumberOfArrays(); i++)
      {
        svtkAbstractArray* aa1 = fd1->GetAbstractArray(i);
        svtkAbstractArray* aa2 = nullptr;
        if (i == 0 && type != VALUES && type != THRESHOLDS)
        {
          aa2 = fd2->GetAbstractArray(i);
        }
        else
        {
          aa2 = fd2->GetAbstractArray(aa1->GetName());
        }
        if (!aa2)
        {
          svtkErrorMacro(<< "Could not find array with name " << aa1->GetName()
                        << " in other selection.");
          return;
        }
        if (aa1->GetDataType() != aa2->GetDataType())
        {
          svtkErrorMacro(<< "Cannot take the union where selection list types "
                        << "do not match.");
          return;
        }
        if (aa1->GetNumberOfComponents() != aa2->GetNumberOfComponents())
        {
          svtkErrorMacro(<< "Cannot take the union where selection list number "
                        << "of components do not match.");
          return;
        }
        // If it is the same array, we are done.
        if (aa1 == aa2)
        {
          return;
        }
        int numComps = aa2->GetNumberOfComponents();
        svtkIdType numTuples = aa2->GetNumberOfTuples();
        for (svtkIdType j = 0; j < numTuples; j++)
        {
          // Avoid duplicates on single-component arrays.
          if (numComps != 1 || aa1->LookupValue(aa2->GetVariantValue(j)) == -1)
          {
            aa1->InsertNextTuple(j, aa2);
          }
        }
      }
      break;
    }
    case FRUSTUM:
    case USER:
    default:
    {
      svtkErrorMacro(<< "Do not know how to take the union of content type " << type << ".");
      return;
    }
  }
}

//----------------------------------------------------------------------------
void svtkSelectionNode::SubtractSelectionList(svtkSelectionNode* other)
{
  int type = this->Properties->Get(CONTENT_TYPE());
  switch (type)
  {
    case GLOBALIDS:
    case INDICES:
    case PEDIGREEIDS:
    {
      svtkDataSetAttributes* fd1 = this->GetSelectionData();
      svtkDataSetAttributes* fd2 = other->GetSelectionData();
      if (fd1->GetNumberOfArrays() != fd2->GetNumberOfArrays())
      {
        svtkErrorMacro(<< "Cannot take subtract selections if the number of arrays do not match.");
        return;
      }
      if (fd1->GetNumberOfArrays() != 1 || fd2->GetNumberOfArrays() != 1)
      {
        svtkErrorMacro(<< "Cannot subtract selections with more than one array.");
        return;
      }

      if (fd1->GetArray(0)->GetDataType() != SVTK_ID_TYPE ||
        fd2->GetArray(0)->GetDataType() != SVTK_ID_TYPE)
      {
        svtkErrorMacro(<< "Can only subtract selections with svtkIdTypeArray lists.");
        return;
      }

      svtkIdTypeArray* fd1_array = (svtkIdTypeArray*)fd1->GetArray(0);
      svtkIdTypeArray* fd2_array = (svtkIdTypeArray*)fd2->GetArray(0);

      if (fd1_array->GetNumberOfComponents() != 1 || fd2_array->GetNumberOfComponents() != 1)
      {
        svtkErrorMacro("Can only subtract selections with single component arrays.");
        return;
      }

      auto fd1Range = svtk::DataArrayValueRange<1>(fd1_array);
      auto fd2Range = svtk::DataArrayValueRange<1>(fd2_array);

      // make sure both arrays are sorted
      std::sort(fd1Range.begin(), fd1Range.end());
      std::sort(fd2Range.begin(), fd2Range.end());

      // set_difference result is bounded by the first set:
      std::vector<svtkIdType> result;
      result.resize(static_cast<std::size_t>(fd1Range.size()));

      auto diffEnd = std::set_difference(
        fd1Range.cbegin(), fd1Range.cend(), fd2Range.cbegin(), fd2Range.cend(), result.begin());
      result.erase(diffEnd, result.end());

      // Copy the result back into fd1_array:
      fd1_array->Reset();
      fd1_array->SetNumberOfTuples(static_cast<svtkIdType>(result.size()));
      fd1Range = svtk::DataArrayValueRange<1>(fd1_array);
      std::copy(result.cbegin(), result.cend(), fd1Range.begin());
      break;
    }
    case BLOCKS:
    case FRUSTUM:
    case LOCATIONS:
    case THRESHOLDS:
    case VALUES:
    case USER:
    default:
    {
      svtkErrorMacro(<< "Do not know how to subtract the given content type " << type << ".");
    }
  };
}

//----------------------------------------------------------------------------
svtkMTimeType svtkSelectionNode::GetMTime()
{
  svtkMTimeType mTime = this->MTime.GetMTime();
  svtkMTimeType propMTime;
  svtkMTimeType fieldMTime;
  if (this->Properties)
  {
    propMTime = this->Properties->GetMTime();
    mTime = (propMTime > mTime ? propMTime : mTime);
  }
  if (this->SelectionData)
  {
    fieldMTime = this->SelectionData->GetMTime();
    mTime = (fieldMTime > mTime ? fieldMTime : mTime);
  }
  return mTime;
}

//----------------------------------------------------------------------------
int svtkSelectionNode::ConvertSelectionFieldToAttributeType(int selectionField)
{
  switch (selectionField)
  {
    case svtkSelectionNode::CELL:
      return svtkDataObject::CELL;
    case svtkSelectionNode::POINT:
      return svtkDataObject::POINT;
    case svtkSelectionNode::FIELD:
      return svtkDataObject::FIELD;
    case svtkSelectionNode::VERTEX:
      return svtkDataObject::VERTEX;
    case svtkSelectionNode::EDGE:
      return svtkDataObject::EDGE;
    case svtkSelectionNode::ROW:
      return svtkDataObject::ROW;
    default:
      svtkGenericWarningMacro("Invalid selection field type: " << selectionField);
      return svtkDataObject::NUMBER_OF_ATTRIBUTE_TYPES;
  }
}

//----------------------------------------------------------------------------
int svtkSelectionNode::ConvertAttributeTypeToSelectionField(int attrType)
{
  switch (attrType)
  {
    case svtkDataObject::CELL:
      return svtkSelectionNode::CELL;
    case svtkDataObject::POINT:
      return svtkSelectionNode::POINT;
    case svtkDataObject::FIELD:
      return svtkSelectionNode::FIELD;
    case svtkDataObject::VERTEX:
      return svtkSelectionNode::VERTEX;
    case svtkDataObject::EDGE:
      return svtkSelectionNode::EDGE;
    case svtkDataObject::ROW:
      return svtkSelectionNode::ROW;
    default:
      svtkGenericWarningMacro("Invalid attribute type: " << attrType);
      return svtkSelectionNode::CELL;
  }
}
